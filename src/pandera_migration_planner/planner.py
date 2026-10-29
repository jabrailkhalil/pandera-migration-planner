"""A small, in-memory planner; schemas validate and never coerce data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.config import ValidationDepth, config_context


@dataclass(frozen=True)
class Issue:
    phase: str
    code: str
    column: str | None
    row_position: int | None
    index: str | None
    value: str | None
    message: str


@dataclass(frozen=True)
class Report:
    source_valid: bool
    target_valid: bool | None
    issues: tuple[Issue, ...]
    affected_columns: tuple[str, ...]
    affected_row_positions: tuple[int, ...]
    _result: Any = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.source_valid and self.target_valid is True


class MigrationError(ValueError):
    def __init__(self, report: Report) -> None:
        self.report = report
        super().__init__(f"Migration rejected: {len(report.issues)} issue(s)")


def _validation_only(schema: Any) -> None:
    """Reject implicit transformations, including those on index schemas."""
    for name in (
        "coerce",
        "parsers",
        "drop_invalid_rows",
        "add_missing_columns",
    ):
        if getattr(schema, name, False):
            raise ValueError(f"Schemas must only validate: {name} is enabled")
    if getattr(schema, "strict", False) == "filter":
        raise ValueError("Schemas must only validate: strict='filter'")
    for name in ("default", "on_missing", "on_missing_columns"):
        if getattr(schema, name, None) is not None:
            raise ValueError(f"Schemas must only validate: {name} is set")
    for column in getattr(schema, "columns", {}).values():
        _validation_only(column)
    if getattr(schema, "index", None) is not None:
        _validation_only(schema.index)
    for index in getattr(schema, "indexes", []):
        _validation_only(index)


def _numpy_dtype(dtype: Any) -> Any:
    dtype = pd.api.types.pandas_dtype(dtype)
    return np.dtype(getattr(dtype, "numpy_dtype", dtype))


def _numeric_dtype(dtype: Any) -> Any:
    try:
        result = _numpy_dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported numeric dtype: {dtype}") from exc
    if result.kind not in "iu" and not (
        result.kind == "f" and result.itemsize in (4, 8)
    ):
        raise ValueError(f"Unsupported numeric dtype: {dtype}")
    return result


def _exact(value: Any) -> Any:
    """Avoid NumPy's lossy int64/float64 comparison promotion."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value), 1
    value = float(value)
    if math.isinf(value):
        return value
    return value.as_integer_ratio()


def _validate(frame: Any, schema: Any, phase: str) -> list[Issue]:
    try:
        with config_context(
            validation_enabled=True,
            validation_depth=ValidationDepth.SCHEMA_AND_DATA,
        ):
            schema.validate(frame.copy(deep=True), lazy=True)
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as exc:
        cases = exc.failure_cases
        if cases is None or cases.empty:
            return [Issue(phase, "schema", None, None, None, None, str(exc))]
        return [
            Issue(
                phase,
                "schema",
                str(row["column"]) if "column" in row else None,
                None,
                repr(row["index"]) if "index" in row else None,
                repr(row["failure_case"]),
                str(row.get("check", exc)),
            )
            for _, row in cases.iterrows()
        ]
    return []


class MigrationPlan:
    """Apply declared renames, then numeric casts, to an independent copy.

    Cast keys refer to names *after* renaming. Lossy casts require an explicit
    opt-in; overflow and invalid conversions are always rejected. Custom schema
    checks must be pure. Pandera parsers and coercion are rejected.
    """

    def __init__(
        self,
        source_schema: pa.DataFrameSchema,
        target_schema: pa.DataFrameSchema,
        *,
        renames: Mapping[str, str] | None = None,
        casts: Mapping[str, str] | None = None,
        allow_lossy: bool = False,
    ) -> None:
        if not isinstance(source_schema, pa.DataFrameSchema) or not isinstance(
            target_schema, pa.DataFrameSchema
        ):
            raise TypeError("Both schemas must be pandas DataFrameSchema")
        if not isinstance(allow_lossy, bool):
            raise TypeError("allow_lossy must be an explicit boolean")
        _validation_only(source_schema)
        _validation_only(target_schema)
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.renames = dict(renames or {})
        self.casts = dict(casts or {})
        if any(not isinstance(x, str) for x in self.renames):
            raise ValueError("Rename keys must be strings")
        if any(not isinstance(x, str) for x in self.renames.values()):
            raise ValueError("Rename destinations must be strings")
        if any(not isinstance(x, str) for x in self.casts):
            raise ValueError("Cast keys must be strings")
        if any(not isinstance(x, str) for x in self.casts.values()):
            raise ValueError("Cast dtypes must be explicit strings")
        for dtype in self.casts.values():
            _numeric_dtype(dtype)
        self.allow_lossy = allow_lossy

    def dry_run(self, frame: pd.DataFrame) -> Report:
        """Validate and simulate the complete plan without changing the input.

        Target validation is skipped (None) when input validation or a declared
        transformation fails. Row positions in conversion diagnostics are zero
        based and remain unambiguous when index labels repeat.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        _validation_only(self.source_schema)
        _validation_only(self.target_schema)
        if not frame.columns.is_unique:
            raise ValueError("Duplicate input column names are unsupported")
        if any(not isinstance(x, str) for x in frame.columns):
            raise ValueError("Input column names must be strings")
        issues = _validate(frame, self.source_schema, "source")
        if issues:
            return Report(False, None, tuple(issues), (), ())
        missing = set(self.renames) - set(frame.columns)
        names = [self.renames.get(c, c) for c in frame.columns]
        missing_casts = set(self.casts) - set(names)
        if missing or missing_casts or len(names) != len(set(names)):
            message = (
                f"Missing rename columns: {sorted(missing)}; missing cast "
                f"columns: {sorted(missing_casts)}; "
                "destinations must be unique"
            )
            issue = Issue("plan", "columns", None, None, None, None, message)
            return Report(True, None, (issue,), (), ())
        result = frame.rename(columns=self.renames).copy(deep=True)
        affected = tuple(dict.fromkeys([*self.renames.values(), *self.casts]))
        rows: set[int] = set(range(len(frame))) if self.renames else set()
        for column, dtype in self.casts.items():
            series = result[column]
            if series.dtype != pd.api.types.pandas_dtype(dtype):
                rows.update(range(len(series)))
            try:
                _numeric_dtype(series.dtype)
            except ValueError as exc:
                issues.append(
                    Issue(
                        "cast",
                        "unsupported",
                        column,
                        None,
                        None,
                        None,
                        str(exc),
                    )
                )
                continue
            target = _numeric_dtype(dtype)
            for pos, value in enumerate(series):
                if pd.isna(value):
                    continue
                overflow = False
                if target.kind in "iu":
                    bounds = np.iinfo(target)
                    overflow = not (
                        math.isfinite(value)
                        and bounds.min <= int(value) <= bounds.max
                    )
                elif math.isfinite(value):
                    overflow = abs(float(value)) > float(np.finfo(target).max)
                if overflow:
                    issues.append(
                        self._issue(
                            series,
                            column,
                            pos,
                            "overflow",
                            f"Value is outside {dtype}'s finite range",
                        )
                    )
            if any(x.column == column for x in issues):
                continue
            try:
                converted = series.astype(dtype)
            except (ValueError, TypeError, OverflowError):
                # Repeat only failed conversions one row at a time to locate
                # invalid input. Never fall back to errors='ignore' or coerce.
                for pos in range(len(series)):
                    try:
                        series.iloc[pos : pos + 1].astype(dtype)
                    except (ValueError, TypeError, OverflowError) as exc:
                        issues.append(
                            self._issue(
                                series,
                                column,
                                pos,
                                "conversion",
                                str(exc),
                            )
                        )
                if not any(x.column == column for x in issues):
                    issues.append(
                        Issue(
                            "cast",
                            "conversion",
                            column,
                            None,
                            None,
                            None,
                            f"Column conversion to {dtype} failed",
                        )
                    )
                continue
            for pos, (old, new) in enumerate(
                zip(series, converted, strict=True)
            ):
                if _exact(old) != _exact(new):
                    rows.add(pos)
                    if not self.allow_lossy:
                        issues.append(
                            self._issue(
                                series,
                                column,
                                pos,
                                "lossy",
                                f"Conversion to {dtype} changes the value",
                            )
                        )
            result[column] = converted
        if issues:
            return Report(
                True, None, tuple(issues), affected, tuple(sorted(rows))
            )
        issues = _validate(result, self.target_schema, "target")
        return Report(
            True,
            not issues,
            tuple(issues),
            affected,
            tuple(sorted(rows)),
            result if not issues else None,
        )

    @staticmethod
    def _issue(
        series: Any,
        column: str,
        pos: int,
        code: str,
        message: str,
    ) -> Issue:
        return Issue(
            "cast",
            code,
            column,
            pos,
            repr(series.index[pos]),
            repr(series.iloc[pos]),
            message,
        )

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return validated migrated data, or raise with the dry-run report."""
        report = self.dry_run(frame)
        if not report.ok:
            raise MigrationError(report)
        return report._result
