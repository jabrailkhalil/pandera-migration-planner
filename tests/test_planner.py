import math

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from pandera_migration_planner import MigrationError, MigrationPlan


def schema(
    dtype: str, name: str = "x", **kwargs: object
) -> pa.DataFrameSchema:
    return pa.DataFrameSchema({name: pa.Column(dtype, **kwargs)}, strict=True)


def plan(source: str, target: str, **kwargs: object) -> MigrationPlan:
    return MigrationPlan(
        schema(source), schema(target), casts={"x": target}, **kwargs
    )


def test_rename_then_cast_preserves_input_and_index() -> None:
    frame = pd.DataFrame({"old": [1.0, 2.0]}, index=["same", "same"])
    original = frame.copy(deep=True)
    migration = MigrationPlan(
        schema("float64", "old"),
        schema("int64", "new"),
        renames={"old": "new"},
        casts={"new": "int64"},
    )
    report = migration.dry_run(frame)
    assert report.ok
    assert report.source_valid and report.target_valid
    assert report.affected_columns == ("new",)
    assert report.affected_row_positions == (0, 1)
    expected = pd.DataFrame(
        {"new": pd.Series([1, 2], dtype="int64").to_numpy()}, index=frame.index
    )
    migrated = migration.apply(frame)
    assert_frame_equal(migrated, expected)
    migrated.iloc[0, 0] = 99
    assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "source,target,values",
    [
        ("int64", "int8", [-128, 0, 127]),
        ("uint64", "int64", [0, 2**63 - 1]),
        ("int64", "float64", [-(2**53), 0, 2**53]),
        ("float64", "float32", [0.0, 0.5, -128.0]),
        ("float64", "int64", [-128.0, 0.0, 127.0]),
        ("Int64", "Float64", [None, 0, 12]),
        ("Float64", "Int64", [None, 0.0, 12.0]),
        ("int64", "int64", []),
        ("float64", "float32", [math.inf, -math.inf]),
    ],
)
def test_exact_casts(source: str, target: str, values: list) -> None:
    frame = pd.DataFrame({"x": pd.Series(values, dtype=source)})
    migration = MigrationPlan(
        schema(source, nullable=True),
        schema(target, nullable=True),
        casts={"x": target},
    )
    expected = pd.DataFrame({"x": pd.Series(values, dtype=target)})
    assert_frame_equal(migration.apply(frame), expected)


@pytest.mark.parametrize(
    "source,target,values,position",
    [
        ("float64", "int64", [1.0, 1.75], 1),
        ("int64", "float64", [0, 2**53 + 1], 1),
        ("uint64", "float64", [2**64 - 1], 0),
        ("float64", "float32", [0.1, 0.5], 0),
    ],
)
def test_lossy_cast_is_blocked_with_exact_row(
    source: str,
    target: str,
    values: list,
    position: int,
) -> None:
    frame = pd.DataFrame({"x": pd.Series(values, dtype=source)})
    frame.index = ["repeated"] * len(frame)
    before = frame.copy(deep=True)
    migration = plan(source, target)
    report = migration.dry_run(frame)
    assert not report.ok and report.target_valid is None
    assert [(i.code, i.row_position, i.index) for i in report.issues] == [
        ("lossy", position, "'repeated'")
    ]
    with pytest.raises(MigrationError) as error:
        migration.apply(frame)
    assert error.value.report.issues == report.issues
    assert_frame_equal(frame, before)


def test_negative_control_plain_pandas_silently_loses_values() -> None:
    values = pd.Series([2**53 + 1], dtype="int64")
    rounded = values.astype("float64")
    # Python integers give an independent oracle. NumPy mixed comparison
    # promotes the integer first and would incorrectly report equality.
    assert int(rounded.iloc[0]) == 2**53
    assert int(rounded.iloc[0]) != int(values.iloc[0])
    assert not plan("int64", "float64").dry_run(values.to_frame("x")).ok


@pytest.mark.parametrize(
    "source,target,values,expected",
    [
        ("float64", "int64", [1.75, -1.75], [1, -1]),
        ("int64", "float64", [2**53 + 1], [float(2**53)]),
        ("float64", "float32", [0.1], [float(np.float32(0.1))]),
    ],
)
def test_explicit_lossy_opt_in(
    source: str,
    target: str,
    values: list,
    expected: list,
) -> None:
    frame = pd.DataFrame({"x": pd.Series(values, dtype=source)})
    assert_frame_equal(
        plan(source, target, allow_lossy=True).apply(frame),
        pd.DataFrame({"x": pd.Series(expected, dtype=target)}),
    )


@pytest.mark.parametrize("allow_lossy", [False, True])
@pytest.mark.parametrize(
    "source,target,value",
    [
        ("int64", "int8", 128),
        ("int64", "uint64", -1),
        ("uint64", "int64", 2**63),
        ("float64", "int64", float(2**63)),
        ("float64", "int64", math.inf),
        ("float64", "float32", 1e100),
    ],
)
def test_overflow_never_wraps(
    source: str,
    target: str,
    value: object,
    allow_lossy: bool,
) -> None:
    frame = pd.DataFrame({"x": pd.Series([value], dtype=source)})
    report = plan(source, target, allow_lossy=allow_lossy).dry_run(frame)
    assert not report.ok
    assert report.issues[0].code == "overflow"
    assert report.issues[0].row_position == 0


@pytest.mark.parametrize(
    "source,target,value",
    [
        ("float64", "int64", np.nan),
        ("Float64", "int64", None),
        ("Int64", "int64", None),
    ],
)
def test_missing_value_cannot_be_cast_to_nonnullable_integer(
    source: str,
    target: str,
    value: object,
) -> None:
    frame = pd.DataFrame({"x": pd.Series([1, value], dtype=source)})
    migration = MigrationPlan(
        schema(source, nullable=True), schema(target), casts={"x": target}
    )
    report = migration.dry_run(frame)
    assert not report.ok
    assert [(i.code, i.row_position) for i in report.issues] == [
        ("conversion", 1)
    ]


@pytest.mark.parametrize(
    "source,target", [("object", "int64"), ("bool", "int64")]
)
def test_unsupported_source_is_not_guessed(source: str, target: str) -> None:
    frame = pd.DataFrame({"x": pd.Series([1], dtype=source)})
    report = plan(source, target).dry_run(frame)
    assert not report.ok and report.issues[0].code == "unsupported"


@pytest.mark.parametrize(
    "dtype",
    [
        "object",
        "str",
        "bool",
        "complex64",
        "datetime64[ns]",
        "float16",
        "nonsense",
    ],
)
def test_unsupported_target_rejected_at_construction(dtype: str) -> None:
    with pytest.raises(ValueError, match="Unsupported numeric dtype"):
        MigrationPlan(
            schema("int64"), pa.DataFrameSchema(), casts={"x": dtype}
        )


@pytest.mark.parametrize(
    "renames,casts",
    [
        ({"missing": "new"}, {}),
        ({"x": "y"}, {}),
        ({"x": "z", "y": "z"}, {}),
        ({"x": "z"}, {"x": "int64"}),
        ({}, {"missing": "int64"}),
    ],
)
def test_missing_or_colliding_columns(renames: dict, casts: dict) -> None:
    frame = pd.DataFrame({"x": [1], "y": [2]})
    migration = MigrationPlan(
        pa.DataFrameSchema(),
        pa.DataFrameSchema(),
        renames=renames,
        casts=casts,
    )
    assert migration.dry_run(frame).issues[0].code == "columns"


def test_simultaneous_rename_cycle_is_explicit() -> None:
    frame = pd.DataFrame({"x": [1], "y": [2]})
    migration = MigrationPlan(
        pa.DataFrameSchema(),
        pa.DataFrameSchema(),
        renames={"x": "y", "y": "x"},
    )
    assert_frame_equal(
        migration.apply(frame), pd.DataFrame({"y": [1], "x": [2]})
    )


def test_noop_returns_independent_copy() -> None:
    frame = pd.DataFrame({"x": [1]})
    migration = MigrationPlan(schema("int64"), schema("int64"))
    report = migration.dry_run(frame)
    assert report.ok and report.affected_row_positions == ()
    result = migration.apply(frame)
    result.iloc[0, 0] = 9
    assert frame.iloc[0, 0] == 1


@pytest.mark.parametrize("phase", ["source", "target"])
def test_schema_failures_are_reported_without_modifying_input(
    phase: str,
) -> None:
    good = schema("int64")
    bad = pa.DataFrameSchema({"x": pa.Column("int64", pa.Check.gt(10))})
    migration = MigrationPlan(
        bad if phase == "source" else good, bad if phase == "target" else good
    )
    frame = pd.DataFrame({"x": [1, 2]})
    before = frame.copy(deep=True)
    report = migration.dry_run(frame)
    assert not report.ok
    assert all(i.phase == phase for i in report.issues)
    assert report.source_valid == (phase == "target")
    assert report.target_valid is (False if phase == "target" else None)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coerce": True},
        {"strict": "filter"},
        {"add_missing_columns": True},
        {"drop_invalid_rows": True},
        {"parsers": pa.Parser(lambda df: df)},
    ],
)
def test_mutating_dataframe_schemas_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="only validate"):
        MigrationPlan(pa.DataFrameSchema(**kwargs), schema("int64"))


@pytest.mark.parametrize(
    "kwargs",
    [{"coerce": True}, {"default": 0}, {"parsers": pa.Parser(lambda x: x)}],
)
def test_mutating_column_schemas_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="only validate"):
        MigrationPlan(schema("int64", **kwargs), schema("int64"))


def test_index_coercion_rejected() -> None:
    with pytest.raises(ValueError, match="only validate"):
        MigrationPlan(
            pa.DataFrameSchema(index=pa.Index(int, coerce=True)),
            pa.DataFrameSchema(),
        )


def test_schema_modified_after_plan_construction_is_checked_again() -> None:
    migration = plan("int64", "int64")
    migration.source_schema.columns["x"].coerce = True
    with pytest.raises(ValueError, match="only validate"):
        migration.dry_run(pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize("columns", [["x", "x"], [0, "x"]])
def test_ambiguous_column_names_rejected(columns: list) -> None:
    migration = MigrationPlan(pa.DataFrameSchema(), pa.DataFrameSchema())
    with pytest.raises(ValueError):
        migration.dry_run(pd.DataFrame([[1, 2]], columns=columns))


@given(st.lists(st.integers(min_value=-128, max_value=127), max_size=40))
def test_property_int8_exact_and_reversible(values: list[int]) -> None:
    frame = pd.DataFrame({"x": pd.Series(values, dtype="int64")})
    narrowed = plan("int64", "int8").apply(frame)
    assert narrowed["x"].tolist() == values
    assert_frame_equal(plan("int8", "int64").apply(narrowed), frame)


@given(st.integers(min_value=2**53 + 1, max_value=2**63 - 1))
def test_property_integer_to_float_uses_independent_exact_oracle(
    value: int,
) -> None:
    report = plan("int64", "float64").dry_run(
        pd.DataFrame({"x": pd.Series([value], dtype="int64")})
    )
    assert report.ok == (int(float(value)) == value)


def test_all_validation_runs_even_when_disabled_by_application() -> None:
    from pandera.config import ValidationDepth, config_context

    bad = pa.DataFrameSchema({"x": pa.Column("int64", pa.Check.gt(10))})
    migration = MigrationPlan(schema("int64"), bad)
    with config_context(
        validation_enabled=False, validation_depth=ValidationDepth.SCHEMA_ONLY
    ):
        report = migration.dry_run(pd.DataFrame({"x": [1]}))
    assert not report.ok
    assert report.source_valid and report.target_valid is False


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_lossy_opt_in_requires_boolean(value: object) -> None:
    with pytest.raises(TypeError, match="explicit boolean"):
        plan("int64", "float64", allow_lossy=value)


@pytest.mark.parametrize(
    "renames,casts",
    [
        ({1: "x"}, {}),
        ({"x": 1}, {}),
        ({}, {1: "int64"}),
        ({}, {"x": None}),
    ],
)
def test_invalid_operation_definitions(renames: dict, casts: dict) -> None:
    with pytest.raises(ValueError):
        MigrationPlan(
            pa.DataFrameSchema(),
            pa.DataFrameSchema(),
            renames=renames,
            casts=casts,
        )


def test_constructor_copies_operation_mappings() -> None:
    casts = {"x": "int64"}
    migration = MigrationPlan(schema("float64"), schema("int64"), casts=casts)
    casts["x"] = "int8"
    assert migration.casts == {"x": "int64"}


def test_invalid_input_type_is_explicit() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        plan("int64", "int64").dry_run([1, 2])


@pytest.mark.parametrize("side", ["source", "target"])
def test_invalid_schema_type_is_explicit(side: str) -> None:
    with pytest.raises(TypeError, match="DataFrameSchema"):
        MigrationPlan(
            None if side == "source" else schema("int64"),
            None if side == "target" else schema("int64"),
        )


def test_multiindex_coercion_rejected() -> None:
    with pytest.raises(ValueError, match="only validate"):
        MigrationPlan(
            pa.DataFrameSchema(
                index=pa.MultiIndex(
                    [pa.Index(int), pa.Index(int, coerce=True)]
                )
            ),
            pa.DataFrameSchema(),
        )


def test_invalid_cast_still_reports_affected_rows() -> None:
    report = plan("int64", "int8").dry_run(pd.DataFrame({"x": [1, 999]}))
    assert report.affected_row_positions == (0, 1)
    assert report.issues[0].row_position == 1


def test_nullable_missing_float_preserved() -> None:
    migration = MigrationPlan(
        schema("float64", nullable=True),
        schema("Float32", nullable=True),
        casts={"x": "Float32"},
    )
    frame = pd.DataFrame({"x": [0.5, np.nan]})
    assert_frame_equal(
        migration.apply(frame),
        pd.DataFrame({"x": pd.Series([0.5, None], dtype="Float32")}),
    )


def test_float_underflow_blocked_without_opt_in() -> None:
    frame = pd.DataFrame({"x": [1e-100]})
    assert plan("float64", "float32").dry_run(frame).issues[0].code == "lossy"
    assert (
        plan("float64", "float32", allow_lossy=True).apply(frame).iat[0, 0]
        == 0
    )


def test_multiple_conversion_errors_collected() -> None:
    migration = MigrationPlan(
        pa.DataFrameSchema(),
        pa.DataFrameSchema(),
        casts={"x": "int8", "y": "int8"},
    )
    frame = pd.DataFrame({"x": [999, 1], "y": [1, 999]})
    report = migration.dry_run(frame)
    assert [(i.column, i.row_position) for i in report.issues] == [
        ("x", 0),
        ("y", 1),
    ]


def test_structure_failure_is_not_success() -> None:
    migration = MigrationPlan(pa.DataFrameSchema(), schema("int64", "missing"))
    report = migration.dry_run(pd.DataFrame({"x": [1]}))
    assert not report.ok and report.target_valid is False
    assert all(i.phase == "target" for i in report.issues)
