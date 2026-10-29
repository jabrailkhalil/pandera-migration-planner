# Pandera migration planner

I am developing a small, independent companion to Pandera for explicit pandas
column migrations. I started this prototype after the discussion in
[Pandera #2626](https://github.com/unionai-oss/pandera/discussions/2626).
It is an early prototype, with no affiliation or endorsement implied.

I keep the migration policy outside Pandera: two `DataFrameSchema` objects
validate the input and output, and explicit mappings declare the operations.
The first version handles column renames and numeric casts in memory.

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -e '.[dev]'
python examples/rename_and_cast.py
python -m pytest -q
```

I have not published this package to PyPI. Install it from this repository.

```python
import pandas as pd
import pandera.pandas as pa
from pandera_migration_planner import MigrationPlan

source = pa.DataFrameSchema({"old_count": pa.Column("float64")}, strict=True)
target = pa.DataFrameSchema({"count": pa.Column("int64")}, strict=True)
plan = MigrationPlan(
    source,
    target,
    renames={"old_count": "count"},
    casts={"count": "int64"},  # names AFTER renaming
)

data = pd.DataFrame({"old_count": [1.0, 2.0]})
report = plan.dry_run(data)
assert report.ok
result = plan.apply(data)  # new DataFrame; data is unchanged
```

I do not infer renames from a schema diff. Renames happen simultaneously, so an
explicit swap is possible; missing columns and destination collisions fail.
Input column labels must be unique strings. Row index labels may repeat.

`dry_run` executes the transformations on a copy and returns:

- `source_valid`, `target_valid` and `ok`. Target validation is `None` if an
  earlier stage failed, rather than a claim that the target was checked.
- Issues with phase, code, column, value and, for cast failures, zero-based row
  position and the original index label. Pandera validation diagnostics retain
  their index labels; I do not invent positions for ambiguous duplicate labels.
- Affected columns and row positions. A renamed column or changed dtype affects
  every row, even when its numeric value stays the same.

`apply` runs the checks again and returns a new frame, or raises `MigrationError`
with the report. It does not write files or overwrite datasets.

## Numeric conversion policy

I support NumPy integer types, float32/float64, and pandas nullable integer and
floating types. I reject object/string, boolean, datetime, complex and float16
casts. This prototype does not parse arbitrary strings as numbers.

I compare finite values using exact integer ratios. Ordinary mixed NumPy
integer/float comparison can hide precision loss: `2**53 + 1` rounds down when
cast to float64. Fractional truncation, underflow and precision loss fail by
default. Set `allow_lossy=True` explicitly to accept those changes. Overflow,
integer wrapping, missing values in nonnullable integers, and failed target
validation remain errors with that option enabled. Infinities may be preserved
by float-to-float casts; signed zero has numeric equality with zero.

I require validation-only schemas. Coercion, parsers, defaults, row dropping,
column insertion and `strict="filter"` are rejected, including index coercion.
The planner enables both schema and data validation within its validation calls.
Custom checks must be pure; pandas deep copies do not recursively clone Python
objects stored in object columns. I do not sandbox user-supplied checks.

## Scope and development

I have not implemented schema inference, schema serialization, inferred column
deletions, nested data, distributed execution, file transactions or a UI.
The implementation scans values in Python to make the precision policy explicit;
I have not benchmarked it for large datasets.

```sh
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m pytest -q
python -m build
```

I use regression cases for precision, overflow, nulls, collisions, duplicate row
labels, schema failures and input preservation, plus property tests with an
independent Python integer oracle. The test matrix and its results are visible
in [Actions](https://github.com/jabrailkhalil/pandera-migration-planner/actions).
This tests the companion package, not Pandera's complete upstream backend suite.
