"""Run: python examples/rename_and_cast.py (after installing the package)."""

import pandas as pd
import pandera.pandas as pa

from pandera_migration_planner import MigrationPlan

source = pa.DataFrameSchema({"old_count": pa.Column("float64")}, strict=True)
target = pa.DataFrameSchema(
    {"count": pa.Column("int64", pa.Check.ge(0))}, strict=True
)
plan = MigrationPlan(
    source, target, renames={"old_count": "count"}, casts={"count": "int64"}
)
data = pd.DataFrame({"old_count": [1.0, 2.0]})
assert plan.dry_run(data).ok
result = plan.apply(data)
assert result["count"].tolist() == [1, 2]
assert list(data.columns) == ["old_count"]
print(result)

bad = pd.DataFrame({"old_count": [1.0, 2.5]})
report = plan.dry_run(bad)
assert not report.ok
assert report.issues[0].code == "lossy"
assert report.issues[0].row_position == 1
print(report.issues[0])
