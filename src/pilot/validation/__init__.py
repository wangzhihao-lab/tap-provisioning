"""Helpers for the validation-round results note (``scripts/sim/validation_report.py``).

``runs`` finds and loads the sweeps of the round and adds the per-week metrics; ``tables`` shapes
them into policies x timing-cell tables, comparisons and weekly learning curves; ``tuning`` reads
what the baseline tuning selected and whether its SLO floor was reachable.
"""

from .runs import (
    BASE_USD_PER_WEEK,
    CASE_ORDER,
    CASES,
    CELL_KEY,
    CELL_ORDER,
    CELLS,
    REALISTIC_CELLS,
    WEEK_HOURS,
    Case,
    Cell,
    RunFile,
    RunSpec,
    cell_of,
    discover,
    expected_runs,
    load_all,
    load_run,
    parse_run,
    same_rows,
    weekly,
    whole_run,
)
from .tables import (
    FEEDBACK_POLICIES,
    METRIC_TITLES,
    METRICS,
    POLICY_ORDER,
    SLO_SUFFIX,
    TIMETABLE_POLICIES,
    Group,
    assemble,
    best_of,
    cell_means,
    compare,
    frontier_row,
    groups_of,
    policy_sort_key,
    steady_state,
    weekly_table,
    wide,
)
from .tuning import discover_tuning, tuning_rows

__all__ = [
    "BASE_USD_PER_WEEK", "CASE_ORDER", "CASES", "CELL_KEY", "CELL_ORDER", "CELLS", "FEEDBACK_POLICIES",
    "METRIC_TITLES", "METRICS", "POLICY_ORDER", "REALISTIC_CELLS", "SLO_SUFFIX",
    "TIMETABLE_POLICIES", "WEEK_HOURS", "Case", "Cell", "Group", "RunFile", "RunSpec",
    "assemble", "best_of", "cell_means", "cell_of", "compare", "discover", "discover_tuning",
    "expected_runs", "frontier_row", "groups_of", "load_all", "load_run", "parse_run",
    "policy_sort_key", "same_rows", "steady_state", "tuning_rows", "weekly", "weekly_table", "whole_run",
    "wide",
]
