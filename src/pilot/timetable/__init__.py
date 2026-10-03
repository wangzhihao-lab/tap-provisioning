"""Institution-wide timetable: parsing, load statistics and figures (T2 / F1).

The teacher name and staff id columns of the export are personal data; they are dropped while the
file is read and never appear in any frame, file, log or figure produced by this package.

``plots`` is imported lazily so that the parsing and statistics side stays importable in
environments without matplotlib.
"""

import importlib
from typing import TYPE_CHECKING, Any

from .parse import (
    BellSchedule,
    TimetableConfig,
    align_weeks,
    count_classes,
    expand_weeks,
    explode_slots,
    format_weeks,
    is_general_elective,
    load_timetable,
    parse_time_code,
    parse_week_part,
)
from .stats import (
    ALL_CAMPUS,
    StatsConfig,
    T2Result,
    compute,
    concentration,
    counts,
    enrollment_summary,
    n_periods_distribution,
    peak_week,
    same_bell,
    session_start_load,
    session_starts,
    slot_load,
    t2_table,
    typical_week,
    week_totals,
    write_t2,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from . import plots

__all__ = [
    "ALL_CAMPUS",
    "BellSchedule",
    "StatsConfig",
    "T2Result",
    "TimetableConfig",
    "align_weeks",
    "compute",
    "concentration",
    "count_classes",
    "counts",
    "enrollment_summary",
    "expand_weeks",
    "explode_slots",
    "format_weeks",
    "is_general_elective",
    "load_timetable",
    "n_periods_distribution",
    "parse_time_code",
    "parse_week_part",
    "peak_week",
    "plots",
    "same_bell",
    "session_start_load",
    "session_starts",
    "slot_load",
    "t2_table",
    "typical_week",
    "week_totals",
    "write_t2",
]


def __getattr__(name: str) -> Any:
    """Import :mod:`pilot.timetable.plots` on first use (it needs matplotlib)."""
    if name == "plots":
        return importlib.import_module(f"{__name__}.plots")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
