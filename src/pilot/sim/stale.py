"""Policy 9's advice: last semester's timetable mapped onto this semester's grid.

The stale baseline answers "what does the calendar advantage cost when the calendar is one
semester out of date?".  Last semester's sections are *not* this semester's sections - courses are
added, dropped, re-timed and re-sized - so the two tables cannot be joined on ``section_id``.  What
survives a semester is the shape of the grid, so the mapping is by ``week x weekday x period_start
x campus``:

1. read last semester's ``slots.csv`` with the same loader the current semester uses, applying the
   same college / campus filter;
2. explode every slot over its teaching weeks and aggregate the enrolment into seats per
   ``(week, weekday, period_start, campus)`` cell, with the seat-weighted mean session duration;
3. scale those seats by the current semester's ``adoption`` - the planner knows the take-up rate it
   is buying capacity for, it just does not know *which* sections will run;
4. feed the aggregate through the same analytic profile ``workload.calendar_forecast`` uses, with
   every perturbation knob at its no-error value: the error of this forecast is structural (the
   timetable moved), not injected.

The off-class component is deliberately *not* stale.  Background traffic is a per-student Poisson
process rather than a timetable artefact, so policy 9 reuses the current semester's off-class
forecast and the whole difference from policy 6 lives in the in-class term.
"""

import logging
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..timetable.parse import BellSchedule, parse_week_part
from .config import WorkloadConfig

logger = logging.getLogger(__name__)

_MIN_PER_WEEK = 7 * 1440
STALE_COLLEGE = "STALE"


def _hm_to_min(hm: str) -> int:
    """``"08:30"`` -> 510."""
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def _expand_weeks(text: str) -> Tuple[int, ...]:
    """``"1-9,11-18"`` -> the tuple of week numbers."""
    out = []
    for part in str(text).split(","):
        out.extend(parse_week_part(part))
    return tuple(sorted(set(out)))


@lru_cache(maxsize=8)
def _aggregate(path: str, college: Optional[str], campus: Optional[str]) -> pd.DataFrame:
    """Seats and mean duration per ``(week, weekday, period_start, campus)`` cell.

    Cached: the table is a property of the export and the slice, not of the run, and every cell of
    a sweep asks for the same one.
    """
    slots = pd.read_csv(path)
    if college:
        slots = slots[slots["college"] == college]
    if campus:
        slots = slots[slots["campus"] == campus]
    slots = slots[slots["enrollment"].fillna(0) > 0]
    if slots.empty:
        return pd.DataFrame(columns=["week", "weekday", "period_start", "campus", "seats",
                                     "bell_min_of_day", "duration_min"])
    bells = BellSchedule()
    rows = []
    for row in slots.itertuples(index=False):
        p0, p1 = int(row.period_start), int(row.period_end)
        start = _hm_to_min(bells.block_start(p0))
        end = _hm_to_min(bells.window(min(p1, len(bells.windows)))[1])
        seats = float(row.enrollment)
        for week in _expand_weeks(row.weeks):
            rows.append((week, int(row.weekday), p0, row.campus, seats, start,
                         max(0.0, end - start) * seats))
    agg = pd.DataFrame(rows, columns=["week", "weekday", "period_start", "campus", "seats",
                                      "bell_min_of_day", "duration_weighted"])
    keys = ["week", "weekday", "period_start", "campus", "bell_min_of_day"]
    out = agg.groupby(keys, as_index=False)[["seats", "duration_weighted"]].sum()
    out["duration_min"] = out["duration_weighted"] / out["seats"].clip(lower=1e-9)
    logger.debug("stale calendar: %d cells from %s", len(out), path)
    return out.drop(columns="duration_weighted")


def stale_sessions(cfg: WorkloadConfig, root: str = ".") -> pd.DataFrame:
    """Last semester's aggregate as a session table with this semester's ``load_sessions`` columns."""
    path = Path(cfg.stale_slots_csv)
    path = path if path.is_absolute() else Path(root) / path
    if not path.exists():
        logger.warning("stale timetable missing: %s (policy 9 will see an empty calendar)", path)
        return pd.DataFrame(columns=["session_id", "section_id", "college", "week", "weekday",
                                     "bell_min", "duration_min", "roster"])
    agg = _aggregate(str(path), cfg.college, cfg.campus)
    rows = []
    for i, week in enumerate(cfg.weeks):
        cell = agg[agg["week"] == week]
        for row in cell.itertuples(index=False):
            bell = i * _MIN_PER_WEEK + (int(row.weekday) - 1) * 1440 + int(row.bell_min_of_day)
            roster = float(row.seats) * cfg.adoption
            if roster <= 0.0 or not 0 <= bell < cfg.horizon_min:
                continue
            rows.append((0, -1, STALE_COLLEGE, int(week), int(row.weekday), int(bell),
                         max(cfg.period_minutes, float(row.duration_min)), roster))
    out = pd.DataFrame(rows, columns=["session_id", "section_id", "college", "week", "weekday",
                                      "bell_min", "duration_min", "roster"])
    out = out.sort_values("bell_min").reset_index(drop=True)
    out["session_id"] = np.arange(len(out))
    return out


def stale_forecast(cfg: WorkloadConfig, root: str, sync_pool: np.ndarray,
                   self_pool: np.ndarray) -> np.ndarray:
    """In-class ``lambda-hat`` of policy 9, on the same minute grid as the current forecast."""
    from .workload import calendar_forecast  # local: workload imports this module

    sessions = stale_sessions(cfg, root)
    if sessions.empty:
        return np.zeros(cfg.horizon_min, dtype=float)
    clean = replace(cfg, roster_noise=0.0, forecast_attendance=1.0, forecast_scale=1.0,
                    bell_shift_frac=0.0)
    lam = calendar_forecast(clean, sessions, np.random.default_rng(0), sync_pool, self_pool)
    logger.info("stale calendar: %d cells, %.0f forecast in-class requests", len(sessions),
                float(lam.sum()))
    return lam


__all__ = ["STALE_COLLEGE", "stale_forecast", "stale_sessions"]
