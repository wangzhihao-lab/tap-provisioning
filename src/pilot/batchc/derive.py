"""Base-tier arithmetic of the batch C sweeps: the cost component above the always-on floor.

The batch C scenario deploys one always-on base replica (``PolicyConfig.min_replicas = 1``) that
every policy pays for, the offline optimum included.  That replica is a *constant*: it adds the
same amount to the numerator and the denominator of ``cost_ratio``, so every policy's total-cost
ratio is pulled towards 1 and the differences that the paper is about get compressed.  The report
therefore works on the component *above* the base tier:

* ``base_usd = M * c_run * minutes`` and ``base_hours = M * minutes / 60`` with ``minutes`` the
  horizon the row covers - ``10080 * weeks`` for a whole-run row (``week = 0``) and ``10080`` for a
  per-teaching-week row (``metrics.weekly_rows`` cuts weeks of ``7 * 1440`` minutes);
* ``offline_total`` is the ``total_cost`` of the ``policy = "offline"`` row of the *same* cell, seed
  and week.  It is not the ``offline_cost`` column, which is the whole-run optimum and would be the
  wrong denominator for a weekly row - that column is only the fallback for a sweep that was run
  without the offline policy, and then only for its whole-run rows;
* ``excess_usd = total_cost - offline_total`` is what the policy paid over the optimum, in USD, and
  is defined whether or not there is a floor;
* ``attr_cost = total_cost - base_usd`` and ``attr_ratio = attr_cost / (offline_total - base_usd)``
  are the attributable cost and the attributable ratio.  The ratio is NaN when the denominator is
  at or below the floor (an optimum that sits exactly on the base tier has nothing above it).

The cell key is read off the header rather than hard-coded: ``sweeps.run_sweep`` writes each row as
``{**params, **shared, ...}``, so a sweep's own grid parameters are exactly the columns that come
before ``sweep`` (the first key of ``shared``).  That is ``sigma`` for the synchrony sweep,
``lead_tier / lead_min / commit_block_min / billing_block_min`` for the lead sweep and
``error_knob / error_value / forecast_scale / bell_shift_frac / p_cancel / price_ratio`` for the
two-axis epsilon sweep.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MINUTES_PER_WEEK: int = 7 * 1440  # metrics._MIN_PER_WEEK: the week a weekly row covers
R1_WEEKS: Tuple[int, ...] = (1, 5)  # novel timetable: week 1 has no history, week 5 the freshman step
DEFAULT_MIN_REPLICAS: int = 1  # batch C runs with --min-replicas 1
DEFAULT_C_RUN: float = 0.0333  # USD per replica-minute (2 USD/h nominal)
OFFLINE_POLICY: str = "offline"
SWEEP_COL: str = "sweep"
RATIO_FLOOR: float = 1e-9  # a denominator at or below this leaves ``attr_ratio`` undefined
DERIVED_COLUMNS: Tuple[str, ...] = ("run_minutes", "base_usd", "base_hours", "offline_total",
                                    "excess_usd", "attr_cost", "attr_ratio", "attr_gpu_hours")


def cell_key_columns(frame: pd.DataFrame) -> List[str]:
    """The sweep's own grid parameters: the columns that come before ``sweep``.

    An empty list means the frame carries no ``sweep`` column (a hand-made frame, or a CSV from
    before the column existed); every row is then treated as belonging to one implicit cell.
    """
    if SWEEP_COL not in frame.columns:
        logger.warning("no '%s' column: treating the frame as a single cell", SWEEP_COL)
        return []
    return list(frame.columns[: frame.columns.get_loc(SWEEP_COL)])


def _weeks_in_run(frame: pd.DataFrame) -> pd.Series:
    """Teaching weeks per row, from the ``weeks`` column or, failing that, the week labels."""
    if "weeks" in frame.columns:
        return pd.to_numeric(frame["weeks"], errors="coerce")
    labels = pd.to_numeric(frame.get("week", pd.Series(dtype=float)), errors="coerce")
    count = int(labels[labels > 0].nunique()) or 1
    logger.warning("no 'weeks' column: assuming %d teaching weeks from the week labels", count)
    return pd.Series(float(count), index=frame.index, dtype=float)


def run_minutes(frame: pd.DataFrame) -> pd.Series:
    """Minutes each row covers: the whole horizon for ``week = 0``, one week otherwise."""
    weeks = _weeks_in_run(frame)
    whole = float(MINUTES_PER_WEEK) * weeks
    if "week" not in frame.columns:
        logger.warning("no 'week' column: treating every row as a whole-run row")
        return whole
    week = pd.to_numeric(frame["week"], errors="coerce").fillna(0.0)
    return whole.where(week == 0, float(MINUTES_PER_WEEK))


def _whole_run_offline_cost(frame: pd.DataFrame) -> pd.Series:
    """``offline_cost`` on the whole-run rows, NaN on the weekly ones.

    The fallback for a sweep that was run without the ``offline`` policy - batch C's SLO-matched
    pair is ``--policies reactive velocity`` - but still carries the optimum of its own stream in
    the shared block: ``run_sweep`` computes the oracle for the cost ratio whatever the policy list
    says, and for a whole-run row that column *is* the offline row's ``total_cost``.  A weekly row
    has no such column and stays NaN rather than being scored against the whole run.
    """
    optimum = pd.to_numeric(frame["offline_cost"], errors="coerce")
    if "week" not in frame.columns:
        return optimum
    week = pd.to_numeric(frame["week"], errors="coerce").fillna(0.0)
    return optimum.where(week == 0)


def offline_total(frame: pd.DataFrame, cell_columns: Optional[Sequence[str]] = None) -> pd.Series:
    """``total_cost`` of the offline optimum of each row's own (cell, seed, week).

    The oracle row is matched on the cell key plus ``seed`` and ``week``, so a weekly row is scored
    against the whole-run optimum *restricted to that week* - which is what ``weekly_rows`` writes
    for ``policy = "offline"`` - and not against the whole-run total.  A sweep run without the
    offline policy at all falls back to its ``offline_cost`` column, whole-run rows only.
    """
    cols = list(cell_key_columns(frame) if cell_columns is None else cell_columns)
    keys = cols + [c for c in ("seed", "week") if c in frame.columns]
    if "policy" not in frame.columns or "total_cost" not in frame.columns:
        logger.warning("no policy/total_cost column: the offline optimum cannot be matched")
        return pd.Series(np.nan, index=frame.index, dtype=float)
    oracle = frame[frame["policy"].astype(str) == OFFLINE_POLICY]
    if oracle.empty and "offline_cost" in frame.columns:
        logger.warning("no '%s' rows: the whole-run rows fall back to the 'offline_cost' column "
                       "and the weekly rows stay NaN", OFFLINE_POLICY)
        return _whole_run_offline_cost(frame)
    if oracle.empty:
        logger.warning("no '%s' rows: excess_usd and attr_ratio stay NaN", OFFLINE_POLICY)
        return pd.Series(np.nan, index=frame.index, dtype=float)
    if not keys:
        return pd.Series(float(pd.to_numeric(oracle["total_cost"]).mean()), index=frame.index)
    grouped = oracle.groupby(keys, dropna=False, as_index=False)["total_cost"].agg(["mean", "size"])
    if int(grouped["size"].max()) > 1:
        logger.warning("more than one offline row per (cell, seed, week); averaging them")
    lookup = grouped.drop(columns="size").rename(columns={"mean": "offline_total"})
    merged = frame[keys].merge(lookup, on=keys, how="left", sort=False)
    merged.index = frame.index
    return pd.to_numeric(merged["offline_total"], errors="coerce")


def derive(frame: pd.DataFrame, min_replicas: int, c_run: float) -> pd.DataFrame:
    """Add the base-tier columns of ``DERIVED_COLUMNS`` to a sweep frame (pure; input untouched).

    Args:
        frame: rows of one sweep CSV, in the schema ``sweeps.run_sweep`` writes.
        min_replicas: the always-on base tier ``M`` every policy pays for.
        c_run: USD per replica-minute.

    Returns:
        A copy of ``frame`` with ``run_minutes``, ``base_usd``, ``base_hours``, ``offline_total``,
        ``excess_usd``, ``attr_cost``, ``attr_ratio`` and ``attr_gpu_hours`` appended.
    """
    out = frame.copy()
    minutes = run_minutes(out)
    base_usd = float(min_replicas) * float(c_run) * minutes
    base_hours = float(min_replicas) * minutes / 60.0
    total = pd.to_numeric(out.get("total_cost", pd.Series(np.nan, index=out.index)),
                          errors="coerce")
    gpu_hours = pd.to_numeric(out.get("gpu_hours", pd.Series(np.nan, index=out.index)),
                              errors="coerce")
    oracle = offline_total(out)
    attr_cost = total - base_usd
    denominator = (oracle - base_usd).where(lambda s: s > RATIO_FLOOR)
    out["run_minutes"] = minutes
    out["base_usd"] = base_usd
    out["base_hours"] = base_hours
    out["offline_total"] = oracle
    out["excess_usd"] = total - oracle
    out["attr_cost"] = attr_cost
    out["attr_ratio"] = attr_cost / denominator
    out["attr_gpu_hours"] = gpu_hours - base_hours
    return out


@dataclass(frozen=True)
class FloorSpec:
    """The always-on base tier a run was made with (``run_sweep.py --min-replicas``).

    ``min_replicas = 0`` is no base tier at all, which is what every sweep before batch C ran with;
    a consumer can then keep its pre-floor behaviour by asking ``on``.  ``excess_usd`` is a
    difference of two costs and so is defined either way.
    """

    min_replicas: int = 0
    c_run: float = 0.0

    @property
    def on(self) -> bool:
        """Whether a base tier was deployed, and results therefore belong above it."""
        return self.min_replicas > 0

    def derived(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``derive`` at this floor."""
        return derive(frame, self.min_replicas, self.c_run)


def regimes(weeks: Sequence[object], r1_weeks: Sequence[int] = R1_WEEKS) -> List[str]:
    """``R1`` for a novel-timetable week (no history, or a step change), ``R2`` otherwise."""
    novel = {int(w) for w in r1_weeks}
    return ["R1" if int(w) in novel else "R2" for w in weeks]


@dataclass(frozen=True)
class CellSelection:
    """One cell of a sweep: the cell-key columns, their values and why this cell was picked."""

    columns: Tuple[str, ...]
    values: Tuple[object, ...]
    rule: str

    @property
    def description(self) -> str:
        """``lead_min = 0.5, lead_tier = d_rep (the smallest lead_min ...)`` for a header line."""
        shown = ", ".join(f"{c} = {v}" for c, v in zip(self.columns, self.values))
        return f"{shown or 'the whole file'} ({self.rule})"

    def rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The rows of ``frame`` that belong to this cell."""
        if not self.columns:
            return frame
        mask = pd.Series(True, index=frame.index)
        for column, value in zip(self.columns, self.values):
            mask &= frame[column].eq(value) | (frame[column].isna() & pd.isna(value))
        return frame[mask]


def reference_cell(frame: pd.DataFrame) -> CellSelection:
    """The cell the main tables report: the replica tier, or the most synchronous cell.

    The lead sweep is ranked by ``lead_min`` ascending (``d_rep``, the 0.5 min replica tier, and the
    commitment / billing block breaks the tie that ``d_inst`` and ``d_res`` have at 8 min); the
    sigma sweep by ``sigma`` descending, its most synchronous cell.  Any other sweep keeps the first
    cell in file order, and a frame without a cell key is one cell.
    """
    columns = cell_key_columns(frame)
    if not columns:
        return CellSelection((), (), "the frame carries no cell key")
    cells = frame[columns].drop_duplicates()
    if "lead_min" in columns:
        rule, order = "the smallest lead_min: the replica tier", True
        cells = cells.sort_values(["lead_min", *[c for c in columns if c != "lead_min"]],
                                  ascending=order)
    elif "sigma" in columns:
        cells = cells.sort_values(["sigma", *[c for c in columns if c != "sigma"]],
                                  ascending=[False, *[True] * (len(columns) - 1)])
        rule = "the largest sigma: the most synchronous cell"
    else:
        rule = "the first cell in file order"
    values = tuple(cells.iloc[0].tolist())
    return CellSelection(tuple(columns), values, rule)


__all__ = ["DEFAULT_C_RUN", "DEFAULT_MIN_REPLICAS", "DERIVED_COLUMNS", "MINUTES_PER_WEEK",
           "OFFLINE_POLICY", "R1_WEEKS", "RATIO_FLOOR", "SWEEP_COL", "CellSelection", "FloorSpec",
           "cell_key_columns", "derive", "offline_total", "reference_cell", "regimes",
           "run_minutes"]
