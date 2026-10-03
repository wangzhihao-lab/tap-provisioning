"""The tables of the batch C report, shaped from the derived sweep frames.

Every builder takes a frame that has already been through ``derive`` and returns a tidy DataFrame,
so the CLI can write each one as CSV and render the same numbers as Markdown without recomputing
anything.  Aggregation is always over the seeds of one cell: one row per (policy, seed) for the
whole-run tables, one per (policy, seed, week) for the weekly one.
"""

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .derive import R1_WEEKS, cell_key_columns, regimes
from .discover import CONFIG_KEY, SweepFile, TuningRun, distinct_scores

logger = logging.getLogger(__name__)

# ``gpu_hours`` sits next to ``attr_gpu_hours`` on purpose: the attributable component alone would
# hide how much of the fleet the always-on base tier already is, which is the whole point of the
# institution-wide scenario, so every table that reports resources reports the total as well.
MAIN_METRICS: Tuple[str, ...] = ("total_cost", "excess_usd", "attr_ratio", "gpu_hours",
                                 "attr_gpu_hours", "starts", "slo_in", "viol_in", "x_peak")
WEEKLY_METRICS: Tuple[str, ...] = ("excess_usd", "attr_ratio", "slo_in", "gpu_hours",
                                   "attr_gpu_hours")
CELL_METRICS: Tuple[str, ...] = ("excess_usd", "attr_ratio", "slo_in")
SWEEP_CELL_METRICS: Tuple[str, ...] = (*CELL_METRICS, "gpu_hours")  # the secondary sweeps
SLO_MATCHED_SUFFIX: str = " (SLO≥0.95)"  # how an SLO-matched baseline is named in a table
SORT_METRIC: str = "excess_usd"  # the main table is ordered by what a policy paid over the optimum
TOLERANCE: float = 1e-6  # a violation has to be worth more than floating-point noise
NAN_CHECKED: Tuple[str, ...] = ("total_cost", "gpu_hours", "slo_in")


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The requested columns as floats; the absent ones are logged once and dropped."""
    present = [c for c in columns if c in frame.columns]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        logger.warning("columns %s are absent; they are left out of the table", missing)
    return frame[present].apply(pd.to_numeric, errors="coerce")


def means(frame: pd.DataFrame, by: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean of each metric over the rows of each group, one column per metric."""
    keys = [c for c in by if c in frame.columns]
    values = _numeric(frame, metrics)
    if not keys or values.empty:
        return pd.DataFrame()
    grouped = frame[keys].join(values).groupby(keys, dropna=False, observed=True)
    return grouped.mean().reset_index()


def mean_sd(frame: pd.DataFrame, by: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Mean and sample standard deviation of each metric, as ``<metric>_mean`` / ``<metric>_sd``."""
    keys = [c for c in by if c in frame.columns]
    values = _numeric(frame, metrics)
    if not keys or values.empty:
        return pd.DataFrame()
    grouped = frame[keys].join(values).groupby(keys, dropna=False, observed=True)
    wide = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    order = [f"{m}{suffix}" for m in values.columns for suffix in ("_mean", "_sd")]
    return wide[order].reset_index()


def main_table(frame: pd.DataFrame, metrics: Sequence[str] = MAIN_METRICS) -> pd.DataFrame:
    """Per policy, mean and sd over seeds of the whole-run metrics, cheapest excess first."""
    table = mean_sd(frame, ("policy",), metrics)
    key = f"{SORT_METRIC}_mean"
    if key in table.columns:
        table = table.sort_values(key, na_position="last").reset_index(drop=True)
    return table


def relabelled(table: pd.DataFrame, suffix: str = SLO_MATCHED_SUFFIX) -> pd.DataFrame:
    """A copy of a table whose ``policy`` values carry ``suffix``.

    The SLO-matched pair of batch C is the *same* two baselines tuned under a constraint
    (``tuning_r30_slo95``), so their rows have to sit in the main table under a name that says
    which selection they come from rather than colliding with the free-tuning rows.
    """
    if table is None or table.empty or "policy" not in table.columns:
        logger.warning("nothing to relabel with '%s'", suffix)
        return pd.DataFrame() if table is None else table
    out = table.copy()
    out["policy"] = out["policy"].astype(str) + suffix
    return out


def appended(table: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """``extra``'s rows under ``table``'s, keeping ``table``'s column order."""
    if extra is None or extra.empty:
        return table
    if table is None or table.empty:
        return extra
    return pd.concat([table, extra], ignore_index=True)[list(table.columns)]


def weekly_table(frame: pd.DataFrame, metrics: Sequence[str] = WEEKLY_METRICS,
                 r1_weeks: Sequence[int] = R1_WEEKS) -> pd.DataFrame:
    """Per policy and teaching week, the mean over seeds, with the R1 / R2 regime of the week."""
    if "week" not in frame.columns:
        logger.warning("no 'week' column: the weekly table is empty")
        return pd.DataFrame()
    weekly = frame[pd.to_numeric(frame["week"], errors="coerce") > 0]
    table = means(weekly, ("policy", "week"), metrics)
    if table.empty:
        return table
    table["week"] = table["week"].astype(int)
    table.insert(2, "regime", regimes(table["week"].tolist(), r1_weeks))
    return table.sort_values(["policy", "week"]).reset_index(drop=True)


def ratio_table(frames: Mapping[int, pd.DataFrame], metrics: Sequence[str] = CELL_METRICS,
                ratios: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Policies against price ratios: one block of rows per metric, one column per ratio."""
    order = [r for r in (ratios if ratios is not None else sorted(frames)) if r in frames]
    blocks: List[pd.DataFrame] = []
    for metric in metrics:
        columns: Dict[str, pd.Series] = {}
        for ratio in order:
            frame = frames[ratio]
            if metric not in frame.columns or "policy" not in frame.columns:
                continue
            values = pd.to_numeric(frame[metric], errors="coerce")
            columns[f"r{ratio}"] = values.groupby(frame["policy"].astype(str)).mean()
        if not columns:
            continue
        block = pd.DataFrame(columns)
        block.index.name = "policy"
        block = block.reset_index()
        block.insert(0, "metric", metric)
        blocks.append(block)
    if not blocks:
        logger.warning("no ratio reached the r-sweep table")
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True).fillna(np.nan)


def cell_table(frame: pd.DataFrame, metrics: Sequence[str] = CELL_METRICS,
               cell_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per cell and policy, the mean over seeds; every cell-key column is kept as a column."""
    keys = list(cell_key_columns(frame) if cell_columns is None else cell_columns)
    table = means(frame, [*keys, "policy"], metrics)
    return table if table.empty else table.sort_values([*keys, "policy"]).reset_index(drop=True)


def _range_text(values: pd.Series) -> str:
    """``0.235..0.702`` for a numeric column, ``n/a`` when it has no finite value.

    ``.. `` rather than a hyphen: a range of a column that can go negative (or hold a signed zero)
    would otherwise read as two minus signs.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "n/a"
    return f"{float(numeric.min()) + 0.0:g}..{float(numeric.max()) + 0.0:g}"


def _peak_ranges(frame: pd.DataFrame) -> str:
    """``static 3-3; reactive 1-4``: the x_peak range of every policy, in one cell of the table."""
    if "x_peak" not in frame.columns or "policy" not in frame.columns:
        return "n/a"
    parts = [f"{policy} {_range_text(rows['x_peak'])}"
             for policy, rows in frame.groupby(frame["policy"].astype(str), sort=True)]
    return "; ".join(parts) if parts else "n/a"


def verification_row(source: SweepFile, frame: pd.DataFrame) -> Dict[str, object]:
    """One row of the verification table: shape, coverage and the two invariants that must hold.

    ``dominance`` counts the whole-run rows that came out cheaper than their own offline optimum,
    which the LP of policy 8 makes impossible, and ``floor`` the rows that spent fewer GPU-hours
    than the always-on base tier, which the deployment makes impossible.  Either one being non-zero
    means the run is not measuring what the paper says it measures.
    """
    whole = frame[pd.to_numeric(frame["week"], errors="coerce") == 0] if "week" in frame else frame
    excess = pd.to_numeric(whole.get("excess_usd", pd.Series(dtype=float)), errors="coerce")
    gpu = pd.to_numeric(frame.get("gpu_hours", pd.Series(dtype=float)), errors="coerce")
    base = pd.to_numeric(frame.get("base_hours", pd.Series(dtype=float)), errors="coerce")
    keys = cell_key_columns(frame)
    row: Dict[str, object] = {
        "file": source.label, "path": str(source.path), "sweep": source.sweep,
        "n_rows": int(len(frame)),
        "n_cells": int(frame[keys].drop_duplicates().shape[0]) if keys else 1,
        "seeds": ",".join(str(s) for s in sorted(frame["seed"].dropna().unique()))
                 if "seed" in frame.columns else "n/a",
        "policies": ",".join(sorted(frame["policy"].astype(str).unique()))
                    if "policy" in frame.columns else "n/a",
    }
    for column in NAN_CHECKED:
        row[f"nan_{column}"] = (int(frame[column].isna().sum()) if column in frame.columns
                                else int(len(frame)))
    row["dominance_violations"] = int((excess < -TOLERANCE).sum())
    row["floor_violations"] = int(((gpu - base) < -TOLERANCE).sum())
    row["x_mean_min"] = (float(pd.to_numeric(frame["x_mean"], errors="coerce").min())
                         if "x_mean" in frame.columns else float("nan"))
    row["x_peak_range_by_policy"] = _peak_ranges(whole)
    row["sigma2_measured_range"] = (_range_text(frame["sigma2_measured"])
                                    if "sigma2_measured" in frame.columns else "n/a")
    return row


VIOLATION_COLUMNS: Tuple[str, ...] = ("dominance_violations", "floor_violations")


def violation_count(table: pd.DataFrame) -> int:
    """Total number of invariant violations in a verification table."""
    columns = [c for c in VIOLATION_COLUMNS if c in table.columns]
    if not columns:
        return 0
    return int(table[columns].to_numpy(dtype=float).sum())


def _config_text(config: Mapping[str, object]) -> str:
    """``hpa_up_util=0.2, hpa_down_util=0.3``: one tuned configuration on one line."""
    if not isinstance(config, Mapping) or not config:
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in config.items())


def _section(section: Mapping[str, object], key: str) -> Mapping[str, object]:
    """One sub-object of a ``best.json`` policy section, or an empty mapping."""
    value = section.get(key)
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float:
    """A JSON scalar as a float, NaN when it is missing or not a number."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def tuning_table(runs: Sequence[TuningRun]) -> pd.DataFrame:
    """Per selection and tuned policy: what was chosen, what the default was, and how far apart.

    ``n_distinct_scores`` is the number of distinct tuning scores in that policy's grid CSV: a grid
    of 54 configurations that produced one score selected nothing, and the reader has to be able to
    see that next to ``rank_of_default``.  ``slice`` and ``slo_floor`` say which selection a row is
    from: batch C tunes r = 30 twice, once freely and once constrained to ``slo_in >= 0.95``.
    """
    rows: List[Dict[str, object]] = []
    for run in runs:
        for policy, section in run.sections().items():
            tuning, heldout = _section(section, "tuning"), _section(section, "heldout")
            base_tuning = _section(section, "default_tuning")
            base_heldout = _section(section, "default_heldout")
            rows.append({
                "price_ratio": int(run.ratio), "slice": run.label, "policy": policy,
                "slo_floor": run.slo_floor,
                "chosen_config": _config_text(_section(section, CONFIG_KEY)),
                "default_config": _config_text(_section(section, "default")),
                "tuning_ratio_chosen": _number(tuning.get("cost_ratio_mean")),
                "tuning_ratio_default": _number(base_tuning.get("cost_ratio_mean")),
                "heldout_ratio_chosen": _number(heldout.get("cost_ratio")),
                "heldout_ratio_default": _number(base_heldout.get("cost_ratio")),
                "heldout_slo_in_chosen": _number(heldout.get("slo_in")),
                "heldout_slo_in_default": _number(base_heldout.get("slo_in")),
                "rank_of_default": _number(section.get("rank_of_default")),
                "grid_size": _number(section.get("grid_size")),
                "n_distinct_scores": _number(distinct_scores(run.grids.get(policy))),
            })
    if not rows:
        logger.warning("no tuning directory was readable; the tuning table is empty")
    return pd.DataFrame(rows)


__all__ = ["CELL_METRICS", "MAIN_METRICS", "NAN_CHECKED", "SLO_MATCHED_SUFFIX", "SORT_METRIC",
           "SWEEP_CELL_METRICS", "TOLERANCE", "VIOLATION_COLUMNS", "WEEKLY_METRICS", "appended",
           "cell_table", "main_table", "mean_sd", "means", "ratio_table", "relabelled",
           "tuning_table", "verification_row", "violation_count", "weekly_table"]
