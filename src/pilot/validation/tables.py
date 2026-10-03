"""Shaping the validation round into the tables the results note shows.

A *group* is one arrival stream and one baseline tuning: ``(case, ratio, tier, tuned_at, slots)``.
Its rows come from up to three runs - the full policy list, the learned-timing planner alone
(``_learn``, older rounds only) and the SLO-matched feedback baselines (``_slo95``) - and
``assemble`` stacks them under one policy column, the SLO-matched pair renamed so that a reader can
tell it from the freely tuned one.

Every table is a mean over seeds of a per-week metric (``runs.load_run``): GPU-hours per week,
in-class SLO attainment, USD per week and USD over the offline optimum per week.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .runs import CELL_ORDER, OFFLINE, RunFile, whole_run

logger = logging.getLogger(__name__)

SLO_SUFFIX: str = " (SLO>=0.95)"  # floor tuning on the synchronous slice
SLO_DISP_SUFFIX: str = " (SLO>=0.95, disp)"  # floor tuning on the dispersed slice
SLO_SUFFIXES: Dict[str, str] = {"slo95": SLO_SUFFIX, "slo95disp": SLO_DISP_SUFFIX}
METRICS: Tuple[str, ...] = ("gpu_h_wk", "slo_in", "usd_wk", "excess_wk")
EXTRA_METRICS: Tuple[str, ...] = ("x_peak", "starts_wk", "ttft_p95_in_s")
METRIC_TITLES: Dict[str, str] = {
    "gpu_h_wk": "GPU-hours per week (one always-on replica = 168)",
    "slo_in": "In-class SLO attainment",
    "usd_wk": "USD per week (running + starts + unserved-request penalties)",
    "excess_wk": "USD per week over the offline optimum",
    "x_peak": "Peak replicas",
    "starts_wk": "Replica starts per week",
    "ttft_p95_in_s": "In-class TTFT p95 (s)",
}
POLICY_ORDER: Tuple[str, ...] = (
    "offline", "informed", "calendar", "calendar_adapt", "calendar_learn", "calendar_learn_prior",
    "bdinterp@0.1",
    "stale_calendar", "forecast_ilp", "guard", "lcp", "velocity", "reactive", "fixed",
    f"velocity{SLO_SUFFIX}", f"reactive{SLO_SUFFIX}", f"velocity{SLO_DISP_SUFFIX}",
    f"reactive{SLO_DISP_SUFFIX}")
TIMETABLE_POLICIES: Tuple[str, ...] = ("calendar", "calendar_adapt", "calendar_learn",
                                       "calendar_learn_prior")
FEEDBACK_POLICIES: Tuple[str, ...] = ("velocity", "reactive", f"velocity{SLO_SUFFIX}",
                                      f"reactive{SLO_SUFFIX}", f"velocity{SLO_DISP_SUFFIX}",
                                      f"reactive{SLO_DISP_SUFFIX}")


@dataclass(frozen=True)
class Group:
    """One stream x tuning of one case, and the runs that feed it."""

    case: str
    ratio: int
    tier: str
    tuned_at: str
    slots: str  # main | heldout_sem2
    runs: Tuple[RunFile, ...]

    @property
    def key(self) -> Tuple[str, int, str, str, str]:
        """Sort/lookup key."""
        return (self.case, self.ratio, self.tier, self.tuned_at, self.slots)

    @property
    def label(self) -> str:
        """Human-readable heading."""
        return self.runs[0].spec.group_label if self.runs else str(self.key)

    @property
    def name(self) -> str:
        """File-name stem: ``r30_d_inst_tuned-d_inst`` (+ ``_heldout_sem2``)."""
        stem = f"r{self.ratio}_{self.tier}_tuned-{self.tuned_at}"
        return stem if self.slots == "main" else f"{stem}_{self.slots}"


def groups_of(runs: Sequence[RunFile]) -> List[Group]:
    """Cluster the runs of the round into groups."""
    buckets: Dict[Tuple[str, int, str, str, str], List[RunFile]] = {}
    for run in runs:
        ratio, tier, tuned_at, slots = run.spec.group
        buckets.setdefault((run.case, ratio, tier, tuned_at, slots), []).append(run)
    return [Group(*key, tuple(sorted(v, key=lambda r: r.run))) for key, v in sorted(buckets.items())]


def assemble(group: Group, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The group's rows under one policy column.

    ``frames`` maps ``run`` name to its loaded frame.  The base run supplies every policy; a
    ``_learn`` run adds ``calendar_learn`` when the base run lacks it; a ``_slo95`` run adds its
    feedback baselines under the SLO-matched label.  Offline rows come from the base run only.
    """
    base = [r for r in group.runs if r.spec.variant in ("main", "heldout_sem2")]
    learn = [r for r in group.runs if r.spec.is_learn_only]
    slo = [r for r in group.runs if r.spec.variant in SLO_SUFFIXES]
    parts: List[pd.DataFrame] = []
    have: set = set()
    if base:
        frame = frames[base[0].run]
        parts.append(frame)
        have = set(frame["policy"].unique())
    for run in learn:
        frame = frames[run.run]
        extra = frame[~frame["policy"].isin(have)]
        if not base:  # the learn run is all there is: keep its offline rows too
            extra = frame
        parts.append(extra)
        have |= set(extra["policy"].unique())
    for run in slo:
        frame = frames[run.run]
        matched = frame[frame["policy"] != OFFLINE].copy()
        matched["policy"] = matched["policy"] + SLO_SUFFIXES[run.spec.variant]
        parts.append(matched)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def policy_sort_key(policy: str) -> Tuple[int, str]:
    """Known policies in ``POLICY_ORDER``, the rest alphabetically after them."""
    return (POLICY_ORDER.index(policy), policy) if policy in POLICY_ORDER else (len(POLICY_ORDER), policy)


def cell_means(frame: pd.DataFrame, metrics: Sequence[str] = METRICS + EXTRA_METRICS
               ) -> pd.DataFrame:
    """Per (policy, cell): mean and sd over seeds of the whole-run per-week metrics."""
    rows = whole_run(frame)
    if rows.empty:
        return pd.DataFrame()
    present = [m for m in metrics if m in rows.columns]
    agg = rows.groupby(["policy", "cell"])[present].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" if s == "mean" else f"{m}_sd" for m, s in agg.columns]
    agg = agg.rename(columns={f"{m}_std": f"{m}_sd" for m in present}).reset_index()
    agg["n_seeds"] = rows.groupby(["policy", "cell"])["seed"].nunique().to_numpy()
    agg = agg.sort_values(["policy", "cell"], key=lambda s: s.map(policy_sort_key) if s.name == "policy" else s)
    return agg.reset_index(drop=True)


def wide(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Policies x timing cells for one metric (means)."""
    if cells.empty or f"{metric}_mean" not in cells.columns:
        return pd.DataFrame()
    table = cells.pivot(index="policy", columns="cell", values=f"{metric}_mean")
    table = table.reindex(columns=[c for c in CELL_ORDER if c in table.columns])
    table = table.loc[sorted(table.index, key=policy_sort_key)]
    return table.reset_index()


def compare(cells: pd.DataFrame, cell: str, reference: str = "velocity") -> pd.DataFrame:
    """Every policy against ``reference`` in one cell: GPU-hour and USD differences, dominance.

    ``dominates`` is true when a policy uses no more GPU-hours and attains no less SLO than the
    reference - a comparison that needs no SLO matching.
    """
    part = cells[cells["cell"] == cell].set_index("policy")
    if part.empty or reference not in part.index:
        return pd.DataFrame()
    ref = part.loc[reference]
    out = pd.DataFrame({
        "policy": part.index,
        "gpu_h_wk": part["gpu_h_wk_mean"].to_numpy(),
        "slo_in": part["slo_in_mean"].to_numpy(),
        "usd_wk": part["usd_wk_mean"].to_numpy(),
        "excess_wk": part["excess_wk_mean"].to_numpy(),
        "d_gpu_h_wk": (part["gpu_h_wk_mean"] - ref["gpu_h_wk_mean"]).to_numpy(),
        "d_gpu_pct": (100.0 * (part["gpu_h_wk_mean"] / ref["gpu_h_wk_mean"] - 1.0)).to_numpy(),
        "d_slo": (part["slo_in_mean"] - ref["slo_in_mean"]).to_numpy(),
        "d_usd_wk": (part["usd_wk_mean"] - ref["usd_wk_mean"]).to_numpy(),
    })
    out["dominates"] = (out["gpu_h_wk"] <= ref["gpu_h_wk_mean"] + 1e-9) & (
        out["slo_in"] >= ref["slo_in_mean"] - 1e-9)
    out["reference"] = reference
    out["cell"] = cell
    return out.sort_values("policy", key=lambda s: s.map(policy_sort_key)).reset_index(drop=True)


def weekly_table(frame: pd.DataFrame, cell: str, policies: Sequence[str], metric: str
                 ) -> pd.DataFrame:
    """Policies x teaching weeks of one metric in one cell (means over seeds)."""
    rows = frame[(pd.to_numeric(frame["week"], errors="coerce") > 0) & (frame["cell"] == cell)
                 & (frame["policy"].isin(policies))]
    if rows.empty or metric not in rows.columns:
        return pd.DataFrame()
    table = rows.pivot_table(index="policy", columns="week", values=metric, aggfunc="mean")
    table.columns = [f"w{int(w)}" for w in table.columns]
    table = table.loc[sorted(table.index, key=policy_sort_key)]
    return table.reset_index()


def steady_state(frame: pd.DataFrame, weeks: Sequence[int] = (2, 3, 4, 5, 6),
                 metrics: Sequence[str] = ("gpu_h_wk", "slo_in", "usd_wk")) -> pd.DataFrame:
    """Per (policy, cell): means over seeds of the per-week rows of ``weeks`` only.

    The learned-timing planner spends week 1 at the bell, so its whole-run numbers mix a blind
    week with its held-out weeks; this table reads every policy on the same later weeks.
    """
    rows = frame[pd.to_numeric(frame["week"], errors="coerce").isin(list(weeks))]
    if rows.empty:
        return pd.DataFrame()
    present = [m for m in metrics if m in rows.columns]
    per_seed = rows.groupby(["policy", "cell", "seed"])[present].mean().reset_index()
    agg = per_seed.groupby(["policy", "cell"])[present].agg(["mean", "std"])
    agg.columns = [f"{m}_{'mean' if s == 'mean' else 'sd'}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["weeks"] = ",".join(str(w) for w in weeks)
    return agg.sort_values(["policy", "cell"], key=lambda s: s.map(policy_sort_key) if s.name == "policy" else s
                           ).reset_index(drop=True)


def best_of(cells: pd.DataFrame, cell: str, policies: Sequence[str], slo_floor: float
            ) -> Optional[pd.Series]:
    """The cheapest (GPU-hours) of ``policies`` that attains ``slo_floor`` in ``cell``; None if none."""
    part = cells[(cells["cell"] == cell) & (cells["policy"].isin(policies))
                 & (cells["slo_in_mean"] >= slo_floor)]
    if part.empty:
        return None
    return part.sort_values("gpu_h_wk_mean").iloc[0]


def frontier_row(cells: pd.DataFrame, cell: str, slo_floor: float) -> Dict[str, object]:
    """Best timetable policy vs best feedback policy at a common SLO floor, in one cell."""
    tt = best_of(cells, cell, TIMETABLE_POLICIES, slo_floor)
    fb = best_of(cells, cell, FEEDBACK_POLICIES, slo_floor)
    row: Dict[str, object] = {"cell": cell, "slo_floor": slo_floor,
                              "timetable_best": None if tt is None else tt["policy"],
                              "timetable_gpu_h_wk": np.nan if tt is None else tt["gpu_h_wk_mean"],
                              "timetable_slo": np.nan if tt is None else tt["slo_in_mean"],
                              "feedback_best": None if fb is None else fb["policy"],
                              "feedback_gpu_h_wk": np.nan if fb is None else fb["gpu_h_wk_mean"],
                              "feedback_slo": np.nan if fb is None else fb["slo_in_mean"]}
    if tt is not None and fb is not None:
        row["saving_pct"] = 100.0 * (1.0 - tt["gpu_h_wk_mean"] / fb["gpu_h_wk_mean"])
    else:
        row["saving_pct"] = np.nan
    return row


__all__ = ["EXTRA_METRICS", "FEEDBACK_POLICIES", "METRICS", "METRIC_TITLES", "POLICY_ORDER",
           "SLO_DISP_SUFFIX", "SLO_SUFFIX", "SLO_SUFFIXES", "TIMETABLE_POLICIES", "Group", "assemble", "best_of", "cell_means",
           "compare", "frontier_row", "groups_of", "policy_sort_key", "steady_state", "weekly_table",
           "wide"]
