"""Split a prompted class day into per-bell bursts and describe each burst.

Theory class A meets Tuesday periods 1-3 (bells 08:30, 09:20, 10:20, with the break before
the third) and the teacher prompts the class to ask the assistant at the start of a period: usually
period 1, sometimes only period 2, sometimes again period 3.  A *burst* is therefore the set of
students whose **first arrival of the day** falls in one bell's window, and the bells tile the
morning: ``[bell_i - pre, bell_{i+1} - pre)``, the last window running ``span`` minutes.

``pre`` is 20 min rather than the 10 min a first pass used, because the observed bursts start up to
17 min before their bell (the earliest cluster of the semester begins at 08:13 for the 08:30 bell);
a 10-minute margin cuts whole bursts in two and merges the remainder into the next window, which is
what produced the "mixed" bursts of the first pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .load import DatasetSpec
from .metrics import peak_to_mean, ramp_10_90, relative_offsets, sigma_shares

logger = logging.getLogger(__name__)

PROMPTED_BURST = "prompted"
MIXED_BURST = "mixed"


@dataclass(frozen=True)
class BurstConfig:
    """How bursts are cut out of a class day and when one counts as prompted."""

    pre_min: float = 20.0          # a burst may start this long before its bell
    span_min: float = 100.0        # window length of the last bell of the day
    min_students: int = 5          # below this a window is not called a burst
    prompted_ramp_min: float = 5.0  # 10-90% ramp at or below this = prompted


def bell_windows(bells: Sequence[float], cfg: BurstConfig) -> List[Tuple[float, float, float]]:
    """``(bell, lo, hi)`` per bell: contiguous windows that tile the class morning."""
    out: List[Tuple[float, float, float]] = []
    for i, bell in enumerate(bells):
        lo = bell - cfg.pre_min
        hi = (bells[i + 1] - cfg.pre_min) if i + 1 < len(bells) else bell + cfg.span_min
        out.append((float(bell), float(lo), float(hi)))
    return out


def _burst_row(date: object, bell: float, label: str, tod: np.ndarray,
               cfg: BurstConfig, window: Tuple[float, float]) -> dict:
    """Metrics of one burst, measured from the burst's own first arrival."""
    rel = relative_offsets(tod)
    ramp = ramp_10_90(rel)
    row = {"date": str(date), "bell": label, "n_students": int(tod.size),
           "first_arrival_offset_min": float(tod.min() - bell),
           "span_min": float(rel.max()), "ramp_10_90_min": ramp,
           "kind": PROMPTED_BURST if ramp <= cfg.prompted_ramp_min else MIXED_BURST,
           "peak_per_min": float(np.bincount(np.floor(rel).astype(int)).max()),
           "p2m_1min": peak_to_mean(tod, window, 1.0),
           "p2m_5min": peak_to_mean(tod, window, 5.0)}
    row.update(sigma_shares(rel))
    return row


def split_bursts(first_arrivals: pd.DataFrame, spec: DatasetSpec,
                 cfg: Optional[BurstConfig] = None) -> pd.DataFrame:
    """One row per (class day, bell) with at least ``min_students`` first arrivals."""
    cfg = cfg or BurstConfig()
    if first_arrivals.empty or not spec.bells:
        return pd.DataFrame()
    days = first_arrivals
    if spec.class_weekday is not None:
        days = days[days["weekday"] == spec.class_weekday]
    rows: List[dict] = []
    for date, group in days.groupby("date"):
        for (bell, lo, hi), label in zip(bell_windows(spec.bell_minutes, cfg), spec.bells):
            tod = group.loc[(group["tod_min"] >= lo) & (group["tod_min"] < hi),
                            "tod_min"].to_numpy(dtype=float)
            if tod.size < cfg.min_students:
                continue
            rows.append(_burst_row(date, bell, label, np.sort(tod), cfg, (lo, hi)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    n_prompted = int((out["kind"] == PROMPTED_BURST).sum())
    logger.info("%s: %d bursts (%d prompted), sigma_2 %.2f-%.2f, ramp %.1f-%.1f min", spec.key,
                len(out), n_prompted, out["sigma_2"].min(), out["sigma_2"].max(),
                out["ramp_10_90_min"].min(), out["ramp_10_90_min"].max())
    return out.sort_values(["date", "bell"]).reset_index(drop=True)


def burst_offsets(first_arrivals: pd.DataFrame, spec: DatasetSpec,
                  cfg: Optional[BurstConfig] = None,
                  kinds: Sequence[str] = (PROMPTED_BURST,)) -> np.ndarray:
    """Per-student offsets (minutes from the burst's first arrival) pooled over the kept bursts.

    This is the offset pool the simulator samples from for the prompted regime, so it is built
    from the same windows ``split_bursts`` describes and keeps only the requested burst kinds.
    """
    cfg = cfg or BurstConfig()
    if first_arrivals.empty or not spec.bells:
        return np.empty(0, dtype=float)
    days = first_arrivals
    if spec.class_weekday is not None:
        days = days[days["weekday"] == spec.class_weekday]
    pool: List[np.ndarray] = []
    for _, group in days.groupby("date"):
        for bell, lo, hi in bell_windows(spec.bell_minutes, cfg):
            tod = np.sort(group.loc[(group["tod_min"] >= lo) & (group["tod_min"] < hi),
                                    "tod_min"].to_numpy(dtype=float))
            if tod.size < cfg.min_students:
                continue
            rel = relative_offsets(tod)
            kind = PROMPTED_BURST if ramp_10_90(rel) <= cfg.prompted_ramp_min else MIXED_BURST
            if kind in kinds:
                pool.append(rel)
    return np.concatenate(pool) if pool else np.empty(0, dtype=float)


__all__ = ["MIXED_BURST", "PROMPTED_BURST", "BurstConfig", "bell_windows", "burst_offsets",
           "split_bursts"]
