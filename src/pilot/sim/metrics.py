"""Cost and service metrics of the simulator design

Cost is the soft-constraint objective of the paper's analysis evaluated on the *fluid* demand: running
cost, switching cost and the graded violation penalty.  This is exactly what the offline LP of
policy 8 minimises, so ``cost_ratio = total / offline_total`` is >= 1 for every other policy by
construction (acceptance test (a)).

Service quality is measured on the request stream that the discrete-event replay produced:
in-class and off-class TTFT p95, SLO attainment at ``a`` and ``b`` seconds, the attainment over the
first minutes after a bell, and the timeout / rejection counts.

The running cost is charged per *billing block* (``ServiceConfig.billing_block_min``): the peak of
each block is paid for the whole block.  At the per-minute default of one minute that is exactly
``c_run * sum_t x_t``; on the reserved tier, where the block is an hour, it is what makes a
committed hour cost more than the minutes it actually used.

``weekly_rows`` splits the same run by teaching week so that week 1 - the blind spot of every
history-driven policy - can be reported apart from the steady state.  The offline and informed
optima stay whole-run quantities: a per-week oracle would be a different, easier problem, so a
weekly ``cost_ratio`` divides by the *whole-run* oracle's cost restricted to that week and may
therefore dip below 1 in a week the oracle chose to under-provision.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CostConfig
from .service import ServiceResult
from .workload import IN_CLASS, ArrivalStream

logger = logging.getLogger(__name__)

_MIN_PER_WEEK = 7 * 1440


@dataclass(frozen=True)
class CostBreakdown:
    """Provisioning cost of one trajectory."""

    gpu_hours: float
    starts: float
    reservation_blocks: float
    run_cost: float
    switch_cost: float
    unserved_in: float
    unserved_off: float
    penalty_cost: float
    total_cost: float
    x_mean: float
    x_peak: float


def _blocks(x: np.ndarray, block_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(peak of each block, length of each block)``; blocks are aligned at multiples of the len."""
    block = max(1, int(round(block_min)))
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    pad = (-x.size) % block
    grid = np.pad(x, (0, pad), mode="constant").reshape(-1, block)
    lengths = np.full(grid.shape[0], float(block))
    if pad:  # a horizon that is not a whole number of blocks pays only the minutes it covers
        lengths[-1] = float(block - pad)
    return grid.max(axis=1), lengths


def reservation_blocks(x: np.ndarray, block_min: float) -> float:
    """Replica-blocks billed: the peak of each billing block, summed over blocks."""
    peaks, _ = _blocks(x, block_min)
    return float(peaks.sum())


def billed_replica_minutes(x: np.ndarray, block_min: float) -> float:
    """Replica-minutes charged when every replica-block is paid for in full.

    At ``block_min = 1`` this is ``sum_t x_t``, the per-minute charge.  At a longer billing block
    the peak of each block is charged for the whole block, so the value can only grow - which is
    what makes a reserved hour dearer than the same trajectory rented by the minute.
    """
    peaks, lengths = _blocks(x, block_min)
    return float((peaks * lengths).sum())


def fluid_cost(x: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray, mu: float,
               cost: CostConfig, billing_block_min: float = 1.0,
               x_prev: float = 0.0) -> CostBreakdown:
    """Evaluate the cost of a replica trajectory against the realised demand.

    ``x_prev`` is the level in force just before the first slot; it is 0 for a whole run and the
    last level of the previous week when one week of a run is scored on its own.
    ``billing_block_min`` is ``ServiceConfig.billing_block_min``: the running cost charges each
    replica-block in full, which is a no-op at the per-minute default of 1.
    """
    x = np.asarray(x, dtype=float)
    served = mu * x
    short_in = np.maximum(lam_in - served, 0.0)
    short_all = np.maximum(lam_in + lam_off - served, 0.0)
    short_off = short_all - short_in
    ups = np.maximum(np.diff(x, prepend=float(x_prev)), 0.0)
    run_cost = cost.c_run * billed_replica_minutes(x, billing_block_min)
    switch_cost = cost.beta_start * float(ups.sum())
    penalty = cost.p_in * float(short_in.sum()) + cost.p_off * float(short_off.sum())
    return CostBreakdown(
        gpu_hours=float(x.sum()) / 60.0, starts=float(ups.sum()),
        reservation_blocks=reservation_blocks(x, billing_block_min), run_cost=run_cost,
        switch_cost=switch_cost, unserved_in=float(short_in.sum()),
        unserved_off=float(short_off.sum()), penalty_cost=penalty,
        total_cost=run_cost + switch_cost + penalty, x_mean=float(x.mean()) if x.size else 0.0,
        x_peak=float(x.max()) if x.size else 0.0)


@dataclass(frozen=True)
class RunMetrics:
    """One (policy, workload) pair reduced to the numbers the simulator design asks for."""

    policy: str
    seed: int
    gpu_hours: float
    starts: float
    reservation_blocks: float
    run_cost: float
    switch_cost: float
    penalty_cost: float
    total_cost: float
    cost_ratio: float
    unserved_in: float
    unserved_off: float
    x_mean: float
    x_peak: float
    n_in: int
    n_off: int
    ttft_p95_in_s: float
    ttft_p95_off_s: float
    slo_in: float
    slo_off: float
    slo_first_in: float
    timeouts: int
    rejected: int
    viol_in: float

    def to_row(self) -> Dict[str, object]:
        """Flat dict for a DataFrame row."""
        return asdict(self)


SYNC_WINDOWS: Tuple[float, ...] = (1.0, 2.0, 5.0)  # the minute windows sigma_1 / _2 / _5 use
MIN_SYNC_STUDENTS = 5  # a session needs this many students before its share means anything


def measure_synchrony(stream: ArrivalStream, sessions: Optional[pd.DataFrame] = None,
                      min_students: int = MIN_SYNC_STUDENTS) -> Dict[str, object]:
    """Measured synchrony of a generated stream: sigma_1 / sigma_2 / sigma_5, per session.

    ``sigma_w`` is the share of students whose *first* request of a session falls within ``w``
    minutes of that session's first request, counting only the in-class requests of that session.
    This is the quantity the pilot measures on the two data sources (0.70 on the prompted sign-ins,
    0.24 on the self-paced ASSISTments in-class offsets) and the axis F5 is drawn against; it is
    *not* ``WorkloadConfig.sigma``, which is only the weight with which the two offset pools are
    mixed.  Sessions with fewer than ``min_students`` students are dropped.

    Returns the per-session shares, their median over sessions (``sigma2_median`` is the number
    the sweeps record as ``sigma2_measured``) and the pooled share over every student of every
    kept session.
    """
    sessions = stream.sessions if sessions is None else sessions
    out: Dict[str, object] = {"n_sessions": 0, "n_students": 0,
                              "session_id": np.zeros(0, dtype=np.int64)}
    for w in SYNC_WINDOWS:
        key = f"sigma{int(w)}"
        out[key] = np.zeros(0, dtype=float)
        out[f"{key}_median"] = float("nan")
        out[f"{key}_pooled"] = float("nan")
    sel = (stream.kind == IN_CLASS) & (stream.session_id >= 0) & (stream.students >= 0)
    if sessions is not None and len(sessions):
        sel &= np.isin(stream.session_id, sessions["session_id"].to_numpy())
    if not sel.any():
        logger.debug("measure_synchrony: no in-class request carries a student index")
        return out
    df = pd.DataFrame({"session": stream.session_id[sel], "student": stream.students[sel],
                       "t": stream.t_arrive_s[sel]})
    first = df.groupby(["session", "student"], sort=True)["t"].min().reset_index()
    lag = (first["t"] - first.groupby("session")["t"].transform("min")) / 60.0
    size = first.groupby("session")["student"].transform("size")
    first = first[size >= int(min_students)]
    if first.empty:
        logger.debug("measure_synchrony: no session reaches %d students", min_students)
        return out
    lag = lag.loc[first.index]
    names = []
    for w in SYNC_WINDOWS:
        key = f"sigma{int(w)}"
        first[key] = (lag <= w + 1e-9).astype(float)
        names.append(key)
    per = first.groupby("session")[names].mean()
    out["session_id"] = per.index.to_numpy()
    out["n_sessions"] = int(len(per))
    out["n_students"] = int(len(first))
    for key in names:
        out[key] = per[key].to_numpy()
        out[f"{key}_median"] = float(np.median(per[key].to_numpy()))
        out[f"{key}_pooled"] = float(first[key].mean())
    logger.debug("measure_synchrony: %d sessions, sigma2 median %.3f pooled %.3f", len(per),
                 out["sigma2_median"], out["sigma2_pooled"])
    return out


def _p95(values: np.ndarray) -> float:
    """95th percentile, NaN when there is nothing to take it of."""
    v = values[np.isfinite(values)]
    return float(np.percentile(v, 95)) if v.size else float("nan")


def _first_minutes_mask(stream: ArrivalStream, first_minutes: int) -> np.ndarray:
    """In-class requests that arrive within ``first_minutes`` of their session's bell."""
    mask = np.zeros(stream.n_requests, dtype=bool)
    if stream.sessions.empty:
        return mask
    bell = np.full(int(stream.sessions["session_id"].max()) + 1, -1, dtype=np.int64)
    bell[stream.sessions["session_id"].to_numpy()] = stream.sessions["bell_min"].to_numpy()
    sel = (stream.kind == IN_CLASS) & (stream.session_id >= 0)
    idx = stream.session_id[sel]
    lag = (stream.t_arrive_s[sel] / 60.0) - bell[idx]
    mask[np.flatnonzero(sel)] = (lag >= 0.0) & (lag < first_minutes)
    return mask


def summarize(policy: str, seed: int, x: np.ndarray, stream: ArrivalStream,
              result: ServiceResult, mu: float, cost: CostConfig,
              billing_block_min: float = 1.0,
              offline_total: Optional[float] = None) -> RunMetrics:
    """Reduce one run to a ``RunMetrics``."""
    brk = fluid_cost(x, stream.lam_in, stream.lam_off, mu, cost, billing_block_min)
    is_in = stream.kind == IN_CLASS
    ttft = result.ttft_s
    ok_in = is_in & result.served & (ttft <= cost.slo_in_s)
    ok_off = (~is_in) & result.served & (ttft <= cost.slo_off_s)
    n_in, n_off = int(is_in.sum()), int((~is_in).sum())
    first = _first_minutes_mask(stream, cost.first_minutes)
    n_first = int(first.sum())
    ratio = brk.total_cost / offline_total if offline_total and offline_total > 0 else float("nan")
    return RunMetrics(
        policy=policy, seed=seed, gpu_hours=brk.gpu_hours, starts=brk.starts,
        reservation_blocks=brk.reservation_blocks, run_cost=brk.run_cost,
        switch_cost=brk.switch_cost, penalty_cost=brk.penalty_cost, total_cost=brk.total_cost,
        cost_ratio=ratio, unserved_in=brk.unserved_in, unserved_off=brk.unserved_off,
        x_mean=brk.x_mean, x_peak=brk.x_peak, n_in=n_in, n_off=n_off,
        ttft_p95_in_s=_p95(ttft[is_in]), ttft_p95_off_s=_p95(ttft[~is_in]),
        slo_in=float(ok_in.sum()) / n_in if n_in else float("nan"),
        slo_off=float(ok_off.sum()) / n_off if n_off else float("nan"),
        slo_first_in=float((ok_in & first).sum()) / n_first if n_first else float("nan"),
        timeouts=int(result.timed_out.sum()), rejected=int(result.rejected.sum()),
        viol_in=1.0 - (float(ok_in.sum()) / n_in) if n_in else float("nan"))


def weekly_rows(policy: str, seed: int, x: np.ndarray, stream: ArrivalStream,
                result: ServiceResult, mu: float, cost: CostConfig,
                weeks: Sequence[int], offline_weekly: Optional[Sequence[float]] = None,
                billing_block_min: float = 1.0) -> List[Dict[str, object]]:
    """One row per teaching week of a run: cost, GPU-hours, starts and in-class SLO attainment.

    ``weeks`` are the teaching-week labels of ``WorkloadConfig.weeks``, in order; week ``i`` covers
    minutes ``[i * 10080, (i + 1) * 10080)``.  ``offline_weekly`` is the whole-run oracle's cost
    restricted to the same weeks, used as the denominator of the weekly ``cost_ratio``.
    """
    x = np.asarray(x, dtype=float)
    is_in = stream.kind == IN_CLASS
    minute = np.minimum((stream.t_arrive_s / 60.0).astype(np.int64), stream.horizon_min - 1)
    ok_in = is_in & result.served & (result.ttft_s <= cost.slo_in_s)
    out: List[Dict[str, object]] = []
    for i, label in enumerate(weeks):
        lo, hi = i * _MIN_PER_WEEK, min(stream.horizon_min, (i + 1) * _MIN_PER_WEEK)
        if hi <= lo:
            continue
        brk = fluid_cost(x[lo:hi], stream.lam_in[lo:hi], stream.lam_off[lo:hi], mu, cost,
                         billing_block_min, x_prev=float(x[lo - 1]) if lo > 0 else 0.0)
        sel = (minute >= lo) & (minute < hi)
        n_in = int((sel & is_in).sum())
        denom = (float(offline_weekly[i]) if offline_weekly is not None
                 and i < len(offline_weekly) else 0.0)
        attained = float((sel & ok_in).sum()) / n_in if n_in else float("nan")
        out.append({"policy": policy, "seed": seed, "week": int(label),
                    "gpu_hours": brk.gpu_hours, "starts": brk.starts,
                    "run_cost": brk.run_cost, "switch_cost": brk.switch_cost,
                    "penalty_cost": brk.penalty_cost, "total_cost": brk.total_cost,
                    "cost_ratio": brk.total_cost / denom if denom > 0 else float("nan"),
                    "unserved_in": brk.unserved_in, "unserved_off": brk.unserved_off,
                    "x_mean": brk.x_mean, "x_peak": brk.x_peak, "n_in": n_in,
                    "n_off": int((sel & ~is_in).sum()), "slo_in": attained,
                    "viol_in": 1.0 - attained if n_in else float("nan")})
    return out


def to_frame(rows) -> pd.DataFrame:
    """Collect ``RunMetrics`` (or dicts) into a DataFrame."""
    return pd.DataFrame([r.to_row() if isinstance(r, RunMetrics) else r for r in rows])


__all__ = ["MIN_SYNC_STUDENTS", "SYNC_WINDOWS", "CostBreakdown", "RunMetrics",
           "billed_replica_minutes", "fluid_cost", "measure_synchrony", "reservation_blocks",
           "summarize", "to_frame", "weekly_rows"]
