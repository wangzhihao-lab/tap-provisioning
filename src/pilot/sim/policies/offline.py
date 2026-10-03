"""Policy 8: the offline optimum, and the informed optimum with lead time ``d``.

Both are linear programs over the soft-constraint cost of the paper's analysis:

    min  sum_t [ c x_t + beta s_t + (p_in - p_off) u_t + p_off w_t ]
    s.t. s_t >= x_t - x_{t-1},  u_t >= lam_in_t - mu x_t,  w_t >= lam_t - mu x_t,  all >= 0

``(p_in - p_off) u + p_off w`` is exactly ``p_in (lam_in - mu x)^+ + p_off (off-class shortfall)``
under strict in-class priority, and is convex because ``p_in > p_off``.

The offline optimum decides every minute with no lead time, so its blocks are the instance's own
commitment granularity (``ServiceConfig.commit_block_min``, 1 on the per-minute tiers).  The
informed optimum keeps full knowledge of the demand but may only change its commitment once per
lead time, i.e. ``x`` is constant on blocks of ``d`` minutes - or on the commitment block when that
is coarser; it is the achievable lower bound.  Both plan on the same blocks
the engine then enforces, so neither is asked for a trajectory the tier cannot buy.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

from .. import register_policy
from ..config import CostConfig
from .base import Policy, PolicyContext, SimState

logger = logging.getLogger(__name__)


def _block_map(horizon: int, block_min: int) -> np.ndarray:
    """Minute -> block index."""
    return np.arange(horizon) // max(1, int(block_min))


def _suffix_argmin(v: np.ndarray, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(min_{y>=x} v[y], one minimiser)`` for every ``x``."""
    smin = np.minimum.accumulate(v[::-1])[::-1]
    cand = np.where(v <= smin + 1e-12, states, states.size)
    return smin, np.minimum.accumulate(cand[::-1])[::-1]


def _prefix_argmin(v: np.ndarray, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(min_{y<=x} v[y], one minimiser)`` for every ``x``."""
    pmin = np.minimum.accumulate(v)
    cand = np.where(v <= pmin + 1e-12, states, -1)
    return pmin, np.maximum.accumulate(cand)


def round_to_replicas(z: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray, mu: float,
                      cost: CostConfig, block_min: int = 1, x_min: float = 0.0,
                      x_init: float = 0.0) -> np.ndarray:
    """Exact integer optimum for ``lam``, over replica counts bounded by the fluid solution ``z``.

    The LP relaxation is loose whenever one replica serves far more than the per-minute demand (a
    single replica is worth 295 req/min under the placeholder calibration), so rounding the fluid
    trajectory is not enough for an oracle that has to dominate every other policy.  This is the
    O(T * X) dynamic program for

        min sum_j [ F_j(x_j) + beta (x_j - x_{j-1})^+ ],  x_j integer in [x_min, X],

    where ``F_j`` aggregates the hit cost over block ``j``, ``X`` comes from the fluid solution and
    ``x_{-1} = x_init``.  Remark R1 of the paper's analysis is therefore evaluated exactly rather than
    bounded.  ``x_min`` is the always-on base tier (``PolicyConfig.min_replicas``): the states start
    there, so the result is the exact optimum *under* the floor, not a floor-less optimum lifted
    onto it.  ``x_init`` is the level in force before the first block, the same boundary as the
    ``x_init`` of ``solve_offline_lp``: 0, the empty pool, for a plan solved from minute 0; for a
    re-solve in mid-run, the level already committed for the slot before it, so that the first
    block pays a start only for the replicas above that level.
    """
    lam_in = np.asarray(lam_in, dtype=float)
    lam_off = np.asarray(lam_off, dtype=float)
    horizon = lam_in.size
    if horizon == 0:
        return np.zeros(0, dtype=float)
    lo = int(min(max(0.0, float(x_min)), cost.x_max))
    top = max(float(np.max(z)) if np.size(z) else 0.0,
              float(np.max(lam_in + lam_off)) / max(1e-9, mu), float(lo))
    n_x = int(min(cost.x_max, np.ceil(top) + 1.0)) + 1
    states = np.arange(lo, n_x, dtype=float)  # replica counts; the DP indexes their positions
    per_min = (cost.c_run * states[None, :]
               + (cost.p_in - cost.p_off) * np.maximum(lam_in[:, None] - mu * states[None, :], 0.0)
               + cost.p_off * np.maximum((lam_in + lam_off)[:, None] - mu * states[None, :], 0.0))
    blk = _block_map(horizon, block_min)
    starts = np.flatnonzero(np.diff(blk, prepend=-1))
    cost_j = np.add.reduceat(per_min, starts, axis=0)
    n_x = states.size
    idx = np.arange(n_x)

    value = cost_j[0] + cost.beta_start * np.maximum(states - x_init, 0.0)
    pred = np.zeros((cost_j.shape[0], n_x), dtype=np.int32)
    for j in range(1, cost_j.shape[0]):
        smin, sarg = _suffix_argmin(value, idx)
        pmin, parg = _prefix_argmin(value - cost.beta_start * states, idx)
        up = pmin + cost.beta_start * states
        use_up = up < smin
        value = cost_j[j] + np.where(use_up, up, smin)
        pred[j] = np.where(use_up, parg, sarg)
    path = np.zeros(cost_j.shape[0], dtype=np.int64)
    path[-1] = int(np.argmin(value))
    for j in range(cost_j.shape[0] - 1, 0, -1):
        path[j - 1] = pred[j][path[j]]
    logger.debug("integer DP: %d blocks, %d states, cost %.3f", cost_j.shape[0], n_x,
                 float(value.min()))
    return states[path][blk]


def solve_offline_lp(lam_in: np.ndarray, lam_off: np.ndarray, mu: float, cost: CostConfig,
                     block_min: int = 1, x_init: float = 0.0, x_min: float = 0.0) -> np.ndarray:
    """Optimal fluid trajectory for the revealed demand; ``x`` is constant on ``block_min`` blocks.

    ``x_min`` is the always-on base tier (``PolicyConfig.min_replicas``): without it the optimum of
    a lightly loaded institution drops almost every off-class request rather than keep one replica
    up, which is not a deployment anyone runs.
    """
    lam_in = np.asarray(lam_in, dtype=float)
    lam_off = np.asarray(lam_off, dtype=float)
    horizon = lam_in.size
    if horizon == 0:
        return np.zeros(0, dtype=float)
    blk = _block_map(horizon, block_min)
    n_b = int(blk[-1]) + 1
    per_block = np.bincount(blk, minlength=n_b).astype(float)

    rows = np.arange(horizon)
    sel = csr_matrix((np.ones(horizon), (rows, blk)), shape=(horizon, n_b))
    eye_b = csr_matrix((np.ones(n_b), (np.arange(n_b), np.arange(n_b))), shape=(n_b, n_b))
    shift = csr_matrix((np.ones(max(0, n_b - 1)), (np.arange(1, n_b), np.arange(n_b - 1))),
                       shape=(n_b, n_b))
    eye_t = csr_matrix((np.ones(horizon), (rows, rows)), shape=(horizon, horizon))
    zero_tb = csr_matrix((horizon, n_b))
    zero_bt = csr_matrix((n_b, horizon))

    a_sw = hstack([eye_b - shift, -eye_b, zero_bt, zero_bt])
    a_in = hstack([-mu * sel, zero_tb, -eye_t, csr_matrix((horizon, horizon))])
    a_all = hstack([-mu * sel, zero_tb, csr_matrix((horizon, horizon)), -eye_t])
    a_ub = vstack([a_sw, a_in, a_all]).tocsr()

    b_sw = np.zeros(n_b)
    b_sw[0] = x_init
    b_ub = np.concatenate([b_sw, -lam_in, -(lam_in + lam_off)])

    obj = np.concatenate([cost.c_run * per_block,
                          np.full(n_b, cost.beta_start),
                          np.full(horizon, cost.p_in - cost.p_off),
                          np.full(horizon, cost.p_off)])
    bounds = [(min(float(x_min), cost.x_max), cost.x_max)] * n_b + [(0.0, None)] * (n_b + 2 * horizon)
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"offline LP failed: {res.message}")
    logger.debug("offline LP: %d blocks, objective %.3f", n_b, res.fun)
    return np.asarray(res.x[:n_b], dtype=float)[blk]


@register_policy("offline")
class OfflinePolicy(Policy):
    """Policy 8: full knowledge, a decision every minute, no lead time."""

    name = "offline"

    @property
    def block_min(self) -> int:
        """Block length of the oracle's own plan: the instance's commitment granularity."""
        return max(1, int(self.ctx.commit_block_min))

    def __init__(self, ctx: PolicyContext, lam_in: Optional[np.ndarray] = None,
                 lam_off: Optional[np.ndarray] = None) -> None:
        self._lam_in = lam_in
        self._lam_off = lam_off
        self._plan: Optional[np.ndarray] = None
        super().__init__(ctx)

    def bind(self, lam_in: np.ndarray, lam_off: np.ndarray) -> None:
        """Hand the realised demand to the oracle (the engine calls this before the run)."""
        self._lam_in, self._lam_off = lam_in, lam_off
        self._plan = None

    def _trajectory(self) -> np.ndarray:
        if self._plan is None:
            if self._lam_in is None or self._lam_off is None:
                raise RuntimeError(f"{self.name} needs the realised demand; call bind() first")
            block = max(1, int(self.block_min))
            z = solve_offline_lp(self._lam_in, self._lam_off, self.ctx.mu, self.ctx.cost, block,
                                 x_min=float(self.ctx.policy.min_replicas))
            self._plan = round_to_replicas(z, self._lam_in, self._lam_off, self.ctx.mu,
                                           self.ctx.cost, block,
                                           x_min=float(self.ctx.policy.min_replicas))
        return self._plan

    def prefix(self, lead: int) -> np.ndarray:
        """The oracle already knows the whole trajectory, so it pre-commits the first ``d``."""
        return self._trajectory()[:max(0, lead)]

    def observe(self, t: int, state: SimState) -> float:
        """Return the oracle trajectory at ``t + d``."""
        plan = self._trajectory()
        idx = min(len(plan) - 1, t + self.ctx.lead_min)
        return float(plan[idx])

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """The oracle's own plan over the block; it is solved on those blocks in the first place."""
        plan = self._trajectory()
        if not len(plan):
            return 0.0
        lo = min(int(start_slot), len(plan) - 1)
        hi = min(max(int(end_slot), lo + 1), len(plan))
        return float(np.asarray(plan)[lo:hi].max())


@register_policy("informed")
class InformedOptimumPolicy(OfflinePolicy):
    """The informed optimum: full knowledge, but the commitment only changes once per lead time."""

    name = "informed"

    @property
    def block_min(self) -> int:
        """Block length: the lead time, or the commitment block when that is the coarser of the two."""
        return max(1, int(self.ctx.lead_min), int(self.ctx.commit_block_min))


__all__ = ["InformedOptimumPolicy", "OfflinePolicy", "round_to_replicas",
           "solve_offline_lp"]
