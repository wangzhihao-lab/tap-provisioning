"""Policy interface: ``observe(t, state) -> replicas effective at t + d``.

Every policy sees the same three objects and nothing else:

``PolicyContext``  what is known before the run starts: the horizon, the fluid capacity mu of one
                   replica, the lead time d in minutes, the commitment block length, the cost model
                   and, for the calendar policies only, the forecast lambda-hat.
``SimState``       what has been revealed by minute ``t``: the realised per-minute demand and token
                   counts (indices ``< t`` only) and the replica count currently warm.
``observe``        returns a non-negative replica target; the engine rounds it up and installs it
                   at minute ``t + d``.
``block_target``   only on a tier that commits capacity in blocks: the level to hold over one
                   block, asked for at the block's decision minute so that the answer can use
                   nothing that minute does not already know.

Policies must never index the demand arrays at or beyond ``t``; the helpers below are the
supported way to read the past.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import CostConfig, PolicyConfig, ServiceConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy may know before minute 0."""

    horizon_min: int
    mu: float
    lead_min: int
    cost: CostConfig
    policy: PolicyConfig
    service: ServiceConfig
    lam_hat_in: np.ndarray
    lam_hat_off: np.ndarray
    trust: float = 1.0
    lam_stale_in: Optional[np.ndarray] = None  # policy 9's in-class forecast, or None
    commit_block_min: int = 1  # capacity is piecewise-constant on blocks of this length
    bells: Optional[np.ndarray] = None  # bell minutes of the timetable slice (the family's grid)
    lam_hat_learn_in: Optional[np.ndarray] = None  # policy 12's forecast (timing learned from past weeks)
    lam_hat_learn_prior_in: Optional[np.ndarray] = None  # policy 12a's forecast (warm-started from an earlier semester)

    @property
    def lam_hat(self) -> np.ndarray:
        """Total forecast demand per minute."""
        return self.lam_hat_in + self.lam_hat_off

    @property
    def stale_in(self) -> np.ndarray:
        """Policy 9's in-class forecast; an empty calendar when last semester is not on disk."""
        return (np.zeros_like(self.lam_hat_in) if self.lam_stale_in is None
                else self.lam_stale_in)


@dataclass
class SimState:
    """What minute ``t`` has revealed.  Only indices strictly below ``t`` are legitimate."""

    lam_in: np.ndarray
    lam_off: np.ndarray
    tokens: np.ndarray
    x_now: float = 0.0
    t: int = 0

    def past(self, arr: np.ndarray, window: int) -> np.ndarray:
        """The last ``window`` revealed values of ``arr`` (shorter near the start of the run)."""
        hi = max(0, self.t)
        lo = max(0, hi - window)
        return arr[lo:hi]

    def past_in(self, window: int) -> np.ndarray:
        """Revealed in-class demand over the last ``window`` minutes."""
        return self.past(self.lam_in, window)

    def past_total(self, window: int) -> np.ndarray:
        """Revealed total demand over the last ``window`` minutes."""
        return self.past(self.lam_in, window) + self.past(self.lam_off, window)

    def last_total(self, default: float = 0.0) -> float:
        """Realised total demand in the last revealed minute."""
        return float(self.lam_in[self.t - 1] + self.lam_off[self.t - 1]) if self.t > 0 else default


class Policy(ABC):
    """Base class; subclasses register themselves through ``pilot.sim.register_policy``."""

    name: str = "policy"

    _last_target: Optional[float] = None  # newest ``decide`` target, the ``block_target`` default

    def __init__(self, ctx: PolicyContext) -> None:
        self.ctx = ctx
        self.reset()

    def reset(self) -> None:
        """Clear per-run state.  Called by ``__init__`` and by the engine before a replay."""

    def initial_level(self) -> float:
        """Replica count during the first ``d`` minutes, before any decision can take effect."""
        return float(self.ctx.policy.min_replicas)

    def prefix(self, lead: int) -> np.ndarray:
        """Targets for minutes ``0 .. lead-1``; open-loop policies override this."""
        return np.full(max(0, lead), self.initial_level(), dtype=float)

    @abstractmethod
    def observe(self, t: int, state: SimState) -> float:
        """Replica target effective at minute ``t + lead_min``."""

    # --- commitment blocks -------------------------------------------------------------------
    def start_run(self) -> None:
        """Forget the last decision.  The engine calls this before the decision loop starts."""
        self._last_target = None

    def decide(self, t: int, state: SimState) -> float:
        """``observe`` plus a record of the target, which ``block_target`` falls back on."""
        value = float(self.observe(t, state))
        self._last_target = value
        return value

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """Level held over the whole commitment block ``[start_slot, end_slot)``.

        The block is decided ``lead_min`` minutes before ``start_slot``, so the value may only
        depend on what that minute knows.  The default is the causal answer for a feedback policy:
        the target it had just produced, held for the block.  A policy that already knows its own
        plan for the whole block at that minute - the timetable policies, history + ILP, the two
        optima, the static pool - overrides this with the maximum of that plan over the block.
        """
        return self.initial_level() if self._last_target is None else float(self._last_target)

    # --- helpers shared by the closed-loop policies ---------------------------------------
    def replicas_for(self, demand_per_min: float, penalty: Optional[float] = None) -> float:
        """Whole replicas for ``demand_per_min`` requests per minute.

        The last replica is only taken when the demand it would cover is worth more than it costs:
        ``penalty * uncovered > c_run``.  Ceiling unconditionally would keep a replica alive for a
        trickle of off-class traffic and make every closed-loop policy pay the static cost.
        """
        cost, mu = self.ctx.cost, max(1e-9, self.ctx.mu)
        need = self.ctx.policy.headroom * max(0.0, demand_per_min) / mu
        lo = math.floor(need + 1e-9)
        uncovered = (need - lo) * mu
        p = cost.p_in if penalty is None else penalty
        take = lo + 1 if p * uncovered > cost.c_run else lo
        return max(float(self.ctx.policy.min_replicas), float(take))

    def clip(self, x: float) -> float:
        """Clamp a target into ``[min_replicas, x_max]``."""
        return float(min(self.ctx.cost.x_max, max(self.ctx.policy.min_replicas, x)))


def hit_cost(x: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray, mu: float,
             cost: CostConfig) -> np.ndarray:
    """Per-minute hit cost ``f_t(x) = c x + h_t(x)`` of the paper's analysis (in-class first)."""
    served = mu * np.asarray(x, dtype=float)
    short_in = np.maximum(lam_in - served, 0.0)
    short_all = np.maximum(lam_in + lam_off - served, 0.0)
    return cost.c_run * x + (cost.p_in - cost.p_off) * short_in + cost.p_off * short_all


def pointwise_target(lam_in: np.ndarray, lam_off: np.ndarray, mu: float,
                     cost: CostConfig) -> np.ndarray:
    """Minimiser of ``f_t`` alone: cover both classes when serving is cheaper than the penalty."""
    if cost.p_off * mu > cost.c_run:
        return (lam_in + lam_off) / max(1e-9, mu)
    if cost.p_in * mu > cost.c_run:
        return lam_in / max(1e-9, mu)
    return np.zeros_like(lam_in, dtype=float)


def as_int_plan(plan: np.ndarray, cost: Optional[CostConfig] = None, floor: float = 0.0
                ) -> np.ndarray:
    """Round a fluid trajectory up to whole replicas (the integrality remark of the paper's analysis).

    Ceiling, not nearest: the integrality remark bounds the cost of an integer trajectory by the
    fluid one *plus* the rounding, which only holds when no slot is rounded down below the fluid
    level.  Every fluid policy goes through this helper (``engine.trajectory`` for the per-minute
    targets, ``as_int_replicas`` for a policy that keeps its own integer trace) so that one rule
    holds for all of them.  ``floor`` is the always-on base tier (``PolicyConfig.min_replicas``):
    a scenario constraint, so the engine applies it to every realised trajectory whatever the
    policy asked for.
    """
    out = np.ceil(np.asarray(plan, dtype=float) - 1e-9)
    hi = cost.x_max if cost else np.inf
    return np.clip(out, min(max(0.0, float(floor)), hi), hi)


def as_int_replicas(value: float, cost: Optional[CostConfig] = None, floor: float = 0.0) -> float:
    """Scalar form of ``as_int_plan``: one fluid level rounded up to whole replicas."""
    return float(as_int_plan(np.asarray([value], dtype=float), cost, floor)[0])


__all__ = ["Policy", "PolicyContext", "SimState", "as_int_plan", "as_int_replicas",
           "hit_cost", "pointwise_target"]
