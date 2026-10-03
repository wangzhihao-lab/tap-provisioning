"""Policy 4: weekly seasonal history forecast plus an LP with switching costs (SageServe type).

The forecast for minute ``m`` is the realised demand at ``m - 7 days``; the plan is the offline LP
of ``offline.py`` solved on that forecast over a receding horizon.  Teaching week 1 has no
same-slot history, so the policy falls back to the reactive rule of policy 2 - which is exactly the
first-week blind spot corollary 3 of the paper's analysis predicts.

``PolicyConfig.forecast_warm_start = "stale_calendar"`` fills that blind spot with the only
calendar a history-driven planner could actually have on day 1: last semester's timetable (policy
9).  The warm start is solved without the planning margin ``kappa``, which is a knob of policies 6
and 7 only, so switching it on changes nothing but the first season.
"""

import logging
from dataclasses import replace
from typing import Optional

import numpy as np

from .. import register_policy
from .base import Policy, PolicyContext, SimState
from .offline import round_to_replicas, solve_offline_lp
from .reactive import ReactivePolicy
from .stale_calendar import StaleCalendarPolicy

WARM_START_STALE = "stale_calendar"

logger = logging.getLogger(__name__)


@register_policy("forecast_ilp")
class ForecastILPPolicy(Policy):
    """Seasonal-naive demand forecast, planned with the same LP the oracle uses."""

    name = "forecast_ilp"

    def __init__(self, ctx: PolicyContext) -> None:
        self._fallback = ReactivePolicy(ctx)
        super().__init__(ctx)

    def reset(self) -> None:
        """Drop the cached plan, the warm start, the first-week fallback and the last target."""
        self._plan: Optional[np.ndarray] = None
        self._plan_lo = 0
        self._warm: Optional[np.ndarray] = None
        self._last_target = None
        if getattr(self, "_fallback", None) is not None:
            self._fallback.reset()

    def _warm_start(self) -> Optional[np.ndarray]:
        """Policy 9's plan, solved once at ``kappa = 1``; ``None`` when the warm start is off."""
        if self.ctx.policy.forecast_warm_start != WARM_START_STALE:
            return None
        if self._warm is None:
            plain = replace(self.ctx, policy=replace(self.ctx.policy, plan_margin=1.0))
            self._warm = StaleCalendarPolicy(plain).trajectory()
            logger.debug("forecast_ilp warm start: %d minutes of stale plan", self._warm.size)
        return self._warm

    def replan_x_init(self, state: SimState) -> float:
        """The target committed for ``target - 1``: where a re-solve from ``target = t + d`` starts.

        That slot was committed by the previous minute's decision, which ``Policy.decide`` keeps
        as ``_last_target``; ``state.x_now`` is the level installed at ``t``, ``d - 1`` minutes
        older.  Before the first decision of a run it falls back on ``state.x_now``.
        """
        return float(state.x_now) if self._last_target is None else float(self._last_target)

    def observe(self, t: int, state: SimState) -> float:
        """Plan on last week's demand; fall back to the reactive rule before the first season."""
        cfg = self.ctx.policy
        season = cfg.ilp_season_days * 1440
        target = t + self.ctx.lead_min
        if target < season:
            warm = self._warm_start()
            if warm is not None and warm.size:
                return float(warm[min(target, warm.size - 1)])
            return self._fallback.observe(t, state)
        if self._plan is None or target >= self._plan_lo + cfg.ilp_replan_min:
            hi = min(self.ctx.horizon_min, target + cfg.ilp_horizon_min)
            idx = np.clip(np.arange(target, hi) - season, 0, max(0, state.t - 1))
            f_in, f_off = state.lam_in[idx], state.lam_off[idx]
            x0 = self.replan_x_init(state)
            traj = solve_offline_lp(f_in, f_off, self.ctx.mu, self.ctx.cost, 1,
                                    x_min=float(self.ctx.policy.min_replicas), x_init=x0)
            self._plan = round_to_replicas(traj, f_in, f_off, self.ctx.mu, self.ctx.cost,
                                           x_min=float(self.ctx.policy.min_replicas), x_init=x0)
            self._plan_lo = target
            logger.debug("forecast_ilp replanned at minute %d over %d minutes", t, idx.size)
        off = target - self._plan_lo
        return float(self._plan[min(off, len(self._plan) - 1)]) if len(self._plan) else 0.0

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """The peak of the plan in force at the block's decision minute, over that block.

        The plan is a receding-horizon one, so the block may reach past the minute the policy
        would next replan at; it does not know that yet, which is exactly why the level is read off
        the plan it holds now.  Before the first season it is the warm start if there is one, and
        otherwise the reactive fallback, which is a feedback rule and keeps the causal default.
        """
        lo, hi = int(start_slot), max(int(end_slot), int(start_slot) + 1)
        if lo < self.ctx.policy.ilp_season_days * 1440:
            warm = self._warm_start()
            if warm is None or not warm.size:
                return super().block_target(start_slot, end_slot)
            idx = np.clip(np.arange(lo, hi), 0, warm.size - 1)
            return float(warm[idx].max())
        if self._plan is None or not len(self._plan):
            return super().block_target(start_slot, end_slot)
        idx = np.clip(np.arange(lo, hi) - self._plan_lo, 0, len(self._plan) - 1)
        return float(np.asarray(self._plan)[idx].max())


__all__ = ["WARM_START_STALE", "ForecastILPPolicy"]
