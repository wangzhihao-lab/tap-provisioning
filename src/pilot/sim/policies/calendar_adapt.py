"""Policy 6a: the timetable plan with an online multiplicative correction.

The simplest way to let a timetable planner learn from what it sees.  Every minute the realised
in-class demand revealed so far is compared with the forecast over the same minutes, and the plan
is re-solved for ``ratio * lambda_hat`` whenever that ratio has moved by more than ``REPLAN_TOL``
since the plan in force was solved (at most once every ``REPLAN_EVERY_MIN`` minutes, and only
for the part of the horizon that can still be changed).  The ratio is a shrunk cumulative
estimate over a trailing window, ``(m0 + sum realised) / (m0 + sum forecast)``, the prior mass
``m0`` being the forecast mass of the first teaching day, so that one odd session does not swing
it.  It corrects a size bias in the forecast - the adoption-error axis of F6 - but neither a
timing error nor a cancelled session, which is the gap the robust branch of BDINTERP-d is there
for.  It is the "calendar plus simple online correction" baseline one would ask for
before crediting the learning-augmented combination with anything.

The ratio and ``m0`` are taken against the forecast the plan is solved for (``forecast()``), so a
subclass that plans on another forecast corrects that one; for policy 6a it is the calendar
forecast ``lambda_hat``.  A re-solve at minute ``t`` replaces the plan from ``t0 = t + d`` on.
Minutes ``t + 1 .. t0 - 1`` are already committed, so the LP and the integer DP both start from the
level the plan committed for ``t0 - 1`` (``replan_x_init``), and the DP pays a start only for
replicas above it.  ``state.x_now`` would be the wrong start: it is the level installed at ``t``,
``d - 1`` minutes older, and inside BDINTERP-d (TAP's advice, ``tap.py``) it is the combined
decision rather than the advice's own sequence.
"""

import logging
from typing import Optional

import numpy as np

from .. import register_policy
from .base import PolicyContext, SimState
from .calendar import CalendarFtPPolicy
from .offline import round_to_replicas, solve_offline_lp

logger = logging.getLogger(__name__)

WINDOW_MIN = 7 * 1440      # trailing window of the ratio estimate
REPLAN_TOL = 0.10          # re-solve once the ratio moved this much (relative)
REPLAN_EVERY_MIN = 60      # and not more often than this
RATIO_LO, RATIO_HI = 0.25, 4.0


@register_policy("calendar_adapt")
class CalendarAdaptPolicy(CalendarFtPPolicy):
    """Policy 6a: follow the timetable plan, rescaled by the observed forecast ratio."""

    name = "calendar_adapt"

    def __init__(self, ctx: PolicyContext) -> None:
        self._planned_ratio = 1.0
        self._last_replan = -10 ** 9
        self._n_replans = 0
        self._m0: Optional[float] = None
        super().__init__(ctx)

    def reset(self) -> None:
        """Forget the corrections of a previous replay."""
        self._planned_ratio = 1.0
        self._last_replan = -10 ** 9
        self._n_replans = 0
        self._plan = None

    @property
    def n_replans(self) -> int:
        """How many times the plan has been re-solved in this run."""
        return self._n_replans

    @property
    def planned_ratio(self) -> float:
        """The ratio the plan in force was solved for (1.0 until the first re-solve)."""
        return self._planned_ratio

    def _prior_mass(self) -> float:
        if self._m0 is None:
            hat = self.forecast()[0]
            first_day = float(hat[:1440].sum())
            self._m0 = first_day if first_day > 0 else max(1.0, float(hat.sum()))
        return self._m0

    def ratio(self, state: SimState) -> float:
        """Shrunk realised / forecast in-class mass over the trailing window, clipped."""
        lo = max(0, state.t - WINDOW_MIN)
        real = float(state.lam_in[lo:state.t].sum())
        hat = float(self.forecast()[0][lo:state.t].sum())
        m0 = self._prior_mass()
        return float(np.clip((m0 + real) / (m0 + hat), RATIO_LO, RATIO_HI))

    def replan_x_init(self, state: SimState, plan: np.ndarray, t0: int) -> float:
        """The plan's own, already committed level at ``t0 - 1``; the empty pool before minute 0."""
        return float(plan[t0 - 1]) if t0 > 0 else 0.0

    def _replan(self, ratio: float, state: SimState) -> None:
        """Re-solve the plan from the first minute a decision can still reach."""
        plan = self.trajectory().copy()
        t0 = min(plan.size, state.t + self.ctx.lead_min)
        if t0 >= plan.size:
            return
        hat_in, hat_off = self.forecast()
        kappa = max(0.0, float(self.ctx.policy.plan_margin))
        scaled_in = kappa * ratio * hat_in[t0:]
        scaled_off = kappa * hat_off[t0:]
        x_min = float(self.ctx.policy.min_replicas)
        x0 = self.replan_x_init(state, plan, t0)
        z = solve_offline_lp(scaled_in, scaled_off, self.ctx.mu, self.ctx.cost, 1,
                             x_init=x0, x_min=x_min)
        plan[t0:] = round_to_replicas(z, scaled_in, scaled_off, self.ctx.mu, self.ctx.cost,
                                      x_min=x_min, x_init=x0)
        self._plan = plan
        self._planned_ratio = ratio
        self._last_replan = state.t
        self._n_replans += 1
        logger.debug("%s: replanned at minute %d with ratio %.3f", self.name, state.t, ratio)

    def observe(self, t: int, state: SimState) -> float:
        """The (possibly rescaled) plan for minute ``t + d``."""
        if t - self._last_replan >= REPLAN_EVERY_MIN:
            ratio = self.ratio(state)
            if abs(ratio / self._planned_ratio - 1.0) > REPLAN_TOL:
                self._replan(ratio, state)
        return super().observe(t, state)


__all__ = ["CalendarAdaptPolicy"]
