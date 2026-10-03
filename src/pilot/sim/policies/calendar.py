"""Policy 6: the timetable as advice.

``calendar`` is follow-the-prediction with full trust: ``x = offline_opt(kappa * lambda_hat)``,
where ``lambda_hat`` is the calendar forecast of ``workload.calendar_forecast``.  Because the
timetable is known before the semester starts, the whole trajectory is available at minute 0 and
the lead time costs it nothing - that is the asymmetry the paper's analysis builds on.

``kappa = PolicyConfig.plan_margin`` is the planning margin: the planner deliberately books for
``kappa`` times the forecast before solving, which buys insurance against an under-forecast at the
price of running cost.  ``kappa = 1`` is the plain follow-the-prediction rule.  The margin is a
property of the *planner*, so it applies to policies 6 and 7 (which uses this class as its advice
black box) and not to the history-driven policy 4.

This is the ADV black box of policy 7; the combination itself lives in ``bdinterp.py``.
``stale_calendar.py`` reuses it with last semester's timetable as the forecast.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from .. import register_policy
from .base import Policy, PolicyContext, SimState
from .offline import round_to_replicas, solve_offline_lp

logger = logging.getLogger(__name__)


@register_policy("calendar")
class CalendarFtPPolicy(Policy):
    """Policy 6: follow the timetable plan, trust 1."""

    name = "calendar"

    def __init__(self, ctx: PolicyContext) -> None:
        self._plan: Optional[np.ndarray] = None
        super().__init__(ctx)

    def forecast(self) -> Tuple[np.ndarray, np.ndarray]:
        """The demand estimate the plan is solved for; ``stale_calendar`` overrides it."""
        return self.ctx.lam_hat_in, self.ctx.lam_hat_off

    def trajectory(self) -> np.ndarray:
        """The advice sequence ADV = offline optimum of ``kappa`` times the calendar forecast."""
        if self._plan is None:
            hat_in, hat_off = self.forecast()
            kappa = max(0.0, float(self.ctx.policy.plan_margin))
            hat_in, hat_off = kappa * hat_in, kappa * hat_off
            z = solve_offline_lp(hat_in, hat_off, self.ctx.mu, self.ctx.cost, 1,
                                 x_min=float(self.ctx.policy.min_replicas))
            self._plan = round_to_replicas(z, hat_in, hat_off, self.ctx.mu, self.ctx.cost,
                                           x_min=float(self.ctx.policy.min_replicas))
        return self._plan

    def prefix(self, lead: int) -> np.ndarray:
        """The timetable is known before minute 0, so the first ``d`` minutes are pre-committed."""
        return self.trajectory()[:max(0, lead)]

    def observe(self, t: int, state: SimState) -> float:
        """The advice for minute ``t + d``."""
        plan = self.trajectory()
        return float(plan[min(len(plan) - 1, t + self.ctx.lead_min)]) if plan.size else 0.0

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """The plan's peak over the block: the whole timetable is known before minute 0."""
        plan = self.trajectory()
        if not plan.size:
            return 0.0
        lo = min(int(start_slot), plan.size - 1)
        hi = min(max(int(end_slot), lo + 1), plan.size)
        return float(plan[lo:hi].max())


__all__ = ["CalendarFtPPolicy"]
