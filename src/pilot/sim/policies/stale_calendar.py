"""Policy 9: last semester's timetable as advice.

The same follow-the-prediction rule as policy 6, solved on the forecast ``sim/stale.py`` builds
from the *previous* semester's slot table.  It isolates how much of policy 6's advantage is the
calendar itself and how much is the calendar being current: the plan still knows that Tuesday
10:20 is a bell and roughly how many seats sit behind it, but it books last semester's sections,
which this semester has re-timed, re-sized, added or dropped.

Only the in-class term is stale; the off-class background is a per-student Poisson process rather
than a timetable artefact, so it is shared with policy 6 (see ``sim/stale.py``).  The planning
margin ``kappa`` applies here too - a stale plan is exactly the case a margin is bought for.
"""

import logging
from typing import Tuple

import numpy as np

from .. import register_policy
from .calendar import CalendarFtPPolicy

logger = logging.getLogger(__name__)


@register_policy("stale_calendar")
class StaleCalendarPolicy(CalendarFtPPolicy):
    """Policy 9: follow last semester's timetable."""

    name = "stale_calendar"

    def forecast(self) -> Tuple[np.ndarray, np.ndarray]:
        """Last semester's in-class forecast, with the current off-class background."""
        return self.ctx.stale_in, self.ctx.lam_hat_off


__all__ = ["StaleCalendarPolicy"]
