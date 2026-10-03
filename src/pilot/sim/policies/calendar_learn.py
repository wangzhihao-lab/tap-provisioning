"""Policies 12 and 12a: the timetable with the trigger timing learned from earlier weeks.

Same follow-the-prediction planner as policy 6, but its forecast places each session's burst at
the offset (and spread) that the section's own realised starts in strictly earlier weeks suggest,
falling back to the pooled estimate for sections without history and to the bell itself in week
1 (``workload.learned_timing``).  Where policy 6 with ``forecast_offset_min`` set knows the timing
distribution, this policy has to learn it: weeks 2 onwards are its held-out test.
"""

import logging
from typing import Tuple

import numpy as np

from .. import register_policy
from .base import PolicyContext
from .calendar import CalendarFtPPolicy

logger = logging.getLogger(__name__)


def learned_prior_forecast(ctx: PolicyContext) -> Tuple[np.ndarray, np.ndarray]:
    """Policy 12a's demand estimate: the prior-warm-started learned timing, off-class background.

    Falls back to policy 12's learned forecast, then to the calendar's, when the stream carries no
    prior (a run without ``--learn-prior-from``).  Shared with TAP's advice (``tap.py``), which
    plans on exactly this forecast.
    """
    hat = ctx.lam_hat_learn_prior_in
    if hat is None:
        logger.warning("calendar_learn_prior: no prior forecast on the stream; using calendar_learn's")
        hat = ctx.lam_hat_learn_in if ctx.lam_hat_learn_in is not None else ctx.lam_hat_in
    return hat, ctx.lam_hat_off


@register_policy("calendar_learn")
class CalendarLearnPolicy(CalendarFtPPolicy):
    """Policy 12: follow the timetable plan with learned trigger timing."""

    name = "calendar_learn"

    def forecast(self) -> Tuple[np.ndarray, np.ndarray]:
        """The learned-timing in-class forecast, with the usual off-class background."""
        hat = self.ctx.lam_hat_learn_in
        if hat is None:
            logger.warning("calendar_learn: no learned forecast on the stream; using the calendar's")
            hat = self.ctx.lam_hat_in
        return hat, self.ctx.lam_hat_off


@register_policy("calendar_learn_prior")
class CalendarLearnPriorPolicy(CalendarFtPPolicy):
    """Policy 12a: policy 12 warm-started with the pooled timing of an earlier semester.

    Identical to ``calendar_learn`` from the first observation of a section onwards; where policy
    12 has nothing yet (all of week 1, a section's first session) it uses the pooled offset and
    spread of the realised starts of the semester named by ``WorkloadConfig.learn_prior_slots_csv``
    instead of the bell.  Runs without that semester fall back to policy 12's forecast.
    """

    name = "calendar_learn_prior"

    def forecast(self) -> Tuple[np.ndarray, np.ndarray]:
        """The prior-warm-started learned forecast, with the usual off-class background."""
        return learned_prior_forecast(self.ctx)


__all__ = ["CalendarLearnPolicy", "CalendarLearnPriorPolicy", "learned_prior_forecast"]
