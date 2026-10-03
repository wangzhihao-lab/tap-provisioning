"""TAP: the planner of the paper's section IV, as one policy (``tap``).

Three stages in series, each an existing component:

``ADV``  ``LearnedPriorAdaptPolicy``: the timetable plan on the learned-timing forecast warm-started
         from an earlier semester (the forecast of policy 12a, ``calendar_learn_prior``), with the
         online size correction of policy 6a (``calendar_adapt``) applied to that same forecast.
``ROB``  ``LazyCapacityPolicy``: integer LCP, lagged by the lead time (policy 5, unchanged).
``TAP``  ``TAPPolicy``: BDINTERP-d over the two (policy 7's combination), with the trust knob fixed
         at ``eps = 0.1`` - the planner is not tuned; an explicit ``tap@eps`` overrides it the way
         ``bdinterp@eps`` does, and ``PolicyConfig.trust`` is never read.

The advice has to be a sequence of its own for the analysis of BDINTERP-d to apply (see the module
docstring of ``bdinterp.py``): both black boxes run inside the combination on the state the engine
hands it, and neither reads the combined decision.  A size re-solve starts from the advice's own
planned level at ``t0 - 1``, never from ``state.x_now``, which inside the combination is the
combined level; that is policy 6a's rule (``CalendarAdaptPolicy.replan_x_init``), so apart from the
forecast the advice is policy 6a's code throughout - the ratio estimate, its window, tolerance and
clip, the re-solve cadence and the level a re-solve starts from.

Causality: the size correction reads only revealed minutes (``state.lam_in[:t]``) and the forecast
over the same minutes; the forecast itself is built from strictly earlier weeks' realised starts
(``workload.learned_timing``), and a whole-horizon solve on it does not leak later weeks into
earlier decisions (``scripts/audit/prefix_invariance.py``; ``scripts/audit/tap_prefix_invariance.py``
repeats the check for this advice and for the combination).
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from .. import register_policy
from .base import PolicyContext
from .bdinterp import BDInterpPolicy
from .calendar_adapt import CalendarAdaptPolicy
from .calendar_learn import learned_prior_forecast

logger = logging.getLogger(__name__)

TAP_TRUST = 0.1  # BDINTERP-d's eps for TAP: the value of the main experiments, never tuned


class LearnedPriorAdaptPolicy(CalendarAdaptPolicy):
    """TAP's advice: policy 12a's forecast, corrected online in size like policy 6a."""

    name = "learn_prior_adapt"

    def __init__(self, ctx: PolicyContext) -> None:
        self._forecast: Optional[Tuple[np.ndarray, np.ndarray]] = None
        super().__init__(ctx)

    def forecast(self) -> Tuple[np.ndarray, np.ndarray]:
        """``calendar_learn_prior``'s forecast (read once; the ratio asks for it every hour)."""
        if self._forecast is None:
            self._forecast = learned_prior_forecast(self.ctx)
        return self._forecast


@register_policy("tap")
class TAPPolicy(BDInterpPolicy):
    """TAP: BDINTERP-d at ``eps = 0.1`` over (learned-prior adaptive plan, LCP)."""

    name = "tap"
    fixed_trust: float = TAP_TRUST  # read by ``sweeps.run_sweep`` for a spec without ``@eps``

    def make_advice(self, ctx: PolicyContext) -> LearnedPriorAdaptPolicy:
        """The ADV black box: the learned-prior plan with the online size correction."""
        return LearnedPriorAdaptPolicy(ctx)

    @property
    def eps(self) -> float:
        """``0.1``, or the value of an explicit ``tap@eps``; ``PolicyConfig.trust`` is ignored.

        The explicit value travels as ``engine.RunContext.trust_override``; a bare
        ``PolicyContext`` (built outside the engine) has none, so TAP runs at 0.1 on it.
        """
        override = getattr(self.ctx, "trust_override", None)
        return max(0.0, float(self.fixed_trust if override is None else override))

    def reset(self) -> None:
        """Clear the advice's corrections too; the ROB branch is reset by the parent."""
        self._adv.reset()
        super().reset()

    def diagnostics(self) -> Dict[str, float]:
        """Share of ADV-phase slots, number of size re-solves, the ratio in force at the end."""
        return {"phase_share": float(self.phase_share()),
                "n_replans": float(getattr(self._adv, "n_replans", 0)),
                "final_ratio": float(getattr(self._adv, "planned_ratio", 1.0)),
                "trust": float(self.eps)}


__all__ = ["LearnedPriorAdaptPolicy", "TAPPolicy", "TAP_TRUST"]
