"""Policy 3: token-velocity extrapolation (TokenScale type).

The slope of the token arrival rate over the last ``velocity_window_min`` minutes is extrapolated
``d`` minutes ahead and converted to a request rate with the running mean tokens per request.  It
reacts one lead time earlier than a threshold rule but still only sees the past, so it cannot
anticipate a bell.

Acting on every slope, as the rule was first written here, makes the baseline pay a start almost
every minute: the per-minute token count is noisy, so its slope changes sign constantly and the
pool chases the noise instead of the trend.  That is a straw man rather than a baseline, so the
default carries the hysteresis band a production autoscaler would have:

* a slope is acted on only once it has kept its sign for ``velocity_persist_min`` consecutive
  slots, and
* after a move the target is held for ``velocity_dwell_min`` slots.

``velocity_hysteresis = False`` restores the original slope-every-minute behaviour.  The cold-start
path (nothing revealed yet) sits outside the band in both modes - there is no slope to confirm.
"""

import logging
from typing import Tuple

import numpy as np

from .. import register_policy
from .base import Policy, SimState

logger = logging.getLogger(__name__)

_FLAT = 1e-9  # a slope below this is noise, not a trend


@register_policy("velocity")
class TokenVelocityPolicy(Policy):
    """Linear extrapolation of the token rate over the lead time, with a hysteresis band."""

    name = "velocity"

    def reset(self) -> None:
        """Clear the held target, the sign run length and the dwell counter."""
        self._target = float(self.ctx.policy.min_replicas)
        self._sign = 0
        self._run = 0
        self._dwell = 0

    def _extrapolate(self, state: SimState) -> Tuple[float, float, bool]:
        """``(target, slope, confirmed)``: the raw rule, and whether a slope existed at all."""
        cfg = self.ctx.policy
        window = max(2, cfg.velocity_window_min + 1)
        tok = state.past(state.tokens, window)
        req = state.past_total(window)
        if tok.size < 2 or req.sum() <= 0.0:
            return self.clip(self.replicas_for(state.last_total())), 0.0, False
        per_req = float(tok.sum() / max(1e-9, req.sum()))
        slope = float(tok[-1] - tok[0]) / max(1, tok.size - 1)
        ahead = float(tok[-1]) + slope * self.ctx.lead_min
        ahead = float(np.clip(ahead, 0.0, cfg.velocity_clip * max(tok.max(), 1.0)))
        return self.clip(self.replicas_for(ahead / max(1e-9, per_req))), slope, True

    def observe(self, t: int, state: SimState) -> float:
        """Extrapolate the token rate ``d`` minutes ahead and size the pool for it."""
        cfg = self.ctx.policy
        target, slope, confirmed = self._extrapolate(state)
        if not cfg.velocity_hysteresis or not confirmed:
            self._target = target
            return target
        self._dwell = max(0, self._dwell - 1)
        sign = 1 if slope > _FLAT else (-1 if slope < -_FLAT else 0)
        self._run = (self._run + 1) if (sign != 0 and sign == self._sign) else int(sign != 0)
        self._sign = sign
        if (self._dwell == 0 and self._run >= max(1, cfg.velocity_persist_min)
                and target != self._target):
            self._target = target
            self._dwell = max(0, cfg.velocity_dwell_min)
        return self.clip(self._target)


__all__ = ["TokenVelocityPolicy"]
