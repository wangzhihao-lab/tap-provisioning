"""Policy 2: reactive threshold autoscaling (HPA / AutoScale type).

Thresholds follow the simulator design: scale up when utilisation stays above 70% for one minute,
scale down when it stays below 30% for five minutes, three minutes of cooldown between moves.

The size of a move is a modelling choice the design left open.  The default here is the
proportional rule Kubernetes HPA actually uses (``desired = ceil(x * util / target)``); the
literal "+/- 1 replica" of the design is available as ``hpa_mode = "step"``.  A one-replica step is
a far weaker baseline under a classroom burst, so the proportional rule is the honest comparison.
"""

import logging
import math

from .. import register_policy
from .base import Policy, SimState

logger = logging.getLogger(__name__)


@register_policy("reactive")
class ReactivePolicy(Policy):
    """Utilisation thresholds with hold times and a cooldown."""

    name = "reactive"

    def reset(self) -> None:
        """Clear the hold counters, the cooldown and the last target."""
        self._up = 0
        self._down = 0
        self._cool = 0
        self._target = float(self.ctx.policy.min_replicas)

    def observe(self, t: int, state: SimState) -> float:
        """Threshold rule on the utilisation of the last revealed minute."""
        cfg = self.ctx.policy
        x = max(1.0, float(state.x_now))
        util = state.last_total() / max(1e-9, self.ctx.mu * x)
        self._up = self._up + 1 if util > cfg.hpa_up_util else 0
        self._down = self._down + 1 if util < cfg.hpa_down_util else 0
        self._cool = max(0, self._cool - 1)
        if self._cool == 0 and self._up >= cfg.hpa_up_hold_min:
            self._target = self._scaled(x, util, up=True)
            self._up, self._down, self._cool = 0, 0, cfg.hpa_cooldown_min
        elif self._cool == 0 and self._down >= cfg.hpa_down_hold_min:
            self._target = self._scaled(x, util, up=False)
            self._up, self._down, self._cool = 0, 0, cfg.hpa_cooldown_min
        else:
            self._target = max(self._target, float(cfg.min_replicas))
        return self.clip(self._target)

    def _scaled(self, x: float, util: float, up: bool) -> float:
        """One scaling move, either proportional (Kubernetes) or a fixed step."""
        cfg = self.ctx.policy
        if cfg.hpa_mode == "step":
            return x + cfg.hpa_step if up else x - cfg.hpa_step
        target_util = cfg.hpa_up_util if up else cfg.hpa_down_util
        desired = math.ceil(x * util / max(1e-9, target_util) - 1e-12)
        return max(1.0, float(desired)) if up else max(float(cfg.min_replicas), float(desired))


__all__ = ["ReactivePolicy"]
