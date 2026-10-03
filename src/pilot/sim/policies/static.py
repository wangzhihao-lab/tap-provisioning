"""Policy 1: static peak provisioning (the upper-bound baseline).

``x`` is constant at the calendar peak demand divided by the fluid capacity of one replica.  It
never violates the SLO once the pool is up, so it is the ceiling on cost and the floor on
violations that every other policy is measured against.
"""

import logging

import numpy as np

from .. import register_policy
from .base import Policy, PolicyContext, SimState

logger = logging.getLogger(__name__)


@register_policy("static")
class StaticPeakPolicy(Policy):
    """Constant capacity at the forecast peak."""

    name = "static"

    def __init__(self, ctx: PolicyContext) -> None:
        super().__init__(ctx)
        peak = float(np.max(ctx.lam_hat)) if ctx.lam_hat.size else 0.0
        self._level = self.clip(self.replicas_for(peak))
        logger.debug("static peak level: %.0f replicas (peak %.1f req/min)", self._level, peak)

    def initial_level(self) -> float:
        """The static pool is provisioned before the run starts."""
        return self._level

    def observe(self, t: int, state: SimState) -> float:
        """Always the peak level."""
        return self._level

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """A constant pool commits the same level whatever the block length is."""
        return self._level


__all__ = ["StaticPeakPolicy"]
