"""Policy 0: the base tier alone - ``x`` is ``PolicyConfig.min_replicas`` for the whole run.

The control for the adoption axis (F9): it says what one always-on replica already absorbs, so
that the point at which timetable-aware scaling starts to matter is read off the data rather than
assumed.  With ``min_replicas = 0`` it provisions nothing and every request is a miss.
"""

import logging

from .. import register_policy
from .base import Policy, PolicyContext, SimState

logger = logging.getLogger(__name__)


@register_policy("fixed")
class FixedTierPolicy(Policy):
    """Never scale: hold the base tier."""

    name = "fixed"

    def __init__(self, ctx: PolicyContext) -> None:
        super().__init__(ctx)
        self._level = float(ctx.policy.min_replicas)
        logger.debug("fixed tier: %.0f replicas for the whole run", self._level)

    def initial_level(self) -> float:
        """The tier is up before the run starts."""
        return self._level

    def observe(self, t: int, state: SimState) -> float:
        """Always the base tier."""
        return self._level

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """A constant level commits the same value whatever the block."""
        return self._level


__all__ = ["FixedTierPolicy"]
