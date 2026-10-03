"""Discrete-event provisioning simulator for batch C.

Calendar -> arrival stream (``workload``) -> replica pool and TTFT (``service``), driven by a
provisioning policy (``policies``) through the event loop in ``engine``.  ``metrics`` turns one run
into the numbers of the simulator design and ``sweeps`` grids them into the CSVs that feed F4-F9.

Policies register themselves here::

    from pilot.sim import PolicyFactory
    cls = PolicyFactory("calendar")
"""

import logging
from typing import Callable, Dict, Type

from .config import (
    LEAD_TIERS,
    CostConfig,
    PolicyConfig,
    RunConfig,
    ServiceConfig,
    SweepConfig,
    WorkloadConfig,
)

logger = logging.getLogger(__name__)

POLICY_FACTORY: Dict[str, Type] = {}


def register_policy(name: str) -> Callable[[Type], Type]:
    """Register a ``Policy`` subclass under ``name``."""

    def decorator(cls: Type) -> Type:
        POLICY_FACTORY[name] = cls
        logger.debug("registered policy %s", name)
        return cls

    return decorator


def PolicyFactory(name: str) -> Type:
    """Return the policy class registered under ``name``."""
    try:
        return POLICY_FACTORY[name]
    except KeyError as exc:
        raise KeyError(f"unknown policy '{name}'; known: {sorted(POLICY_FACTORY)}") from exc


from . import policies  # noqa: E402,F401  (import for the registration side effects)
from .engine import RunResult, derive_rngs, run_policy, run_all  # noqa: E402
from .metrics import CostBreakdown, RunMetrics, fluid_cost, weekly_rows  # noqa: E402
from .policies import (  # noqa: E402
    BDInterpPolicy,
    Policy,
    PolicyContext,
    SimState,
    StaleCalendarPolicy,
)
from .service import ServiceResult, load_service_config, replay  # noqa: E402
from .stale import stale_forecast, stale_sessions  # noqa: E402
from .sweeps import SWEEPS, SweepResult, run_sweep  # noqa: E402
from .workload import (  # noqa: E402
    ArrivalStream,
    calendar_forecast,
    forecast_error,
    generate,
    load_sessions,
)

__all__ = [
    "ArrivalStream",
    "BDInterpPolicy",
    "CostBreakdown",
    "CostConfig",
    "LEAD_TIERS",
    "POLICY_FACTORY",
    "Policy",
    "PolicyConfig",
    "PolicyContext",
    "PolicyFactory",
    "RunConfig",
    "RunMetrics",
    "RunResult",
    "SWEEPS",
    "ServiceConfig",
    "ServiceResult",
    "SimState",
    "StaleCalendarPolicy",
    "SweepConfig",
    "SweepResult",
    "WorkloadConfig",
    "calendar_forecast",
    "derive_rngs",
    "fluid_cost",
    "forecast_error",
    "generate",
    "load_service_config",
    "load_sessions",
    "register_policy",
    "replay",
    "run_all",
    "run_policy",
    "run_sweep",
    "stale_forecast",
    "stale_sessions",
    "weekly_rows",
]
