"""Provisioning policies 1-8 of the simulator design

Importing this package registers every policy in ``pilot.sim.POLICY_FACTORY``.  Policy 0
(``fixed``) is the base tier alone and policy 6a (``calendar_adapt``) the timetable plan with an
online size correction, both added on 2026-09-17 as controls.  Policy 7
(``bdinterp``, also registered under its former name ``trust``) is BDINTERP-d of the BDINTERP-d analysis; policy 9
(``stale_calendar``) is policy 6 driven by last semester's timetable.  ``tap`` is the planner of the
paper's section IV: BDINTERP-d at eps = 0.1 over the learned-prior plan with the online size
correction, and LCP (``tap.py``).
"""

from .base import (
    Policy,
    PolicyContext,
    SimState,
    as_int_plan,
    as_int_replicas,
    hit_cost,
    pointwise_target,
)
from .bdinterp import BDInterpPolicy
from .calendar import CalendarFtPPolicy
from .calendar_adapt import CalendarAdaptPolicy
from .calendar_learn import CalendarLearnPolicy, CalendarLearnPriorPolicy
from .fixed import FixedTierPolicy
from .forecast_ilp import ForecastILPPolicy
from .guard import GuardPolicy
from .lcp import LazyCapacityPolicy
from .offline import (
    InformedOptimumPolicy,
    OfflinePolicy,
    round_to_replicas,
    solve_offline_lp,
)
from .reactive import ReactivePolicy
from .stale_calendar import StaleCalendarPolicy
from .static import StaticPeakPolicy
from .tap import LearnedPriorAdaptPolicy, TAPPolicy
from .velocity import TokenVelocityPolicy

__all__ = [
    "BDInterpPolicy",
    "CalendarAdaptPolicy",
    "CalendarLearnPolicy",
    "CalendarLearnPriorPolicy",
    "CalendarFtPPolicy",
    "FixedTierPolicy",
    "ForecastILPPolicy",
    "GuardPolicy",
    "InformedOptimumPolicy",
    "LazyCapacityPolicy",
    "LearnedPriorAdaptPolicy",
    "OfflinePolicy",
    "Policy",
    "PolicyContext",
    "ReactivePolicy",
    "SimState",
    "StaleCalendarPolicy",
    "StaticPeakPolicy",
    "TAPPolicy",
    "TokenVelocityPolicy",
    "as_int_plan",
    "as_int_replicas",
    "hit_cost",
    "pointwise_target",
    "round_to_replicas",
    "solve_offline_lp",
]
