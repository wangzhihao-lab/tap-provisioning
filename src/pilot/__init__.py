"""Timetable-aware provisioning of LLM teaching-assistant services: data analysis and simulator.

Q1: are education arrivals sharper and more calendar-predictable than generic LLM traffic?
Q2: provisioning lead time vs. burst ramp (measurement harness lives in scripts/q2).
The provisioning simulator and its policies live in ``pilot.sim``.
"""

from .config import (
    AnalysisConfig,
    FigureConfig,
    LoaderConfig,
    MockConfig,
    PredictConfig,
    SessionDetectConfig,
)
from .q1 import run_q1
from .sources import SourceFactory, register_source

__all__ = [
    "AnalysisConfig",
    "FigureConfig",
    "LoaderConfig",
    "MockConfig",
    "PredictConfig",
    "SessionDetectConfig",
    "SourceFactory",
    "register_source",
    "run_q1",
]
