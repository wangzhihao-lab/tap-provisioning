"""Synthetic calendar-driven arrivals: in-class synchronized bursts, background traffic, deadline ramps."""

import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from ..config import MockConfig, SessionSpec
from . import register_source
from .base import ArrivalSource

logger = logging.getLogger(__name__)

# hour-of-day weights for self-paced background traffic (sums to 1)
_DIURNAL = np.array(
    [0.2, 0.1, 0.05, 0.02, 0.02, 0.05, 0.3, 0.8, 1.2, 1.5, 1.6, 1.4, 1.0, 1.3, 1.6, 1.7, 1.5, 1.2,
     1.4, 2.0, 2.4, 2.2, 1.5, 0.7]
)
_DIURNAL = _DIURNAL / _DIURNAL.sum()


def _session_starts(cfg: MockConfig, spec: SessionSpec) -> List[datetime]:
    """All start times of one recurring session across the configured weeks."""
    first = datetime.fromisoformat(cfg.start_date)
    starts = []
    for w in range(cfg.weeks):
        day = first + timedelta(days=7 * w + spec.weekday)
        starts.append(day.replace(hour=spec.start_hour, minute=spec.start_minute))
    return starts


def _trigger_times(cfg: MockConfig, spec: SessionSpec, start: datetime) -> List[pd.Timestamp]:
    """Trigger instants inside one session; the generator spreads them evenly."""
    step = spec.duration_min / (cfg.triggers_per_session + 1)
    return [pd.Timestamp(start + timedelta(minutes=step * j))
            for j in range(1, cfg.triggers_per_session + 1)]


def _burst(rng: np.random.Generator, cfg: MockConfig, t0: datetime, n: int) -> np.ndarray:
    """Arrival offsets (minutes) of a synchronized in-class burst triggered at t0."""
    k = rng.binomial(n, cfg.sync_fraction)
    delays = rng.lognormal(mean=np.log(cfg.sync_delay_median_min), sigma=cfg.sync_delay_sigma, size=k)
    return np.array([t0 + timedelta(minutes=float(d)) for d in delays], dtype="datetime64[ns]")


def _background(rng: np.random.Generator, cfg: MockConfig, n_students: int) -> np.ndarray:
    """Self-paced traffic: Poisson per hour with a diurnal profile, whole horizon."""
    first = datetime.fromisoformat(cfg.start_date)
    hours = cfg.weeks * 7 * 24
    per_hour = cfg.background_rate_per_student_day * n_students * _DIURNAL[np.arange(hours) % 24]
    counts = rng.poisson(per_hour)
    out = []
    for h, c in enumerate(counts):
        if c:
            base = first + timedelta(hours=h)
            out.extend(base + timedelta(minutes=float(m)) for m in rng.uniform(0, 60, size=c))
    return np.array(out, dtype="datetime64[ns]")


def _deadline_ramp(rng: np.random.Generator, cfg: MockConfig, n_students: int) -> np.ndarray:
    """Extra arrivals whose density grows exponentially toward each weekly deadline."""
    first = datetime.fromisoformat(cfg.start_date)
    out = []
    for w in range(cfg.weeks):
        deadline = (first + timedelta(days=7 * w + cfg.deadline_weekday)).replace(hour=cfg.deadline_hour)
        n = rng.poisson(cfg.deadline_extra_per_student * n_students)
        # hours before the deadline: exponential with mean ramp/3, truncated to the ramp
        before = np.minimum(rng.exponential(cfg.deadline_ramp_hours / 3.0, size=n), cfg.deadline_ramp_hours)
        out.extend(deadline - timedelta(hours=float(b)) for b in before)
    return np.array(out, dtype="datetime64[ns]")


@register_source("mock")
class MockSource(ArrivalSource):
    """Calendar-driven synthetic workload with known ground truth (used for tests and Q1 dry runs)."""

    timetable_kind = "timetable"

    def __init__(self, cfg: MockConfig | None = None) -> None:
        self.cfg = cfg or MockConfig()

    def sessions(self) -> pd.DataFrame:
        """Timetable as a frame: start, duration_min, n_students, group, triggers."""
        rows = [
            {"start": s, "duration_min": spec.duration_min, "n_students": spec.n_students,
             "group": spec.group, "triggers": _trigger_times(self.cfg, spec, s)}
            for spec in self.cfg.sessions
            for s in _session_starts(self.cfg, spec)
        ]
        return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.seed)
        frames = []
        for spec in self.cfg.sessions:
            for start in _session_starts(self.cfg, spec):
                # triggers spread evenly inside the session
                for t0 in _trigger_times(self.cfg, spec, start):
                    ts = _burst(rng, self.cfg, t0, spec.n_students)
                    frames.append(pd.DataFrame({"ts": ts, "group": spec.group, "kind": "in_class"}))
            frames.append(pd.DataFrame({"ts": _background(rng, self.cfg, spec.n_students), "group": spec.group, "kind": "background"}))
            frames.append(pd.DataFrame({"ts": _deadline_ramp(rng, self.cfg, spec.n_students), "group": spec.group, "kind": "deadline"}))
        df = self.validate(pd.concat(frames, ignore_index=True))
        logger.info("mock workload: %d arrivals, %d sessions", len(df), len(self.sessions()))
        return df
