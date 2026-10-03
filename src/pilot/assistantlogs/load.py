"""Load the anonymised assistant tables (``data/raw/xuexitong/assistant``) and slice them per dataset.

One *dataset* is one teaching class under one teaching design: the lab course (self-paced use
inside class), theory class A (the teacher prompts everyone at the start of a period) and
theory class B (out-of-class use with a recurring homework deadline).  The tables the
anonymiser writes hold several classes per file, one per sheet, so a dataset is a (file, sheet)
pair plus the teaching facts needed to place its arrivals on the timetable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

SELF_PACED = "self_paced_in_class"
PROMPTED = "prompted_in_class"
OFF_CLASS = "off_class_deadline"


def hm_to_min(hm: str) -> float:
    """``"08:30"`` -> 510.0 minutes after midnight."""
    h, m = str(hm).split(":")
    return float(int(h) * 60 + int(m))


@dataclass(frozen=True)
class DatasetSpec:
    """One teaching class: where its anonymised rows are and what its calendar looks like."""

    key: str
    label: str
    stem: str                       # csv stem under the raw directory
    regime: str
    roster: int
    sheet: Optional[str] = None     # None = the file's only session sheet
    bells: Tuple[str, ...] = ()     # class-day bells, earliest first
    class_weekday: Optional[int] = None   # 0 = Monday
    window: Tuple[str, str] = ("00:00", "23:59")  # class window used for peak-to-mean
    burst_pre_min: float = 20.0     # a burst may start this long before its bell
    burst_span_min: float = 100.0   # length of the last bell's window
    deadline_weekday: Optional[int] = None
    deadline_hm: str = "23:59"

    @property
    def bell_minutes(self) -> Tuple[float, ...]:
        """Bells as minutes after midnight."""
        return tuple(hm_to_min(b) for b in self.bells)

    @property
    def window_minutes(self) -> Tuple[float, float]:
        """Class window as minutes after midnight."""
        return hm_to_min(self.window[0]), hm_to_min(self.window[1])


@dataclass(frozen=True)
class AssistantDataset:
    """The two anonymised frames of one dataset, with derived day / time-of-day columns."""

    spec: DatasetSpec
    sessions: pd.DataFrame
    messages: pd.DataFrame

    @property
    def student_messages(self) -> pd.DataFrame:
        """Student turns only (the arrivals that reach the service)."""
        return self.messages[self.messages["role"] == "student"]

    @property
    def class_days(self) -> Tuple[object, ...]:
        """Dates on which the class meets, from the messages themselves."""
        days = self.student_messages
        if self.spec.class_weekday is not None:
            days = days[days["weekday"] == self.spec.class_weekday]
        return tuple(sorted(days["date"].unique()))


def _with_time_columns(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add ``date``, ``weekday`` and ``tod_min`` (minutes after midnight) from a timestamp column."""
    out = frame.copy()
    ts = pd.to_datetime(out[col])
    out["date"] = ts.dt.date
    out["weekday"] = ts.dt.dayofweek
    out["tod_min"] = ts.dt.hour * 60.0 + ts.dt.minute + ts.dt.second / 60.0
    return out


def load_dataset(spec: DatasetSpec, raw_dir: Path) -> AssistantDataset:
    """Read ``<stem>_sessions.csv`` / ``<stem>_messages.csv`` and keep this dataset's sheet."""
    sessions = pd.read_csv(raw_dir / f"{spec.stem}_sessions.csv",
                           parse_dates=["created", "ended"])
    messages = pd.read_csv(raw_dir / f"{spec.stem}_messages.csv", parse_dates=["ts"])
    if spec.sheet is not None:
        sessions = sessions[sessions["sheet"] == spec.sheet].reset_index(drop=True)
    messages = messages.merge(sessions[["session", "user", "sheet"]], on="session", how="inner")
    sessions = _with_time_columns(sessions, "created")
    messages = _with_time_columns(messages, "ts")
    logger.info("%s: %d sessions, %d messages, %d users, roster %d", spec.key, len(sessions),
                len(messages), sessions["user"].nunique(), spec.roster)
    return AssistantDataset(spec, sessions, messages)


__all__ = ["OFF_CLASS", "PROMPTED", "SELF_PACED", "AssistantDataset", "DatasetSpec", "hm_to_min",
           "load_dataset"]
