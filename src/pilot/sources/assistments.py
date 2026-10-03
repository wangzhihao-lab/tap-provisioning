"""Loader for the ASSISTments 2012-13 problem log.

The release has wall-clock ``start_time``/``end_time`` and a ``student_class_id``, which makes it
the only public dataset that can show classroom-synchronized bursts (see
refs/scans/D_week0_sweep_I1.md section 1.4). It carries no session id, so class sessions are
recovered by gap clustering: within one class, a new session starts after ``gap_min`` minutes of
inactivity, and a session counts only if at least ``min_students`` distinct students appear in it.

Student identifiers are replaced by a one-way digest as soon as the file is read and are used only
to count distinct students; no identifier is ever written to an output frame.
"""

import hashlib
import logging
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

from ..config import LoaderConfig, SessionDetectConfig
from ..metrics.sessions import gap_sessions
from . import register_source
from .base import ArrivalSource, parse_timestamps

logger = logging.getLogger(__name__)

_TS_COL = "start_time"
_GROUP_COL = "student_class_id"
_USER_COL = "user_id"
_OPTIONAL = ("end_time", "teacher_id", "school_id")


def _digest(value: object) -> str:
    """One-way digest of an already pseudonymous id; used for counting only."""
    return hashlib.blake2s(str(value).encode("utf-8"), digest_size=8).hexdigest()


def _available_columns(path: Path, encoding: str) -> List[str]:
    header = pd.read_csv(path, nrows=0, encoding=encoding)
    return [str(c) for c in header.columns]


def _chunks(path: Path, usecols: List[str], cfg: LoaderConfig,
            encoding: str) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(path, usecols=usecols, chunksize=cfg.chunk_rows, encoding=encoding,
                         dtype={_GROUP_COL: "string", _USER_COL: "string"})
    seen = 0
    for chunk in reader:
        if cfg.max_rows and seen + len(chunk) > cfg.max_rows:
            chunk = chunk.iloc[: cfg.max_rows - seen]
        seen += len(chunk)
        yield chunk
        if cfg.max_rows and seen >= cfg.max_rows:
            logger.info("stopped after max_rows=%d", cfg.max_rows)
            return


@register_source("assistments")
class AssistmentsSource(ArrivalSource):
    """ASSISTments 2012-13 problem log as an arrival stream plus inferred class sessions."""

    workload_class = "education"
    n_paths = 1

    def __init__(self, path: Path, cfg: Optional[LoaderConfig] = None,
                 detect: Optional[SessionDetectConfig] = None, group: str = "assistments") -> None:
        self.path = Path(path)
        self.cfg = cfg or LoaderConfig()
        self.detect = detect or SessionDetectConfig()
        self.group = group
        self._events: Optional[pd.DataFrame] = None

    def _read(self) -> pd.DataFrame:
        """Parse the CSV once into (ts, group, uid); cached on the instance."""
        if self._events is not None:
            return self._events
        if not self.path.exists():
            raise FileNotFoundError(f"ASSISTments CSV not found: {self.path}")
        encoding = "utf-8"
        try:
            columns = _available_columns(self.path, encoding)
        except UnicodeDecodeError:
            encoding = "latin-1"
            columns = _available_columns(self.path, encoding)
            logger.warning("%s is not utf-8; falling back to latin-1", self.path.name)
        if _TS_COL not in columns:
            raise ValueError(
                f"{self.path.name} has no '{_TS_COL}' column (columns: {columns[:12]}...). "
                "ASSISTments 2009-10 has no wall-clock timestamps and cannot be used here."
            )
        usecols = [c for c in (_TS_COL, _GROUP_COL, _USER_COL, *_OPTIONAL) if c in columns]
        frames = []
        for chunk in _chunks(self.path, usecols, self.cfg, encoding):
            part = pd.DataFrame({"ts": parse_timestamps(chunk[_TS_COL])})
            part["group"] = (chunk[_GROUP_COL].fillna("unknown").astype(str)
                             if _GROUP_COL in chunk.columns else self.group)
            if _USER_COL in chunk.columns:
                lut = {v: _digest(v) for v in chunk[_USER_COL].dropna().unique()}
                part["uid"] = chunk[_USER_COL].map(lut)
            frames.append(part.dropna(subset=["ts"]))
        events = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
        for col in ("group", "uid"):  # thousands of classes, tens of thousands of students
            if col in events.columns:
                events[col] = events[col].astype("category")
        if events.empty:
            raise ValueError(f"{self.path.name}: no parsable timestamps in '{_TS_COL}'")
        logger.info("%s: %d problem events, %d classes, %s to %s", self.path.name, len(events),
                    events["group"].nunique(), events["ts"].min(), events["ts"].max())
        self._events = events
        return events

    def load(self) -> pd.DataFrame:
        events = self._read()
        df = events.loc[:, ["ts", "group"]].copy()
        df["kind"] = "problem"
        return self.validate(df)

    def sessions(self) -> pd.DataFrame:
        """Class sessions recovered by gap clustering, one row per session."""
        out = gap_sessions(self._read(), self.detect)
        out["triggers"] = [[t] for t in out["start"]]
        logger.info("%s: %d class sessions (>= %d students)", self.path.name, len(out),
                    self.detect.min_students)
        return out
