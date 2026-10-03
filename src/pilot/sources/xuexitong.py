"""Loaders for Xuexitong (Chaoxing) teacher-side exports.

Only timestamps, group labels and event kinds leave this module; student identifiers are
replaced by salted hashes and names are dropped. Column names follow what the exports show on
screen; unknown layouts raise instead of guessing.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from . import register_source
from .base import ArrivalSource

logger = logging.getLogger(__name__)

_SALT_ENV = "PILOT_ID_SALT"


def hash_id(value: object) -> str:
    """Salted SHA-256 of a student id; the salt comes from the environment, never from disk."""
    salt = os.environ.get(_SALT_ENV)
    if not salt:
        raise RuntimeError(f"set {_SALT_ENV} before loading identifiable exports")
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16]


def _pick(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Return the first column whose name contains any candidate string."""
    for cand in candidates:
        for col in df.columns:
            if cand in str(col):
                return str(col)
    raise KeyError(f"none of {list(candidates)} found in columns {list(df.columns)}")


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


@register_source("xxt_signin")
class SignInExport(ArrivalSource):
    """One or more sign-in detail exports; each row is one student's check-in time."""

    n_paths = -1

    def __init__(self, paths: List[Path], group: str = "signin") -> None:
        self.paths = [Path(p) for p in paths]
        self.group = group

    def load(self) -> pd.DataFrame:
        frames = []
        for p in self.paths:
            raw = _read_any(p)
            ts_col = _pick(raw, ["签到时间", "时间"])
            df = pd.DataFrame({"ts": pd.to_datetime(raw[ts_col], errors="coerce"), "group": self.group, "kind": "signin"})
            df["session"] = p.stem
            frames.append(df)
            logger.info("%s: %d check-ins", p.name, df["ts"].notna().sum())
        return self.validate(pd.concat(frames, ignore_index=True))


@register_source("xxt_homework")
class HomeworkExport(ArrivalSource):
    """Homework-grading export; each row is one submission with its submission time (提交时间)."""

    n_paths = -1

    def __init__(self, paths: List[Path], group: str = "homework") -> None:
        self.paths = [Path(p) for p in paths]
        self.group = group

    def load(self) -> pd.DataFrame:
        frames = []
        for p in self.paths:
            raw = _read_any(p)
            ts_col = _pick(raw, ["提交时间"])
            df = pd.DataFrame({"ts": pd.to_datetime(raw[ts_col], errors="coerce"), "group": self.group, "kind": "homework"})
            df["assignment"] = p.stem
            ip_col: Optional[str] = next((c for c in raw.columns if "ip" in str(c).lower()), None)
            if ip_col:
                df["on_campus"] = raw[ip_col].astype(str).str.startswith("121.251.")
            frames.append(df)
        return self.validate(pd.concat(frames, ignore_index=True))


@register_source("xxt_ai_sessions")
class AiSessionExport(ArrivalSource):
    """AI-assistant session-history export; each row is one session with its start, end and message count."""

    n_paths = -1

    def __init__(self, paths: List[Path], group: str = "ai_ta") -> None:
        self.paths = [Path(p) for p in paths]
        self.group = group

    def load(self) -> pd.DataFrame:
        frames = []
        for p in self.paths:
            raw = _read_any(p)
            start = _pick(raw, ["创建时间", "开始时间"])
            df = pd.DataFrame({"ts": pd.to_datetime(raw[start], errors="coerce"), "group": self.group, "kind": "ai_session"})
            for src, dst in (("结束时间", "ts_end"), ("消息数", "n_messages"), ("渠道", "channel")):
                try:
                    col = _pick(raw, [src])
                    df[dst] = pd.to_datetime(raw[col], errors="coerce") if dst == "ts_end" else raw[col]
                except KeyError:
                    logger.warning("%s: column %s not found", p.name, src)
            try:
                df["uid"] = raw[_pick(raw, ["学工号"])].map(hash_id)
            except KeyError:
                logger.warning("%s: no 学工号 column, sessions left unlinked", p.name)
            frames.append(df)
        return self.validate(pd.concat(frames, ignore_index=True))
