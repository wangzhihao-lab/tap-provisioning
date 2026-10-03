"""Anonymise a Xuexitong AI-assistant session-history export into timestamp-only tables.

Input: the xlsx exported from robot-lc.chaoxing.com (columns 会话ID, 学工号, 消息数, 渠道,
创建时间, 结束时间, 机器人, 会话详情, ...).  Output, next to the input or under ``--out``:

* ``<stem>_sessions.csv``  one row per session: hashed user, hashed session, bot id, channel,
  created/ended time, message count, duration.  No names, IDs, IPs, locations or content.
* ``<stem>_messages.csv``  one row per message found inside 会话详情 (the export prints one line
  per turn): hashed session, turn index, timestamp, ``role`` (student / assistant) and
  ``n_chars``, the length of the message body.  The body itself is counted and dropped.

Every sheet that carries the session columns is read (a colleague may split teaching classes into
one sheet each) and kept apart by the ``sheet`` column; sheets that do not carry them - a summary
or parameter sheet, say - are skipped with a warning instead of being concatenated as empty
session rows.

Hashing is salted; keep the salt outside the repository so the tables cannot be joined back to
the roster.  ``--salt-file`` defaults to ``~/.tap_assistant_salt``.
Run: ``uv run python scripts/q1/xxt_assistant_sessions.py FILE.xlsx``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from pilot.assistantlogs.parse import ASSISTANT, detail_frame

logger = logging.getLogger(__name__)

SESSION_COLUMNS: Tuple[str, ...] = ("会话ID", "学工号", "消息数", "渠道", "创建时间", "结束时间",
                                    "机器人")
DETAIL_COLUMN = "会话详情"
SALT_FILES: Tuple[str, ...] = ("~/.tap_assistant_salt",)


def _hash(value: object, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:12]


def read_salt(path: Optional[Path] = None) -> str:
    """The secret salt, from ``path`` or the first default location that exists."""
    candidates = [path] if path is not None else [Path(p).expanduser() for p in SALT_FILES]
    for cand in candidates:
        if cand is not None and cand.expanduser().exists():
            salt = cand.expanduser().read_text(encoding="utf-8").strip()
            if salt:
                return salt
    raise SystemExit(f"no salt found; tried {[str(c) for c in candidates]}")


def session_sheets(xlsx: Path) -> List[Tuple[str, pd.DataFrame]]:
    """The sheets of ``xlsx`` that hold session rows, in file order."""
    kept: List[Tuple[str, pd.DataFrame]] = []
    for name, frame in pd.read_excel(xlsx, sheet_name=None).items():
        missing = [c for c in SESSION_COLUMNS if c not in frame.columns]
        if missing:
            logger.warning("%s: sheet %r has no session columns (%s missing), skipped", xlsx.name,
                           name, len(missing))
            continue
        kept.append((str(name), frame))
    if not kept:
        raise ValueError(f"{xlsx.name}: no sheet carries the columns {list(SESSION_COLUMNS)}")
    return kept


def session_table(sheets: List[Tuple[str, pd.DataFrame]], salt: str) -> pd.DataFrame:
    """One row per session, with every identifier replaced by a salted hash."""
    raw = pd.concat([f.assign(**{"_sheet": n}) for n, f in sheets], ignore_index=True)
    created = pd.to_datetime(raw["创建时间"], errors="coerce")
    ended = pd.to_datetime(raw["结束时间"], errors="coerce")
    out = pd.DataFrame({
        "sheet": raw["_sheet"].astype(str),
        "session": [_hash(v, salt) for v in raw["会话ID"]],
        "user": [_hash(v, salt) for v in raw["学工号"]],
        "bot": [_hash(v, salt) for v in raw["机器人"]],
        "channel": raw["渠道"].astype(str),
        "created": created,
        "ended": ended,
        "n_messages": pd.to_numeric(raw["消息数"], errors="coerce").fillna(0).astype(int),
    })
    out["duration_min"] = (out["ended"] - out["created"]).dt.total_seconds() / 60.0
    return out


def message_table(sheets: List[Tuple[str, pd.DataFrame]], sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per message: hashed session, turn, timestamp, role and character count."""
    raw = pd.concat([f for _, f in sheets], ignore_index=True)
    if DETAIL_COLUMN not in raw.columns:
        logger.warning("export has no %s column; no message table", DETAIL_COLUMN)
        return pd.DataFrame(columns=["session", "turn", "ts", "role", "n_chars"])
    return detail_frame(sessions["session"], raw[DETAIL_COLUMN], raw["学工号"].astype(str))


def _log_integrity(name: str, sessions: pd.DataFrame, messages: pd.DataFrame) -> None:
    """Report the checks a reader of the tables would want: counts, roles, window fit."""
    joined = messages.merge(sessions[["session", "created", "ended"]], on="session", how="left")
    inside = ((joined["ts"] >= joined["created"]) & (joined["ts"] <= joined["ended"])).mean()
    declared = int(sessions["n_messages"].sum())
    logger.info("%s: %d sessions (%d with messages) in %d sheets, %d users, %s -> %s", name,
                len(sessions), int((sessions["n_messages"] > 0).sum()),
                sessions["sheet"].nunique(), sessions["user"].nunique(),
                sessions["created"].min(), sessions["created"].max())
    logger.info("%s: %d messages parsed vs %d declared, %.1f%% assistant, %.3f inside "
                "[created, ended]", name, len(messages), declared,
                100.0 * float((messages["role"] == ASSISTANT).mean()) if len(messages) else 0.0,
                float(inside) if len(joined) else float("nan"))


def convert(xlsx: Path, out_dir: Path, salt: str) -> Tuple[Path, Path]:
    """Write the two anonymised tables for one export file; return their paths."""
    sheets = session_sheets(xlsx)
    sessions = session_table(sheets, salt)
    messages = message_table(sheets, sessions)
    _log_integrity(xlsx.name, sessions, messages)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / xlsx.stem.split("_历史会话")[0].replace("单位", "unit")
    s_path, m_path = Path(f"{stem}_sessions.csv"), Path(f"{stem}_messages.csv")
    sessions.to_csv(s_path, index=False)
    messages.to_csv(m_path, index=False)
    return s_path, m_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("xlsx", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/raw/xuexitong/assistant"))
    ap.add_argument("--salt", default="", help="secret salt; keep it out of the repository")
    ap.add_argument("--salt-file", type=Path, default=None,
                    help=f"file holding the salt (default: first of {list(SALT_FILES)})")
    args = ap.parse_args()
    salt = args.salt or read_salt(args.salt_file)
    for f in args.xlsx:
        convert(f, args.out, salt)


if __name__ == "__main__":
    main()
