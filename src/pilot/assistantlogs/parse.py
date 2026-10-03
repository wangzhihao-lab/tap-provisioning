"""Parse one Xuexitong ``会话详情`` cell into timestamp / role / length rows.

The export prints one line per message, ``<speaker> <YYYY-MM-DD HH:MM:SS> <text>``, where the
speaker is the student's (already pseudonymous) account or the assistant's display name.  This
module is the only place the text is looked at, and all it does with it is count characters: the
rows it returns carry the turn index, the timestamp, the role and ``n_chars``.  Nothing that could
identify a person or reproduce a message leaves this function.

A message that itself contains newlines arrives as several lines; a line without the
``speaker + timestamp`` prefix is folded into the preceding message's character count (plus one
for the newline), so ``n_chars`` is the length of the whole message body.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

TS_PATTERN = r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d(?::\d\d)?"
TS_RE = re.compile(TS_PATTERN)
LINE_RE = re.compile(rf"^(?P<speaker>\S+)[ \t]+(?P<ts>{TS_PATTERN})[ \t]?(?P<text>.*)$", re.S)
# display names the platform uses for the bot side of the conversation
ASSISTANT_RE = re.compile(r"(AI|助教|助手|机器人|robot|bot)", re.IGNORECASE)

STUDENT = "student"
ASSISTANT = "assistant"


@dataclass(frozen=True)
class MessageRow:
    """One message of one session: when it was sent, by which side, and how long it was."""

    turn: int
    ts: pd.Timestamp
    role: str
    n_chars: int


def speaker_role(speaker: str, user: Optional[str] = None) -> str:
    """``student`` or ``assistant`` for one speaker token.

    The session's own (pseudonymous) account id is the reliable signal when the caller has it:
    every session in the exports seen so far has exactly one student, and the student's lines are
    prefixed with that id.  Otherwise the assistant's display name is matched by pattern.
    """
    if user is not None and speaker == str(user):
        return STUDENT
    return ASSISTANT if ASSISTANT_RE.search(speaker) else STUDENT


def parse_detail(detail: object, user: Optional[str] = None) -> List[MessageRow]:
    """Rows for one ``会话详情`` cell; an empty list when the cell holds no message."""
    if not isinstance(detail, str):
        return []
    rows: List[MessageRow] = []
    for raw in detail.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        match = LINE_RE.match(line)
        if match is None:
            if rows:  # continuation of a multi-line message: only its length is kept
                rows[-1] = replace(rows[-1], n_chars=rows[-1].n_chars + len(line.strip()) + 1)
            continue
        rows.append(MessageRow(
            turn=len(rows),
            ts=pd.Timestamp(match.group("ts")),
            role=speaker_role(match.group("speaker"), user),
            n_chars=len(match.group("text").strip()),
        ))
    return rows


def detail_frame(sessions: pd.Series, details: pd.Series,
                 users: Optional[pd.Series] = None) -> pd.DataFrame:
    """``session, turn, ts, role, n_chars`` for every message of every session.

    ``sessions`` holds the (hashed) session key of each row of ``details``; ``users`` the session's
    account id, used only to tell the two roles apart.
    """
    out: List[dict] = []
    user_iter = users if users is not None else pd.Series([None] * len(details))
    for key, detail, user in zip(sessions, details, user_iter):
        for row in parse_detail(detail, None if pd.isna(user) else user):
            out.append({"session": key, "turn": row.turn, "ts": row.ts, "role": row.role,
                        "n_chars": row.n_chars})
    frame = pd.DataFrame(out, columns=["session", "turn", "ts", "role", "n_chars"])
    logger.debug("parsed %d messages (%d assistant)", len(frame),
                 int((frame["role"] == ASSISTANT).sum()) if len(frame) else 0)
    return frame


__all__ = ["ASSISTANT", "ASSISTANT_RE", "LINE_RE", "STUDENT", "TS_PATTERN", "TS_RE", "MessageRow",
           "detail_frame", "parse_detail", "speaker_role"]
