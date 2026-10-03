#!/usr/bin/env python3
"""Turn one real prompted burst into a replay trace for the box (batch B', F10).

Reads the anonymised message table of the theory class (timestamps, roles and character counts
only; no text, no ids) and writes ``traces/<name>.csv`` with one row per student request:
``offset_s`` from the burst's first request, a ``student`` index, ``prompt_chars`` and the
length of the reply the platform's assistant actually produced (``reply_chars``; the median when
the export shows no reply).  The box replays the offsets against a live vLLM server and asks for
``ceil(reply_chars / chars_per_token)`` output tokens per request.

Run from the repository root: ``uv run python scripts/q2/make_replay_trace.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("make_replay_trace")

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/xuexitong/assistant"


def build(sheet: str, day: str, start: str, end: str, stem: str = "course_theory") -> pd.DataFrame:
    """Student turns of ``sheet`` on ``day`` inside ``[start, end)`` as replay rows."""
    sessions = pd.read_csv(RAW / f"{stem}_sessions.csv", parse_dates=["created"])
    messages = pd.read_csv(RAW / f"{stem}_messages.csv", parse_dates=["ts"])
    messages = messages.merge(sessions[["session", "sheet", "user"]], on="session")
    same_day = messages[(messages["sheet"] == sheet)
                        & (messages["ts"].dt.date == pd.Timestamp(day).date())]
    same_day = same_day.sort_values(["session", "turn"]).reset_index(drop=True)
    nxt = same_day.groupby("session").shift(-1)
    student = same_day[same_day["role"] == "student"].copy()
    student["reply_chars"] = nxt.loc[student.index, "n_chars"].where(
        nxt.loc[student.index, "role"] == "assistant")
    lo, hi = pd.Timestamp(f"{day} {start}"), pd.Timestamp(f"{day} {end}")
    student = student[(student["ts"] >= lo) & (student["ts"] < hi)].sort_values("ts")
    if student.empty:
        raise SystemExit(f"no student turns in {sheet} on {day} {start}-{end}")
    median_reply = float(student["reply_chars"].median())
    t0 = student["ts"].iloc[0]
    codes = {u: i for i, u in enumerate(pd.unique(student["user"]))}
    out = pd.DataFrame({
        "offset_s": (student["ts"] - t0).dt.total_seconds().round(1).to_numpy(),
        "student": student["user"].map(codes).astype(int).to_numpy(),
        "prompt_chars": student["n_chars"].astype(int).to_numpy(),
        "reply_chars": student["reply_chars"].fillna(median_reply).round().astype(int)
        .to_numpy(),
    })
    logger.info("%s %s %s-%s: %d requests, %d students, %.0f s span, reply median %.0f chars",
                sheet, day, start, end, len(out), len(codes), out["offset_s"].max(),
                median_reply)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", default="class_1")
    ap.add_argument("--day", default="2026-03-24")
    ap.add_argument("--start", default="08:20")
    ap.add_argument("--end", default="08:50")
    ap.add_argument("--name", default="burst_20260324_0830")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "traces")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    frame = build(args.sheet, args.day, args.start, args.end)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.name}.csv"
    frame.to_csv(path, index=False)
    meta = {"source": "data/raw/xuexitong/assistant (anonymised; timestamps and lengths only)",
            "sheet": args.sheet, "day": args.day, "window": [args.start, args.end],
            "n_requests": int(len(frame)), "n_students": int(frame["student"].nunique()),
            "span_s": float(frame["offset_s"].max()),
            "first_request_wallclock": f"{args.day} {args.start} window, first student turn",
            "reply_chars_median": float(frame["reply_chars"].median())}
    (args.out / f"{args.name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("wrote %s and its .json", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
