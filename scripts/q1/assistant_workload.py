"""Batch A: workload characterisation of the Xuexitong AI-assistant logs, and simulator calibration.

Reads the anonymised tables written by ``xxt_assistant_sessions.py`` (timestamps, roles and character
counts only) and writes, under ``results/q1/assistant/``:

``per_day.csv``            one row per class day of the two in-class datasets
``bursts.csv``             one row per (class day, bell) burst of the prompted class
``arrivals_per_min.csv``   minute-level arrival series relative to the day's first bell
``period_profile.csv``     the lab course's pooled 5-min profile over its four periods
``gaps_lengths.csv``       within-session gaps and message lengths, one row per dataset
``weekly_profile.csv``     weekday x hour session counts of the out-of-class class
``sessions_per_student.csv`` its per-student session counts over the semester
``deadline_ramp.csv``      sessions per hour in the 72 h before each inferred deadline
``calibration.json``       the numbers the simulator's WorkloadConfig is calibrated against
``batch_a_summary.md``     the written summary, with the headline numbers and the caveats

and, under ``data/offsets/``, the two arrival-offset pools the simulator samples
(``prompted_ai_offsets_min.csv``, ``selfpaced_ai_offsets_min.csv``).

Run: ``uv run python scripts/q1/assistant_workload.py`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pilot.assistantlogs import (
    OFF_CLASS,
    PROMPTED,
    SELF_PACED,
    AssistantDataset,
    BurstConfig,
    DatasetSpec,
    binned_counts,
    burst_offsets,
    deadline_ramp,
    first_arrivals,
    gap_stats,
    length_stats,
    load_dataset,
    peak_to_mean,
    questions_per_student,
    ramp_10_90,
    relative_offsets,
    sigma_shares,
    split_bursts,
    weekly_profile,
)
from pilot.assistantlogs.report import write_summary

logger = logging.getLogger(__name__)

# The genuine platform exports of 2026-09-17 (``course_lab.xlsx``, ``course_theory.xlsx`` with one sheet
# per teaching class).  Bells are the published period starts (``pilot.timetable.parse.BellSchedule``):
# the prompted class meets Tuesday periods 1-3 (08:30, 09:15, 10:20 after the break); its bursts
# begin 15-30 min *before* each bell and end at it, so a burst window opens 35 min before its bell.
DATASETS: Tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="dsv_lab", label="lab course, self-paced in class",
        stem="course_lab", sheet="Sheet1",
        regime=SELF_PACED, roster=185, bells=("14:00", "14:45", "15:50", "16:35"),
        class_weekday=2, window=("13:30", "17:40"),
    ),
    DatasetSpec(
        key="ds_prompted", label="theory class A, prompted in class",
        stem="course_theory", sheet="class_1",
        regime=PROMPTED, roster=125, bells=("08:30", "09:15", "10:20"),
        class_weekday=1, window=("07:50", "12:00"), burst_pre_min=35.0,
    ),
    DatasetSpec(
        key="ds_offclass", label="theory class B, out of class with a biweekly deadline",
        stem="course_theory", sheet="class_2",
        regime=OFF_CLASS, roster=110, deadline_weekday=0, deadline_hm="23:59",
    ),
)


@dataclass(frozen=True)
class Paths:
    """Where the tables are read from and written to."""

    raw: Path
    assistant: Path
    offsets: Path

    @staticmethod
    def under(root: Path) -> "Paths":
        """The repository's standard layout, relative to its root."""
        return Paths(raw=root / "data/raw/xuexitong/assistant",
                     assistant=root / "results/q1/assistant",
                     offsets=root / "data/offsets")


def _session_opened(ds: AssistantDataset) -> pd.DataFrame:
    """Per (date, user) first session-*creation* time: the arrival proxy the first notes used.

    It is kept as a secondary column so that the earlier numbers stay auditable: a session is
    created when the student opens the assistant, which is 2-45 s before their first question and
    includes the sessions that were opened and never used.
    """
    frame = ds.sessions
    idx = frame.groupby(["date", "user"])["created"].idxmin()
    return frame.loc[idx, ["date", "user", "tod_min"]]


def per_day_table(ds: AssistantDataset, arrivals: pd.DataFrame) -> pd.DataFrame:
    """One row per class day: synchrony, ramp, peak-to-mean, adoption and questions."""
    spec = ds.spec
    if spec.class_weekday is None:
        return pd.DataFrame()
    days = arrivals[arrivals["weekday"] == spec.class_weekday]
    opened = _session_opened(ds)
    questions = questions_per_student(ds.student_messages)
    window = spec.window_minutes
    bell = spec.bell_minutes[0]
    rows: List[dict] = []
    for date, group in days.groupby("date"):
        tod = np.sort(group["tod_min"].to_numpy(dtype=float))
        rel = relative_offsets(tod)
        q = questions.loc[questions["date"] == date, "questions"].to_numpy(dtype=float)
        row = {"dataset": spec.key, "regime": spec.regime, "date": str(date),
               "n_students": int(tod.size), "roster": spec.roster,
               "adoption": float(tod.size) / spec.roster,
               "n_sessions": int((ds.sessions["date"] == date).sum()),
               "ramp_10_90_min": ramp_10_90(rel),
               "first_arrival_offset_min": float(tod.min() - bell),
               "median_arrival_offset_min": float(np.median(tod) - bell),
               "p2m_1min": peak_to_mean(tod, window, 1.0),
               "p2m_5min": peak_to_mean(tod, window, 5.0),
               "n_student_msgs": int(q.sum()),
               "questions_median": float(np.median(q)) if q.size else float("nan"),
               "questions_mean": float(q.mean()) if q.size else float("nan"),
               "questions_p90": float(np.quantile(q, 0.9)) if q.size else float("nan")}
        row.update(sigma_shares(rel))
        opened_rel = relative_offsets(opened.loc[opened["date"] == date, "tod_min"])
        row["n_opened"] = int(opened_rel.size)
        row["sigma_2_opened"] = sigma_shares(opened_rel, (2,))["sigma_2"]
        row["ramp_10_90_opened_min"] = ramp_10_90(opened_rel)
        rows.append(row)
    return pd.DataFrame(rows)


def arrival_series(ds: AssistantDataset, arrivals: pd.DataFrame,
                   bin_minutes: float = 1.0) -> pd.DataFrame:
    """Minute-level first arrivals and student turns, offset from the day's first bell."""
    spec = ds.spec
    if spec.class_weekday is None:
        return pd.DataFrame()
    lo, hi = spec.window_minutes
    bell = spec.bell_minutes[0]
    msgs = ds.student_messages
    rows: List[dict] = []
    for date, group in arrivals[arrivals["weekday"] == spec.class_weekday].groupby("date"):
        starts, firsts = binned_counts(group["tod_min"], lo, hi, bin_minutes)
        _, turns = binned_counts(msgs.loc[msgs["date"] == date, "tod_min"], lo, hi, bin_minutes)
        for start, n_first, n_turn in zip(starts, firsts, turns):
            rows.append({"dataset": spec.key, "date": str(date),
                         "minute_from_bell": float(start - bell),
                         "first_arrivals": int(n_first), "student_msgs": int(n_turn)})
    return pd.DataFrame(rows)


def period_profile(ds: AssistantDataset, arrivals: pd.DataFrame,
                   bin_minutes: float = 5.0) -> pd.DataFrame:
    """Pooled arrivals per 5-min bin over the class window (F3b), in minutes after midnight."""
    spec = ds.spec
    lo, hi = spec.window_minutes
    days = arrivals[arrivals["weekday"] == spec.class_weekday]
    starts, firsts = binned_counts(days["tod_min"], lo, hi, bin_minutes)
    msgs = ds.student_messages
    _, turns = binned_counts(msgs.loc[msgs["weekday"] == spec.class_weekday, "tod_min"], lo, hi,
                             bin_minutes)
    n_days = max(1, days["date"].nunique())
    return pd.DataFrame({"dataset": spec.key, "tod_min": starts,
                         "clock": [f"{int(s) // 60:02d}:{int(s) % 60:02d}" for s in starts],
                         "first_arrivals": firsts.astype(int),
                         "student_msgs": turns.astype(int),
                         "first_arrivals_per_day": firsts / n_days})


def infer_deadlines(ds: AssistantDataset, min_share: float = 0.25) -> List[pd.Timestamp]:
    """Deadline instants inferred from the data: the busiest ``deadline_weekday`` evenings.

    The homework of the out-of-class class is due every other Monday night, and the data show it:
    the deadline Mondays carry 55-94 sessions each and the Mondays between them 2-3.  A Monday is
    called a deadline when its session count reaches ``min_share`` of the busiest Monday's.
    """
    spec = ds.spec
    if spec.deadline_weekday is None:
        return []
    mondays = ds.sessions[ds.sessions["weekday"] == spec.deadline_weekday]
    counts = mondays.groupby("date").size()
    if counts.empty:
        return []
    keep = counts[counts >= min_share * counts.max()]
    hour, minute = (int(x) for x in spec.deadline_hm.split(":"))
    out = [pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=minute) for d in keep.index]
    gaps = np.diff([t.normalize().value for t in out]) / (86400 * 1e9)
    logger.info("%s: %d deadlines inferred (%s), spacing %s days, off-deadline %s sessions",
                spec.key, len(out), [t.date().isoformat() for t in out], sorted(set(gaps.tolist())),
                sorted(set(counts[counts < min_share * counts.max()].tolist())))
    return out


def deadline_baseline(ds: AssistantDataset, deadlines: Sequence[pd.Timestamp],
                      hours: int = 72) -> Dict[str, float]:
    """Sessions per hour inside the pre-deadline windows against the rest of the record."""
    ts = pd.to_datetime(ds.sessions["created"])
    if ts.empty or not deadlines:
        return {}
    inside = np.zeros(len(ts), dtype=bool)
    for due in deadlines:
        delta = (pd.Timestamp(due) - ts).dt.total_seconds() / 3600.0
        inside |= ((delta > 0) & (delta <= hours)).to_numpy()
    span_h = (ts.max() - ts.min()).total_seconds() / 3600.0
    in_h = float(hours * len(deadlines))
    return {"sessions_in_window": int(inside.sum()), "sessions_outside": int((~inside).sum()),
            "per_hour_in_window": round(float(inside.sum()) / in_h, 3),
            "per_hour_outside": round(float((~inside).sum()) / max(1.0, span_h - in_h), 3),
            "window_hours": float(in_h), "record_hours": round(float(span_h), 1)}


def _describe(values: np.ndarray) -> Dict[str, float]:
    """``median / mean / p90 / max`` of a sample, rounded for the calibration file."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    return {"n": int(vals.size), "median": round(float(np.median(vals)), 3),
            "mean": round(float(vals.mean()), 3), "p90": round(float(np.quantile(vals, 0.9)), 3),
            "max": round(float(vals.max()), 3)}


def calibration(per_day: pd.DataFrame, bursts: pd.DataFrame, gaps: pd.DataFrame,
                sessions_per_student: pd.DataFrame) -> dict:
    """The numbers the simulator is calibrated against (``results/q1/assistant/calibration.json``)."""
    out: dict = {"source": "results/q1/assistant (batch A)", "adoption": {},
                 "questions_per_student_per_class": {}, "student_turn_gap_s": {},
                 "response_gap_s": {}, "student_msg_chars": {}, "assistant_msg_chars": {}}
    for key, part in per_day.groupby("dataset"):
        adopt = part["adoption"].to_numpy(dtype=float)
        out["adoption"][key] = {"roster": int(part["roster"].iloc[0]), "days": len(part),
                                "median": round(float(np.median(adopt)), 3),
                                "min": round(float(adopt.min()), 3),
                                "max": round(float(adopt.max()), 3),
                                "students_median": float(np.median(part["n_students"]))}
    for key, part in per_day.groupby("dataset"):
        out["questions_per_student_per_class"][key] = {
            "median": round(float(part["questions_median"].median()), 3),
            "mean": round(float(part["questions_mean"].mean()), 3),
            "p90": round(float(part["questions_p90"].median()), 3)}
    for _, row in gaps.iterrows():
        key = row["dataset"]
        out["student_turn_gap_s"][key] = {"median": round(float(row["student_turn_gap_s_median"]), 1),
                                          "p75": round(float(row["student_turn_gap_s_p75"]), 1),
                                          "n": int(row["student_turn_gap_s_n"])}
        out["response_gap_s"][key] = {"median": round(float(row["response_gap_s_median"]), 1),
                                      "n": int(row["response_gap_s_n"])}
        out["student_msg_chars"][key] = {"median": round(float(row["student_chars_median"]), 1),
                                         "p90": round(float(row["student_chars_p90"]), 1)}
        out["assistant_msg_chars"][key] = {
            "median": round(float(row["assistant_chars_median"]), 1),
            "p90": round(float(row["assistant_chars_p90"]), 1)}
    if not bursts.empty:
        prompted = bursts[bursts["kind"] == "prompted"]
        out["prompted_bursts"] = {
            "n_bursts": len(prompted), "n_days": int(prompted["date"].nunique()),
            "sigma_2_min": round(float(prompted["sigma_2"].min()), 3),
            "sigma_2_median": round(float(prompted["sigma_2"].median()), 3),
            "sigma_2_max": round(float(prompted["sigma_2"].max()), 3),
            "ramp_10_90_min": _describe(prompted["ramp_10_90_min"].to_numpy())}
    if not sessions_per_student.empty:
        counts = sessions_per_student["sessions"].to_numpy(dtype=float)
        out["sessions_per_student_semester"] = _describe(counts)
    return out


def _write(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV (creating its directory) and log the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("wrote %s (%d rows)", path, len(frame))
    return path


def write_pools(datasets: Dict[str, AssistantDataset], arrivals: Dict[str, pd.DataFrame],
                paths: Paths, cfg: BurstConfig) -> Dict[str, int]:
    """The two offset pools: prompted bursts and the self-paced lab days."""
    sizes: Dict[str, int] = {}
    prompted = burst_offsets(arrivals["ds_prompted"], datasets["ds_prompted"].spec, cfg)
    _write(pd.DataFrame({"offset_min": np.round(prompted, 4)}),
           paths.offsets / "prompted_ai_offsets_min.csv")
    sizes["prompted_ai_offsets_min.csv"] = int(prompted.size)
    lab = datasets["dsv_lab"]
    days = arrivals["dsv_lab"]
    days = days[days["weekday"] == lab.spec.class_weekday]
    pool = np.concatenate([relative_offsets(g["tod_min"]) for _, g in days.groupby("date")])
    _write(pd.DataFrame({"offset_min": np.round(pool, 4)}),
           paths.offsets / "selfpaced_ai_offsets_min.csv")
    sizes["selfpaced_ai_offsets_min.csv"] = int(pool.size)
    return sizes


def run(root: Path, specs: Sequence[DatasetSpec] = DATASETS,
        cfg: Optional[BurstConfig] = None) -> dict:
    """Compute every table, write them and return the calibration dictionary."""
    cfg = cfg or BurstConfig()
    paths = Paths.under(root)
    datasets = {s.key: load_dataset(s, paths.raw) for s in specs}
    arrivals = {k: first_arrivals(d.student_messages) for k, d in datasets.items()}

    per_day = pd.concat([per_day_table(d, arrivals[k]) for k, d in datasets.items()],
                        ignore_index=True)
    bursts = pd.concat([split_bursts(arrivals[k], d.spec, cfg).assign(dataset=k)
                        for k, d in datasets.items()], ignore_index=True)
    series = pd.concat([arrival_series(d, arrivals[k]) for k, d in datasets.items()],
                       ignore_index=True)
    gaps = pd.DataFrame([{"dataset": k, "label": d.spec.label, **gap_stats(d.messages),
                          **length_stats(d.messages)} for k, d in datasets.items()])
    off = datasets["ds_offclass"]
    weekly = weekly_profile(off.sessions).assign(dataset=off.spec.key)
    # values only, sorted: the distribution without any per-student ordering
    spstud = pd.DataFrame({"sessions": sorted(off.sessions.groupby("user").size().tolist()),
                           "dataset": off.spec.key})
    deadlines = infer_deadlines(off)
    ramp = deadline_ramp(off.sessions, deadlines).assign(dataset=off.spec.key)
    profile = period_profile(datasets["dsv_lab"], arrivals["dsv_lab"])

    _write(per_day, paths.assistant / "per_day.csv")
    _write(bursts, paths.assistant / "bursts.csv")
    _write(series, paths.assistant / "arrivals_per_min.csv")
    _write(profile, paths.assistant / "period_profile.csv")
    _write(gaps, paths.assistant / "gaps_lengths.csv")
    _write(weekly, paths.assistant / "weekly_profile.csv")
    _write(spstud, paths.assistant / "sessions_per_student.csv")
    _write(ramp, paths.assistant / "deadline_ramp.csv")

    cal = calibration(per_day, bursts, gaps, spstud)
    cal["deadlines"] = [t.isoformat(sep=" ") for t in deadlines]
    cal["deadline_window"] = deadline_baseline(off, deadlines)
    mondays = off.sessions[off.sessions["weekday"] == off.spec.deadline_weekday]
    cal["monday_sessions"] = {str(k): int(v) for k, v in mondays.groupby("date").size().items()}
    cal["offclass_roster"] = int(off.spec.roster)
    cal["pool_rows"] = write_pools(datasets, arrivals, paths, cfg)
    (paths.assistant / "calibration.json").write_text(
        json.dumps(cal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", paths.assistant / "calibration.json")
    write_summary(paths.assistant / "batch_a_summary.md", datasets, per_day, bursts, gaps, weekly,
                  spstud, ramp, cal)
    return cal


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                    help="repository root (default: inferred from this file)")
    args = ap.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
