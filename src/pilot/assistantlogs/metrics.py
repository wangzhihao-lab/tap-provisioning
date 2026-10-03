"""Workload metrics of the anonymised assistant logs: synchrony, ramp, peak-to-mean, gaps, lengths.

Every in-class metric is computed on **per-student first arrivals**: the first student turn that
student sent that day.  That is the quantity the provisioning problem cares about (one student
walking up to the service) and the one ``sigma_k`` is defined on throughout the paper: the share of
first arrivals landing within k minutes of the first arrival of the day (or of the burst).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KS: Tuple[int, ...] = (1, 2, 5, 15)
RAMP_LO, RAMP_HI = 0.1, 0.9


def first_arrivals(student_messages: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, user): the student's first turn of that day.

    Columns: ``date, user, ts, tod_min, weekday``.
    """
    if student_messages.empty:
        return pd.DataFrame(columns=["date", "user", "ts", "tod_min", "weekday"])
    idx = student_messages.groupby(["date", "user"])["ts"].idxmin()
    cols = ["date", "user", "ts", "tod_min", "weekday"]
    return student_messages.loc[idx, cols].sort_values("ts").reset_index(drop=True)


def sigma_shares(rel_min: np.ndarray, ks: Sequence[int] = KS) -> Dict[str, float]:
    """``{"sigma_1": ..., ...}``: share of arrivals strictly within k minutes of the reference."""
    rel = np.asarray(rel_min, dtype=float)
    if rel.size == 0:
        return {f"sigma_{k}": float("nan") for k in ks}
    return {f"sigma_{k}": float((rel < float(k)).mean()) for k in ks}


def ramp_10_90(rel_min: np.ndarray, lo: float = RAMP_LO, hi: float = RAMP_HI) -> float:
    """Minutes between the 10% and 90% quantiles of the arrival offsets."""
    rel = np.asarray(rel_min, dtype=float)
    if rel.size < 2:
        return float("nan")
    return float(np.quantile(rel, hi) - np.quantile(rel, lo))


def relative_offsets(tod_min: Iterable[float], origin: Optional[float] = None) -> np.ndarray:
    """Arrival offsets in minutes from ``origin`` (default: the earliest arrival)."""
    arr = np.sort(np.asarray(list(tod_min), dtype=float))
    if arr.size == 0:
        return arr
    return arr - (float(arr[0]) if origin is None else float(origin))


def binned_counts(tod_min: Iterable[float], lo: float, hi: float,
                  bin_minutes: float) -> Tuple[np.ndarray, np.ndarray]:
    """Counts of ``tod_min`` in ``[lo, hi)`` bins of ``bin_minutes``; returns (bin start, count)."""
    edges = np.arange(lo, hi + bin_minutes, bin_minutes, dtype=float)
    counts, _ = np.histogram(np.asarray(list(tod_min), dtype=float), bins=edges)
    return edges[:-1], counts.astype(float)


def peak_to_mean(tod_min: Iterable[float], window: Tuple[float, float],
                 bin_minutes: float) -> float:
    """Peak-to-mean ratio of the binned arrival series over the class window."""
    _, counts = binned_counts(tod_min, window[0], window[1], bin_minutes)
    if counts.size == 0 or counts.mean() == 0.0:
        return float("nan")
    return float(counts.max() / counts.mean())


def questions_per_student(student_messages: pd.DataFrame) -> pd.DataFrame:
    """Student turns per (date, user): the questions one student asks in one class."""
    if student_messages.empty:
        return pd.DataFrame(columns=["date", "user", "questions"])
    out = (student_messages.groupby(["date", "user"]).size().rename("questions").reset_index())
    return out


def _quantiles(values: np.ndarray, name: str) -> Dict[str, float]:
    """``n / median / mean / p75 / p90 / max`` of one sample, prefixed with ``name``."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        keys = ("n", "median", "mean", "p75", "p90", "max")
        return {f"{name}_{k}": float("nan") for k in keys} | {f"{name}_n": 0.0}
    return {f"{name}_n": float(vals.size), f"{name}_median": float(np.median(vals)),
            f"{name}_mean": float(vals.mean()), f"{name}_p75": float(np.quantile(vals, 0.75)),
            f"{name}_p90": float(np.quantile(vals, 0.90)), f"{name}_max": float(vals.max())}


def turn_gaps(messages: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Within-session gaps in seconds, split by what the two turns are.

    ``student_turn``  student turn -> that student's next turn in the same session (think time
    plus reading the answer), which is what the simulator's ``inter_request_mean_s`` stands for.
    ``response``      student turn -> the assistant turn that answers it (service latency as the
    platform logged it, not a measurement of our own service).
    """
    student_turn: list = []
    response: list = []
    ordered = messages.sort_values(["session", "turn"])
    for _, part in ordered.groupby("session", sort=False):
        ts = part["ts"].to_numpy()
        roles = part["role"].to_numpy()
        stu_idx = np.flatnonzero(roles == "student")
        if stu_idx.size > 1:
            deltas = np.diff(ts[stu_idx]) / np.timedelta64(1, "s")
            student_turn.extend(deltas.tolist())
        for i in stu_idx:
            nxt = i + 1
            if nxt < roles.size and roles[nxt] == "assistant":
                response.append(float((ts[nxt] - ts[i]) / np.timedelta64(1, "s")))
    return {"student_turn": np.asarray(student_turn, dtype=float),
            "response": np.asarray(response, dtype=float)}


def gap_stats(messages: pd.DataFrame) -> Dict[str, float]:
    """Quantiles of both gap families, flattened into one row."""
    gaps = turn_gaps(messages)
    out: Dict[str, float] = {}
    for name, vals in gaps.items():
        out.update(_quantiles(vals, f"{name}_gap_s"))
    return out


def length_stats(messages: pd.DataFrame) -> Dict[str, float]:
    """Quantiles of the message length (characters) of each role."""
    out: Dict[str, float] = {}
    for role in ("student", "assistant"):
        vals = messages.loc[messages["role"] == role, "n_chars"].to_numpy(dtype=float)
        out.update(_quantiles(vals, f"{role}_chars"))
    return out


def weekly_profile(frame: pd.DataFrame, value: str = "sessions") -> pd.DataFrame:
    """Counts by (weekday, hour) over the whole record; zero-filled 7 x 24."""
    grid = pd.MultiIndex.from_product([range(7), range(24)], names=["weekday", "hour"])
    if frame.empty:
        return pd.DataFrame({value: 0}, index=grid).reset_index()
    hour = (pd.to_datetime(frame["ts"] if "ts" in frame else frame["created"]).dt.hour)
    counts = (pd.DataFrame({"weekday": frame["weekday"], "hour": hour})
              .groupby(["weekday", "hour"]).size().rename(value))
    return counts.reindex(grid, fill_value=0).reset_index()


def deadline_ramp(frame: pd.DataFrame, deadlines: Sequence[pd.Timestamp],
                  hours: int = 72) -> pd.DataFrame:
    """Sessions per hour in the ``hours`` before each deadline instant.

    Columns: ``deadline, hours_before, sessions`` with ``hours_before`` counted backwards
    (``0`` = the last hour before the deadline).
    """
    ts = pd.to_datetime(frame["created"] if "created" in frame else frame["ts"])
    rows = []
    for due in deadlines:
        due = pd.Timestamp(due)
        delta = (due - ts).dt.total_seconds() / 3600.0
        inside = delta[(delta > 0) & (delta <= hours)]
        binned = np.floor(inside).astype(int).value_counts()
        for h in range(hours):
            rows.append({"deadline": due.date().isoformat(), "hours_before": h,
                         "sessions": int(binned.get(h, 0))})
    return pd.DataFrame(rows, columns=["deadline", "hours_before", "sessions"])


__all__ = ["KS", "RAMP_HI", "RAMP_LO", "binned_counts", "deadline_ramp", "first_arrivals",
           "gap_stats", "length_stats", "peak_to_mean", "questions_per_student", "ramp_10_90",
           "relative_offsets", "sigma_shares", "turn_gaps", "weekly_profile"]
