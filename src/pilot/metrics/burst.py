"""Q1 burst metrics: per-minute counts, ramp time, synchronization fraction, peak-to-mean."""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def minute_counts(ts: pd.Series, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
                  bin_minutes: int = 1) -> pd.Series:
    """Arrivals per ``bin_minutes`` over [start, end], zero-filled."""
    ts = pd.to_datetime(ts).dropna()
    if ts.empty:
        return pd.Series(dtype="int64")
    start = (start or ts.min()).floor(f"{bin_minutes}min")
    end = (end or ts.max()).ceil(f"{bin_minutes}min")
    idx = pd.date_range(start, end, freq=f"{bin_minutes}min")
    counts = ts.dt.floor(f"{bin_minutes}min").value_counts().reindex(idx, fill_value=0).sort_index()
    counts.name = "arrivals"
    return counts


@dataclass(frozen=True)
class BurstStats:
    """Shape of one burst window."""

    n: int
    ramp_min: float           # minutes between the lo and hi cumulative quantiles
    peak_per_min: int
    sync_fraction: dict       # {k_minutes: fraction of window arrivals within k min of t0}


def ramp_time(ts: pd.Series, lo: float = 0.1, hi: float = 0.9) -> float:
    """Minutes between the ``lo`` and ``hi`` cumulative quantiles of arrival times."""
    ts = pd.to_datetime(ts).dropna().sort_values()
    if len(ts) < 2:
        return float("nan")
    q_lo, q_hi = ts.quantile(lo), ts.quantile(hi)
    return float((q_hi - q_lo).total_seconds() / 60.0)


def burst_stats(ts: pd.Series, t0: pd.Timestamp, window_min: int = 15, ks: Sequence[int] = (1, 2, 5),
                lo: float = 0.1, hi: float = 0.9) -> BurstStats:
    """Describe the arrivals in [t0, t0 + window) relative to the trigger time ``t0``."""
    ts = pd.to_datetime(ts)
    t0 = pd.Timestamp(t0)
    win = ts[(ts >= t0) & (ts < t0 + pd.Timedelta(minutes=window_min))]
    n = int(len(win))
    if n == 0:
        return BurstStats(0, float("nan"), 0, {k: float("nan") for k in ks})
    counts = minute_counts(win, t0, t0 + pd.Timedelta(minutes=window_min))
    sync = {k: float(((win - t0) < pd.Timedelta(minutes=k)).mean()) for k in ks}
    return BurstStats(n, ramp_time(win, lo, hi), int(counts.max()), sync)


def peak_to_mean(counts: pd.Series) -> float:
    """Peak-to-mean ratio of a count series (nan for an empty or all-zero series)."""
    if counts.empty or counts.mean() == 0:
        return float("nan")
    return float(counts.max() / counts.mean())


def per_session_bursts(df: pd.DataFrame, sessions: pd.DataFrame, triggers_per_session: int,
                       window_min: int = 15, ks: Sequence[int] = (1, 2, 5)) -> pd.DataFrame:
    """Burst statistics for every trigger of every session in a timetable frame.

    ``sessions`` needs columns start, duration_min, group; triggers are placed evenly inside each
    session (the same convention as the mock generator). With real data, pass the recorded trigger
    times instead through ``sessions['triggers']`` (a list per row).
    """
    rows = []
    by_group = {g: part["ts"] for g, part in df.groupby("group", sort=False, observed=True)}
    empty = pd.Series(dtype="datetime64[ns]")
    for _, s in sessions.iterrows():
        sub = by_group.get(s["group"], empty)
        triggers = s["triggers"] if "triggers" in sessions.columns else [
            pd.Timestamp(s["start"]) + pd.Timedelta(minutes=s["duration_min"] / (triggers_per_session + 1) * j)
            for j in range(1, triggers_per_session + 1)
        ]
        for j, t0 in enumerate(triggers, start=1):
            st = burst_stats(sub, t0, window_min, ks)
            rows.append({"group": s["group"], "start": s["start"], "trigger": j, "t0": t0, "n": st.n,
                         "ramp_min": st.ramp_min, "peak_per_min": st.peak_per_min,
                         **{f"sync_{k}min": v for k, v in st.sync_fraction.items()}})
    out = pd.DataFrame(rows)
    logger.info("%d bursts; median ramp %.2f min; median sync_1min %.2f",
                len(out), out["ramp_min"].median(), out["sync_1min"].median())
    return out


def signin_distribution(minute_counts_: pd.Series) -> dict:
    """Summary used for the sign-in comparison: share in the top minute and within 5 minutes of it."""
    if minute_counts_.empty:
        return {}
    top = minute_counts_.idxmax()
    total = int(minute_counts_.sum())
    within5 = int(minute_counts_[(minute_counts_.index >= top) & (minute_counts_.index < top + pd.Timedelta(minutes=5))].sum())
    return {"total": total, "top_minute": str(top), "share_top_minute": minute_counts_.max() / total,
            "share_5min_from_top": within5 / total, "peak_to_mean": peak_to_mean(minute_counts_)}
