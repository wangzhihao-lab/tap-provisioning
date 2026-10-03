"""Session detection.

Two ways to cut an arrival series into comparable units:

* ``gap_sessions`` clusters a class log into class sessions (a new session starts after a gap of
  inactivity), which is how ASSISTments 2012-13 and the Xuexitong exports get their sessions --
  neither carries an explicit session id.
* ``onset_times`` finds load onsets in a trace that has no timetable at all (BurstGPT, Azure),
  so that the same burst metrics can be computed on generic traffic.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import SessionDetectConfig
from .burst import minute_counts

logger = logging.getLogger(__name__)

SESSION_COLUMNS = ("group", "start", "end", "duration_min", "n_students", "n_events")


def gap_sessions(df: pd.DataFrame, cfg: Optional[SessionDetectConfig] = None,
                 uid_col: str = "uid") -> pd.DataFrame:
    """Cluster each group's arrivals into sessions separated by >= ``cfg.gap_min`` of inactivity.

    Args:
        df: arrivals with columns ts and group; ``uid_col`` is used only to count distinct
            students and never leaves this function.
        cfg: detection thresholds.
        uid_col: name of the (already anonymous) student column, if present.

    Returns:
        One row per session with columns group, start, end, duration_min, n_students, n_events.
    """
    cfg = cfg or SessionDetectConfig()
    if df.empty:
        return pd.DataFrame(columns=list(SESSION_COLUMNS))

    cols = [c for c in ("ts", "group", uid_col) if c in df.columns]
    work = df.loc[:, cols].sort_values(["group", "ts"], kind="mergesort").reset_index(drop=True)
    gap = work.groupby("group", sort=False)["ts"].diff()
    is_new = gap.isna() | (gap >= pd.Timedelta(minutes=cfg.gap_min))
    work["session_id"] = is_new.groupby(work["group"], sort=False).cumsum()

    agg = {"start": ("ts", "min"), "end": ("ts", "max"), "n_events": ("ts", "size")}
    if uid_col in work.columns:
        agg["n_students"] = (uid_col, "nunique")
    out = work.groupby(["group", "session_id"], sort=False, observed=True).agg(**agg).reset_index()

    if "n_students" not in out.columns:
        out["n_students"] = np.nan
        keep = out["n_events"] >= cfg.min_events
    else:
        keep = out["n_students"] >= cfg.min_students
    out = out.loc[keep].copy()
    out["duration_min"] = (out["end"] - out["start"]).dt.total_seconds() / 60.0
    out = out.loc[:, list(SESSION_COLUMNS)].sort_values("start").reset_index(drop=True)
    logger.info("gap clustering: %d sessions in %d groups (gap >= %d min, >= %d students)",
                len(out), out["group"].nunique(), cfg.gap_min, cfg.min_students)
    return out


def _forward_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Total arrivals in [i, i + window) for every bin i."""
    n = len(values)
    cum = np.concatenate([[0.0], np.cumsum(values)])
    return cum[np.minimum(np.arange(n) + window, n)] - cum[:n]


def onset_times(ts: pd.Series, cfg: Optional[SessionDetectConfig] = None, bin_minutes: int = 1,
                window_min: int = 15) -> pd.DatetimeIndex:
    """Times at which the arrival rate rises above ``cfg.rise_factor`` x its trailing median.

    A bin counts as busy only if the following ``window_min`` minutes carry at least
    ``cfg.min_events`` arrivals, which keeps single arrivals in a sparse stretch from registering
    as bursts. An onset must then be preceded by ``cfg.quiet_min`` minutes that are not busy, so a
    long busy stretch yields one onset rather than one per bin. This is the reactive analogue of a
    class start: the first moment at which a threshold autoscaler could know that load is rising.
    The rule assumes a reasonably dense trace, which is what the generic controls are; on a
    sparse log a single arrival can sit above a zero trailing median, so education logs are
    sessionised with ``gap_sessions`` instead.
    """
    cfg = cfg or SessionDetectConfig()
    counts = minute_counts(ts, bin_minutes=bin_minutes)
    if counts.empty:
        return pd.DatetimeIndex([])

    window = max(1, cfg.baseline_window_min // bin_minutes)
    baseline = counts.shift(1).rolling(window, min_periods=window // 2).median()
    thresh = np.maximum(cfg.rise_factor * baseline.to_numpy(dtype=float), cfg.min_rate_per_bin)
    values = counts.to_numpy(dtype=float)
    volume = _forward_sum(values, max(1, window_min // bin_minutes)) >= cfg.min_events
    hot = pd.Series((values >= thresh) & volume, index=counts.index).fillna(False)

    quiet_bins = max(1, cfg.quiet_min // bin_minutes)
    was_hot = hot.shift(1, fill_value=False).rolling(quiet_bins, min_periods=1).max()
    starts = hot & (was_hot == 0)
    idx = pd.DatetimeIndex(counts.index[starts.to_numpy(dtype=bool)])
    if len(idx) > cfg.max_onsets:
        logger.warning("%d onsets found, keeping the first %d", len(idx), cfg.max_onsets)
        idx = idx[: cfg.max_onsets]
    logger.info("onset detection: %d onsets over %d bins", len(idx), len(counts))
    return idx
