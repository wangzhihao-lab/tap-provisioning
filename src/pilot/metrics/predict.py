"""Arrival forecasting at horizon H: history-only vs. calendar-informed predictors.

Q1 asks whether education arrivals are more *calendar-predictable* than generic LLM traffic.
Two families of predictors are compared on the same per-bin series:

* history-only -- seasonal naive (the same weekday-and-time slot averaged over the previous
  ``season_weeks`` weeks) and a seasonal exponential-smoothing variant of it. Both fall back to a
  causal expanding mean while no previous week exists, which is exactly the semester-first-week
  situation the calendar term is meant to cover.
* calendar-informed -- the history-only estimate of the *background* (bins outside any session)
  plus a timetable term: enrollment x a per-session usage profile (arrivals per student per bin
  since session start) learned from previous sessions of the same course, backing off to a
  cross-course pooled profile and finally to a parametric prior.

Everything a predictor uses is available at the forecast origin t - H: lagged weeks are at least
one season old, expanding means are shifted by H bins, and a session only contributes to a profile
once it has finished at least H bins before the session being predicted.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import PredictConfig
from .burst import minute_counts

logger = logging.getLogger(__name__)


def to_series(df: pd.DataFrame, bin_minutes: int = 1) -> pd.Series:
    """Arrivals per bin over the whole span of an arrival frame, zero-filled."""
    return minute_counts(df["ts"], bin_minutes=bin_minutes).astype(float)


def _causal_level(values: np.ndarray, horizon_bins: int) -> np.ndarray:
    """Expanding mean of everything up to the forecast origin, NaN-aware."""
    n = len(values)
    out = np.full(n, np.nan)
    if n <= horizon_bins:
        return out
    ok = ~np.isnan(values)
    cum = np.cumsum(np.where(ok, values, 0.0))
    cnt = np.cumsum(ok.astype(float))
    keep = cnt[: n - horizon_bins] > 0
    tail = np.full(n - horizon_bins, np.nan)
    tail[keep] = cum[: n - horizon_bins][keep] / cnt[: n - horizon_bins][keep]
    out[horizon_bins:] = tail
    return out


def seasonal_naive(values: np.ndarray, season_bins: int, n_weeks: int) -> np.ndarray:
    """Mean of the same slot over the previous ``n_weeks`` seasons, skipping missing slots."""
    n = len(values)
    total = np.zeros(n)
    count = np.zeros(n)
    for w in range(1, n_weeks + 1):
        lag = w * season_bins
        if lag >= n:
            break
        past = values[: n - lag]
        ok = ~np.isnan(past)
        total[lag:][ok] += past[ok]
        count[lag:][ok] += 1.0
    out = np.full(n, np.nan)
    hit = count > 0
    out[hit] = total[hit] / count[hit]
    return out


def seasonal_ewma(values: np.ndarray, season_bins: int, alpha: float) -> np.ndarray:
    """Seasonal exponential smoothing: one smoothed level per slot of the weekly season."""
    n = len(values)
    out = np.full(n, np.nan)
    state = np.full(season_bins, np.nan)
    for i in range(n):
        slot = i % season_bins
        out[i] = state[slot]
        v = values[i]
        if np.isnan(v):
            continue
        state[slot] = v if np.isnan(state[slot]) else alpha * v + (1.0 - alpha) * state[slot]
    return out


def _with_fallback(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Use ``primary`` where it exists, ``fallback`` elsewhere."""
    out = primary.copy()
    gap = np.isnan(out)
    out[gap] = fallback[gap]
    return out


def history_only(values: np.ndarray, cfg: PredictConfig, season_bins: int,
                 horizon_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Seasonal-naive and seasonal-EWMA predictions, both backed by a causal expanding mean."""
    level = _causal_level(values, horizon_bins)
    snaive = _with_fallback(seasonal_naive(values, season_bins, cfg.season_weeks), level)
    ewma = _with_fallback(seasonal_ewma(values, season_bins, cfg.ewma_alpha), level)
    return snaive, ewma


def prior_profile(cfg: PredictConfig, profile_bins: int) -> np.ndarray:
    """Cross-course prior: a decaying burst of ``prior_sync_fraction`` queries per student."""
    scale = max(cfg.prior_decay_min / cfg.bin_minutes, 1e-6)
    weights = np.exp(-np.arange(profile_bins, dtype=float) / scale)
    return cfg.prior_sync_fraction * weights / weights.sum()


def _group_positions(arrivals: pd.DataFrame, index: pd.DatetimeIndex,
                     bin_minutes: int) -> Dict[str, np.ndarray]:
    """Bin positions of every arrival, split by group and sorted."""
    offs = (pd.to_datetime(arrivals["ts"]) - index[0]).dt.total_seconds().to_numpy()
    pos = np.floor(offs / (60.0 * bin_minutes)).astype(int)
    frame = pd.DataFrame({"pos": pos, "group": arrivals["group"].astype(str).to_numpy()})
    return {str(g): np.sort(part["pos"].to_numpy())
            for g, part in frame.groupby("group", sort=False, observed=True)}


def _window_counts(positions: np.ndarray, start: int, length: int) -> np.ndarray:
    """Arrivals per bin in [start, start + length) for one group."""
    lo = int(np.searchsorted(positions, start, side="left"))
    hi = int(np.searchsorted(positions, start + length, side="left"))
    if hi <= lo:
        return np.zeros(length)
    return np.bincount(positions[lo:hi] - start, minlength=length)[:length].astype(float)


def _session_positions(sessions: pd.DataFrame, index: pd.DatetimeIndex,
                       bin_minutes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map session starts onto bin positions; returns (positions, enrollments, groups)."""
    if sessions is None or sessions.empty:
        return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=object)
    starts = pd.to_datetime(sessions["start"])
    offs = (starts - index[0]).dt.total_seconds().to_numpy() / (60.0 * bin_minutes)
    pos = np.rint(offs).astype(int)
    enroll = (pd.to_numeric(sessions.get("n_students"), errors="coerce").to_numpy(dtype=float)
              if "n_students" in sessions.columns else np.full(len(sessions), np.nan))
    enroll = np.where(np.isfinite(enroll) & (enroll > 0), enroll, 1.0)
    groups = sessions["group"].astype(str).to_numpy() if "group" in sessions.columns \
        else np.array(["all"] * len(sessions), dtype=object)
    keep = (pos >= 0) & (pos < len(index))
    order = np.argsort(pos[keep], kind="mergesort")
    return pos[keep][order], enroll[keep][order], groups[keep][order]


def calendar_forecast(values: np.ndarray, index: pd.DatetimeIndex, sessions: pd.DataFrame,
                      cfg: PredictConfig, arrivals: Optional[pd.DataFrame] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Background estimate plus the timetable term; returns (calendar, background).

    With ``arrivals`` the per-session profile is learned from that session's *own* group, which is
    what makes the term usable when many courses run at once: learning it from the aggregate
    series would let every concurrent course claim the whole load. Without it (a single-group
    series) the aggregate residual is used instead.
    """
    n = len(values)
    season_bins = max(1, int(cfg.season_days * 24 * 60 // cfg.bin_minutes))
    horizon_bins = max(1, int(np.ceil(cfg.horizon_min / cfg.bin_minutes)))
    profile_bins = max(1, int(np.ceil(cfg.profile_len_min / cfg.bin_minutes)))
    pos, enroll, groups = _session_positions(sessions, index, cfg.bin_minutes)

    occupied = np.zeros(n, dtype=bool)
    for p in pos:
        occupied[p: p + profile_bins] = True
    background = np.where(occupied, np.nan, values)
    bg = _with_fallback(seasonal_naive(background, season_bins, cfg.season_weeks),
                        _causal_level(background, horizon_bins))
    bg = np.where(np.isnan(bg), 0.0, bg)
    if len(pos) == 0:
        return bg, bg

    # the residual is signed: clipping it here would bias the learned profile upward on the
    # mostly-empty bins of a session window
    resid = values - bg
    by_group = (_group_positions(arrivals, index, cfg.bin_minutes)
                if arrivals is not None and len(np.unique(groups)) > 1 else None)
    profiles = np.zeros((len(pos), profile_bins))
    for j, (p, e, g) in enumerate(zip(pos, enroll, groups)):
        if by_group is not None:
            chunk = _window_counts(by_group.get(str(g), np.empty(0, dtype=int)), p, profile_bins)
        else:
            chunk = resid[p: p + profile_bins]
        profiles[j, : len(chunk)] = chunk / e
    cum_all = np.vstack([np.zeros(profile_bins), np.cumsum(profiles, axis=0)])
    per_group: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for g in np.unique(groups):
        sel = np.flatnonzero(groups == g)
        per_group[g] = (pos[sel], np.vstack([np.zeros(profile_bins), np.cumsum(profiles[sel], 0)]))
    prior = prior_profile(cfg, profile_bins)

    timetable = np.zeros(n)
    for i, (p, e, g) in enumerate(zip(pos, enroll, groups)):
        limit = p - horizon_bins - profile_bins
        g_pos, g_cum = per_group[g]
        k_g = int(np.searchsorted(g_pos, limit, side="right"))
        if k_g > 0:
            profile = g_cum[k_g] / k_g
        else:
            k_all = int(np.searchsorted(pos, limit, side="right"))
            profile = cum_all[k_all] / k_all if k_all > 0 else prior
        span = min(profile_bins, n - p)
        timetable[p: p + span] += e * profile[:span]
    logger.debug("calendar term over %d sessions, profile %d bins", len(pos), profile_bins)
    return np.clip(bg + timetable, 0.0, None), bg


@dataclass(frozen=True)
class ForecastScores:
    """Errors of each predictor on one source."""

    n_bins: int
    n_sessions: int
    mae_snaive: float
    mae_ewma: float
    mae_calendar: float
    mase_ewma: float
    mase_calendar: float
    mase_calendar_first: float
    mase_calendar_later: float
    mae_window_snaive: float
    mae_window_calendar: float
    window_gain: float

    def as_dict(self) -> Dict[str, float]:
        """Flat mapping for the summary table."""
        return asdict(self)


def _mae(truth: np.ndarray, pred: np.ndarray) -> float:
    if truth.size == 0:
        return float("nan")
    return float(np.mean(np.abs(truth - pred)))


def mase(truth: np.ndarray, pred: np.ndarray, reference: np.ndarray,
         eps: float = 1e-9) -> float:
    """Mean absolute error scaled by the seasonal-naive benchmark on the same bins."""
    denom = _mae(truth, reference)
    if not np.isfinite(denom) or denom < eps:
        logger.warning("seasonal-naive MAE is %.3g; MASE undefined", denom)
        return float("nan")
    return _mae(truth, pred) / denom


def _window_mask(index: pd.DatetimeIndex, windows: Sequence[pd.Timestamp],
                 window_min: int) -> np.ndarray:
    """Bins that fall inside [t0, t0 + window_min) for any trigger t0."""
    mask = np.zeros(len(index), dtype=bool)
    span = pd.Timedelta(minutes=window_min)
    for t0 in windows:
        t0 = pd.Timestamp(t0)
        mask |= (index >= t0) & (index < t0 + span)
    return mask


def evaluate_forecasts(y: pd.Series, sessions: Optional[pd.DataFrame],
                       windows: Sequence[pd.Timestamp], cfg: PredictConfig,
                       window_min: int = 15, arrivals: Optional[pd.DataFrame] = None
                       ) -> Tuple[ForecastScores, pd.DataFrame]:
    """Score the history-only and calendar-informed predictors on one arrival series.

    Args:
        y: arrivals per bin, zero-filled, on a regular grid.
        sessions: timetable with start, group and (optionally) n_students.
        windows: trigger instants used for the session-start-window error.
        cfg: predictor parameters.
        window_min: length of the session-start window in minutes.
        arrivals: the raw arrival frame, so per-session profiles can be learned per group.

    Returns:
        The scores and a per-bin frame with the truth and every prediction.
    """
    index = pd.DatetimeIndex(y.index)
    values = y.to_numpy(dtype=float)
    season_bins = max(1, int(cfg.season_days * 24 * 60 // cfg.bin_minutes))
    horizon_bins = max(1, int(np.ceil(cfg.horizon_min / cfg.bin_minutes)))
    snaive, ewma = history_only(values, cfg, season_bins, horizon_bins)
    cal, _ = calendar_forecast(values, index, sessions, cfg, arrivals)

    frame = pd.DataFrame({"y": values, "snaive": snaive, "ewma": ewma, "calendar": cal},
                         index=index)
    valid = np.isfinite(snaive) & np.isfinite(ewma) & np.isfinite(cal)
    valid[:horizon_bins] = False
    cutoff = index[0] + pd.Timedelta(days=cfg.first_period_days)
    first = valid & np.asarray(index < cutoff)
    later = valid & ~first
    win = valid & _window_mask(index, windows, window_min)

    def cut(mask: np.ndarray, col: str) -> np.ndarray:
        return frame[col].to_numpy()[mask]

    scores = ForecastScores(
        n_bins=int(valid.sum()),
        n_sessions=0 if sessions is None else int(len(sessions)),
        mae_snaive=_mae(cut(valid, "y"), cut(valid, "snaive")),
        mae_ewma=_mae(cut(valid, "y"), cut(valid, "ewma")),
        mae_calendar=_mae(cut(valid, "y"), cut(valid, "calendar")),
        mase_ewma=mase(cut(valid, "y"), cut(valid, "ewma"), cut(valid, "snaive"),
                       cfg.min_denominator),
        mase_calendar=mase(cut(valid, "y"), cut(valid, "calendar"), cut(valid, "snaive"),
                           cfg.min_denominator),
        mase_calendar_first=mase(cut(first, "y"), cut(first, "calendar"), cut(first, "snaive"),
                                 cfg.min_denominator),
        mase_calendar_later=mase(cut(later, "y"), cut(later, "calendar"), cut(later, "snaive"),
                                 cfg.min_denominator),
        mae_window_snaive=_mae(cut(win, "y"), cut(win, "snaive")),
        mae_window_calendar=_mae(cut(win, "y"), cut(win, "calendar")),
        window_gain=float("nan"),
    )
    gain = (1.0 - scores.mae_window_calendar / scores.mae_window_snaive
            if scores.mae_window_snaive > cfg.min_denominator else float("nan"))
    scores = ForecastScores(**{**scores.as_dict(), "window_gain": gain})
    logger.info("forecast scores: MASE calendar %.3f (first period %.3f), EWMA %.3f",
                scores.mase_calendar, scores.mase_calendar_first, scores.mase_ewma)
    return scores, frame
