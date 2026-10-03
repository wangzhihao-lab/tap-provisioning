"""Forecasting: the calendar term must pay off when no same-slot history exists yet."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pilot.config import PredictConfig
from pilot.metrics.predict import (
    evaluate_forecasts,
    mase,
    prior_profile,
    seasonal_ewma,
    seasonal_naive,
    to_series,
)

START = pd.Timestamp("2026-09-07 00:00:00")  # a Monday
WEEKS = 4
COURSES = ({"group": "c1", "hour": 10, "n_students": 100, "peak": (40, 15, 5)},
           {"group": "c2", "hour": 14, "n_students": 50, "peak": (20, 8, 2)})


def _synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly classes with a burst at the session start plus a flat background."""
    arrivals: list[pd.Timestamp] = []
    groups: list[str] = []
    rows = []
    for half_hour in range(WEEKS * 7 * 48):
        arrivals.append(START + pd.Timedelta(minutes=30 * half_hour))
        groups.append("background")
    for week in range(WEEKS):
        for course in COURSES:
            start = START + pd.Timedelta(days=7 * week, hours=course["hour"])
            for minute, count in enumerate(course["peak"]):
                for i in range(count + week):  # a little week-to-week drift
                    arrivals.append(start + pd.Timedelta(minutes=minute, seconds=i))
                    groups.append(course["group"])
            rows.append({"group": course["group"], "start": start,
                         "n_students": course["n_students"], "triggers": [start]})
    df = pd.DataFrame({"ts": arrivals, "group": groups, "kind": "q"}).sort_values("ts")
    return df.reset_index(drop=True), pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


@pytest.fixture(name="synthetic")
def synthetic_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _synthetic()


def _cfg() -> PredictConfig:
    return dataclasses.replace(PredictConfig(), profile_len_min=60, first_period_days=7)


def test_calendar_beats_seasonal_naive_in_week_one(synthetic) -> None:
    df, sessions = synthetic
    cfg = _cfg()
    scores, _ = evaluate_forecasts(to_series(df, cfg.bin_minutes), sessions,
                                   list(sessions["start"]), cfg)
    assert scores.mase_calendar_first < 0.8, "no same-slot history exists in week 1"
    assert scores.mase_calendar < 1.0


def test_calendar_stays_comparable_once_history_exists(synthetic) -> None:
    df, sessions = synthetic
    cfg = _cfg()
    scores, _ = evaluate_forecasts(to_series(df, cfg.bin_minutes), sessions,
                                   list(sessions["start"]), cfg)
    assert scores.mase_calendar_later <= 1.15
    assert scores.mae_window_calendar < scores.mae_window_snaive
    assert scores.window_gain > 0.0


def test_predictions_are_causal(synthetic) -> None:
    """Dropping everything after the first week must not change the week-1 predictions."""
    df, sessions = synthetic
    cfg = _cfg()
    cutoff = START + pd.Timedelta(days=7)
    full = evaluate_forecasts(to_series(df, cfg.bin_minutes), sessions,
                              list(sessions["start"]), cfg)[1]
    head_df = df[df["ts"] < cutoff]
    head_sessions = sessions[sessions["start"] < cutoff]
    head = evaluate_forecasts(to_series(head_df, cfg.bin_minutes), head_sessions,
                              list(head_sessions["start"]), cfg)[1]
    common = head.index.intersection(full.index)
    np.testing.assert_allclose(full.loc[common, "calendar"], head.loc[common, "calendar"],
                               atol=1e-9)


def test_seasonal_naive_averages_previous_weeks() -> None:
    season = 4
    values = np.array([1.0, 0, 0, 0, 3.0, 0, 0, 0, 5.0, 0, 0, 0], dtype=float)
    out = seasonal_naive(values, season_bins=season, n_weeks=2)
    assert np.isnan(out[0])
    assert out[4] == 1.0
    assert out[8] == pytest.approx(2.0)  # mean of 1 and 3


def test_seasonal_ewma_tracks_the_slot() -> None:
    values = np.array([1.0, 0, 3.0, 0, 5.0, 0], dtype=float)
    out = seasonal_ewma(values, season_bins=2, alpha=0.5)
    assert np.isnan(out[0])
    assert out[2] == 1.0
    assert out[4] == pytest.approx(2.0)  # 0.5 * 3 + 0.5 * 1


def test_mase_is_undefined_against_a_perfect_benchmark() -> None:
    truth = np.array([1.0, 2.0])
    assert np.isnan(mase(truth, truth, truth))


def test_prior_profile_carries_the_configured_mass() -> None:
    cfg = _cfg()
    profile = prior_profile(cfg, 60)
    assert profile.sum() == pytest.approx(cfg.prior_sync_fraction)
    assert profile[0] > profile[1] > profile[2]


def test_no_sessions_falls_back_to_history(synthetic) -> None:
    df, _ = synthetic
    cfg = _cfg()
    scores, frame = evaluate_forecasts(to_series(df, cfg.bin_minutes), None, [], cfg)
    assert scores.n_sessions == 0
    assert np.isnan(scores.mae_window_snaive)
    assert frame["calendar"].notna().any()
