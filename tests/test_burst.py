"""Burst metrics on hand-built arrival patterns with known answers."""

import numpy as np
import pandas as pd

from pilot.metrics.burst import burst_stats, minute_counts, peak_to_mean, ramp_time

T0 = pd.Timestamp("2026-09-09 10:00:00")


def _at(offsets_s: list[float]) -> pd.Series:
    return pd.Series([T0 + pd.Timedelta(seconds=s) for s in offsets_s])


def test_ramp_time_on_a_uniform_burst() -> None:
    # 100 arrivals one second apart: the 10% and 90% quantiles sit 79.2 s apart
    ramp = ramp_time(_at([float(i) for i in range(100)]))
    assert abs(ramp - 79.2 / 60.0) < 1e-6


def test_ramp_time_needs_two_points() -> None:
    assert np.isnan(ramp_time(_at([0.0])))


def test_sync_fractions_are_exact() -> None:
    offsets = [10.0] * 50 + [90.0] * 20 + [240.0] * 30
    stats = burst_stats(_at(offsets), T0, window_min=15, ks=(1, 2, 5))
    assert stats.n == 100
    assert stats.sync_fraction[1] == 0.5
    assert stats.sync_fraction[2] == 0.7
    assert stats.sync_fraction[5] == 1.0
    assert stats.peak_per_min == 50


def test_arrivals_after_the_window_are_ignored() -> None:
    stats = burst_stats(_at([10.0, 20.0, 3600.0]), T0, window_min=15, ks=(1,))
    assert stats.n == 2


def test_empty_window_is_reported_as_nan() -> None:
    stats = burst_stats(_at([3600.0]), T0, window_min=15, ks=(1, 2))
    assert stats.n == 0
    assert np.isnan(stats.ramp_min)
    assert all(np.isnan(v) for v in stats.sync_fraction.values())


def test_minute_counts_fill_the_gaps() -> None:
    counts = minute_counts(_at([0.0, 30.0, 600.0]), bin_minutes=1)
    assert counts.iloc[0] == 2
    assert counts.iloc[1] == 0
    assert counts.loc[T0 + pd.Timedelta(minutes=10)] == 1
    assert peak_to_mean(counts) > 1.0


def test_peak_to_mean_of_an_empty_series_is_nan() -> None:
    assert np.isnan(peak_to_mean(pd.Series(dtype="int64")))
