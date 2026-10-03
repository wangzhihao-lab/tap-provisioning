"""The synthetic source must be well formed and reproducible."""

import dataclasses

import pandas as pd

from pilot.config import MockConfig
from pilot.sources.mock import MockSource


def test_load_schema() -> None:
    df = MockSource(dataclasses.replace(MockConfig(), weeks=2)).load()
    assert list(df.columns) == ["ts", "group", "kind"]
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert df["ts"].is_monotonic_increasing
    assert set(df["kind"]) == {"in_class", "background", "deadline"}
    assert set(df["group"]) == {"class-A", "class-B"}
    assert len(df) > 0


def test_load_is_deterministic() -> None:
    cfg = dataclasses.replace(MockConfig(), weeks=2)
    first, second = MockSource(cfg).load(), MockSource(cfg).load()
    pd.testing.assert_frame_equal(first, second)


def test_seed_changes_the_draw() -> None:
    cfg = dataclasses.replace(MockConfig(), weeks=2)
    other = dataclasses.replace(cfg, seed=cfg.seed + 1)
    assert not MockSource(cfg).load().equals(MockSource(other).load())


def test_sessions_match_the_timetable() -> None:
    cfg = dataclasses.replace(MockConfig(), weeks=3)
    sessions = MockSource(cfg).sessions()
    assert len(sessions) == cfg.weeks * len(cfg.sessions)
    assert set(sessions.columns) >= {"start", "duration_min", "n_students", "group", "triggers"}
    assert all(len(t) == cfg.triggers_per_session for t in sessions["triggers"])
    assert sessions["start"].is_monotonic_increasing


def test_bursts_land_on_the_triggers() -> None:
    cfg = dataclasses.replace(MockConfig(), weeks=1)
    src = MockSource(cfg)
    df, sessions = src.load(), src.sessions()
    t0 = sessions.loc[0, "triggers"][0]
    window = df[(df["ts"] >= t0) & (df["ts"] < t0 + pd.Timedelta(minutes=5))]
    expected = cfg.sync_fraction * sessions.loc[0, "n_students"]
    assert len(window) > 0.5 * expected
