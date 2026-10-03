"""Session detection, the Q1 orchestrator and the CLI."""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from pilot.cli import build_sources, main
from pilot.config import AnalysisConfig, MockConfig, PredictConfig, SessionDetectConfig
from pilot.metrics.sessions import gap_sessions, onset_times
from pilot.q1 import run_q1
from pilot.sources.mock import MockSource

START = pd.Timestamp("2026-09-09 10:00:00")


def _class_log(n_students: int = 6) -> pd.DataFrame:
    rows = []
    for session, offset_h in enumerate((0, 4)):
        for student in range(n_students):
            for item in range(4):
                rows.append({"ts": START + pd.Timedelta(hours=offset_h, minutes=item * 5,
                                                        seconds=student),
                             "group": "class-A", "uid": f"u{student}", "session": session})
    return pd.DataFrame(rows)


def test_gap_clustering_splits_on_inactivity() -> None:
    sessions = gap_sessions(_class_log(), SessionDetectConfig())
    assert len(sessions) == 2
    assert (sessions["n_students"] == 6).all()
    assert sessions["duration_min"].iloc[0] == pytest.approx(15.0, abs=0.1)


def test_gap_clustering_drops_thin_sessions() -> None:
    detect = dataclasses.replace(SessionDetectConfig(), min_students=7)
    assert gap_sessions(_class_log(), detect).empty


def _rise(quiet_per_min: int = 2, surge_per_min: int = 30) -> pd.Series:
    """Four hours at a steady low rate, then ten minutes at a much higher one."""
    quiet = [START + pd.Timedelta(minutes=m, seconds=30 * k)
             for m in range(240) for k in range(quiet_per_min)]
    surge = [START + pd.Timedelta(minutes=240 + m, seconds=int(60 * k / surge_per_min))
             for m in range(10) for k in range(surge_per_min)]
    return pd.Series(quiet + surge)


def test_onset_detection_finds_a_rise_after_a_quiet_stretch() -> None:
    onsets = onset_times(_rise(), SessionDetectConfig())
    assert list(onsets) == [START + pd.Timedelta(minutes=240)]


def test_onset_detection_ignores_a_steady_rate() -> None:
    steady = [START + pd.Timedelta(minutes=m, seconds=30 * k)
              for m in range(300) for k in range(2)]
    assert len(onset_times(pd.Series(steady), SessionDetectConfig())) == 0


def test_run_q1_writes_every_output(tmp_path: Path) -> None:
    cfg = dataclasses.replace(MockConfig(), weeks=3)
    summary = run_q1({"mock": MockSource(cfg)}, tmp_path)
    expected = ["mock_bursts.csv", "summary.csv", "summary.md", "ramp_cdf.pdf", "ramp_cdf.png",
                "sigma_hist.pdf", "sigma_hist.png", "predictability.pdf", "predictability.png"]
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0, name
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_bursts"] == cfg.weeks * len(cfg.sessions) * cfg.triggers_per_session
    assert 0.0 < row["ramp_min_median"] < 5.0
    assert row["sync_1min_median"] > 0.5
    assert row["p2m_1min"] > row["p2m_5min"]
    assert row["mase_calendar_first"] < 1.0
    bursts = pd.read_csv(tmp_path / "mock_bursts.csv")
    assert {"t0", "ramp_min", "sync_2min"} <= set(bursts.columns)
    assert "Q1 summary" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_run_q1_accepts_a_source_without_sessions(tmp_path: Path) -> None:
    analysis = dataclasses.replace(AnalysisConfig(), bin_minutes=1)
    predict = dataclasses.replace(PredictConfig(), profile_len_min=30)
    summary = run_q1({"flat": _FlatSource()}, tmp_path, analysis, predict)
    assert summary.loc[0, "workload_class"] == "generic"
    assert summary.loc[0, "timetable"] == "detected"


class _FlatSource(MockSource):
    """A trace-like source: no timetable, so Q1 has to detect its own onsets."""

    workload_class = "generic"
    timetable_kind = "detected"

    def sessions(self):  # type: ignore[override]
        return None

    def load(self) -> pd.DataFrame:
        quiet = [START + pd.Timedelta(minutes=i) for i in range(0, 600, 3)]
        surge = [START + pd.Timedelta(minutes=600, seconds=s) for s in range(0, 1800, 3)]
        df = pd.DataFrame({"ts": quiet + surge, "group": "trace", "kind": "request"})
        return self.validate(df)


def test_cli_mock_dump(tmp_path: Path) -> None:
    out = tmp_path / "arrivals.csv"
    assert main(["mock", "--out", str(out), "--weeks", "1"]) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == ["ts", "group", "kind"]
    assert len(df) > 0


def test_build_sources_requires_paths() -> None:
    assert list(build_sources(["mock"], [], [])) == ["mock"]
    with pytest.raises(ValueError, match="needs --path"):
        build_sources(["assistments"], [], [])
    with pytest.raises(ValueError, match="needs --paths"):
        build_sources(["xxt_signin"], [], [])
    with pytest.raises(ValueError, match="unused"):
        build_sources(["mock"], ["stray.csv"], [])


def test_cli_reports_a_missing_path() -> None:
    with pytest.raises(SystemExit):
        main(["q1", "--source", "assistments"])
