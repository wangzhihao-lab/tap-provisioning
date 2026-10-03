"""Real-export loaders: schema handling, session inference and clear failures."""

from pathlib import Path

import pandas as pd
import pytest

from pilot.config import LoaderConfig
from pilot.sources import SourceFactory
from pilot.sources.assistments import AssistmentsSource
from pilot.sources.generic_traces import AzureTrace2023, BurstGPTTrace, SageServeTrace

CLASS_START = pd.Timestamp("2012-09-05 10:00:00")


def _assistments_csv(path: Path, n_students: int = 6) -> Path:
    rows = []
    for session, offset_h in enumerate((0, 3)):
        base = CLASS_START + pd.Timedelta(hours=offset_h)
        for student in range(n_students):
            for item in range(3):
                start = base + pd.Timedelta(seconds=30 * item + student)
                rows.append({"start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                             "end_time": (start + pd.Timedelta(seconds=20)).isoformat(),
                             "user_id": 1000 + student, "student_class_id": 77,
                             "teacher_id": 5, "school_id": 9, "skill": f"s{session}"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_assistments_arrivals_and_sessions(tmp_path: Path) -> None:
    src = AssistmentsSource(_assistments_csv(tmp_path / "assist.csv"))
    df = src.load()
    assert list(df.columns) == ["ts", "group", "kind"]
    assert set(df["kind"]) == {"problem"}
    assert set(df["group"]) == {"77"}
    sessions = src.sessions()
    assert len(sessions) == 2
    assert (sessions["n_students"] == 6).all()
    assert all(len(t) == 1 for t in sessions["triggers"])
    assert sessions["start"].iloc[0] == CLASS_START


def test_assistments_never_emits_identifiers(tmp_path: Path) -> None:
    src = AssistmentsSource(_assistments_csv(tmp_path / "assist.csv"))
    for frame in (src.load(), src.sessions()):
        assert not any("user" in str(c).lower() or "uid" in str(c).lower() for c in frame.columns)


def test_assistments_honours_max_rows(tmp_path: Path) -> None:
    cfg = LoaderConfig(chunk_rows=7, max_rows=10)
    src = AssistmentsSource(_assistments_csv(tmp_path / "assist.csv"), cfg=cfg)
    assert len(src.load()) == 10


def test_assistments_rejects_a_file_without_wall_clock_time(tmp_path: Path) -> None:
    path = tmp_path / "assist_2009.csv"
    pd.DataFrame({"order_id": [1, 2], "user_id": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="2009-10"):
        AssistmentsSource(path).load()


def test_burstgpt_anchors_relative_seconds(tmp_path: Path) -> None:
    path = tmp_path / "burstgpt.csv"
    pd.DataFrame({"Timestamp": [0, 60, 120], "Model": ["ChatGPT"] * 3,
                  "Request tokens": [10, 20, 30], "Response tokens": [1, 2, 3]}).to_csv(path,
                                                                                       index=False)
    df = BurstGPTTrace(path).load()
    origin = pd.Timestamp(LoaderConfig().burstgpt_origin)
    assert df["ts"].iloc[0] == origin
    assert df["ts"].iloc[-1] == origin + pd.Timedelta(minutes=2)
    assert list(df["group"]) == ["burstgpt"] * 3
    assert df["req_tokens"].tolist() == [10, 20, 30]
    assert BurstGPTTrace(path).workload_class == "generic"


def test_azure_trace_parses_wall_clock(tmp_path: Path) -> None:
    path = tmp_path / "azure.csv"
    pd.DataFrame({"TIMESTAMP": ["2023-11-11 00:00:00.100", "2023-11-11 00:01:00.200"],
                  "ContextTokens": [5, 6], "GeneratedTokens": [1, 2]}).to_csv(path, index=False)
    df = AzureTrace2023(path).load()
    assert df["ts"].iloc[0].year == 2023
    assert df["resp_tokens"].tolist() == [1, 2]


def test_sageserve_reports_an_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "ES_26.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no arrival-time column"):
        SageServeTrace(path).load()


def test_sageserve_accepts_a_recognisable_schema(tmp_path: Path) -> None:
    path = tmp_path / "ES_26.csv"
    pd.DataFrame({"arrival_time": ["2025-01-01 00:00:00"], "input_tokens": [3],
                  "output_tokens": [4]}).to_csv(path, index=False)
    df = SageServeTrace(path).load()
    assert len(df) == 1
    assert df["req_tokens"].iloc[0] == 3


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AssistmentsSource(tmp_path / "nope.csv").load()
    with pytest.raises(FileNotFoundError):
        BurstGPTTrace(tmp_path / "nope.csv").load()


def test_registry_exposes_every_source() -> None:
    for name in ("mock", "assistments", "burstgpt", "azure_2023", "azure_2024", "sageserve",
                 "xxt_signin", "xxt_homework", "xxt_ai_sessions"):
        assert SourceFactory(name) is not None
    with pytest.raises(KeyError):
        SourceFactory("does_not_exist")
