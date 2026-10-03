"""Per-session sign-in metrics of Xuexitong sign-in detail exports (scripts/q1/signin_sessions.py).

The workbooks are synthetic. Their shape copies the one recorded in
the data audit (2026-09-09, 118 enrolled, 106 signed, 82 of them inside the
same minute and 102 inside five), so the numbers the script reports can be checked by hand.
No real student appears anywhere.
"""

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
FIXTURES = TESTS / "fixtures"
SAMPLE = FIXTURES / "xxt_signin_sample.xlsx"

COLUMNS = ["序号", "姓名", "学号", "班级", "签到状态", "签到时间", "签到方式", "发起时间"]
CREATED = "2026-09-08 19:17:00"
SAMPLE_T0 = "2026-09-09 10:18:00"
# seconds after t0: 82 inside the first minute, then 7/3/2/8 in the next four, then four stragglers
SAMPLE_OFFSETS: List[int] = (
    [i % 60 for i in range(82)]
    + [60 + 5 * k for k in range(7)]
    + [120 + 7 * k for k in range(3)]
    + [180 + 9 * k for k in range(2)]
    + [240 + 6 * k for k in range(8)]
    + [420, 840, 2280, 3420]
)
SAMPLE_ABSENT = 12
NAMES = [f"测试学生{i:03d}" for i in range(len(SAMPLE_OFFSETS) + SAMPLE_ABSENT)]


def _load_script():
    """Import the script by path; scripts/ is not a package."""
    path = TESTS.parent / "scripts" / "q1" / "signin_sessions.py"
    spec = importlib.util.spec_from_file_location("signin_sessions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sig = _load_script()


def signin_frame(t0: str, offsets: Sequence[int], absent: int = 0,
                 created: str = CREATED, clazz: str = "课程A2025-3班") -> pd.DataFrame:
    """One sign-in detail sheet: signed rows carry a 签到时间, 未签 rows leave that cell empty."""
    base = pd.Timestamp(t0)
    rows = []
    for i, sec in enumerate(offsets):
        ts = base + pd.Timedelta(seconds=int(sec))
        rows.append({"序号": i + 1, "姓名": NAMES[i], "学号": f"2025{i:07d}", "班级": clazz,
                     "签到状态": "已签", "签到时间": ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "签到方式": "手势签到", "发起时间": created})
    for j in range(absent):
        i = len(offsets) + j
        rows.append({"序号": i + 1, "姓名": NAMES[i], "学号": f"2025{i:07d}", "班级": clazz,
                     "签到状态": "未签", "签到时间": None, "签到方式": None,
                     "发起时间": created})
    return pd.DataFrame(rows, columns=COLUMNS)


def build_workbook(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False, sheet_name="签到详情")
    return path


def sample_workbook() -> Path:
    """The committed fixture: 118 rows, 106 of them with a parsable time."""
    if not SAMPLE.exists():
        build_workbook(SAMPLE, signin_frame(SAMPLE_T0, SAMPLE_OFFSETS, SAMPLE_ABSENT))
    return SAMPLE


@pytest.fixture()
def exports(tmp_path: Path) -> Path:
    """Three sign-in activities: the 106-row burst, a 30-row spread, a 5-row leftover."""
    folder = tmp_path / "signin"
    folder.mkdir()
    shutil.copy(sample_workbook(), folder / SAMPLE.name)
    build_workbook(folder / "signin_20260902_1010.xlsx",
                   signin_frame("2026-09-02 10:10:00", [int(k * 1200 / 29) for k in range(30)],
                                created="2026-09-02 10:08:00"))
    build_workbook(folder / "signin_20260826_0800.xlsx",
                   signin_frame("2026-08-26 08:00:00", [0, 12, 30, 65, 130],
                                created="2026-08-26 07:58:00"))
    return folder


def _table(exports: Path) -> pd.DataFrame:
    paths = sig.collect_paths([str(exports / "*.xlsx")])
    assert len(paths) == 3
    return sig.session_table(paths, sig.MetricConfig()).set_index("session")


def _run(out: Path, exports: Path, compare: Optional[Path] = None) -> Path:
    argv = ["--paths", str(exports / "*.xlsx"), "--out", str(out)]
    if compare is not None:
        argv += ["--compare", str(compare)]
    assert sig.main(argv) == 0
    return out


@pytest.fixture()
def out_dir(tmp_path: Path, exports: Path) -> Path:
    compare = tmp_path / "student_arrival_sessions_min10.csv"
    pd.DataFrame({"ramp_10_90_min": [0.4, 2.5, 7.0, 40.0],
                  "in_class_proxy": [True, True, True, False]}).to_csv(compare, index=False)
    return _run(tmp_path / "out", exports, compare)


def test_only_rows_with_a_parsable_time_are_counted(exports: Path) -> None:
    table = _table(exports)
    assert list(table.index) == ["signin_20260826_0800", "signin_20260902_1010",
                                "xxt_signin_sample"]
    assert table.loc["xxt_signin_sample", "n"] == 106  # 118 rows, 12 未签 have no time
    assert table.loc["signin_20260902_1010", "n"] == 30
    assert table.loc["signin_20260826_0800", "n"] == 5


def test_sample_session_is_one_synchronised_minute(exports: Path) -> None:
    row = _table(exports).loc["xxt_signin_sample"]
    assert row["sync_1min"] == pytest.approx(82 / 106)
    assert row["sync_2min"] == pytest.approx(89 / 106)
    assert row["sync_5min"] == pytest.approx(102 / 106)
    assert row["share_within_15min"] == pytest.approx(104 / 106)
    assert row["span_min"] == pytest.approx(57.0)
    assert row["median_arrival_min"] < 1.0
    assert 0.0 < row["ramp_10_90_min"] < 5.0
    assert row["date"] == "2026-09-09"
    assert row["weekday"] == pd.Timestamp(SAMPLE_T0).weekday()
    assert row["hour"] == 10


def test_a_spread_out_session_ramps_slower(exports: Path) -> None:
    table = _table(exports)
    spread = table.loc["signin_20260902_1010"]
    assert spread["span_min"] == pytest.approx(20.0, abs=0.1)
    assert spread["ramp_10_90_min"] == pytest.approx(16.0, abs=0.6)
    assert spread["ramp_10_90_min"] > table.loc["xxt_signin_sample", "ramp_10_90_min"]


def test_start_time_column_gives_the_secondary_metrics(exports: Path) -> None:
    row = _table(exports).loc["xxt_signin_sample"]
    assert row["t_start"] == "2026-09-08 19:17:00"
    # the activity was created the evening before, so delays from it dwarf the t0-relative ones
    assert row["median_arrival_min_from_start"] == pytest.approx(901.0, abs=1.0)
    assert row["sync_5min_from_start"] == 0.0
    assert row["ramp_10_90_min"] > 0.0


def test_a_missing_start_column_is_not_fatal(tmp_path: Path) -> None:
    frame = signin_frame("2026-09-09 10:18:00", [0, 30, 90]).drop(columns=["发起时间"])
    path = build_workbook(tmp_path / "no_start.xlsx", frame)
    assert sig.read_start_time(path) is None
    row = sig.session_table([path], sig.MetricConfig()).iloc[0]
    assert row["n"] == 3
    assert "t_start" not in row.index


def test_main_writes_every_output_and_counts_the_dropped_session(out_dir: Path) -> None:
    expected = ["signin_sessions.csv", "signin_summary.csv", "signin_summary.md",
                "signin_ramp_cdf.pdf", "signin_ramp_cdf.png"]
    for name in expected:
        assert (out_dir / name).stat().st_size > 0, name
    sessions = pd.read_csv(out_dir / "signin_sessions.csv")
    assert set(sessions["session"]) == {"xxt_signin_sample", "signin_20260902_1010"}
    summary = pd.read_csv(out_dir / "signin_summary.csv")
    assert summary["n_sessions"].eq(2).all()
    assert summary["n_sessions_dropped"].eq(1).all()  # the 5-row activity
    assert summary["n_signins"].eq(136).all()
    assert set(summary["metric"]) >= {"n", "ramp_10_90_min", "sync_1min", "share_within_15min"}
    text = (out_dir / "signin_summary.md").read_text(encoding="utf-8")
    assert "| ramp_10_90_min |" in text
    assert "1 session(s) dropped for n < 10" in text


def test_compare_curve_uses_only_the_in_class_rows(tmp_path: Path) -> None:
    compare = tmp_path / "compare.csv"
    pd.DataFrame({"ramp_10_90_min": [1.0, 2.0, 3.0, 4.0],
                  "in_class_proxy": [True, False, True, False]}).to_csv(compare, index=False)
    frame = sig.load_compare(compare)
    assert list(frame["ramp_10_90_min"]) == [1.0, 3.0]
    assert sig.load_compare(tmp_path / "compare.csv") is not None


def test_no_name_or_id_reaches_any_output(out_dir: Path, exports: Path) -> None:
    table = _table(exports).reset_index()
    frame_text = " ".join(map(str, table.columns)) + table.to_csv(index=False)
    files = sorted(p for p in out_dir.iterdir() if p.is_file())
    assert len(files) == 5
    blobs = [p.read_bytes() for p in files]
    for token in ["姓名", "学号", "签到状态", "测试学生", "2025000"]:
        assert token not in frame_text
        for path, blob in zip(files, blobs):
            assert token.encode("utf-8") not in blob, f"{token} in {path.name}"
    for name in NAMES:
        assert name not in frame_text
        for path, blob in zip(files, blobs):
            assert name.encode("utf-8") not in blob, f"{name} in {path.name}"
