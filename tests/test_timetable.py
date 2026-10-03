"""Tests for pilot.timetable against the anonymized 31-row fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from pilot.timetable import parse, stats

FIXTURE = Path(__file__).parent / "fixtures" / "timetable_sample.txt"
PERSONAL = ("授课教师", "教工号", "teacher", "staff_id")


@pytest.fixture(scope="module")
def sections() -> pd.DataFrame:
    return parse.load_timetable(FIXTURE)


@pytest.fixture(scope="module")
def slots(sections: pd.DataFrame) -> pd.DataFrame:
    return parse.explode_slots(sections)


@pytest.fixture(scope="module")
def week_rows(slots: pd.DataFrame) -> pd.DataFrame:
    return parse.expand_weeks(slots)


def _one(frame: pd.DataFrame, name: str) -> pd.Series:
    hit = frame[frame["course_name"].str.startswith(name)]
    assert len(hit) == 1, f"{name}: expected one row, got {len(hit)}"
    return hit.iloc[0]


# --- time codes -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("3010203", (3, (1, 2, 3))),
        ("10708", (1, (7, 8))),
        ("1091011", (1, (9, 10, 11))),
        ("405060708", (4, (5, 6, 7, 8))),
        ("10102030405060708", (1, (1, 2, 3, 4, 5, 6, 7, 8))),
        ("109101112", (1, (9, 10, 11, 12))),
        ("70102030405060708", (7, (1, 2, 3, 4, 5, 6, 7, 8))),
    ],
)
def test_parse_time_code(code: str, expected: Tuple[int, Tuple[int, ...]]) -> None:
    assert parse.parse_time_code(code) == expected


@pytest.mark.parametrize("code", ["", "abc", "30102x", "301020", "80102", "10113"])
def test_parse_time_code_rejects_junk(code: str) -> None:
    assert parse.parse_time_code(code) is None


def test_week_parts_and_classes() -> None:
    assert parse.parse_week_part("4-18")[0] == 4 and parse.parse_week_part("4-18")[-1] == 18
    assert parse.parse_week_part("10") == (10,)
    assert parse.count_classes("2024专业01[1-2]班,2024专业02[1-4]班") == 6
    assert parse.count_classes("临班24") == 1
    assert parse.is_general_elective("临班373") and not parse.is_general_elective("2024专业011班")
    assert parse.format_weeks((1, 2, 3, 5, 7, 8)) == "1-3,5,7-8"


def test_align_weeks_rules() -> None:
    one_to_one, ambiguous = parse.align_weeks("1-18,1-18", 2)
    assert not ambiguous and len(one_to_one) == 2
    grouped, ambiguous = parse.align_weeks("1-2,4-18,4", 2)
    assert not ambiguous and grouped[1] == (4,)
    union, ambiguous = parse.align_weeks("1,2", 3)
    assert ambiguous and union[0] == (1, 2) and len(union) == 3


# --- sections -------------------------------------------------------------------------------
def test_section_count(sections: pd.DataFrame) -> None:
    assert len(sections) == 31
    assert list(sections.columns) == list(parse.SECTION_COLUMNS)


def test_blank_campus_row(sections: pd.DataFrame) -> None:
    blank = sections[sections["campus"] == ""]
    assert len(blank) == 1
    assert blank.iloc[0]["course_name"].startswith("课程14")
    assert blank.iloc[0]["college"] == "学院07"
    assert blank.iloc[0]["enrollment"] == 20


def test_blank_room_slot(slots: pd.DataFrame) -> None:
    blank = slots[slots["room"] == ""]
    assert len(blank) == 1
    assert blank.iloc[0]["course_name"] == "课程13"
    assert (blank.iloc[0]["weekday"], blank.iloc[0]["period_start"]) == (2, 5)


def test_government_budget_row_resolves_by_grouping(sections: pd.DataFrame,
                                                    slots: pd.DataFrame) -> None:
    section = _one(sections, "课程06")
    assert not section["ambiguous"]
    mine = slots[slots["section_id"] == section["section_id"]]
    assert len(mine) == 4
    assert [parse.format_weeks(w) for w in mine["weeks"]] == ["1-5,7-18", "10", "5", "1-18"]
    assert list(zip(mine["weekday"], mine["period_start"])) == [(2, 1), (4, 7), (4, 7), (2, 3)]


def test_ambiguous_row_takes_the_week_union(sections: pd.DataFrame, slots: pd.DataFrame) -> None:
    section = _one(sections, "课程27")
    assert section["ambiguous"]
    mine = slots[slots["section_id"] == section["section_id"]]
    assert len(mine) == 7
    assert mine["weeks"].nunique() == 1
    assert parse.format_weeks(mine.iloc[0]["weeks"]) == "1-11"


def test_long_and_late_slots(slots: pd.DataFrame) -> None:
    monday_eight = slots[(slots["course_name"] == "课程11")].iloc[0]
    assert (monday_eight["weekday"], monday_eight["period_start"],
            monday_eight["period_end"], monday_eight["n_periods"]) == (1, 1, 8, 8)
    late = slots[slots["course_name"] == "课程12"].iloc[0]
    assert (late["weekday"], late["period_start"], late["period_end"]) == (1, 9, 12)
    sunday = slots[slots["weekday"] == 7]
    assert len(sunday) == 1 and sunday.iloc[0]["n_periods"] == 8


def test_large_section_and_odd_fields(sections: pd.DataFrame) -> None:
    big = _one(sections, "课程23")
    assert big["enrollment"] == 615 and big["n_classes"] == 19
    assert _one(sections, "课程18")["credits"] == pytest.approx(2.5)
    assert _one(sections, "课程17")["enrollment"] == 100  # alphanumeric staff id
    assert _one(sections, "课程16")["enrollment"] == 41              # name with a space
    assert _one(sections, "课程15")["enrollment"] == 0            # placeholder section
    assert int(sections["general_elective"].sum()) == 2


def test_no_personal_columns(sections: pd.DataFrame, slots: pd.DataFrame,
                             week_rows: pd.DataFrame) -> None:
    for frame in (sections, slots, week_rows):
        assert not [c for c in frame.columns if c in PERSONAL]
        joined = " ".join(str(v) for v in frame.astype(str).to_numpy().ravel())
        assert "教师" not in joined and "20000001" not in joined


def test_frame_sizes_are_consistent(sections: pd.DataFrame, slots: pd.DataFrame,
                                    week_rows: pd.DataFrame) -> None:
    assert len(slots) == sum(len(str(t).split(",")) for t in sections["time_raw"])
    assert len(week_rows) == int(slots["n_weeks"].sum())
    assert set(slots["section_id"]) <= set(sections["section_id"])
    assert slots.groupby("section_id")["slot_index"].max().max() == 6


def test_tab_separated_layout(tmp_path: Path) -> None:
    header = "\t".join(list(parse.HEAD_COLUMNS) + [parse.ROOM_COLUMN] + list(parse.TAIL_COLUMNS))
    row = "\t".join([" ", " 学院07 ", " 临班9 ", " 90000001 ", " 课程90 ", " 1-4 ", " 40102 ", "  ",
                     " X ", " 20000001 ", " 通选 ", " 1 ", "40 "])
    path = tmp_path / "tabs.txt"
    path.write_text("　　\n" + header + "\n" + row + "\n", encoding="utf-8")
    frame = parse.load_timetable(path)
    assert len(frame) == 1
    assert frame.iloc[0]["campus"] == "" and frame.iloc[0]["room_raw"] == ""
    assert frame.iloc[0]["enrollment"] == 40 and frame.iloc[0]["general_elective"]


# --- statistics and figures -----------------------------------------------------------------
def test_stats_end_to_end(tmp_path: Path, sections: pd.DataFrame, slots: pd.DataFrame,
                          week_rows: pd.DataFrame) -> None:
    result = stats.compute(sections, slots, week_rows)
    load = result.slot_load
    total = load[load["campus"] == stats.ALL_CAMPUS]["seats"].sum()
    assert total == load[load["campus"] != stats.ALL_CAMPUS]["seats"].sum()
    assert result.counts["n_sections"] == 31
    assert result.counts["n_weekend_slots"] == 5
    assert result.enrollment["max"] == 615
    assert 1.0 < result.concentration["peak_to_mean"] < 50.0
    assert 0.0 < result.concentration["share_top5"] <= result.concentration["share_top10"] <= 1.0
    assert set(result.same_bell["bell"]) <= set(parse.BellSchedule().bells())
    assert not bool(result.same_bell["off_block"].any())
    assert (result.same_bell["seats_max"] >= result.same_bell["seats_typ"]).all()
    csv_path = stats.write_t2(result, tmp_path)
    assert csv_path.exists() and (tmp_path / "T2_summary.md").exists()
    table = pd.read_csv(csv_path)
    assert list(table.columns) == ["statistic", "value", "unit", "note"]
    assert (table["statistic"] == "Course sections").any()
    assert table.loc[table["statistic"] == "Course sections", "value"].iloc[0] == 31


def test_figures(tmp_path: Path, sections: pd.DataFrame, slots: pd.DataFrame,
                 week_rows: pd.DataFrame) -> None:
    plots = pytest.importorskip("pilot.timetable.plots")
    result = stats.compute(sections, slots, week_rows)
    paths = plots.make_all(sections, result.typical_week, result.session_starts, tmp_path)
    assert len(paths) == 9 and all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert {p.suffix for p in paths} == {".pdf", ".png", ".svg"}


def test_missing_input_raises() -> None:
    with pytest.raises(FileNotFoundError):
        parse.load_timetable(FIXTURE.parent / "does_not_exist.txt")
