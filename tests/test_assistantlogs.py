"""Assistant-log parsing, burst splitting and the AI offset-pool selector."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.assistantlogs import (
    PROMPTED,
    BurstConfig,
    DatasetSpec,
    bell_windows,
    burst_offsets,
    deadline_ramp,
    first_arrivals,
    gap_stats,
    length_stats,
    parse_detail,
    peak_to_mean,
    ramp_10_90,
    sigma_shares,
    split_bursts,
)
from pilot.sim import WorkloadConfig

REPO = Path(__file__).resolve().parents[1]
DETAIL = ("S001 2026-03-10 08:30:05 how do I fix this\n"
          "AI助教 2026-03-10 08:30:29 first check the index bound\n"
          "and then rerun it\n"
          "S001 2026-03-10 08:34:05 thanks")


def test_parse_detail_keeps_only_time_role_and_length() -> None:
    """Roles, timestamps and body lengths come out; a wrapped line folds into its message."""
    rows = parse_detail(DETAIL, user="S001")
    assert [r.role for r in rows] == ["student", "assistant", "student"]
    assert [r.turn for r in rows] == [0, 1, 2]
    assert rows[0].n_chars == len("how do I fix this")
    # the continuation line adds its own length plus the newline, and never the text itself
    assert rows[1].n_chars == len("first check the index bound") + len("and then rerun it") + 1
    assert rows[2].ts == pd.Timestamp("2026-03-10 08:34:05")
    assert not hasattr(rows[0], "text")
    assert parse_detail(float("nan")) == []


def test_parse_detail_without_user_falls_back_to_the_bot_name() -> None:
    """Without the account id the assistant is recognised by its display name."""
    rows = parse_detail(DETAIL)
    assert [r.role for r in rows] == ["student", "assistant", "student"]


def _synthetic_day() -> pd.DataFrame:
    """One class day: 20 students in a 2-min burst at the bell, 10 spread over an hour."""
    bell = 8 * 60 + 30
    burst = bell - 3 + np.linspace(0.0, 2.0, 20)
    spread = bell + 60 + np.linspace(0.0, 55.0, 10)
    tod = np.concatenate([burst, spread])
    return pd.DataFrame({"date": ["2026-03-10"] * tod.size, "weekday": 1,
                         "user": [f"u{i}" for i in range(tod.size)], "tod_min": tod,
                         "ts": pd.Timestamp("2026-03-10") + pd.to_timedelta(tod, unit="m")})


def test_sigma_and_ramp_on_a_known_burst() -> None:
    """sigma_k counts the share inside k minutes; the ramp is the 10-90% spread."""
    rel = np.array([0.0, 0.5, 1.5, 3.0, 40.0])
    shares = sigma_shares(rel)
    assert shares["sigma_1"] == pytest.approx(0.4)
    assert shares["sigma_2"] == pytest.approx(0.6)
    assert shares["sigma_5"] == pytest.approx(0.8)
    assert ramp_10_90(np.arange(0.0, 11.0)) == pytest.approx(8.0)
    assert np.isnan(ramp_10_90(np.array([1.0])))
    assert peak_to_mean([0.0, 0.2, 5.0], (0.0, 10.0), 5.0) == pytest.approx(2.0 / 1.5)


def test_bell_windows_tile_the_morning_without_gaps() -> None:
    """Windows start ``pre`` before their bell and end where the next one starts."""
    cfg = BurstConfig(pre_min=20.0, span_min=100.0)
    windows = bell_windows((510.0, 560.0, 620.0), cfg)
    assert [w[1] for w in windows] == [490.0, 540.0, 600.0]
    assert [w[2] for w in windows] == [540.0, 600.0, 720.0]


def test_split_bursts_separates_prompted_from_mixed() -> None:
    """A 2-min cluster at the bell is prompted; an hour-long spread in one window is not."""
    spec = DatasetSpec(key="t", label="t", stem="t", regime=PROMPTED, roster=30,
                       bells=("08:30", "09:20", "10:20"), class_weekday=1)
    frame = _synthetic_day()
    bursts = split_bursts(frame, spec, BurstConfig())
    kinds = dict(zip(bursts["bell"], bursts["kind"]))
    assert kinds["08:30"] == "prompted"
    assert bursts.loc[bursts["bell"] == "08:30", "n_students"].iloc[0] == 20
    assert kinds["09:20"] == "mixed"
    pool = burst_offsets(frame, spec, BurstConfig())
    assert pool.size == 20 and pool.max() == pytest.approx(2.0)


def test_first_arrivals_and_session_shape_metrics() -> None:
    """One row per (day, student), and the two gap families are measured apart."""
    msgs = pd.DataFrame({
        "session": ["a"] * 4,
        "turn": [0, 1, 2, 3],
        "role": ["student", "assistant", "student", "assistant"],
        "n_chars": [10, 100, 12, 120],
        "ts": pd.to_datetime(["2026-03-10 08:30:00", "2026-03-10 08:30:20",
                              "2026-03-10 08:33:00", "2026-03-10 08:33:30"]),
        "user": ["u1"] * 4,
        "date": ["2026-03-10"] * 4,
        "weekday": [1] * 4,
        "tod_min": [510.0, 510.33, 513.0, 513.5],
    })
    arrivals = first_arrivals(msgs[msgs["role"] == "student"])
    assert len(arrivals) == 1 and arrivals["tod_min"].iloc[0] == 510.0
    gaps = gap_stats(msgs)
    assert gaps["student_turn_gap_s_median"] == pytest.approx(180.0)
    assert gaps["response_gap_s_median"] == pytest.approx(25.0)
    lengths = length_stats(msgs)
    assert lengths["student_chars_median"] == pytest.approx(11.0)
    assert lengths["assistant_chars_median"] == pytest.approx(110.0)


def test_deadline_ramp_bins_backwards_from_the_deadline() -> None:
    """``hours_before = 0`` is the last hour before the due instant."""
    sessions = pd.DataFrame({"created": pd.to_datetime(
        ["2026-03-23 23:30", "2026-03-23 22:30", "2026-03-21 10:00", "2026-03-24 00:30"])})
    ramp = deadline_ramp(sessions, [pd.Timestamp("2026-03-23 23:59")], hours=72)
    counts = ramp.set_index("hours_before")["sessions"]
    assert counts.loc[0] == 1 and counts.loc[1] == 1
    assert counts.sum() == 3  # the session after the deadline is not in the window


def test_offset_source_selects_the_ai_pools() -> None:
    """``offset_source`` switches both pools at once and the default stays on the measured pair."""
    base = WorkloadConfig()
    assert base.offset_source == "signin"
    assert base.sync_offsets_csv.endswith("signin_offsets_min.csv")
    ai = WorkloadConfig(offset_source="ai")
    assert ai.sync_offsets_csv.endswith("prompted_ai_offsets_min.csv")
    assert ai.self_offsets_csv.endswith("selfpaced_ai_offsets_min.csv")
    with pytest.raises(ValueError):
        WorkloadConfig(offset_source="nope")


def test_ai_offset_pools_on_disk_are_sharper_than_the_signin_pair() -> None:
    """The batch-A pools exist and carry the 2-minute shares the summary reports."""
    cfg = WorkloadConfig(offset_source="ai")
    for rel in (cfg.sync_offsets_csv, cfg.self_offsets_csv):
        if not (REPO / rel).exists():
            pytest.skip(f"missing {rel}; run scripts/q1/assistant_workload.py first")
    prompted = pd.read_csv(REPO / cfg.sync_offsets_csv)["offset_min"].to_numpy(dtype=float)
    self_paced = pd.read_csv(REPO / cfg.self_offsets_csv)["offset_min"].to_numpy(dtype=float)
    # genuine exports of 2026-09-17: prompted bursts 0.84, the self-paced lab 0.20 (summary section 6)
    assert float((prompted <= 2.0).mean()) == pytest.approx(0.84, abs=0.02)
    assert float((self_paced <= 2.0).mean()) == pytest.approx(0.20, abs=0.02)
    assert self_paced.max() > 45.0  # the self-paced pool is written uncapped


def test_published_tables_carry_no_per_student_row() -> None:
    """Nothing under results/q1/assistant identifies a student, not even by hash."""
    out = REPO / "results/q1/assistant"
    if not out.exists():
        pytest.skip("run scripts/q1/assistant_workload.py first")
    for path in sorted(out.glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=1).columns)
        assert not ({"user", "session", "学工号", "会话ID"} & columns), path.name
