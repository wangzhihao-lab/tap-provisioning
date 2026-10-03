"""Measured synchrony sigma_2, and the difference between it and the mixing weight w.

``WorkloadConfig.sigma`` is *w*, the share of students drawn from the prompted (sign-in) offset
pool.  What the paper plots is sigma_2, the share of per-student first arrivals within 2 min of
their session's first arrival, which ``metrics.measure_synchrony`` reads off the generated stream.
The two are not the same number: the pools overlap in the first two minutes, so w = 0 already
carries the self-paced source's own 0.24 and w = 1 reproduces the sign-in source's 0.70.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.sim import WorkloadConfig
from pilot.sim.metrics import measure_synchrony
from pilot.sim.workload import IN_CLASS, OFF_CLASS, ArrivalStream, generate

REPO = Path(__file__).resolve().parents[1]
COLLEGE = "college_18"
# the two pilot anchors: prompted sign-ins and self-paced ASSISTments in-class offsets
SIGNIN_SIGMA2 = 0.70
ASSISTMENTS_SIGMA2 = 0.24
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
_CACHE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def guard() -> None:
    """Skip when the timetable slice or the public offset traces are not on disk."""
    cfg = WorkloadConfig()
    for rel in (cfg.slots_csv, cfg.sync_offsets_csv, cfg.self_offsets_csv,
                cfg.azure_trace_csv):
        if not (REPO / rel).exists():
            pytest.skip(f"missing input {rel}; run scripts/t2/run_t2.py first")


def measured(w: float) -> dict:
    """``measure_synchrony`` of the quick slice at mixing weight ``w`` (cached per value)."""
    if w not in _CACHE:
        cfg = WorkloadConfig(college=COLLEGE, weeks=(1, 2), adoption=0.10, sigma=w)
        _CACHE[w] = measure_synchrony(generate(cfg, np.random.default_rng(0), str(REPO)))
    return _CACHE[w]


def test_measured_sigma2_reproduces_the_two_data_anchors() -> None:
    """All-prompted lands on the sign-in anchor, all-self-paced on the ASSISTments one."""
    prompted, self_paced = measured(1.0), measured(0.0)
    assert prompted["n_sessions"] > 10 and prompted["n_students"] > 100
    assert prompted["sigma2_pooled"] == pytest.approx(SIGNIN_SIGMA2, abs=0.05)
    assert self_paced["sigma2_pooled"] == pytest.approx(ASSISTMENTS_SIGMA2, abs=0.05)
    assert prompted["sigma2_median"] == pytest.approx(SIGNIN_SIGMA2, abs=0.05)
    assert self_paced["sigma2_median"] == pytest.approx(ASSISTMENTS_SIGMA2, abs=0.05)


def test_the_knob_is_not_the_measured_synchrony() -> None:
    """w and sigma_2 are different numbers: the pools overlap, so w = 0 is not "no synchrony"."""
    assert measured(0.0)["sigma2_median"] > 0.15  # the self-paced source is already this bursty
    assert measured(1.0)["sigma2_median"] < 1.0  # and the prompted one is not fully synchronous


def test_measured_sigma2_is_monotone_in_the_mixing_weight() -> None:
    """sigma_2 is non-decreasing in w, so the knob still orders the F5 x axis."""
    medians = [measured(w)["sigma2_median"] for w in W_GRID]
    pooled = [measured(w)["sigma2_pooled"] for w in W_GRID]
    assert all(a <= b + 1e-9 for a, b in zip(medians, medians[1:])), medians
    assert all(a <= b + 1e-9 for a, b in zip(pooled, pooled[1:])), pooled
    assert medians[-1] - medians[0] > 0.3  # and it spans the range the anchors sit in


def test_the_three_windows_are_nested() -> None:
    """sigma_1 <= sigma_2 <= sigma_5 in every session, and sigma_5 is not yet saturated."""
    for w in (0.0, 1.0):
        m = measured(w)
        assert np.all(m["sigma1"] <= m["sigma2"] + 1e-9)
        assert np.all(m["sigma2"] <= m["sigma5"] + 1e-9)
        assert m["sigma1_pooled"] <= m["sigma2_pooled"] <= m["sigma5_pooled"]
        assert m["sigma5_pooled"] < 1.0


def _hand_stream(rows: list) -> ArrivalStream:
    """``(session, student, minute)`` triples as an in-class stream, plus one off-class request."""
    sid = np.array([r[0] for r in rows] + [-1], dtype=np.int64)
    stu = np.array([r[1] for r in rows] + [-1], dtype=np.int64)
    t = np.array([r[2] * 60.0 for r in rows] + [0.0], dtype=float)
    kind = np.array([IN_CLASS] * len(rows) + [OFF_CLASS], dtype=np.int8)
    zeros = np.zeros(60, dtype=float)
    sessions = pd.DataFrame({"session_id": sorted({r[0] for r in rows}), "bell_min": 0})
    return ArrivalStream(t, kind, sid, np.ones(t.size, dtype=np.int64),
                         np.ones(t.size, dtype=np.int64), zeros, zeros, zeros, zeros,
                         sessions, 60, None, stu)


def test_sessions_below_the_student_floor_are_dropped() -> None:
    """A session needs five students before its share is counted, and only its own requests."""
    big = [(0, s, 0.0 if s < 4 else 30.0) for s in range(10)]  # 4 of 10 inside the window
    small = [(1, s, 0.0) for s in range(3)]  # perfectly synchronous, but too small to count
    m = measure_synchrony(_hand_stream(big + small))
    assert m["n_sessions"] == 1 and list(m["session_id"]) == [0]
    assert m["n_students"] == 10
    assert m["sigma2_pooled"] == pytest.approx(0.4)
    assert m["sigma2_median"] == pytest.approx(0.4)
    assert measure_synchrony(_hand_stream(small))["n_sessions"] == 0


def test_a_students_first_request_is_what_counts() -> None:
    """Follow-up requests never move a student in or out of the window."""
    late_followups = [(0, s, m) for s in range(6) for m in (1.0, 20.0, 40.0)]
    m = measure_synchrony(_hand_stream(late_followups))
    assert m["sigma2_pooled"] == pytest.approx(1.0)  # every first request is at the same minute
    mixed = [(0, s, float(s)) for s in range(6)]  # first arrivals 0..5 min apart
    shares = measure_synchrony(_hand_stream(mixed))
    assert shares["sigma1_pooled"] == pytest.approx(2 / 6)
    assert shares["sigma2_pooled"] == pytest.approx(3 / 6)
    assert shares["sigma5_pooled"] == pytest.approx(6 / 6)
