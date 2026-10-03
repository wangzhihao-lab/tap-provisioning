"""Trigger timing: the realised burst may start before the bell (batch A: mean -8.7 min)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pilot.sim import RunConfig, WorkloadConfig, engine
from pilot.sim.sweeps import SWEEPS, SweepConfig

REPO = Path(__file__).resolve().parents[1]


def _cfg(**workload) -> RunConfig:
    base = dict(college="college_18", weeks=(1,), adoption=0.2, sigma=0.7, p_cancel=0.0)
    base.update(workload)
    return RunConfig(workload=WorkloadConfig(**base), seed=7, root=str(REPO))


@pytest.fixture(scope="module", autouse=True)
def slice_present() -> None:
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def test_constant_offset_shifts_arrivals_and_draws_nothing() -> None:
    """A deterministic offset moves every in-class arrival by exactly that much and leaves the
    random stream untouched, so the realised counts are the reference counts shifted."""
    ref = engine.build_stream(_cfg())
    early = engine.build_stream(_cfg(trigger_offset_mean_min=-9.0))
    assert np.array_equal(early.lam_in[:-9], ref.lam_in[9:])
    assert np.array_equal(early.lam_hat_in, ref.lam_hat_in)  # the planner still expects the bell
    assert np.allclose(early.sessions["trigger_offset_min"], -9.0)


def test_forecast_offset_moves_the_planner_not_the_arrivals() -> None:
    ref = engine.build_stream(_cfg())
    learned = engine.build_stream(_cfg(forecast_offset_min=-9.0))
    assert np.array_equal(learned.lam_in, ref.lam_in)
    assert np.array_equal(learned.lam_hat_in[:-9], ref.lam_hat_in[9:])
    cfg = _cfg(forecast_offset_min=-9.0)
    grid_ref = engine.trigger_grid(_cfg(), ref)
    grid_new = engine.trigger_grid(cfg, learned)
    assert np.array_equal(grid_new, grid_ref - 9)


def test_random_offsets_follow_the_knobs() -> None:
    stream = engine.build_stream(_cfg(trigger_offset_mean_min=-8.7, trigger_offset_sd_min=5.3))
    off = stream.sessions["trigger_offset_min"].to_numpy()
    assert off.min() >= -25.0 and off.max() <= 10.0
    assert abs(off.mean() + 8.7) < 2.0 and 3.0 < off.std() < 8.0


def test_offset_sweep_cells() -> None:
    cells = list(SWEEPS["offset"](_cfg(), SweepConfig(), quick=False))
    assert len(cells) == 12  # A-F of the main evaluation, G-L of the sensitivity sweeps
    assert [c[0]["forecast_offset"] for c in cells[:6]] == [0.0, 0.0, -8.7, 0.0, -8.7, -8.7]
    assert [c[0]["trigger_offset_common_sd"] for c in cells[:6]] == [0.0, 0.0, 0.0, 5.3, 5.3, 3.7]
    # G-J keep the total spread of cell C (6.64 min) while the common share rises
    for c in cells[6:10]:
        total = c[0]["trigger_offset_sd"] ** 2 + c[0]["trigger_offset_course_sd"] ** 2 + c[0]["trigger_offset_common_sd"] ** 2
        assert abs(total - 44.09) < 0.2
    assert all(c[1].workload.trigger_offset_common_sd_min == c[0]["trigger_offset_common_sd"]
               for c in cells)
    assert replace(cells[1][1].workload, trigger_offset_mean_min=0.0, trigger_offset_sd_min=0.0,
                   trigger_offset_course_sd_min=0.0) == cells[0][1].workload


def test_common_shock_keeps_sections_of_one_bell_together() -> None:
    stream = engine.build_stream(_cfg(trigger_offset_mean_min=-8.7, trigger_offset_common_sd_min=5.3))
    s = stream.sessions
    per_bell = s.groupby("bell_min")["trigger_offset_min"].nunique()
    assert (per_bell == 1).all()  # every section on a bell shares that bell's draw
    assert s["trigger_offset_min"].nunique() > 5  # but the bells differ from each other


def test_widened_forecast_keeps_its_mass_and_spreads_it() -> None:
    """A distribution-aware planner spreads each trigger's forecast without changing its total."""
    point = engine.build_stream(_cfg(forecast_offset_min=-9.0))
    wide = engine.build_stream(_cfg(forecast_offset_min=-9.0, forecast_offset_sd_min=5.3))
    assert np.array_equal(wide.lam_in, point.lam_in)
    assert wide.lam_hat_in.sum() == pytest.approx(point.lam_hat_in.sum(), rel=0.02)
    assert wide.lam_hat_in.max() < 0.6 * point.lam_hat_in.max()
    # the spread is centred: the forecast's centre of mass around the biggest trigger is unchanged
    t = np.arange(point.lam_hat_in.size)
    lo, hi = max(0, int(np.argmax(point.lam_hat_in)) - 60), int(np.argmax(point.lam_hat_in)) + 60
    c_point = (t[lo:hi] * point.lam_hat_in[lo:hi]).sum() / point.lam_hat_in[lo:hi].sum()
    c_wide = (t[lo:hi] * wide.lam_hat_in[lo:hi]).sum() / wide.lam_hat_in[lo:hi].sum()
    assert abs(c_point - c_wide) < 2.0
