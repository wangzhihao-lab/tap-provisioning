"""Workload generation: sessions, the sigma mixture, the calendar forecast and the token pools."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.sim import WorkloadConfig
from pilot.sim.stale import stale_sessions
from pilot.sim.workload import (
    IN_CLASS,
    OFF_CLASS,
    calendar_forecast,
    forecast_error,
    generate,
    load_offset_pool,
    load_sessions,
    load_token_pool,
    off_class_intensity,
)

REPO = Path(__file__).resolve().parents[1]
SLOTS = REPO / "data" / "timetable" / "slots_2026-2027-1.csv"
COLLEGE = "college_04"


def small_workload(**kwargs) -> WorkloadConfig:
    """One small college, one teaching week; the caller overrides what it needs."""
    base = dict(college=COLLEGE, weeks=(1,), adoption=0.4, sigma=0.7)
    base.update(kwargs)
    return WorkloadConfig(**base)


@pytest.fixture(scope="module")
def guard() -> None:
    """Skip the module when the timetable slice or the public traces are not on disk."""
    cfg = WorkloadConfig()
    for rel in (cfg.slots_csv, cfg.sync_offsets_csv, cfg.self_offsets_csv,
                cfg.azure_trace_csv):
        if not (REPO / rel).exists():
            pytest.skip(f"missing input {rel}; run scripts/t2/run_t2.py first")


def test_offset_pools_are_clipped_not_truncated(guard) -> None:
    """The loader caps at the session length by clipping, so no late offset is dropped.

    The old loader threw away every offset above 2 min (sign-ins) or 15 min (ASSISTments), which
    renormalised the pools and pushed their 2-minute share to 1.00 and 0.42.  Keeping every row and
    clipping it at the cap leaves that share at the pooled raw value the pilot measured, which is
    what makes the generated sigma_2 reproduce the data.
    """
    cfg = WorkloadConfig()
    assert cfg.offset_source == "signin"  # the measured pair stays the default
    assert (cfg.sync_offsets_csv, cfg.self_offsets_csv) == (cfg.signin_offsets_csv,
                                                            cfg.assistments_offsets_csv)
    raw = {name: pd.read_csv(REPO / path)["offset_min"].to_numpy(dtype=float)
           for name, path in (("sync", cfg.sync_offsets_csv),
                              ("self", cfg.self_offsets_csv))}
    sync = load_offset_pool(REPO / cfg.sync_offsets_csv, cfg.pool_cap)
    self_paced = load_offset_pool(REPO / cfg.self_offsets_csv, cfg.pool_cap)
    assert cfg.pool_cap == cfg.period_minutes == 45.0
    for pool, name in ((sync, "sync"), (self_paced, "self")):
        col = raw[name]
        kept = col[(col >= 0.0) & np.isfinite(col)]
        assert pool.size == kept.size  # nothing is dropped any more
        assert 0.0 <= pool.min() and pool.max() <= cfg.pool_cap
        assert (pool > 15.0).any()  # offsets the old 15 min cap removed are still there
    assert self_paced.mean() > sync.mean()  # self-paced arrivals are later by construction
    # the 2-minute share, which is what sigma_2 measures, survives the cap
    assert float((sync <= 2.0).mean()) == pytest.approx(0.704, abs=0.01)
    assert float((self_paced <= 2.0).mean()) == pytest.approx(0.244, abs=0.01)


def test_token_pool_matches_the_trace(guard) -> None:
    """Prompt and output lengths come from the Azure conversation trace."""
    ctx, gen = load_token_pool(REPO / WorkloadConfig().azure_trace_csv)
    assert ctx.size == gen.size == 19366
    assert gen.min() >= 1 and ctx.min() >= 0


def test_sessions_land_on_bells(guard) -> None:
    """Every session starts on a block bell and inside its teaching week."""
    rng = np.random.default_rng(0)
    cfg = small_workload(weeks=(1, 2))
    sessions = load_sessions(cfg, rng, str(REPO))
    assert not sessions.empty
    assert (sessions["bell_min"] >= 0).all()
    assert (sessions["bell_min"] < cfg.horizon_min).all()
    minute_of_day = sessions["bell_min"] % 1440
    assert set(minute_of_day.unique()) <= {510, 620, 840, 950, 1120, 1220}
    assert (sessions["roster"] > 0).all()
    assert (sessions["duration_min"] >= cfg.period_minutes).all()


def test_sigma_sharpens_the_burst(guard) -> None:
    """A larger mixing weight w concentrates the same requests into the minutes after the bell.

    The total in-class demand does not depend on w - only when it arrives does - so the statistic
    is the mass in the first two minutes after each bell and the mean of the twenty busiest
    minutes, both of which are far less noisy than the single peak minute.
    """
    burst, busiest, totals = [], [], []
    for w in (0.0, 1.0):
        stream = generate(small_workload(sigma=w), np.random.default_rng(1), str(REPO))
        bells = stream.sessions["bell_min"].to_numpy()
        burst.append(sum(float(stream.lam_in[b:b + 2].sum()) for b in bells))
        busiest.append(float(np.sort(stream.lam_in)[-20:].mean()))
        totals.append(float(stream.lam_in.sum()))
    assert totals[1] == pytest.approx(totals[0], rel=0.05)  # same demand, different timing
    assert burst[1] > 2.5 * burst[0]
    assert busiest[1] > 1.3 * busiest[0]


def test_forecast_tracks_the_realised_demand(guard) -> None:
    """lambda-hat has the same mass and the same bells as lambda, without the sampling noise."""
    cfg = small_workload(roster_noise=0.0, attendance_lo=1.0, attendance_hi=1.0, p_cancel=0.0,
                         forecast_attendance=1.0)
    stream = generate(cfg, np.random.default_rng(2), str(REPO))
    assert stream.lam_hat_in.sum() == pytest.approx(stream.lam_in.sum(), rel=0.15)
    assert stream.lam_hat_in.max() < stream.lam_in.max()  # the analytic profile is smoother
    busy = stream.lam_in > 0
    assert stream.lam_hat_in[busy].mean() > stream.lam_hat_in[~busy].mean()


def test_error_sources_open_a_gap(guard) -> None:
    """Roster noise, attendance and cancellation push lambda away from lambda-hat."""
    clean = generate(small_workload(roster_noise=0.0, attendance_lo=1.0, attendance_hi=1.0,
                                    p_cancel=0.0), np.random.default_rng(3), str(REPO))
    noisy = generate(small_workload(roster_noise=0.2, attendance_lo=0.7, attendance_hi=0.8,
                                    p_cancel=0.2), np.random.default_rng(3), str(REPO))

    def l1(s):
        return float(np.abs(s.lam_in - s.lam_hat_in).sum() / max(1.0, s.lam_hat_in.sum()))

    assert l1(noisy) > l1(clean)


def test_stream_shape_and_kinds(guard) -> None:
    """The stream is sorted, in range, and carries both request kinds."""
    stream = generate(small_workload(), np.random.default_rng(4), str(REPO))
    assert stream.n_requests > 0
    assert np.all(np.diff(stream.t_arrive_s) >= 0)
    assert stream.t_arrive_s.max() < stream.horizon_min * 60
    assert set(np.unique(stream.kind)) <= {IN_CLASS, OFF_CLASS}
    assert (stream.session_id[stream.kind == OFF_CLASS] == -1).all()
    assert stream.lam_in.sum() == (stream.kind == IN_CLASS).sum()
    assert stream.lam_off.sum() == (stream.kind == OFF_CLASS).sum()
    assert stream.output_tokens.min() >= 1


def test_off_class_intensity_has_a_deadline_ramp(guard) -> None:
    """The 48 h before a deadline carry more background than the two days before that."""
    cfg = small_workload(weeks=(1, 2), deadline_weekday=4, deadline_extra_per_student=2.0)
    rng = np.random.default_rng(5)
    sessions = load_sessions(cfg, rng, str(REPO))
    lam = off_class_intensity(cfg, sessions)
    due = 4 * 1440 + cfg.deadline_hour * 60
    assert lam[due - 2880:due].sum() > 1.2 * lam[due - 5760:due - 2880].sum()


def test_calendar_forecast_is_zero_without_sessions(guard) -> None:
    """No adopting section means no in-class forecast."""
    cfg = small_workload(adoption=0.0)
    rng = np.random.default_rng(6)
    sessions = load_sessions(cfg, rng, str(REPO))
    pool = np.array([0.0, 1.0])
    assert sessions.empty
    assert calendar_forecast(cfg, sessions, rng, pool, pool).sum() == 0.0


def test_forecast_scale_biases_the_plan_and_shows_up_in_eta(guard) -> None:
    """``forecast_scale`` moves lambda-hat proportionally; the measured error follows it."""
    etas = {}
    for s in (0.5, 1.0, 2.0):
        cfg = small_workload(forecast_scale=s, p_cancel=0.0, bell_shift_frac=0.0)
        stream = generate(cfg, np.random.default_rng(7), str(REPO))
        etas[s] = forecast_error(stream.lam_in, stream.lam_off, stream.lam_hat_in,
                                 stream.lam_hat_off)[1]
        if s != 1.0:
            base = generate(small_workload(p_cancel=0.0), np.random.default_rng(7), str(REPO))
            assert stream.lam_hat_in.sum() == pytest.approx(s * base.lam_hat_in.sum(), rel=1e-9)
    assert etas[0.5] > etas[1.0] and etas[2.0] > etas[1.0]  # both directions are an error


def test_bell_shift_moves_the_forecast_without_changing_its_mass(guard) -> None:
    """``bell_shift_frac`` is a pure timing error: same total, different minutes, larger eta."""
    exact = generate(small_workload(p_cancel=0.0), np.random.default_rng(8), str(REPO))
    shifted = generate(small_workload(p_cancel=0.0, bell_shift_frac=1.0),
                       np.random.default_rng(8), str(REPO))
    assert shifted.lam_hat_in.sum() == pytest.approx(exact.lam_hat_in.sum(), rel=0.05)
    assert not np.allclose(shifted.lam_hat_in, exact.lam_hat_in)
    e0 = forecast_error(exact.lam_in, exact.lam_off, exact.lam_hat_in, exact.lam_hat_off)[1]
    e1 = forecast_error(shifted.lam_in, shifted.lam_off, shifted.lam_hat_in,
                        shifted.lam_hat_off)[1]
    assert e1 > e0


def test_the_knobs_are_off_by_default(guard) -> None:
    """Every perturbation knob is exact at its default, so a default run is unperturbed."""
    cfg = WorkloadConfig()
    assert (cfg.forecast_scale, cfg.bell_shift_frac) == (1.0, 0.0)
    a = generate(small_workload(), np.random.default_rng(9), str(REPO))
    b = generate(small_workload(forecast_scale=1.0, bell_shift_frac=0.0),
                 np.random.default_rng(9), str(REPO))
    assert np.array_equal(a.lam_hat_in, b.lam_hat_in)
    assert np.array_equal(a.lam_in, b.lam_in)  # no random number is consumed either


def test_eta_norm_is_a_ratio_of_the_realised_demand(guard) -> None:
    """``eta_norm`` is ``eta`` per realised request, and 0 only for a perfect forecast."""
    stream = generate(small_workload(), np.random.default_rng(10), str(REPO))
    eta, eta_norm = forecast_error(stream.lam_in, stream.lam_off, stream.lam_hat_in,
                                   stream.lam_hat_off)
    assert eta > 0.0
    assert eta_norm == pytest.approx(eta / (stream.lam_in.sum() + stream.lam_off.sum()))
    zero, zero_norm = forecast_error(stream.lam_in, stream.lam_off, stream.lam_in, stream.lam_off)
    assert zero == 0.0 and zero_norm == 0.0


def test_the_stale_calendar_covers_the_same_grid(guard) -> None:
    """Policy 9's forecast is built from last semester and lands on the current bells."""
    cfg = small_workload(weeks=(1, 2))
    if not (REPO / cfg.stale_slots_csv).exists():
        pytest.skip("last semester's slot table is missing; run scripts/t2/run_t2.py for it")
    sessions = stale_sessions(cfg, str(REPO))
    assert not sessions.empty
    assert set(sessions["bell_min"] % 1440) <= {510, 620, 840, 950, 1120, 1220}
    stream = generate(cfg, np.random.default_rng(11), str(REPO))
    assert stream.lam_stale_in is not None and stream.lam_stale_in.sum() > 0.0
    # a different calendar, but the same order of magnitude as the current one
    assert not np.allclose(stream.lam_stale_in, stream.lam_hat_in)
    assert 0.1 < stream.lam_stale_in.sum() / stream.lam_hat_in.sum() < 10.0
