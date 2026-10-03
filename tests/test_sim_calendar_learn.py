"""Policy 12: the timetable with trigger timing learned from earlier weeks (held-out by construction)."""

from pathlib import Path

import numpy as np
import pytest

from pilot.sim import RunConfig, WorkloadConfig, engine
from pilot.sim.workload import calendar_forecast, learned_timing, load_offset_pool, timing_prior

REPO = Path(__file__).resolve().parents[1]
WEEK = 7 * 1440


def _cfg(**workload) -> RunConfig:
    base = dict(college="college_18", weeks=(1, 2, 3), adoption=0.2, sigma=0.7, p_cancel=0.0)
    base.update(workload)
    return RunConfig(workload=WorkloadConfig(**base), seed=5, root=str(REPO))


@pytest.fixture(scope="module", autouse=True)
def slice_present() -> None:
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def test_week_one_is_the_bell_and_later_weeks_learn_a_constant_offset() -> None:
    stream = engine.build_stream(_cfg(trigger_offset_mean_min=-9.0))
    s = stream.sessions
    offs, sds = learned_timing(s, s["realised_start_min"].to_numpy())
    w = s["week"].to_numpy()
    assert np.all(offs[w == 1] == 0.0) and np.all(sds[w == 1] == 0.0)
    # observed offset = true offset (-9) + jitter + the earliest student's delay: a few minutes above -9
    late = offs[w >= 2]
    assert -9.0 < late.mean() < -4.0
    # the learned forecast of weeks 2-3 leads the bell-only forecast; week 1 is identical.  Both
    # forecasts are rebuilt with the same generator so that the roster-noise draws coincide and
    # the centroid difference is the timing effect alone (on this small slice the independent
    # draws of the stream's own forecasts move a weekly centroid by tens of minutes).
    pools = (load_offset_pool(REPO / WorkloadConfig().signin_offsets_csv, WorkloadConfig().pool_cap),
             load_offset_pool(REPO / WorkloadConfig().self_offsets_csv, WorkloadConfig().pool_cap))
    wl = _cfg(trigger_offset_mean_min=-9.0).workload
    ref = calendar_forecast(wl, s, np.random.default_rng(11), *pools, np.zeros(len(s)), np.zeros(len(s)))
    learn = calendar_forecast(wl, s, np.random.default_rng(11), *pools, offs, sds)
    assert np.allclose(learn[:WEEK], ref[:WEEK])  # week 1: bell only, same draws -> identical
    t = np.arange(WEEK, 3 * WEEK)
    c_ref = (t * ref[WEEK:]).sum() / ref[WEEK:].sum()
    c_learn = (t * learn[WEEK:]).sum() / learn[WEEK:].sum()
    assert -9.0 < c_learn - c_ref < -3.0
    # the stream's own learned forecast is the same construction with its own fixed generator
    assert stream.lam_hat_learn_in is not None and abs(stream.lam_hat_learn_in.sum() / learn.sum() - 1.0) < 0.05


def test_learned_planner_runs_and_beats_the_bell_only_plan_under_a_shift() -> None:
    cfg = _cfg(trigger_offset_mean_min=-9.0, trigger_offset_sd_min=3.0)
    stream = engine.build_stream(cfg)
    runs = engine.run_all(cfg, policies=("calendar", "calendar_learn", "offline"), stream=stream)
    assert runs["calendar_learn"].metrics.slo_in >= runs["calendar"].metrics.slo_in
    assert runs["calendar_learn"].metrics.total_cost >= runs["offline"].metrics.total_cost - 1e-6


def test_stream_is_bit_identical_to_before_the_learned_forecast() -> None:
    """Adding the learned forecast must not move any random draw of the realised stream."""
    a = engine.build_stream(_cfg())
    b = engine.build_stream(_cfg())
    assert np.array_equal(a.t_arrive_s, b.t_arrive_s) and np.array_equal(a.lam_hat_in, b.lam_hat_in)
    assert a.lam_hat_learn_in is not None and a.lam_hat_learn_in.sum() > 0


PRIOR_SLOTS = "data/timetable/slots_2025-2026-2.csv"


@pytest.fixture(scope="module")
def prior_present() -> None:
    if not (REPO / PRIOR_SLOTS).exists():
        pytest.skip("previous-semester slice missing; run scripts/t2/run_t2.py on 2025-2026-2 first")


def test_prior_fills_week_one_and_leaves_the_stream_and_policy_12_untouched(prior_present: None) -> None:
    """Policy 12a: week 1 sits on the earlier semester's pooled timing; nothing else moves."""
    plain = engine.build_stream(_cfg(trigger_offset_mean_min=-9.0, trigger_offset_sd_min=3.0))
    cfg = _cfg(trigger_offset_mean_min=-9.0, trigger_offset_sd_min=3.0, learn_prior_slots_csv=PRIOR_SLOTS)
    stream = engine.build_stream(cfg)
    assert np.array_equal(plain.t_arrive_s, stream.t_arrive_s)
    assert np.array_equal(plain.lam_hat_learn_in, stream.lam_hat_learn_in)
    assert stream.lam_hat_learn_prior_in is not None
    s = stream.sessions
    prior_off, prior_sd = learned_timing(s, s["realised_start_min"].to_numpy(), prior=(-6.0, 2.5))
    w = s["week"].to_numpy()
    assert np.all(prior_off[w == 1] == -6.0) and np.all(prior_sd[w == 1] == 2.5)
    plain_off, _ = learned_timing(s, s["realised_start_min"].to_numpy())
    assert np.array_equal(prior_off[w >= 2], plain_off[w >= 2])  # history overrides the prior
    # in week 1 the prior forecast leads the bell-only one (the earlier semester started early too);
    # same generator for both so the comparison isolates the timing
    pools = (load_offset_pool(REPO / WorkloadConfig().signin_offsets_csv, WorkloadConfig().pool_cap),
             load_offset_pool(REPO / WorkloadConfig().self_offsets_csv, WorkloadConfig().pool_cap))
    est_off, est_sd = learned_timing(s, s["realised_start_min"].to_numpy(),
                                     prior=timing_prior(cfg.workload, str(REPO), seed=3))
    bell = calendar_forecast(cfg.workload, s, np.random.default_rng(11), *pools, plain_off, np.zeros(len(s)))
    warm = calendar_forecast(cfg.workload, s, np.random.default_rng(11), *pools, est_off, est_sd)
    t = np.arange(WEEK)
    c_bell = (t * bell[:WEEK]).sum() / bell[:WEEK].sum()
    c_warm = (t * warm[:WEEK]).sum() / warm[:WEEK].sum()
    assert -10.0 < c_warm - c_bell < -3.0
    runs = engine.run_all(cfg, policies=("calendar_learn", "calendar_learn_prior", "offline"), stream=stream)
    assert runs["calendar_learn_prior"].metrics.slo_in >= runs["calendar_learn"].metrics.slo_in
