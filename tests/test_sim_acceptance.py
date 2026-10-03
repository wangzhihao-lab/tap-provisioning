"""Acceptance tests (a)-(e) of the simulator design

(a) the offline optimum is never beaten;
(b) with the calendar switched off, policies 6 and 7 cost what policy 5 costs;
(c) the reactive policy's violations fall as the lead time falls;
(d) the service model matches the calibration curve (in test_sim_service.py, which owns the
    concurrency grid);
(e) the same config and seed give identical results twice.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pilot.sim import RunConfig, ServiceConfig, WorkloadConfig
from pilot.sim import engine

REPO = Path(__file__).resolve().parents[1]


def config(lead_min: float = 8.0, **workload) -> RunConfig:
    """A small but non-degenerate instance: one college, one or two teaching weeks."""
    base = dict(college="college_18", weeks=(1,), adoption=0.2, sigma=0.7)
    base.update(workload)
    return RunConfig(workload=WorkloadConfig(**base), service=ServiceConfig(lead_min=lead_min),
                     seed=11, root=str(REPO))


@pytest.fixture(scope="module", autouse=True)
def guard() -> None:
    """Skip when the timetable slice has not been built."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def test_a_offline_optimum_dominates() -> None:
    """(a) No policy costs less than policy 8 on the same instance, and none beats the LP bound."""
    cfg = config()
    stream = engine.build_stream(cfg)
    ctx = engine.make_context(cfg, stream)
    runs = engine.run_all(cfg, stream=stream)
    oracle = runs["offline"].metrics.total_cost
    bound = engine.lp_lower_bound(cfg, stream, ctx)
    for name, run in runs.items():
        assert run.metrics.total_cost >= oracle - 1e-6, f"{name} beat the offline optimum"
        assert run.metrics.total_cost >= bound - 1e-6, f"{name} beat the fluid LP bound"
        assert run.metrics.cost_ratio >= 1.0 - 1e-9
    assert oracle >= bound - 1e-6


def test_b_no_calendar_degrades_to_the_robust_policy() -> None:
    """(b) With no classes and a flat background, policies 6 and 7 cost what policy 5 costs.

    The background has to be flat: ``lambda_hat`` carries the diurnal profile, which is calendar
    information in its own right, so with the default diurnal shape the equality would not hold.
    """
    cfg = config(college="college_04", adoption=1.0, sigma=0.0, attendance_lo=0.0,
                 attendance_hi=0.0, forecast_attendance=0.0, diurnal=(1.0,) * 24,
                 r_off_per_student_day=3.0, deadline_extra_per_student=0.0)
    stream = engine.build_stream(cfg)
    ctx = engine.make_context(cfg, stream)
    assert stream.lam_in.sum() == 0.0 and stream.lam_hat_in.sum() == 0.0
    assert stream.lam_off.mean() > 1.0  # non-degenerate load, away from the 0/1 replica boundary
    costs = {n: engine.run_policy(n, cfg, stream, ctx).metrics.total_cost
             for n in ("lcp", "calendar", "bdinterp")}
    for name in ("calendar", "bdinterp"):
        assert abs(costs[name] - costs["lcp"]) <= 0.05 * costs["lcp"], f"{name} vs lcp: {costs}"


def test_c_reactive_violations_fall_with_the_lead_time() -> None:
    """(c) Policy 2's in-class violations shrink as the lead time goes to zero.

    Two caveats, both properties of the baseline rather than of the simulator.  The violation rate
    does not reach zero at d = 0: a threshold rule needs a minute of utilisation before it can
    react and the classroom burst rises in two.  And the decrease is only monotone while d stays
    within the burst, which is the range the claim is about; a lead time of an hour makes the pool
    oscillate and the rate is no longer ordered by d.  (That is a lead time, not the d_res tier:
    the reserved tier runs on the instance lead time and constrains the *commitment* instead.)
    """
    cfg = config(adoption=0.6)
    stream = engine.build_stream(cfg)
    viol = []
    for lead in (0.0, 1.0, 8.0):
        ctx = engine.make_context(replace(cfg, service=replace(cfg.service, lead_min=lead)),
                                  stream)
        assert ctx.lead_min == int(lead)
        viol.append(engine.run_policy("reactive", cfg, stream, ctx).metrics.viol_in)
    assert all(a <= b + 1e-9 for a, b in zip(viol, viol[1:])), viol
    assert viol[0] < 0.5 * viol[-1], viol


def test_d_service_model_matches_the_calibration_curve() -> None:
    """(d) Delegated to test_sim_service.py, which sweeps the whole concurrency grid."""
    from test_sim_service import test_ttft_matches_the_calibration_table as check
    for n in (1, 8, 16, 32):
        check(n)


def test_e_same_seed_same_bytes(tmp_path) -> None:
    """(e) Two runs of the same config and seed are identical down to the CSV bytes."""
    cfg = config(weeks=(1,), adoption=0.1)
    first = engine.run_all(cfg)
    second = engine.run_all(cfg)
    assert set(first) == set(second)
    for name in first:
        assert np.array_equal(first[name].x, second[name].x)
        assert np.array_equal(np.nan_to_num(first[name].service.ttft_s, nan=-1.0),
                              np.nan_to_num(second[name].service.ttft_s, nan=-1.0))
        assert first[name].metrics.to_row() == second[name].metrics.to_row()
    from pilot.sim.metrics import to_frame
    paths = []
    for i, runs in enumerate((first, second)):
        path = tmp_path / f"run{i}.csv"
        to_frame([r.metrics for r in runs.values()]).to_csv(path, index=False)
        paths.append(path)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_e_streams_are_paired_across_policies() -> None:
    """Every policy in one comparison sees the same arrival stream."""
    cfg = config(weeks=(1,), adoption=0.1)
    a = engine.build_stream(cfg)
    b = engine.build_stream(cfg)
    assert np.array_equal(a.t_arrive_s, b.t_arrive_s)
    assert np.array_equal(a.prompt_tokens, b.prompt_tokens)
    assert np.array_equal(a.lam_in, b.lam_in)
