"""The two controls added on 2026-09-17: the base tier alone and the timetable with an online
size correction (a calendar planner that learns from what it sees is the baseline BDINTERP-d
has to beat on the forecast-error axis)."""


from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pilot.sim import PolicyConfig, RunConfig, ServiceConfig, WorkloadConfig, engine

REPO = Path(__file__).resolve().parents[1]


def _config(**workload) -> RunConfig:
    base = dict(college="college_18", weeks=(1,), adoption=0.2, sigma=0.7)
    base.update(workload)
    return RunConfig(workload=WorkloadConfig(**base), service=ServiceConfig(lead_min=0.5),
                     policy=PolicyConfig(min_replicas=1), seed=11, root=str(REPO))


@pytest.fixture(scope="module", autouse=True)
def guard() -> None:
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def test_fixed_holds_the_base_tier_only() -> None:
    cfg = _config()
    stream = engine.build_stream(cfg)
    runs = engine.run_all(cfg, policies=("fixed", "offline"), stream=stream)
    x = runs["fixed"].x
    assert float(x.min()) == float(x.max()) == 1.0
    assert runs["fixed"].metrics.starts == 1.0
    assert runs["fixed"].metrics.total_cost >= runs["offline"].metrics.total_cost - 1e-6


def test_online_correction_undoes_a_size_bias() -> None:
    """With a forecast twice the truth the corrected plan buys less than plain FtP and stays
    above the optimum; with an exact forecast the two plans coincide up to the ratio's drift."""
    biased = _config(forecast_scale=2.0)
    stream = engine.build_stream(biased)
    runs = engine.run_all(biased, policies=("calendar", "calendar_adapt", "offline"),
                          stream=stream)
    ftp, adapt, opt = (runs[k].metrics for k in ("calendar", "calendar_adapt", "offline"))
    assert adapt.gpu_hours < ftp.gpu_hours
    assert adapt.total_cost < ftp.total_cost
    assert adapt.total_cost >= opt.total_cost - 1e-6
    exact = replace(biased, workload=replace(biased.workload, forecast_scale=1.0))
    stream = engine.build_stream(exact)
    runs = engine.run_all(exact, policies=("calendar", "calendar_adapt"), stream=stream)
    a, b = runs["calendar"].metrics.total_cost, runs["calendar_adapt"].metrics.total_cost
    assert abs(a - b) <= 0.15 * a


def test_online_correction_is_causal() -> None:
    """The plan in force at minute t uses realised demand strictly before t."""
    cfg = _config(forecast_scale=2.0)
    stream = engine.build_stream(cfg)
    ctx = engine.make_context(cfg, stream)
    policy = engine.make_policy("calendar_adapt", ctx, stream)
    policy.reset()
    state = engine.SimState(stream.lam_in, stream.lam_off, np.zeros(stream.horizon_min))
    state.t = 0
    assert policy.ratio(state) == pytest.approx(1.0)  # nothing revealed yet: the prior alone
    state.t = stream.horizon_min
    late = policy.ratio(state)
    assert 0.25 <= late < 1.0  # the truth is half the forecast, so the ratio moves down
