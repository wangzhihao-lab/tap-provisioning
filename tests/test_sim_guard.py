"""The bell-aware guard (policy 11): lemma 1's standing-guard witness S in the simulator."""

from pathlib import Path

import numpy as np
import pytest

from pilot.sim import PolicyConfig, RunConfig, ServiceConfig, WorkloadConfig, engine
from pilot.sim.policies.guard import PROFILE_MIN

REPO = Path(__file__).resolve().parents[1]


def _config(**workload) -> RunConfig:
    base = dict(college="college_18", weeks=(1,), adoption=0.2, sigma=0.7)
    base.update(workload)
    return RunConfig(workload=WorkloadConfig(**base), service=ServiceConfig(lead_min=0.5),
                     policy=PolicyConfig(min_replicas=1), seed=11, root=str(REPO))


@pytest.fixture(scope="module", autouse=True)
def slice_present() -> None:
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def test_guard_knows_the_family_only() -> None:
    cfg = _config()
    stream = engine.build_stream(cfg)
    ctx = engine.make_context(cfg, stream)
    policy = engine.make_policy("guard", ctx, stream)
    assert ctx.bells is not None and len(policy.bells) == len(ctx.bells)
    assert set(np.unique(stream.sessions["bell_min"]).astype(int)) <= set(policy.bells)  # bells plus later triggers
    assert policy.profile.shape == (PROFILE_MIN,) and policy.profile.max() == pytest.approx(1.0)
    assert policy.a_max == pytest.approx(max(float(ctx.lam_hat_in[b: b + PROFILE_MIN].max())
                                            for b in policy.bells))


def test_guard_stands_at_a_max_on_every_bell() -> None:
    """During the blind minutes of each bell the level is the family maximum, not the bell's own."""
    cfg = _config()
    stream = engine.build_stream(cfg)
    ctx = engine.make_context(cfg, stream)
    runs = engine.run_all(cfg, policies=("guard", "offline", "reactive"), stream=stream)
    policy = engine.make_policy("guard", ctx, stream)
    x = runs["guard"].x
    for tau in policy.bells[:20]:
        need = policy.a_max * policy.profile[0] / ctx.mu
        assert x[tau] >= min(ctx.cost.x_max, np.floor(need))
    assert runs["guard"].metrics.slo_in >= runs["reactive"].metrics.slo_in
    assert runs["guard"].metrics.total_cost >= runs["offline"].metrics.total_cost - 1e-6


def test_guard_is_causal_and_corrects_downward() -> None:
    cfg = _config()
    stream = engine.build_stream(cfg)
    ctx = engine.make_context(cfg, stream)
    policy = engine.make_policy("guard", ctx, stream)
    policy.reset()
    state = engine.SimState(stream.lam_in, stream.lam_off, np.zeros(stream.horizon_min))
    tau = policy.bells[0]
    state.t = tau - ctx.lead_min  # decides the level in force at the bell itself
    at_bell = policy.observe(state.t, state)
    state.t = tau + 5  # five minutes of the burst revealed
    later = policy.observe(state.t, state)
    assert later <= at_bell
