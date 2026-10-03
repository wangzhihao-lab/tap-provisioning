"""Policy registry, the lead-time contract, and the behaviour each policy is supposed to have."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pilot.sim import (
    POLICY_FACTORY,
    PolicyConfig,
    PolicyFactory,
    RunConfig,
    ServiceConfig,
    WorkloadConfig,
)
from pilot.sim import engine
from pilot.sim.policies import BDInterpPolicy, Policy, SimState, solve_offline_lp
from pilot.sim.policies.offline import round_to_replicas

REPO = Path(__file__).resolve().parents[1]
EXPECTED = {"static", "reactive", "velocity", "forecast_ilp", "lcp", "lcp_grid", "calendar",
            "bdinterp", "trust", "offline", "informed", "stale_calendar"}


def base_config(**service) -> RunConfig:
    """One small college, two teaching weeks, the d_inst lead tier."""
    wl = WorkloadConfig(college="college_04", weeks=(1, 2), adoption=0.4, sigma=0.7)
    return RunConfig(workload=wl, service=ServiceConfig(lead_min=8.0, **service), root=str(REPO))


@pytest.fixture(scope="module")
def instance():
    """A shared arrival stream plus its policy context (the paired design)."""
    cfg = base_config()
    if not (REPO / cfg.workload.slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")
    stream = engine.build_stream(cfg)
    return cfg, stream, engine.make_context(cfg, stream)


def test_registry_holds_every_policy() -> None:
    """Importing the package registers policies 1-9, policy 7 under both of its keys."""
    assert EXPECTED <= set(POLICY_FACTORY)
    assert all(issubclass(PolicyFactory(k), Policy) for k in EXPECTED)
    with pytest.raises(KeyError):
        PolicyFactory("nope")


def test_every_policy_returns_a_whole_non_negative_pool(instance) -> None:
    """Trajectories are integral, non-negative and bounded by the configured diameter."""
    cfg, stream, ctx = instance
    for name in engine.DEFAULT_POLICIES:
        x = engine.run_policy(name, cfg, stream, ctx).x
        assert x.shape == (stream.horizon_min,)
        assert np.all(x >= 0.0) and np.all(x <= cfg.cost.x_max)
        assert np.allclose(x, np.round(x))


def test_a_decision_at_t_lands_at_t_plus_d(instance) -> None:
    """The engine installs ``observe(t)`` at minute ``t + d`` and nowhere else."""
    _, stream, ctx = instance
    lead = ctx.lead_min
    assert lead > 0

    class Ramp(Policy):
        name = "ramp"

        def observe(self, t: int, state: SimState) -> float:
            return float(t % 7)

    x = engine.trajectory(Ramp(ctx), stream, ctx)
    assert np.all(x[:lead] == 0.0)
    expected = np.array([(t - lead) % 7 for t in range(lead, stream.horizon_min)], dtype=float)
    assert np.array_equal(x[lead:], expected)


def test_static_is_flat_and_never_the_cheapest(instance) -> None:
    """Policy 1 holds the calendar peak for the whole run."""
    cfg, stream, ctx = instance
    run = engine.run_policy("static", cfg, stream, ctx)
    assert len(np.unique(run.x)) == 1
    assert run.metrics.starts <= 1


def test_lcp_is_lazier_than_the_reactive_rule(instance) -> None:
    """Policy 5 changes the pool far less often than policy 2."""
    cfg, stream, ctx = instance
    lcp = engine.run_policy("lcp", cfg, stream, ctx).metrics
    reactive = engine.run_policy("reactive", cfg, stream, ctx).metrics
    assert lcp.starts < reactive.starts


def test_forecast_ilp_falls_back_in_the_first_week(instance) -> None:
    """Policy 4 has no same-slot history in week 1, so it runs the reactive rule there."""
    cfg, stream, ctx = instance
    ilp = engine.run_policy("forecast_ilp", cfg, stream, ctx).x
    reactive = engine.run_policy("reactive", cfg, stream, ctx).x
    week1 = 7 * 1440
    assert np.array_equal(ilp[:week1], reactive[:week1])
    assert not np.array_equal(ilp[week1:], reactive[week1:])


def test_calendar_precommits_the_lead_window(instance) -> None:
    """Policy 6 knows the timetable before minute 0, so it does not start empty at a bell."""
    cfg, stream, ctx = instance
    plan = PolicyFactory("calendar")(ctx).trajectory()
    assert plan.size == stream.horizon_min
    assert plan.max() > 0.0
    bell = int(stream.sessions["bell_min"].iloc[0])
    assert plan[bell:bell + 5].max() >= plan[max(0, bell - 60):bell - 30].max()


def test_policy_7_is_bdinterp_under_both_of_its_keys(instance) -> None:
    """Policy 7 is BDINTERP-d; ``trust`` stays as the key the older CSVs use.

    The algorithm itself is covered by ``test_sim_bdinterp.py``; here only the registration, the
    alias and the meaning of the knob (``ctx.trust`` is the epsilon of the BDINTERP-d analysis, 0 = pure advice).
    """
    assert PolicyFactory("trust") is PolicyFactory("bdinterp") is BDInterpPolicy
    cfg, stream, _ = instance
    ctx = engine.make_context(cfg, stream, trust=0.0)
    alias = engine.run_policy("trust", cfg, stream, ctx).x
    assert np.array_equal(alias, engine.run_policy("bdinterp", cfg, stream, ctx).x)
    assert np.array_equal(alias, engine.run_policy("calendar", cfg, stream, ctx).x)


def test_informed_optimum_is_blockwise_and_dearer_than_the_oracle(instance) -> None:
    """The informed optimum may only move once per lead time, so it cannot beat policy 8."""
    cfg, stream, ctx = instance
    offline = engine.run_policy("offline", cfg, stream, ctx)
    informed = engine.run_policy("informed", cfg, stream, ctx)
    assert informed.metrics.total_cost >= offline.metrics.total_cost - 1e-6
    blocks = informed.x[: (informed.x.size // ctx.lead_min) * ctx.lead_min]
    assert np.all(np.ptp(blocks.reshape(-1, ctx.lead_min), axis=1) == 0.0)


def test_lp_and_integer_rounding_on_a_hand_instance() -> None:
    """The LP covers demand when serving is cheaper than the penalty; the DP returns integers."""
    cost = base_config().cost
    mu = 100.0
    lam_in = np.zeros(120)
    lam_in[40:60] = 250.0
    lam_off = np.full(120, 20.0)
    z = solve_offline_lp(lam_in, lam_off, mu, cost, 1)
    assert z.shape == (120,)
    assert z[50] == pytest.approx(2.7, abs=0.05)  # (250 + 20) / 100
    x = round_to_replicas(z, lam_in, lam_off, mu, cost)
    assert np.allclose(x, np.round(x))
    assert x[50] >= 2.0
    assert x.max() <= np.ceil(z.max()) + 1


def test_reactive_step_mode_moves_one_replica(instance) -> None:
    """The literal +/- 1 rule of the simulator design is still available through ``hpa_mode``."""
    cfg, stream, _ = instance
    step = replace(cfg, policy=replace(cfg.policy, hpa_mode="step"))
    ctx = engine.make_context(step, stream)
    x = engine.run_policy("reactive", step, stream, ctx).x
    assert np.abs(np.diff(x)).max() <= 1.0


def test_plan_margin_books_more_and_only_for_policies_6_and_7(instance) -> None:
    """kappa multiplies the calendar plan; policies 4, 5 and 8 do not see it."""
    cfg, stream, ctx = instance
    plain = engine.run_policy("calendar", cfg, stream, ctx)
    wide = replace(cfg, policy=replace(cfg.policy, plan_margin=1.5))
    wide_ctx = engine.make_context(wide, stream)
    padded = engine.run_policy("calendar", wide, stream, wide_ctx)
    assert padded.x.sum() > plain.x.sum()
    assert padded.metrics.unserved_in <= plain.metrics.unserved_in + 1e-9
    assert engine.run_policy("bdinterp", wide, stream, wide_ctx).x.sum() > plain.x.sum()
    for name in ("forecast_ilp", "lcp", "offline"):
        assert np.array_equal(engine.run_policy(name, wide, stream, wide_ctx).x,
                              engine.run_policy(name, cfg, stream, ctx).x), name
    assert PolicyConfig().plan_margin == 1.0  # the default changes nothing


def test_velocity_hysteresis_cuts_the_starts(instance) -> None:
    """Policy 3 with the band moves far less often, and the flag restores the old rule."""
    cfg, stream, ctx = instance
    assert cfg.policy.velocity_hysteresis
    band = engine.run_policy("velocity", cfg, stream, ctx).metrics
    raw_cfg = replace(cfg, policy=replace(cfg.policy, velocity_hysteresis=False))
    raw = engine.run_policy("velocity", raw_cfg, stream, engine.make_context(raw_cfg, stream))
    assert band.starts < raw.metrics.starts
    dwell = cfg.policy.velocity_dwell_min
    x = engine.run_policy("velocity", cfg, stream, ctx).x
    moves = np.flatnonzero(np.diff(x) != 0.0)
    assert moves.size == 0 or np.diff(moves).min() >= dwell - 1  # d-slot dwell, lead aside


def test_stale_calendar_is_a_worse_calendar_but_still_a_calendar(instance) -> None:
    """Policy 9 plans from last semester: dearer than policy 6, far cheaper than the reactive rule."""
    cfg, stream, ctx = instance
    if stream.lam_stale_in is None or stream.lam_stale_in.sum() == 0.0:
        pytest.skip("last semester's slot table is missing")
    oracle = engine.run_policy("offline", cfg, stream, ctx).metrics.total_cost
    stale = engine.run_policy("stale_calendar", cfg, stream, ctx, oracle).metrics
    fresh = engine.run_policy("calendar", cfg, stream, ctx, oracle).metrics
    reactive = engine.run_policy("reactive", cfg, stream, ctx, oracle).metrics
    assert stale.cost_ratio >= fresh.cost_ratio - 1e-9
    assert stale.cost_ratio < reactive.cost_ratio
    assert "stale_calendar" in engine.DEFAULT_POLICIES


def test_forecast_ilp_can_warm_start_from_the_stale_plan(instance) -> None:
    """Policy 4's first week is the stale plan when asked for, and the reactive rule otherwise."""
    cfg, stream, ctx = instance
    if stream.lam_stale_in is None or stream.lam_stale_in.sum() == 0.0:
        pytest.skip("last semester's slot table is missing")
    warm_cfg = replace(cfg, policy=replace(cfg.policy, forecast_warm_start="stale_calendar"))
    warm_ctx = engine.make_context(warm_cfg, stream)
    warm = engine.run_policy("forecast_ilp", warm_cfg, stream, warm_ctx).x
    week1 = min(7 * 1440, stream.horizon_min)
    stale = engine.run_policy("stale_calendar", warm_cfg, stream, warm_ctx).x
    reactive = engine.run_policy("reactive", cfg, stream, ctx).x
    assert np.array_equal(warm[:week1], stale[:week1])
    assert not np.array_equal(warm[:week1], reactive[:week1])
    assert PolicyConfig().forecast_warm_start is None
