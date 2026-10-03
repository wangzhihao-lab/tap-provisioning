"""Lead time, commitment granularity and billing granularity as three separate constraints.

``LEAD_TIERS`` used to carry one number per tier and spend it on whichever meaning a consumer
needed: ``d_res`` was a 60-minute *lead*, while ``metrics`` billed every run in 60-minute blocks
whatever tier it was on.  The tier table now names the three constraints apart - availability delay,
the block the capacity has to be constant on, and the block a replica is billed for - and the two
rented tiers set both blocks to 1, which has to leave every number they produce untouched.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pilot.sim import CostConfig, RunConfig, ServiceConfig, WorkloadConfig, engine
from pilot.sim.config import LEAD_TIERS, with_lead_tier
from pilot.sim.engine import commit_blocks
from pilot.sim.metrics import billed_replica_minutes, fluid_cost, reservation_blocks

REPO = Path(__file__).resolve().parents[1]
BLOCK = LEAD_TIERS["d_res"].commit_block_min
_CACHE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def guard() -> None:
    """Skip when the timetable slice has not been built."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def run(tier: str, policies=("reactive", "calendar", "lcp", "offline", "informed")) -> dict:
    """Every policy on one small instance at one tier (cached: the stream is the same)."""
    if tier not in _CACHE:
        workload = WorkloadConfig(college="college_18", weeks=(1,), adoption=0.2, sigma=0.7)
        cfg = RunConfig(workload=workload, service=with_lead_tier(ServiceConfig(), tier),
                        seed=11, root=str(REPO))
        stream = engine.build_stream(cfg)
        ctx = engine.make_context(cfg, stream)
        _CACHE[tier] = (cfg, stream, {n: engine.run_policy(n, cfg, stream, ctx) for n in policies})
    return _CACHE[tier]


def test_the_tier_table_keeps_the_three_constraints_apart() -> None:
    """d_rep and d_inst differ only in lead time; d_res adds the hour of commitment and billing."""
    rep, inst, res = LEAD_TIERS["d_rep"], LEAD_TIERS["d_inst"], LEAD_TIERS["d_res"]
    assert (rep.lead_min, rep.commit_block_min, rep.billing_block_min) == (0.5, 1, 1)
    assert (inst.lead_min, inst.commit_block_min, inst.billing_block_min) == (8.0, 1, 1)
    assert res.lead_min == inst.lead_min  # a reservation still runs on an instance
    assert res.commit_block_min == res.billing_block_min == 60
    service = with_lead_tier(ServiceConfig(), "d_res")
    assert (service.lead_min, service.commit_block_min, service.billing_block_min) == (8.0, 60, 60)
    assert ServiceConfig().commit_block_min == ServiceConfig().billing_block_min == 1


def test_blocks_of_one_are_a_no_op() -> None:
    """On the rented tiers the block machinery may not touch a trajectory or a bill."""
    x = np.array([0.0, 3.0, 1.0, 1.0, 7.0, 2.0])
    assert np.array_equal(commit_blocks(x, 1), x)
    assert billed_replica_minutes(x, 1) == pytest.approx(x.sum())
    assert reservation_blocks(x, 1) == pytest.approx(x.sum())
    cost = CostConfig()
    zeros = np.zeros(x.size)
    assert fluid_cost(x, zeros, zeros, 1.0, cost, 1.0).run_cost == pytest.approx(
        cost.c_run * x.sum())
    for tier in ("d_rep", "d_inst"):
        cfg, stream, runs = run(tier)
        for name, result in runs.items():
            billed = billed_replica_minutes(result.x, cfg.service.billing_block_min)
            assert billed == pytest.approx(float(result.x.sum())), name
            assert result.metrics.run_cost == pytest.approx(cfg.cost.c_run * result.x.sum()), name


def test_a_commitment_block_holds_one_level_for_every_policy() -> None:
    """With a 60-minute commitment the installed capacity is constant inside every block."""
    _, _, runs = run("d_res")
    for name, result in runs.items():
        grid = result.x[: (result.x.size // BLOCK) * BLOCK].reshape(-1, BLOCK)
        assert np.all(np.ptp(grid, axis=1) == 0.0), name
        assert result.metrics.starts <= result.x.size / BLOCK  # at most one move per block


def test_block_billing_never_charges_less_than_the_minutes() -> None:
    """Charging each replica-block in full can only raise the running cost."""
    rng = np.random.default_rng(0)
    x = np.floor(rng.uniform(0.0, 6.0, size=600))
    per_minute = billed_replica_minutes(x, 1)
    assert billed_replica_minutes(x, 60) > per_minute  # a varying trajectory really pays more
    for block in (1, 5, 60, 120):
        assert billed_replica_minutes(x, block) >= per_minute - 1e-9
    flat = np.full(600, 3.0)
    assert billed_replica_minutes(flat, 60) == pytest.approx(flat.sum())  # nothing to round up
    partial = np.array([2.0, 5.0, 1.0])  # a horizon shorter than one block pays its own minutes
    assert billed_replica_minutes(partial, 60) == pytest.approx(5.0 * 3)


def test_the_reserved_hour_is_paid_in_full() -> None:
    """Every committed replica-block is charged whole, so the bill is never below the minutes."""
    cfg, _, runs = run("d_res")
    for name, result in runs.items():
        minutes = float(result.x.sum())
        assert result.metrics.run_cost >= cfg.cost.c_run * minutes - 1e-6, name
        # commitment and billing share the block here, so the two coincide exactly
        assert result.metrics.run_cost == pytest.approx(cfg.cost.c_run * minutes)
        assert result.metrics.reservation_blocks * BLOCK == pytest.approx(minutes)


def test_the_hourly_commitment_binds_the_feedback_policies() -> None:
    """d_res and d_inst share a lead time, so any difference is the hourly commitment alone.

    A feedback policy pays for it, because the level it commits an hour ahead is the one it wanted
    at that minute and the bell arrives later: it moves far less often and misses more.  A policy
    that plans from the timetable does not, because its plan for the hour already exists when the
    hour is committed.  The oracle re-plans on the block, and since the hour-constant trajectories
    are a subset of the per-minute ones and cost the same there, it can never come out below the
    unconstrained optimum.
    """
    inst_cfg, _, inst = run("d_inst")
    res_cfg, _, res = run("d_res")
    assert inst_cfg.service.lead_min == res_cfg.service.lead_min
    for name in ("reactive", "lcp"):
        # the commitment shows in the trajectory: hour-constant, hence far fewer starts.  Whether
        # it also costs more depends on the background density - with the sparse off-class
        # background measured on 2026-09-17 the per-minute feedback policy thrashes between
        # requests and pays more than the hour-committed one - so the cost direction is not
        # asserted here; the relative gain of the timetable policy below is.
        x = res[name].x
        assert np.all(x.reshape(-1, BLOCK) == x.reshape(-1, BLOCK)[:, :1]), name
        assert res[name].metrics.starts < inst[name].metrics.starts, name
    ratio = {tier: runs["calendar"].metrics.total_cost / runs["offline"].metrics.total_cost
             for tier, runs in (("d_inst", inst), ("d_res", res))}
    assert ratio["d_res"] < ratio["d_inst"], ratio  # the timetable policy gains, relatively
    assert res["offline"].metrics.total_cost >= inst["offline"].metrics.total_cost - 1e-6


def test_the_block_level_uses_only_the_decision_minute() -> None:
    """``Policy.block_target`` may not look past the block's decision minute, ``start - d``.

    Every block is compared against the same policy's *unconstrained* per-minute targets, which the
    block = 1 path produces on the identical stream (nothing feeds the block level back into
    ``SimState``, so both runs drive the policy through the same states).  A feedback policy has to
    land on the target it held at the decision minute; a policy whose plan does not depend on the
    block lands on that plan's peak over the block, and the ceiling commutes with the maximum so
    the two are comparable after rounding.  The two optima are left out on purpose: they re-solve
    their LP *on* the commitment block, so their blocked plan is a different and better one than
    the peak of their unconstrained plan - that is covered by the dominance test below.
    """
    cfg, stream, _ = run("d_res")
    blocked = engine.make_context(cfg, stream)
    free = replace(cfg, service=replace(cfg.service, commit_block_min=1))
    free_ctx = engine.make_context(free, stream)
    horizon = stream.horizon_min
    starts = [a for a in range(0, horizon, BLOCK) if a >= blocked.lead_min]  # decided in the run
    looked_ahead = 0
    for name in ("reactive", "velocity", "lcp", "static", "calendar", "stale_calendar"):
        x = engine.run_policy(name, cfg, stream, blocked).x
        raw = engine.run_policy(name, free, stream, free_ctx).x
        feedback = name in ("reactive", "velocity", "lcp")
        for a in starts:
            hi = min(a + BLOCK, horizon)
            want = raw[a] if feedback else raw[a:hi].max()
            assert x[a] == pytest.approx(want), (name, a)
            assert np.all(x[a:hi] == x[a]), (name, a)
            looked_ahead += bool(feedback and raw[a:hi].max() > raw[a])
    assert looked_ahead > 0, "no block where a feedback policy wanted more later: test is vacuous"


def test_the_oracle_still_dominates_under_block_commitment() -> None:
    """Acceptance test (a) has to survive the tier: the oracle plans on the same blocks."""
    cfg, stream, runs = run("d_res")
    oracle = runs["offline"].metrics.total_cost
    bound = engine.lp_lower_bound(cfg, stream)
    for name, result in runs.items():
        assert result.metrics.total_cost >= oracle - 1e-6, name
        assert result.metrics.total_cost >= bound - 1e-6, name


def test_a_longer_billing_block_alone_leaves_the_trajectory_alone() -> None:
    """Billing and commitment are independent: billing by the hour does not flatten anything."""
    cfg, stream, _ = run("d_inst")
    ctx = engine.make_context(cfg, stream)
    per_minute = engine.run_policy("reactive", cfg, stream, ctx)
    billed = replace(cfg, service=replace(cfg.service, billing_block_min=60))
    hourly = engine.run_policy("reactive", billed, stream, engine.make_context(billed, stream))
    assert np.array_equal(per_minute.x, hourly.x)  # the commitment block is still 1
    assert hourly.metrics.gpu_hours == pytest.approx(per_minute.metrics.gpu_hours)
    assert hourly.metrics.run_cost > per_minute.metrics.run_cost  # but the idle hour is paid for
