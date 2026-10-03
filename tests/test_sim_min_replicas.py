"""The always-on base tier (``PolicyConfig.min_replicas``) is a deployment constraint.

Every realised trajectory sits on it, and the oracles optimise *under* it.  A floor-less optimum
lifted onto the floor is feasible but not optimal - and before this test existed the integer DP of
the offline oracle dropped the floor the LP had honoured, so the denominator of the
institution-wide scenario was the optimum of a different problem from the one the online policies
were solving.
"""

from pathlib import Path

import numpy as np
import pytest

from pilot.sim import CostConfig, PolicyConfig, RunConfig, ServiceConfig, WorkloadConfig, engine
from pilot.sim.policies import round_to_replicas
from pilot.sim.policies.base import hit_cost
from pilot.sim.policies.offline import solve_offline_lp

REPO = Path(__file__).resolve().parents[1]
MU = 1000.0
COST = CostConfig(c_run=0.0333, beta_start=0.0333, p_in=6 * 0.0333 / MU, p_off=0.0333 / MU)


def _burst(horizon: int = 720, at: int = 300, width: int = 5, height: float = 3000.0
           ) -> tuple[np.ndarray, np.ndarray]:
    """One bell-shaped burst in an otherwise idle day."""
    lam_in = np.zeros(horizon)
    lam_in[at:at + width] = height
    return lam_in, np.zeros(horizon)


def _total(x: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray) -> float:
    """Cost of a trajectory: hit cost plus the switching cost from ``x_0 = 0``."""
    ups = np.maximum(np.diff(x, prepend=0.0), 0.0)
    return float(hit_cost(x, lam_in, lam_off, MU, COST).sum() + COST.beta_start * ups.sum())


def test_integer_dp_optimises_under_the_floor() -> None:
    """With ``x_min = 1`` the DP returns the exact optimum over ``[1, X]``, not a lifted path."""
    lam_in, lam_off = _burst()
    z0 = solve_offline_lp(lam_in, lam_off, MU, COST, 1)
    z1 = solve_offline_lp(lam_in, lam_off, MU, COST, 1, x_min=1.0)
    path0 = round_to_replicas(z0, lam_in, lam_off, MU, COST, 1)
    path1 = round_to_replicas(z1, lam_in, lam_off, MU, COST, 1, x_min=1.0)
    assert path0.min() == 0.0, "without a floor nothing runs outside the burst"
    assert path1.min() >= 1.0
    assert path1.max() >= path0.max()
    lifted = np.maximum(path0, 1.0)
    assert _total(path1, lam_in, lam_off) <= _total(lifted, lam_in, lam_off) + 1e-9
    assert _total(path1, lam_in, lam_off) >= _total(path0, lam_in, lam_off) - 1e-9


def test_floor_is_capped_by_the_quota() -> None:
    """A floor above ``x_max`` collapses onto the quota rather than breaking the DP."""
    lam_in, lam_off = _burst()
    cost = CostConfig(c_run=COST.c_run, beta_start=COST.beta_start, p_in=COST.p_in,
                      p_off=COST.p_off, x_max=3.0)
    z = solve_offline_lp(lam_in, lam_off, MU, cost, 1, x_min=5.0)
    path = round_to_replicas(z, lam_in, lam_off, MU, cost, 1, x_min=5.0)
    assert path.min() == path.max() == 3.0


@pytest.fixture(scope="module")
def floored_runs() -> tuple:
    """Every policy on one college, one week, with a base tier of one replica."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")
    cfg = RunConfig(workload=WorkloadConfig(college="college_18", weeks=(1,), adoption=0.2,
                                            sigma=0.7),
                    service=ServiceConfig(lead_min=0.5), policy=PolicyConfig(min_replicas=1),
                    seed=11, root=str(REPO))
    stream = engine.build_stream(cfg)
    return engine.run_all(cfg, stream=stream), stream


def test_every_policy_sits_on_the_base_tier(floored_runs: tuple) -> None:
    """The realised trajectory of every policy, the oracles included, never drops below the tier."""
    runs, stream = floored_runs
    for name, run in runs.items():
        assert float(np.min(run.x)) >= 1.0, f"{name} went below the base tier"
    assert runs["offline"].metrics.gpu_hours >= stream.horizon_min / 60.0 - 1e-6


def test_offline_still_dominates_under_the_floor(floored_runs: tuple) -> None:
    """Acceptance test (a) of the simulator design holds in the floored scenario too."""
    runs, _ = floored_runs
    oracle = runs["offline"].metrics.total_cost
    for name, run in runs.items():
        assert run.metrics.total_cost >= oracle - 1e-6, f"{name} beat the floored optimum"
