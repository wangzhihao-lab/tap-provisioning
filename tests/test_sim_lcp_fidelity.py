"""Policy 5 against the algorithm it claims to be: LCP(0) of Lin et al. (2011).

``reference_lcp0`` below is an independent, deliberately naive implementation of the paper: at
every slot it re-solves the two offline prefix problems of section III from scratch, over integer
replica counts, and projects the previous capacity onto ``[x_t^L, x_t^U]``.  It is ``O(t)`` passes
per slot and only usable on a short instance, which is the point - it shares no code path with the
incremental version that ships in ``policies/lcp.py``.

Two things are measured on 20 random 300-slot instances:

* the shipped ``lcp`` reproduces the reference exactly (this is the fidelity claim);
* the superseded ``lcp_grid`` approximation - constant capacity over a geometric grid of lookbacks
  - lands far outside the 2% band that would have let it stay, which is why it was replaced.
"""

from typing import Tuple

import numpy as np
import pytest

from pilot.sim import CostConfig, PolicyConfig, PolicyFactory, ServiceConfig
from pilot.sim.metrics import fluid_cost
from pilot.sim.policies.base import Policy, PolicyContext, SimState, as_int_plan, hit_cost

COST = CostConfig()
MU = 296.0  # one replica under the placeholder calibration, in requests per minute
LEAD = 8
SLOTS = 300
SEEDS = tuple(range(20))
TOLERANCE = 0.02  # the band the approximation would have had to stay inside


def random_instance(seed: int, n: int = SLOTS) -> Tuple[np.ndarray, np.ndarray]:
    """A gamma background with three to seven decaying bursts on top of it."""
    rng = np.random.default_rng(seed)
    lam_off = rng.gamma(2.0, 30.0, size=n)
    lam_in = np.zeros(n, dtype=float)
    for start in rng.choice(n - 25, size=int(rng.integers(3, 8)), replace=False):
        width = int(rng.integers(5, 20))
        lam_in[start:start + width] += (rng.uniform(200.0, 1500.0)
                                        * np.exp(-np.arange(width) / 5.0))
    return lam_in, lam_off


def context(n: int) -> PolicyContext:
    """A policy context with no calendar at all: LCP never looks at one."""
    return PolicyContext(horizon_min=n, mu=MU, lead_min=LEAD, cost=COST, policy=PolicyConfig(),
                         service=ServiceConfig(), lam_hat_in=np.zeros(n), lam_hat_off=np.zeros(n))


def drive(policy: Policy, lam_in: np.ndarray, lam_off: np.ndarray,
          ctx: PolicyContext) -> np.ndarray:
    """``engine.trajectory`` without a request stream: decide at ``t``, install at ``t + d``."""
    n, lead = lam_in.size, ctx.lead_min
    plan = np.zeros(n + lead + 1, dtype=float)
    pre = np.asarray(policy.prefix(lead), dtype=float)
    if lead:
        plan[:lead] = np.pad(pre, (0, max(0, lead - pre.size)), mode="edge")[:lead]
    state = SimState(lam_in, lam_off, np.zeros(n))
    for t in range(n):
        state.t = t
        state.x_now = float(plan[t])
        plan[t + lead] = policy.observe(t, state)
    return as_int_plan(plan[:n], ctx.cost)


def reference_lcp0(lam_in: np.ndarray, lam_off: np.ndarray, ctx: PolicyContext) -> np.ndarray:
    """LCP(0) of Lin et al. (2011), re-solving both prefix problems from scratch every slot."""
    n, lead, beta = lam_in.size, ctx.lead_min, ctx.cost.beta_start
    states = np.arange(int(np.ceil((lam_in + lam_off).max() / ctx.mu)) + 3, dtype=float)
    plan = np.zeros(n + lead + 1, dtype=float)
    x = 0.0
    for t in range(n):
        low = high = None
        for u in range(t):
            f = hit_cost(states, float(lam_in[u]), float(lam_off[u]), ctx.mu, ctx.cost)
            if low is None:  # x_0 = 0: reaching x costs beta x, leaving 0 costs nothing
                low, high = f + beta * states, f.copy()
                continue
            low = f + np.minimum(np.minimum.accumulate(low - beta * states) + beta * states,
                                 np.minimum.accumulate(low[::-1])[::-1])
            high = f + np.minimum(np.minimum.accumulate(high),
                                  np.minimum.accumulate((high + beta * states)[::-1])[::-1]
                                  - beta * states)
        lo = 0.0 if low is None else float(np.argmin(low))
        hi = lo if high is None else max(lo, float(states.size - 1
                                                   - int(np.argmin(high[::-1]))))
        x = float(np.clip(x if t > 0 else 0.0, lo, hi))
        plan[t + lead] = x
    return as_int_plan(plan[:n], ctx.cost)


def _cost(x: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray) -> float:
    """Total cost of one trajectory on one instance."""
    return fluid_cost(x, lam_in, lam_off, MU, COST).total_cost


@pytest.fixture(scope="module")
def deviations() -> dict:
    """Per-seed cost deviation of the shipped policy and of the superseded approximation."""
    out: dict = {"exact": [], "grid": [], "identical": []}
    for seed in SEEDS:
        lam_in, lam_off = random_instance(seed)
        ctx = context(lam_in.size)
        ref = reference_lcp0(lam_in, lam_off, ctx)
        base = _cost(ref, lam_in, lam_off)
        exact = drive(PolicyFactory("lcp")(ctx), lam_in, lam_off, ctx)
        grid = drive(PolicyFactory("lcp_grid")(ctx), lam_in, lam_off, ctx)
        out["identical"].append(bool(np.array_equal(exact, ref)))
        out["exact"].append(abs(_cost(exact, lam_in, lam_off) - base) / base)
        out["grid"].append(abs(_cost(grid, lam_in, lam_off) - base) / base)
    return out


def test_the_shipped_policy_is_lcp0(deviations: dict) -> None:
    """Policy 5 reproduces the paper's algorithm slot for slot on every seed."""
    assert all(deviations["identical"]), deviations["identical"]
    assert max(deviations["exact"]) <= 1e-12


def test_the_geometric_grid_approximation_missed_the_band(deviations: dict) -> None:
    """The approximation policy 5 used to be is far outside the 2% band, hence the replacement."""
    worst = max(deviations["grid"])
    assert worst > TOLERANCE, f"the approximation now fits in {TOLERANCE:.0%}: {worst:.3%}"
    assert worst > 0.5  # it was 85% at the time of the replacement; a wide guard, not a target


def test_the_bounds_are_ordered_and_lazy() -> None:
    """``x_L <= x_U`` at every slot, and the capacity only ever moves to one of the two bounds."""
    lam_in, lam_off = random_instance(3)
    ctx = context(lam_in.size)
    policy = PolicyFactory("lcp")(ctx)
    policy.reset()
    state = SimState(lam_in, lam_off, np.zeros(lam_in.size))
    seen = []
    for t in range(lam_in.size):
        state.t = t
        policy.observe(t, state)
        lo, hi = policy._bounds()
        assert lo <= hi
        seen.append((lo, hi, policy._x))
    assert all(lo <= x <= hi for lo, hi, x in seen)
    assert any(x == lo for lo, _, x in seen) and any(x == hi for _, hi, x in seen)
