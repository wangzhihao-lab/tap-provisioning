"""A re-solve in mid-run starts from the level already committed, in the LP and in the integer DP.

Three policies re-solve their plan while the run is going: policy 6a (``calendar_adapt``) after a
size correction, TAP's advice (``tap.LearnedPriorAdaptPolicy``) the same way, and policy 4
(``forecast_ilp``) on its receding horizon.  A decision at minute ``t`` can first reach slot
``t0 = t + d``; slots ``t + 1 .. t0 - 1`` are already committed, so the re-solve has to start from
the level committed for ``t0 - 1``.  ``state.x_now`` is the level installed at ``t``, ``d - 1``
slots older.  Before 2026-09-25 only the LP received a start level, and the integer DP
(``offline.round_to_replicas``) charged a start for every replica of its first block, as if the
pool were empty.

(a) a one-slot example, and the DP against brute force from a non-empty pool;
(b) with ``x_init`` left out the DP is bit for bit the function before the fix;
(c) policy 6a at ``d = 8`` re-solves from its plan at ``t0 - 1`` and pays starts only above it;
(d) policy 4 hands the LP and the DP the target it committed one minute earlier;
(e) TAP's advice re-solves by policy 6a's rule, from its own sequence.
"""

import itertools
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pytest

from pilot.sim import CostConfig, PolicyConfig, ServiceConfig, engine
from pilot.sim.metrics import fluid_cost
from pilot.sim.policies import calendar_adapt, forecast_ilp
from pilot.sim.policies.base import PolicyContext, SimState
from pilot.sim.policies.calendar_adapt import CalendarAdaptPolicy
from pilot.sim.policies.forecast_ilp import ForecastILPPolicy
from pilot.sim.policies.offline import round_to_replicas, solve_offline_lp
from pilot.sim.policies.tap import LearnedPriorAdaptPolicy, TAPPolicy

LEAD = 8  # the instance tier: x_now lags the start of a re-solve by seven slots
MU = 1.0
EXAMPLE_COST = CostConfig(c_run=1.0, beta_start=1.0, p_in=30.0, p_off=5.0, x_max=4.0)
ADAPT_COST = CostConfig(c_run=1.0, beta_start=3.0, p_in=30.0, p_off=5.0, x_max=12.0)


# --- (b)'s reference: offline.round_to_replicas and its helpers before x_init (verbatim) ---------
def _ref_block_map(horizon: int, block_min: int) -> np.ndarray:
    """Minute -> block index."""
    return np.arange(horizon) // max(1, int(block_min))


def _ref_suffix_argmin(v: np.ndarray, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(min_{y>=x} v[y], one minimiser)`` for every ``x``."""
    smin = np.minimum.accumulate(v[::-1])[::-1]
    cand = np.where(v <= smin + 1e-12, states, states.size)
    return smin, np.minimum.accumulate(cand[::-1])[::-1]


def _ref_prefix_argmin(v: np.ndarray, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(min_{y<=x} v[y], one minimiser)`` for every ``x``."""
    pmin = np.minimum.accumulate(v)
    cand = np.where(v <= pmin + 1e-12, states, -1)
    return pmin, np.maximum.accumulate(cand)


def _ref_round_to_replicas(z: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray, mu: float,
                           cost: CostConfig, block_min: int = 1, x_min: float = 0.0) -> np.ndarray:
    """The integer DP as of 2026-09-24 (docstring and debug log dropped)."""
    lam_in = np.asarray(lam_in, dtype=float)
    lam_off = np.asarray(lam_off, dtype=float)
    horizon = lam_in.size
    if horizon == 0:
        return np.zeros(0, dtype=float)
    lo = int(min(max(0.0, float(x_min)), cost.x_max))
    top = max(float(np.max(z)) if np.size(z) else 0.0,
              float(np.max(lam_in + lam_off)) / max(1e-9, mu), float(lo))
    n_x = int(min(cost.x_max, np.ceil(top) + 1.0)) + 1
    states = np.arange(lo, n_x, dtype=float)  # replica counts; the DP indexes their positions
    per_min = (cost.c_run * states[None, :]
               + (cost.p_in - cost.p_off) * np.maximum(lam_in[:, None] - mu * states[None, :], 0.0)
               + cost.p_off * np.maximum((lam_in + lam_off)[:, None] - mu * states[None, :], 0.0))
    blk = _ref_block_map(horizon, block_min)
    starts = np.flatnonzero(np.diff(blk, prepend=-1))
    cost_j = np.add.reduceat(per_min, starts, axis=0)
    n_x = states.size
    idx = np.arange(n_x)

    value = cost_j[0] + cost.beta_start * states
    pred = np.zeros((cost_j.shape[0], n_x), dtype=np.int32)
    for j in range(1, cost_j.shape[0]):
        smin, sarg = _ref_suffix_argmin(value, idx)
        pmin, parg = _ref_prefix_argmin(value - cost.beta_start * states, idx)
        up = pmin + cost.beta_start * states
        use_up = up < smin
        value = cost_j[j] + np.where(use_up, up, smin)
        pred[j] = np.where(use_up, parg, sarg)
    path = np.zeros(cost_j.shape[0], dtype=np.int64)
    path[-1] = int(np.argmin(value))
    for j in range(cost_j.shape[0] - 1, 0, -1):
        path[j - 1] = pred[j][path[j]]
    return states[path][blk]


# --- shared helpers -------------------------------------------------------------------------------
def _true_cost(x: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray, cost: CostConfig,
               x_init: float, mu: float = MU) -> float:
    """The cost of ``x`` with ``x_init`` in force before its first slot."""
    return fluid_cost(np.asarray(x, dtype=float), lam_in, lam_off, mu, cost,
                      x_prev=x_init).total_cost


def _brute_force(lam_in: np.ndarray, lam_off: np.ndarray, cost: CostConfig, x_init: float,
                 lo: int, block_min: int = 1, mu: float = MU) -> float:
    """Cheapest integer trajectory in ``[lo, x_max]``, constant on blocks, from ``x_init``."""
    blk = np.arange(lam_in.size) // block_min
    levels = range(lo, int(cost.x_max) + 1)
    return min(_true_cost(np.asarray(choice, dtype=float)[blk], lam_in, lam_off, cost, x_init, mu)
               for choice in itertools.product(levels, repeat=int(blk[-1]) + 1))


def _context(lam_hat_in: np.ndarray, cost: CostConfig, **policy: object) -> PolicyContext:
    """A hand-built context at ``d = 8``: forecast ``lam_hat_in`` (TAP's too), no off-class."""
    horizon = lam_hat_in.size
    return PolicyContext(horizon_min=horizon, mu=MU, lead_min=LEAD, cost=cost,
                         policy=PolicyConfig(**{"min_replicas": 1, **policy}),
                         service=ServiceConfig(lead_min=float(LEAD)),
                         lam_hat_in=lam_hat_in, lam_hat_off=np.zeros(horizon),
                         lam_hat_learn_prior_in=lam_hat_in)


def _stream(lam_in: np.ndarray) -> SimpleNamespace:
    """The fields of an ``ArrivalStream`` that ``engine.trajectory`` reads."""
    return SimpleNamespace(horizon_min=lam_in.size, lam_in=lam_in, lam_off=np.zeros(lam_in.size),
                           t_arrive_s=np.zeros(0), prompt_tokens=np.zeros(0),
                           output_tokens=np.zeros(0))


def _bursts(horizon: int, bells: List[int], width: int, height: float) -> np.ndarray:
    """Triangular bursts of ``width`` minutes and peak ``height`` starting at every bell."""
    shape = height * (1.0 - np.abs(np.linspace(-1.0, 1.0, width)))
    out = np.zeros(horizon)
    for bell in bells:
        seg = out[bell:bell + width]
        seg += shape[:seg.size]
    return out


def _adapt_instance() -> Tuple[np.ndarray, np.ndarray]:
    """Twelve hours of bursts, realised at twice the forecast: the size ratio keeps moving."""
    hat = _bursts(720, list(range(30, 700, 90)), width=40, height=4.0)
    return hat, 2.0 * hat


def _spy(monkeypatch: pytest.MonkeyPatch, module: object) -> Dict[str, list]:
    """Log the ``x_init`` that ``module``'s LP and DP calls receive (``None``: not passed)."""
    log: Dict[str, list] = {"lp": [], "dp": []}
    lp, dp = module.solve_offline_lp, module.round_to_replicas

    def lp_spy(*args, **kwargs):
        log["lp"].append(kwargs.get("x_init"))
        return lp(*args, **kwargs)

    def dp_spy(*args, **kwargs):
        log["dp"].append(kwargs.get("x_init"))
        return dp(*args, **kwargs)

    monkeypatch.setattr(module, "solve_offline_lp", lp_spy)
    monkeypatch.setattr(module, "round_to_replicas", dp_spy)
    return log


class _ReplanLog:
    """Mixin for policy 6a and TAP's advice: per re-solve, the minute, ``t0`` and ``x_now``."""

    def reset(self) -> None:
        self.replans: List[Tuple[int, int, float]] = []
        super().reset()

    def _replan(self, ratio: float, state: SimState) -> None:
        t0 = min(self.trajectory().size, state.t + self.ctx.lead_min)
        if t0 < self.trajectory().size:
            self.replans.append((int(state.t), int(t0), float(state.x_now)))
        super()._replan(ratio, state)


class _RecordedAdapt(_ReplanLog, CalendarAdaptPolicy):
    """Policy 6a with its re-solves logged."""


class _RecordedAdvice(_ReplanLog, LearnedPriorAdaptPolicy):
    """TAP's advice with its re-solves logged."""


class _RecordedTAP(TAPPolicy):
    """TAP over the logged advice."""

    def make_advice(self, ctx: PolicyContext) -> LearnedPriorAdaptPolicy:
        return _RecordedAdvice(ctx)


# --- (a) ------------------------------------------------------------------------------------------
def test_one_slot_example_keeps_the_committed_replica() -> None:
    """One slot left, two replicas committed, demand 1.05: keep both (2.0), not drop one (2.5)."""
    lam_in, lam_off, x_init = np.array([1.05]), np.array([0.0]), 2.0
    z = solve_offline_lp(lam_in, lam_off, MU, EXAMPLE_COST, x_init=x_init, x_min=1.0)
    assert z.tolist() == pytest.approx([1.05])
    fixed = round_to_replicas(z, lam_in, lam_off, MU, EXAMPLE_COST, x_min=1.0, x_init=x_init)
    empty = round_to_replicas(z, lam_in, lam_off, MU, EXAMPLE_COST, x_min=1.0)
    assert fixed.tolist() == [2.0]
    assert empty.tolist() == [1.0]
    assert _true_cost(fixed, lam_in, lam_off, EXAMPLE_COST, x_init) == pytest.approx(2.0)
    assert _true_cost(empty, lam_in, lam_off, EXAMPLE_COST, x_init) == pytest.approx(2.5)


@pytest.mark.parametrize("seed", range(24))
def test_dp_from_a_non_empty_pool_is_the_exact_integer_optimum(seed: int) -> None:
    """Against every trajectory in ``[x_min, x_max]``, on blocks of one and two minutes."""
    rng = np.random.default_rng(1000 + seed)
    block = (1, 2)[seed % 2]
    lo = (0, 1, 2)[seed % 3]
    horizon = int(rng.integers(1, 5))
    cost = CostConfig(c_run=float(rng.uniform(0.3, 2.0)), beta_start=float(rng.uniform(0.0, 4.0)),
                      p_in=float(rng.uniform(3.0, 30.0)), p_off=float(rng.uniform(0.1, 2.5)),
                      x_max=5.0)
    lam_in = np.round(rng.uniform(0.0, 4.5, horizon), 2)
    lam_off = np.round(rng.uniform(0.0, 1.0, horizon), 2)
    x_init = float(rng.integers(0, 6))
    z = solve_offline_lp(lam_in, lam_off, MU, cost, block, x_init=x_init, x_min=float(lo))
    path = round_to_replicas(z, lam_in, lam_off, MU, cost, block, x_min=float(lo), x_init=x_init)
    best = _brute_force(lam_in, lam_off, cost, x_init, lo, block)
    assert _true_cost(path, lam_in, lam_off, cost, x_init) == pytest.approx(best, abs=1e-9)


# --- (b) ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(36))
def test_default_start_is_bit_for_bit_the_previous_dp(seed: int) -> None:
    """``x_init`` left out, or 0, gives the pre-fix DP exactly: blocks, floors, quotas, ties."""
    rng = np.random.default_rng(seed)
    block = (1, 2, 3, 5)[seed % 4]
    x_min = (0.0, 1.0, 2.0)[seed % 3]
    horizon = int(rng.integers(1, 80))
    cost = CostConfig(c_run=float(rng.uniform(0.2, 2.0)), beta_start=float(rng.uniform(0.0, 6.0)),
                      p_in=float(rng.uniform(3.0, 40.0)), p_off=float(rng.uniform(0.05, 2.5)),
                      x_max=(3.0, 6.0, 2000.0)[(seed // 3) % 3])
    lam_in = rng.gamma(0.7, 3.0, horizon) * (rng.random(horizon) < 0.6)
    lam_off = rng.uniform(0.0, 1.2, horizon)
    if seed % 2:  # whole demands put ties in front of the DP's tie-breaks
        lam_in, lam_off = np.round(lam_in), np.round(lam_off)
    mu = float(rng.uniform(0.5, 3.0))
    z = solve_offline_lp(lam_in, lam_off, mu, cost, block, x_min=x_min)
    ref = _ref_round_to_replicas(z, lam_in, lam_off, mu, cost, block, x_min=x_min)
    for new in (round_to_replicas(z, lam_in, lam_off, mu, cost, block, x_min=x_min),
                round_to_replicas(z, lam_in, lam_off, mu, cost, block, x_min=x_min, x_init=0.0)):
        assert new.dtype == ref.dtype and new.shape == ref.shape
        assert new.tobytes() == ref.tobytes()


# --- (c) ------------------------------------------------------------------------------------------
def test_calendar_adapt_resolves_from_its_plan_at_t0_minus_1(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The one-slot case inside policy 6a: plan 4 at ``t``, 2 at ``t0 - 1``; the DP keeps the 2."""
    rest = np.array([1.05, 0.2, 0.2, 0.2])  # the part of the horizon the re-solve can change
    t = 24
    t0 = t + LEAD
    hat = np.concatenate([np.zeros(t0), rest])
    policy = CalendarAdaptPolicy(_context(hat, EXAMPLE_COST))
    plan = np.ones(hat.size)
    plan[t], plan[t0 - 1] = 4.0, 2.0
    policy._plan = plan.copy()
    state = SimState(np.zeros(hat.size), np.zeros(hat.size), np.zeros(hat.size), x_now=4.0, t=t)
    assert policy.replan_x_init(state, plan, t0) == 2.0
    assert policy.replan_x_init(state, plan, 0) == 0.0

    log = _spy(monkeypatch, calendar_adapt)
    policy._replan(1.0, state)
    assert log == {"lp": [2.0], "dp": [2.0]}
    new = policy.trajectory()
    assert np.array_equal(new[:t0], plan[:t0])  # committed slots are left alone
    assert new[t0:].tolist() == [2.0, 1.0, 1.0, 1.0]

    zero = np.zeros(rest.size)
    best = _brute_force(rest, zero, EXAMPLE_COST, 2.0, lo=1)
    assert best == pytest.approx(5.0)
    assert _true_cost(new[t0:], rest, zero, EXAMPLE_COST, 2.0) == pytest.approx(best)
    z = solve_offline_lp(rest, zero, MU, EXAMPLE_COST, 1, x_init=2.0, x_min=1.0)
    empty = round_to_replicas(z, rest, zero, MU, EXAMPLE_COST, x_min=1.0)  # the pre-fix DP call
    assert empty.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert _true_cost(empty, rest, zero, EXAMPLE_COST, 2.0) == pytest.approx(5.5)


def test_calendar_adapt_in_the_engine_starts_every_resolve_from_the_committed_level(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Every re-solve of a run hands the LP and the DP the installed level at ``t0 - 1``."""
    hat, lam = _adapt_instance()
    ctx = _context(hat, ADAPT_COST)
    log = _spy(monkeypatch, calendar_adapt)
    policy = _RecordedAdapt(ctx)
    x = engine.trajectory(policy, _stream(lam), ctx)
    assert policy.n_replans == len(policy.replans) >= 2
    committed = [float(x[t0 - 1]) for _, t0, _ in policy.replans]
    assert log["lp"] == committed and log["dp"] == committed
    assert any(x_now != c for (_, _, x_now), c in zip(policy.replans, committed)), \
        "at d = 8 some re-solve has to start from a level x_now does not show"


# --- (d) ------------------------------------------------------------------------------------------
class _RecordedILP(ForecastILPPolicy):
    """Policy 4 logging, per re-solve, the minute, the level installed then and the start level."""

    def reset(self) -> None:
        self.replans: List[Tuple[int, float, float]] = []
        super().reset()

    def replan_x_init(self, state: SimState) -> float:
        value = super().replan_x_init(state)
        self.replans.append((int(state.t), float(state.x_now), value))
        return value


def test_forecast_ilp_resolves_from_the_target_it_committed_a_minute_earlier(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Receding horizon: the LP and the DP both start from ``x[target - 1]``, not ``x[t]``."""
    lam = np.tile(_bursts(1440, list(range(0, 1440, 60)), width=40, height=5.0), 3)
    cost = CostConfig(c_run=1.0, beta_start=3.0, p_in=30.0, p_off=5.0, x_max=10.0)
    ctx = _context(np.zeros(lam.size), cost, ilp_season_days=1, ilp_replan_min=170,
                   ilp_horizon_min=360)
    log = _spy(monkeypatch, forecast_ilp)
    policy = _RecordedILP(ctx)
    x = engine.trajectory(policy, _stream(lam), ctx)
    assert len(policy.replans) >= 10
    committed = [float(x[t + LEAD - 1]) for t, _, _ in policy.replans]
    assert [value for _, _, value in policy.replans] == committed
    assert log["lp"] == committed and log["dp"] == committed
    assert any(x_now != c for (_, x_now, _), c in zip(policy.replans, committed)), \
        "at d = 8 some re-solve has to start from a level x_now does not show"


def test_forecast_ilp_start_level_falls_back_on_x_now_before_any_decision() -> None:
    """The start is the previous ``decide``; ``reset`` and ``start_run`` both forget it."""
    policy = ForecastILPPolicy(_context(np.zeros(60), EXAMPLE_COST))
    state = SimState(np.zeros(60), np.zeros(60), np.zeros(60), x_now=3.0, t=5)
    assert policy.replan_x_init(state) == 3.0
    assert policy.decide(5, state) == 1.0  # the first week's reactive fallback, at the floor
    assert policy.replan_x_init(state) == 1.0
    policy.reset()
    assert policy.replan_x_init(state) == 3.0
    policy.decide(5, state)
    policy.start_run()
    assert policy.replan_x_init(state) == 3.0


# --- (e) ------------------------------------------------------------------------------------------
def test_tap_advice_resolves_by_the_rule_of_policy_6a(monkeypatch: pytest.MonkeyPatch) -> None:
    """Policy 6a's start rule, applied to the advice's own sequence inside the combination."""
    assert LearnedPriorAdaptPolicy.replan_x_init is CalendarAdaptPolicy.replan_x_init
    hat, lam = _adapt_instance()
    ctx = _context(hat, ADAPT_COST)
    log = _spy(monkeypatch, calendar_adapt)
    tap = _RecordedTAP(ctx)
    engine.trajectory(tap, _stream(lam), ctx)
    advice = np.asarray(tap._adv_seq[:lam.size])
    replans = tap._adv.replans
    assert tap._adv.n_replans == len(replans) >= 2
    own = [float(advice[t0 - 1]) for _, t0, _ in replans]
    assert log["lp"] == own and log["dp"] == own
    assert any(x_now != a for (_, _, x_now), a in zip(replans, own)), \
        "some re-solve has to start from a level the combined x_now does not show"

    alone = _RecordedAdvice(ctx)
    x_alone = engine.trajectory(alone, _stream(lam), ctx)
    assert np.array_equal(x_alone, advice)  # a sequence of its own, as BDINTERP-d needs
    assert [r[:2] for r in alone.replans] == [r[:2] for r in replans]
    assert log["dp"][len(own):] == own  # and alone it starts every re-solve from the same levels
