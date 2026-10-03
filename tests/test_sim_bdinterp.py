"""Policy 7, BDINTERP-d of the BDINTERP-d analysis.

Four properties, in the order lemma B proves them:

(i)   **consistency.**  With a calendar that carries no forecast error, the combined policy costs at
      most ``(1 + eps)`` times policy 6 plus the additive ``O(d c_max)`` of (B1).
(ii)  **robustness.**  With a calendar that cancels every class while the demand is real, it stays
      within a constant factor of policy 5 - (B2) allows ``1 + (1 + eps)/delta = 9`` at ``eps = 1``,
      the assertion below is the looser 12.  That instance runs at ``d = 0`` and with no off-class
      background, because robustness is a statement about *ROB being good*: with the 8-minute
      instance lead time and a burst that is over in five, policy 5 misses the classes as badly as
      a plan that booked none of them, so the lead time rather than the algorithm decides the
      comparison and the phase test never has a reason to leave the advice.
(iii) **determinism.**  Both black boxes and the combination are deterministic given the stream.
(iv)  **availability.**  Committing slot ``t + d`` reads no cost later than slot ``t``: the phase
      test and the contraction proxy only ever touch the revealed prefix.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from pilot.sim import RunConfig, ServiceConfig, WorkloadConfig, engine
from pilot.sim.policies.base import SimState
from pilot.sim.policies.bdinterp import ADV_PHASE, BDInterpPolicy

REPO = Path(__file__).resolve().parents[1]
COLLEGE = "college_04"
PERFECT: Dict[str, object] = {"roster_noise": 0.0, "attendance_lo": 1.0, "attendance_hi": 1.0,
                              "p_cancel": 0.0, "forecast_attendance": 1.0}
# the plan books no class at all, and there is no background traffic to hide behind
CANCELLED: Dict[str, object] = {"forecast_attendance": 0.0, "adoption": 1.0,
                                "r_off_per_student_day": 0.0, "deadline_extra_per_student": 0.0}
_CACHE: Dict[Tuple, Tuple] = {}


@pytest.fixture(scope="module", autouse=True)
def guard() -> None:
    """Skip when the timetable slice has not been built."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def instance(workload: Dict[str, object], eps: float, lead: float = 8.0) -> Tuple:
    """``(cfg, stream, ctx)`` for one workload variant, with the trust knob set to ``eps``."""
    key = (tuple(sorted(workload.items())), eps, lead)
    if key not in _CACHE:
        base: Dict[str, object] = {"college": COLLEGE, "weeks": (1,), "adoption": 0.4,
                                   "sigma": 0.7}
        base.update(workload)
        cfg = RunConfig(workload=WorkloadConfig(**base), service=ServiceConfig(lead_min=lead),
                        seed=7, root=str(REPO))
        stream = engine.build_stream(cfg)
        _CACHE[key] = (cfg, stream, engine.make_context(cfg, stream, trust=eps))
    return _CACHE[key]


def run_bdinterp(cfg: RunConfig, stream, ctx) -> Tuple[np.ndarray, BDInterpPolicy]:
    """Drive policy 7 by hand so that its phase log and fluid trajectory can be read."""
    policy = engine.make_policy("bdinterp", ctx, stream)
    policy.reset()
    return engine.trajectory(policy, stream, ctx), policy


def c_max(cfg: RunConfig, stream, x_ref: float) -> float:
    """``c x_max + p_in lam_max + beta x_max / 2`` of the BDINTERP-d analysis

    ``CostConfig.x_max`` is a box bound (2000 replicas) rather than a reachable level, so the slack
    is evaluated at the peak the two candidate sequences actually reach; anything larger would make
    the additive term of (B1) vacuous.
    """
    cost = cfg.cost
    lam_max = float((stream.lam_in + stream.lam_off).max())
    return cost.c_run * x_ref + cost.p_in * lam_max + 0.5 * cost.beta_start * x_ref


def test_i_a_perfect_calendar_is_followed_and_costs_what_policy_6_costs() -> None:
    """(i) Consistency: ``C_ALG <= (1 + eps) C_ADV + O(d c_max)``, here with equality to ``C_ADV``."""
    eps = 0.1
    cfg, stream, ctx = instance(PERFECT, eps)
    calendar = engine.run_policy("calendar", cfg, stream, ctx)
    ours, policy = run_bdinterp(cfg, stream, ctx)
    cost = engine.run_policy("bdinterp", cfg, stream, ctx).metrics.total_cost
    lcp = engine.run_policy("lcp", cfg, stream, ctx)
    slack = 4.0 * (ctx.lead_min + 1) * c_max(cfg, stream, max(calendar.x.max(), lcp.x.max()))
    assert cost <= (1.0 + eps) * calendar.metrics.total_cost + slack
    # the advice is never worth less than delta = eps/4 of the robust cost, so the phase test
    # stays in the ADV branch for the whole run and the two trajectories coincide exactly
    assert policy.phase_share() == 1.0
    assert np.array_equal(ours, calendar.x)
    assert cost == pytest.approx(calendar.metrics.total_cost)


def test_ii_a_calendar_that_cancels_every_class_stays_near_the_robust_policy() -> None:
    """(ii) Robustness: a plan that books no class at all keeps policy 7 within 12x policy 5."""
    eps = 1.0
    cfg, stream, ctx = instance(CANCELLED, eps, lead=0.0)
    assert stream.lam_hat_in.sum() == 0.0 and stream.lam_in.sum() > 0.0
    lcp = engine.run_policy("lcp", cfg, stream, ctx).metrics.total_cost
    calendar = engine.run_policy("calendar", cfg, stream, ctx).metrics.total_cost
    cost = engine.run_policy("bdinterp", cfg, stream, ctx).metrics.total_cost
    _, policy = run_bdinterp(cfg, stream, ctx)
    assert calendar > 2.0 * lcp, "the instance has to punish the advice for the test to mean anything"
    assert cost <= 12.0 * lcp
    assert policy.phase_share() < 1.0  # the contraction towards s_t is exercised


def test_integerisation_is_the_shared_ceiling() -> None:
    """Policy 7 integerises with ``as_int_plan`` like every other fluid policy: ceiling, clipped.

    Two halves.  On a real instance the committed trace has to be the ceiling of the fluid one and
    the engine has to install exactly that.  The ROB branch then gets a fractional decision fed to
    it by hand - ``nu`` is a ratio of two integer gaps, so on the instances of this module it lands
    on 0 or 1 and the fluid decision is already whole - to show that the rule is the ceiling and
    not the nearest integer it replaced.
    """
    cfg, stream, ctx = instance(CANCELLED, 1.0, lead=0.0)
    x, policy = run_bdinterp(cfg, stream, ctx)
    fluid = np.asarray(policy.x_fluid, dtype=float)
    committed = np.asarray(policy.x_int, dtype=float)
    assert policy.phase_share() < 1.0  # the contraction towards s_t is the branch under test
    expected = np.clip(np.ceil(fluid - 1e-9), cfg.policy.min_replicas, cfg.cost.x_max)
    assert np.array_equal(committed, expected)
    assert np.all(committed >= fluid - 1e-9)  # never below the fluid level (remark R1)
    assert np.array_equal(x, expected[: x.size])  # the engine installs exactly that trace

    fresh = BDInterpPolicy(ctx)
    fresh.reset()
    fresh._c_adv, fresh._c_rob, fresh._q = 1000.0, 0.0, 0.0  # phase test fails -> ROB branch
    fresh._adv_seq, fresh._rob_seq, fresh.x_fluid = [0.0], [4.0], [1.0]  # nu = 3/4
    assert fresh._commit(1.0, 0.0) == 1.0  # y = 0.75 -> ceiling 1, nearest would have given 0
    assert fresh.x_fluid[-1] == pytest.approx(0.75)


def test_ii_epsilon_selects_between_the_two_black_boxes() -> None:
    """``eps = 0`` is policy 6 exactly; a large ``eps`` is policy 5 once the advice has a price."""
    cfg, stream, ctx = instance({}, 0.0)
    ours, policy = run_bdinterp(cfg, stream, ctx)
    assert policy.phase_share() == 1.0
    assert np.array_equal(ours, engine.run_policy("calendar", cfg, stream, ctx).x)
    cfg, stream, ctx = instance({}, 1e6)
    ours, policy = run_bdinterp(cfg, stream, ctx)
    first_rob = next(i for i, p in enumerate(policy.phase) if p != ADV_PHASE)
    robust = engine.run_policy("lcp", cfg, stream, ctx).x
    assert first_rob <= ctx.lead_min + 60
    assert np.array_equal(ours[first_rob + 1:], robust[first_rob + 1:])


def test_iii_the_policy_is_deterministic() -> None:
    """(iii) Two runs on the same stream give the same integer and fluid trajectories."""
    cfg, stream, ctx = instance({}, 0.5)
    first_x, first = run_bdinterp(cfg, stream, ctx)
    second_x, second = run_bdinterp(cfg, stream, ctx)
    assert np.array_equal(first_x, second_x)
    assert first.x_fluid == second.x_fluid and first.phase == second.phase
    assert first.x_int == second.x_int
    denom = engine.run_policy("offline", cfg, stream, ctx).metrics.total_cost
    left = engine.run_policy("bdinterp", cfg, stream, ctx, denom).metrics.to_row()
    right = engine.run_policy("bdinterp", cfg, stream, ctx, denom).metrics.to_row()
    assert left == right  # every field, cost ratio included (it is NaN without a denominator)
    # the rounding is the shared ceiling of ``as_int_plan``, and both traces are kept
    assert len(first.x_fluid) == len(first.x_int) == len(first.phase)
    assert np.array_equal(np.asarray(first.x_int), np.ceil(np.asarray(first.x_fluid) - 1e-9))


def _touched(key, n: int) -> int:
    """Largest slot index a read of ``key`` can reach."""
    if isinstance(key, slice):
        stop = n if key.stop is None else (key.stop if key.stop >= 0 else key.stop + n)
        return int(min(stop, n)) - 1
    if isinstance(key, (int, np.integer)):
        return int(key) if key >= 0 else int(key) + n
    return n - 1


class _Watched(np.ndarray):
    """Demand array that logs the largest slot index every direct read touches."""

    def __new__(cls, values: np.ndarray, log: List[int]) -> "_Watched":
        obj = np.asarray(values, dtype=float).view(cls)
        obj.log = log
        obj.watch = True
        return obj

    def __array_finalize__(self, obj) -> None:
        """Views and ufunc results inherit the log but never report on their own arithmetic."""
        self.log = getattr(obj, "log", None)
        self.watch = False

    def __getitem__(self, key):
        """Record the read, then behave like an ordinary array."""
        if getattr(self, "watch", False) and self.log is not None:
            self.log.append(_touched(key, self.shape[0]))
        return super().__getitem__(key)


def test_iv_no_cost_after_the_committing_minute_is_read() -> None:
    """(iv) Deciding ``x_{t+d}`` at minute ``t`` touches no demand later than slot ``t``."""
    cfg, stream, ctx = instance({}, 0.5)
    log: List[int] = []
    state = SimState(_Watched(stream.lam_in, log), _Watched(stream.lam_off, log),
                     np.zeros(stream.horizon_min))
    policy = engine.make_policy("bdinterp", ctx, stream)
    policy.reset()
    policy.prefix(ctx.lead_min)
    assert not log, "the pre-committed prefix must not look at the demand at all"
    for t in range(stream.horizon_min):
        log.clear()
        state.t = t
        state.x_now = 0.0
        policy.observe(t, state)
        reach = max(log, default=-1)
        assert reach <= t, f"slot {t + ctx.lead_min} read demand of slot {reach} > {t}"
        # the simulator reveals minute m only once it is over, so the prefix is one slot shorter
        # than the BDINTERP-d analysis assumes: BDINTERP-d with d -> d + 1
        assert reach <= t - 1
