"""TAP (``tap``): BDINTERP-d at eps = 0.1 over the learned-prior adaptive plan and LCP.

What has to hold, in the order the construction plan lists it:

* the refactor of policy 6a (ratio and prior mass read ``forecast()``, the re-solve start is a
  hook) leaves ``calendar_adapt`` minute-for-minute where it was - checked against a verbatim copy
  of the pre-refactor class, with only the start of a re-solve moved to the committed level at
  ``t0 - 1`` in the LP and the DP (2026-09-25, ``test_sim_replan_boundary.py``);
* TAP's advice is ``calendar_learn_prior`` exactly while its size ratio does not move, and when it
  does move the ratio is measured against its own forecast and re-solves start from its own plan;
* TAP is BDINTERP-d exactly when its advice is the same sequence as policy 7's;
* ``tap`` runs at eps = 0.1 whatever ``PolicyConfig.trust`` says, ``tap@0.1`` is the same run, and
  an explicit ``tap@eps`` overrides it;
* committing slot ``t + d`` reads no realised demand at or after minute ``t``;
* ``run_sweep.py`` writes one diagnostics row per TAP run.
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from pilot.sim import PolicyConfig, RunConfig, ServiceConfig, SweepConfig, WorkloadConfig, engine
from pilot.sim import sweeps
from pilot.sim.policies.base import SimState
from pilot.sim.policies.bdinterp import BDInterpPolicy
from pilot.sim.policies.calendar import CalendarFtPPolicy
from pilot.sim.policies.calendar_adapt import (
    RATIO_HI,
    RATIO_LO,
    REPLAN_EVERY_MIN,
    REPLAN_TOL,
    WINDOW_MIN,
    CalendarAdaptPolicy,
)
from pilot.sim.policies.offline import round_to_replicas, solve_offline_lp
from pilot.sim.policies.tap import TAP_TRUST, LearnedPriorAdaptPolicy, TAPPolicy

REPO = Path(__file__).resolve().parents[1]
COLLEGE = "college_04"
PRIOR_SLOTS = "data/timetable/slots_2025-2026-2.csv"
PERFECT: Dict[str, object] = {"roster_noise": 0.0, "attendance_lo": 1.0, "attendance_hi": 1.0,
                              "p_cancel": 0.0, "forecast_attendance": 1.0}
CANCELLED: Dict[str, object] = {"forecast_attendance": 0.0, "adoption": 1.0,
                                "r_off_per_student_day": 0.0, "deadline_extra_per_student": 0.0}
_CACHE: Dict[Tuple, Tuple] = {}


@pytest.fixture(scope="module", autouse=True)
def slices_present() -> None:
    """Skip when the timetable slices have not been built."""
    for rel in (WorkloadConfig().slots_csv, PRIOR_SLOTS):
        if not (REPO / rel).exists():
            pytest.skip(f"{rel} missing; run scripts/t2/run_t2.py first")


def instance(workload: Optional[Dict[str, object]] = None, lead: float = 8.0,
             trust: Optional[float] = None, prior: bool = True,
             policy_trust: float = 0.5) -> Tuple[RunConfig, object, object]:
    """``(cfg, stream, ctx)`` of one small week; ``trust`` is an explicit ``name@eps``."""
    workload = dict(workload or {})
    key = (tuple(sorted(workload.items())), lead, trust, prior, policy_trust)
    if key not in _CACHE:
        base: Dict[str, object] = {"college": COLLEGE, "weeks": (1,), "adoption": 0.4, "sigma": 0.7,
                                   "learn_prior_slots_csv": PRIOR_SLOTS if prior else None}
        base.update(workload)
        cfg = RunConfig(workload=WorkloadConfig(**base), service=ServiceConfig(lead_min=lead),
                        policy=PolicyConfig(trust=policy_trust), seed=7, root=str(REPO))
        stream = engine.build_stream(cfg)
        _CACHE[key] = (cfg, stream, engine.make_context(cfg, stream, trust=trust))
    return _CACHE[key]


def trajectory(policy, stream, ctx) -> np.ndarray:
    """Reset ``policy`` and run the engine's decision loop on it."""
    policy.reset()
    return engine.trajectory(policy, stream, ctx)


class _LegacyCalendarAdapt(CalendarFtPPolicy):
    """``CalendarAdaptPolicy`` as it was before the refactor (copied verbatim, not registered)."""

    def __init__(self, ctx) -> None:
        self._planned_ratio = 1.0
        self._last_replan = -10 ** 9
        self._m0: Optional[float] = None
        super().__init__(ctx)

    def reset(self) -> None:
        self._planned_ratio = 1.0
        self._last_replan = -10 ** 9
        self._plan = None

    def _prior_mass(self) -> float:
        if self._m0 is None:
            hat = self.ctx.lam_hat_in
            first_day = float(hat[:1440].sum())
            self._m0 = first_day if first_day > 0 else max(1.0, float(hat.sum()))
        return self._m0

    def ratio(self, state: SimState) -> float:
        lo = max(0, state.t - WINDOW_MIN)
        real = float(state.lam_in[lo:state.t].sum())
        hat = float(self.ctx.lam_hat_in[lo:state.t].sum())
        m0 = self._prior_mass()
        return float(np.clip((m0 + real) / (m0 + hat), RATIO_LO, RATIO_HI))

    def _replan(self, ratio: float, state: SimState) -> None:
        plan = self.trajectory().copy()
        t0 = min(plan.size, state.t + self.ctx.lead_min)
        if t0 >= plan.size:
            return
        hat_in, hat_off = self.forecast()
        kappa = max(0.0, float(self.ctx.policy.plan_margin))
        scaled_in = kappa * ratio * hat_in[t0:]
        scaled_off = kappa * hat_off[t0:]
        x_min = float(self.ctx.policy.min_replicas)
        z = solve_offline_lp(scaled_in, scaled_off, self.ctx.mu, self.ctx.cost, 1,
                             x_init=float(state.x_now), x_min=x_min)
        plan[t0:] = round_to_replicas(z, scaled_in, scaled_off, self.ctx.mu, self.ctx.cost,
                                      x_min=x_min)
        self._plan = plan
        self._planned_ratio = ratio
        self._last_replan = state.t

    def observe(self, t: int, state: SimState) -> float:
        if t - self._last_replan >= REPLAN_EVERY_MIN:
            ratio = self.ratio(state)
            if abs(ratio / self._planned_ratio - 1.0) > REPLAN_TOL:
                self._replan(ratio, state)
        return super().observe(t, state)


class _LegacyWithCommittedStart(_LegacyCalendarAdapt):
    """The pre-refactor class with the one change of 2026-09-25: LP and DP start at ``t0 - 1``."""

    def _replan(self, ratio: float, state: SimState) -> None:
        plan = self.trajectory().copy()
        t0 = min(plan.size, state.t + self.ctx.lead_min)
        if t0 >= plan.size:
            return
        hat_in, hat_off = self.forecast()
        kappa = max(0.0, float(self.ctx.policy.plan_margin))
        scaled_in = kappa * ratio * hat_in[t0:]
        scaled_off = kappa * hat_off[t0:]
        x_min = float(self.ctx.policy.min_replicas)
        x0 = float(plan[t0 - 1]) if t0 > 0 else 0.0
        z = solve_offline_lp(scaled_in, scaled_off, self.ctx.mu, self.ctx.cost, 1,
                             x_init=x0, x_min=x_min)
        plan[t0:] = round_to_replicas(z, scaled_in, scaled_off, self.ctx.mu, self.ctx.cost,
                                      x_min=x_min, x_init=x0)
        self._plan = plan
        self._planned_ratio = ratio
        self._last_replan = state.t


class _PinnedAdvice(LearnedPriorAdaptPolicy):
    """TAP's advice with the size ratio held at 1: it never re-solves."""

    def ratio(self, state: SimState) -> float:
        return 1.0


class _TAPPinned(TAPPolicy):
    """TAP whose advice never re-solves."""

    def make_advice(self, ctx) -> LearnedPriorAdaptPolicy:
        return _PinnedAdvice(ctx)


def test_calendar_adapt_matches_its_pre_refactor_code_minute_by_minute() -> None:
    """Policy 6a still reads the calendar forecast; a re-solve starts at its plan at ``t0 - 1``."""
    for workload in ({}, {"forecast_scale": 0.5}):
        cfg, stream, ctx = instance(workload, prior=False)
        new, old = CalendarAdaptPolicy(ctx), _LegacyWithCommittedStart(ctx)
        x_new, x_old = trajectory(new, stream, ctx), trajectory(old, stream, ctx)
        assert new.n_replans > 0, "the instance has to re-solve for the comparison to mean anything"
        assert np.array_equal(x_new, x_old)
        assert np.array_equal(new.trajectory(), old.trajectory())
        assert new.planned_ratio == old._planned_ratio


def test_advice_is_calendar_learn_prior_while_its_ratio_holds() -> None:
    """No size correction -> the learned-prior plan itself, minute for minute."""
    cfg, stream, ctx = instance()
    assert ctx.lam_hat_learn_prior_in is not None
    advice = _PinnedAdvice(ctx)
    x = trajectory(advice, stream, ctx)
    assert advice.n_replans == 0
    assert np.array_equal(x, engine.run_policy("calendar_learn_prior", cfg, stream, ctx).x)


def test_advice_corrects_its_own_forecast_and_restarts_from_its_own_plan() -> None:
    """The ratio uses the learned-prior forecast; a re-solve starts from the advice's level."""
    cfg, stream, ctx = instance()
    advice = LearnedPriorAdaptPolicy(ctx)
    advice.reset()
    t = 3 * 1440 + 600
    state = SimState(stream.lam_in, stream.lam_off, np.zeros(stream.horizon_min), x_now=17.0, t=t)
    hat = ctx.lam_hat_learn_prior_in
    first_day = float(hat[:1440].sum())
    m0 = first_day if first_day > 0 else max(1.0, float(hat.sum()))
    lo = max(0, t - WINDOW_MIN)
    expected = np.clip((m0 + stream.lam_in[lo:t].sum()) / (m0 + hat[lo:t].sum()), RATIO_LO, RATIO_HI)
    assert advice.ratio(state) == pytest.approx(float(expected), rel=1e-12)
    plan = np.arange(20, dtype=float)
    assert advice.replan_x_init(state, plan, 5) == 4.0  # its own plan at t0 - 1, not x_now
    assert advice.replan_x_init(state, plan, 0) == 0.0
    assert CalendarAdaptPolicy(ctx).replan_x_init(state, plan, 5) == 4.0  # policy 6a: same rule
    trajectory(advice, stream, ctx)
    assert advice.n_replans > 0  # attendance below the booked roster moves the ratio


def test_tap_is_bdinterp_when_the_advice_is_the_same_sequence() -> None:
    """Same advice sequence -> same fluid decisions, phases and installed trajectory."""
    for workload, eps, lead in ((PERFECT, 0.1, 8.0), (CANCELLED, 1.0, 0.0)):
        _, stream, ctx = instance(workload, lead=lead, trust=eps, prior=False)
        same = replace(ctx, lam_hat_learn_prior_in=ctx.lam_hat_in)  # advice forecast = policy 6's
        tap, bd = _TAPPinned(same), BDInterpPolicy(same)
        assert tap.eps == bd.eps == eps
        x_tap, x_bd = trajectory(tap, stream, same), trajectory(bd, stream, same)
        assert np.array_equal(x_tap, x_bd)
        assert tap.x_fluid == bd.x_fluid and tap.phase == bd.phase
    assert tap.phase_share() < 1.0  # the second instance exercises the robust branch


def test_tap_runs_at_its_fixed_trust_unless_one_is_named() -> None:
    """``PolicyConfig.trust`` is ignored; ``tap@0.1`` is ``tap``; ``tap@eps`` overrides."""
    cfg, stream, _ = instance(policy_trust=0.9)
    ctx = engine.make_context(cfg, stream)
    assert ctx.trust == 0.9 and ctx.trust_override is None
    assert TAPPolicy(ctx).eps == TAP_TRUST == 0.1
    assert BDInterpPolicy(ctx).eps == 0.9  # policy 7 still reads the configured knob
    assert TAPPolicy(engine.make_context(cfg, stream, trust=0.4)).eps == 0.4
    plain = engine.run_policy("tap", cfg, stream, ctx)
    named = engine.run_policy("tap", cfg, stream, engine.make_context(cfg, stream, trust=0.1))
    assert np.array_equal(plain.x, named.x)
    assert plain.diagnostics == named.diagnostics and plain.diagnostics["trust"] == 0.1

    cell = SweepConfig(trigger_offset=((0.0,) * 6,))  # timing as in ``instance``: the same stream
    result = sweeps.run_sweep("offset", cfg, cell, seeds=[7], policies=["offline", "tap", "tap@0.1"])
    frame = result.frame
    a = frame[frame["policy"] == "tap"].drop(columns="policy").reset_index(drop=True)
    b = frame[frame["policy"] == "tap@0.1"].drop(columns="policy").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert (a["trust"] == 0.1).all()
    diag = result.diagnostics
    assert diag is not None and list(diag["policy"]) == ["tap", "tap@0.1"]
    assert diag["phase_share"].between(0.0, 1.0).all() and (diag["n_replans"] >= 0).all()
    assert diag["final_ratio"].between(RATIO_LO, RATIO_HI).all()


class _Watched(np.ndarray):
    """Demand array that logs the largest slot index every direct read touches."""

    def __new__(cls, values: np.ndarray, log: List[int]) -> "_Watched":
        obj = np.asarray(values, dtype=float).view(cls)
        obj.log = log
        obj.watch = True
        return obj

    def __array_finalize__(self, obj) -> None:
        self.log = getattr(obj, "log", None)
        self.watch = False

    def __getitem__(self, key):
        if getattr(self, "watch", False) and self.log is not None:
            n = self.shape[0]
            if isinstance(key, slice):
                stop = n if key.stop is None else (key.stop if key.stop >= 0 else key.stop + n)
                self.log.append(int(min(stop, n)) - 1)
            elif isinstance(key, (int, np.integer)):
                self.log.append(int(key) if key >= 0 else int(key) + n)
            else:
                self.log.append(n - 1)
        return super().__getitem__(key)


def test_tap_reads_no_demand_at_or_after_the_committing_minute() -> None:
    """Deciding ``x_{t+d}`` at minute ``t`` touches realised demand of slots ``< t`` only."""
    _, stream, ctx = instance()
    log: List[int] = []
    state = SimState(_Watched(stream.lam_in, log), _Watched(stream.lam_off, log),
                     np.zeros(stream.horizon_min))
    policy = engine.make_policy("tap", ctx, stream)
    policy.reset()
    policy.start_run()
    policy.prefix(ctx.lead_min)
    assert not log, "the pre-committed prefix must not look at the demand at all"
    for t in range(stream.horizon_min):
        log.clear()
        state.t = t
        state.x_now = 0.0
        policy.observe(t, state)
        assert max(log, default=-1) <= t - 1
    assert policy._adv.n_replans > 0  # the size correction ran on the watched arrays


def _load_run_sweep():
    """``scripts/sim/run_sweep.py`` as a module."""
    path = REPO / "scripts" / "sim" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("_tap_test_run_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_sweep_writes_one_diagnostics_row_per_tap_run(tmp_path: Path) -> None:
    """``<out>/diagnostics.csv``: case, tier, cell, seed, policy, phase_share, n_replans, ratio."""
    run_sweep = _load_run_sweep()
    argv = ["--sweep", "offset", "--college", COLLEGE, "--weeks", "1", "--max-sections", "12",
            "--offset-cells", "A", "C", "--seeds", "3", "--policies", "offline", "tap",
            "--learn-prior-from", PRIOR_SLOTS, "--root", str(REPO), "--case", "unit",
            "--out", str(tmp_path)]
    assert run_sweep.main(argv) == 0
    diag = pd.read_csv(tmp_path / "diagnostics.csv")
    assert list(diag.columns[:10]) == ["case", "tier", "lead_min", "cell", "seed", "policy",
                                       "phase_share", "n_replans", "final_ratio", "trust"]
    assert list(diag["cell"]) == ["A", "C"] and set(diag["case"]) == {"unit"}
    assert set(diag["tier"]) == {"d_inst"} and (diag["trust"] == 0.1).all()
    rows = pd.read_csv(tmp_path / "offset" / "offset.csv")
    assert set(rows["policy"]) == {"offline", "tap"}
