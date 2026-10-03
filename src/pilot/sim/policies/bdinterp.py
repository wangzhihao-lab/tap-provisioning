"""Policy 7: BDINTERP-d, black-box interpolation with the decision committed ``d`` slots ahead.

the BDINTERP-d analysis adapts Algorithm 3 of Christianson, Handina and
Wierman (2022) to a lead time: the decision for slot ``t`` is committed at ``t - d``, so the phase
test and the contraction amount may only use costs the committing minute has already seen.  The two
black boxes are the simulator's own policies:

``ADV``  policy 6 (``calendar``): ``x~`` is the integer optimum of the calendar forecast, known for
         the whole horizon before minute 0.
``ROB``  policy 5 (``lcp``): ``s_t`` is the lazy projection computed at minute ``t - d``, which is
         exactly when this policy needs it.

Both run *inside* this policy, driven by the state object the engine hands it, so their sequences
are the ones policies 5 and 6 produce on their own (neither reads ``state.x_now``).

One slot, for slot ``u = t + d`` committed at minute ``t``:

1. take ``x~_u`` and ``s_u``;
2. phase test: ``C_ROB(1, .) >= delta * C_ADV(1, .)`` -> ``x_u = x~_u`` (the advice is worth its
   price), otherwise
3. ``nu = ||x_{u-1} - s_{u-1}|| / ||x~_{u-1} - s_{u-1}||`` (0 when the denominator vanishes),
   ``y_u = nu x~_u + (1 - nu) s_u``, and ``x_u`` is ``y_u`` moved ``gamma q_u`` of metric length
   towards ``s_u`` without crossing it.

The committed value is that fluid decision rounded *up* to whole replicas and clipped to
``[min_replicas, x_max]`` - the same ``as_int_plan`` rule the engine applies to every other policy,
so the integrality remark of the paper's analysis covers this policy too.  ``x_fluid`` keeps the fluid
decision the theory analyses.

The metric is the switching cost ``||a - b|| = (beta/2) |a - b|`` of the BDINTERP-d analysis, ``q_u`` is
the one-slot proxy ``C_ADV`` of the newest revealed slot, and ``gamma = delta = eps / 4`` for the
trust knob ``eps = ctx.trust``: ``eps -> 0`` is pure advice, large ``eps`` is pure LCP.

Two departures from the BDINTERP-d analysis, both forced by the information model:

* **one slot of extra lag.**  Section 2 lets the decision for slot ``t`` use ``f_1..f_{t-d}``.  The
  simulator reveals minute ``m`` only once it is over, so minute ``t - d`` knows ``f_1..f_{t-d-1}``;
  the phase test and ``q`` therefore run on that prefix.  This is BDINTERP-d with ``d -> d + 1``,
  so lemma B applies verbatim with an additive ``O((d+1) c_max)`` instead of ``O(d c_max)``.
* **accounting metric.**  ``C_ADV`` and ``C_ROB`` are accumulated with the metric switching cost
  ``(beta/2)|dx|``, while the engine scores a run with the one-sided ``beta (dx)^+``; the BDINTERP-d analysis
  section 4.3 is the conversion, worth at most ``beta x_max / 2`` per sequence.

A free-switching instance (``beta_start = 0``) degenerates: the metric collapses, the contraction
saturates at once and the policy is LCP after its first priced slot.
"""

import logging
import math
from typing import List, Sequence, Tuple

import numpy as np

from .. import register_policy
from .base import Policy, PolicyContext, SimState, as_int_replicas, hit_cost
from .calendar import CalendarFtPPolicy
from .lcp import LazyCapacityPolicy

logger = logging.getLogger(__name__)

_TINY = 1e-12
ADV_PHASE, ROB_PHASE = "ADV", "ROB"


def _at(seq: Sequence[float], i: int) -> float:
    """``seq[i]`` with the convention that everything before slot 0 is an empty pool."""
    return float(seq[i]) if 0 <= i < len(seq) else 0.0


def _take(arr: np.ndarray, i: int) -> float:
    """``arr[i]``, edge-clamped (the engine pads a short prefix the same way), 0 when empty."""
    return float(arr[min(i, arr.size - 1)]) if arr.size else 0.0


@register_policy("trust")  # legacy name: the sweeps and CSVs written before the BDINTERP-d analysis landed
@register_policy("bdinterp")
class BDInterpPolicy(Policy):
    """Policy 7: BDINTERP-d over (calendar advice, LCP), trust knob ``eps``."""

    name = "bdinterp"

    def __init__(self, ctx: PolicyContext) -> None:
        self._adv = self.make_advice(ctx)
        self._rob = LazyCapacityPolicy(ctx)
        super().__init__(ctx)

    def make_advice(self, ctx: PolicyContext) -> CalendarFtPPolicy:
        """The ADV black box: policy 6 here; ``tap`` (``tap.py``) swaps in its own advice."""
        return CalendarFtPPolicy(ctx)

    # --- knobs -------------------------------------------------------------------------------
    @property
    def eps(self) -> float:
        """Trust knob: ``0`` follows the calendar, large values follow LCP."""
        return max(0.0, float(self.ctx.trust))

    @property
    def gamma(self) -> float:
        """Contraction rate; ``gamma = delta = eps / 4``."""
        return 0.25 * self.eps

    @property
    def delta(self) -> float:
        """Phase-test threshold; equal to ``gamma`` by the choice ``2 gamma + 2 delta = eps``."""
        return 0.25 * self.eps

    @property
    def half_beta(self) -> float:
        """The metric factor ``beta / 2``; floored so that a free-switching instance still runs."""
        return max(0.5 * self.ctx.cost.beta_start, _TINY)

    @property
    def diameter(self) -> float:
        """``D = beta x_max / 2``: the bound on ``||x~_t - s_t||`` lemma B's robustness term uses."""
        return 0.5 * self.ctx.cost.beta_start * self.ctx.cost.x_max

    # --- per-run state -----------------------------------------------------------------------
    def reset(self) -> None:
        """Clear both components, the running costs and the recorded trajectory."""
        self._rob.reset()
        self._c_adv = 0.0
        self._c_rob = 0.0
        self._q = 0.0
        self._settled = 0
        self._adv_seq: List[float] = []
        self._rob_seq: List[float] = []
        self.x_fluid: List[float] = []
        self.x_int: List[float] = []
        self.phase: List[str] = []
        logger.debug("bdinterp eps=%.3f gamma=delta=%.3f D=%.2f", self.eps, self.gamma,
                     self.diameter)

    # --- cost accounting ---------------------------------------------------------------------
    def _slot_cost(self, state: SimState, u: int, seq: Sequence[float]) -> float:
        """``C_A(u, u) = f_u(A_u) + ||A_u - A_{u-1}||`` on the *realised* demand of slot ``u``.

        The always-on base tier (``PolicyConfig.min_replicas``) is a sunk cost that both branches
        pay whatever they decide, so its running cost is left out of the phase test: with it in,
        ``C_ADV / C_ROB`` is pinned near 1 by the shared constant and the robust branch never
        engages even under a twenty-fold over-forecast (batch C attempt 3, 2026-09-18).  The
        guarantee of the BDINTERP-d analysis is unaffected: it is a statement about the decision-dependent cost,
        to which a common additive constant adds the same amount on both sides.
        """
        cur, prev = _at(seq, u), _at(seq, u - 1)
        f = hit_cost(np.array([cur]), state.lam_in[u:u + 1], state.lam_off[u:u + 1], self.ctx.mu,
                     self.ctx.cost)
        sunk = self.ctx.cost.c_run * float(self.ctx.policy.min_replicas)
        return max(0.0, float(f[0]) - sunk) + self.half_beta * abs(cur - prev)

    def _settle(self, state: SimState, upto: int) -> None:
        """Fold every slot revealed since the last call into ``C_ADV``, ``C_ROB`` and ``q``."""
        for u in range(self._settled, max(self._settled, upto)):
            q = self._slot_cost(state, u, self._adv_seq)
            self._c_adv += q
            self._c_rob += self._slot_cost(state, u, self._rob_seq)
            self._q = q  # q_u = C_ADV of the newest revealed slot (the delayed proxy)
        self._settled = max(self._settled, upto)

    # --- one slot of BDINTERP-d ----------------------------------------------------------------
    def _decide(self, adv: float, rob: float) -> Tuple[float, str]:
        """The fluid decision for the next slot, and the phase it was taken in."""
        if self._c_rob >= self.delta * self._c_adv:
            return adv, ADV_PHASE
        adv_prev = _at(self._adv_seq, len(self._adv_seq) - 1)
        rob_prev = _at(self._rob_seq, len(self._rob_seq) - 1)
        x_prev = _at(self.x_fluid, len(self.x_fluid) - 1)
        span = abs(adv_prev - rob_prev)
        nu = 0.0 if span <= _TINY else min(1.0, abs(x_prev - rob_prev) / span)
        y = nu * adv + (1.0 - nu) * rob
        shrink = self.gamma * self._q / self.half_beta  # metric length -> replicas
        return rob + math.copysign(max(abs(y - rob) - shrink, 0.0), y - rob), ROB_PHASE

    def _commit(self, adv: float, rob: float) -> float:
        """Decide one slot, record the fluid and the integer value, return the integer one."""
        x, phase = self._decide(float(adv), float(rob))
        self._adv_seq.append(float(adv))
        self._rob_seq.append(float(rob))
        self.x_fluid.append(float(x))
        self.phase.append(phase)
        self.x_int.append(self.clip(as_int_replicas(x, self.ctx.cost)))
        return self.x_int[-1]

    # --- policy interface ----------------------------------------------------------------------
    def prefix(self, lead: int) -> np.ndarray:
        """Slots ``0..d-1``: nothing is revealed yet, so the phase test returns the advice.

        ``C_ROB(1, .) = C_ADV(1, .) = 0`` satisfies the test, which is the same pre-commitment
        policy 6 makes - the timetable is known before minute 0 and the robust component has seen
        no demand at all.
        """
        adv, rob = self._adv.prefix(lead), self._rob.prefix(lead)
        return np.asarray([self._commit(_take(adv, u), _take(rob, u))
                           for u in range(max(0, lead))], dtype=float)

    def observe(self, t: int, state: SimState) -> float:
        """Commit slot ``t + d`` from what minute ``t`` has revealed (slots ``< t``)."""
        self._settle(state, t)
        adv = self._adv.observe(t, state)
        rob = self._rob.observe(t, state)
        return self._commit(adv, rob)

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """Block level: ADV over the whole block, ROB frozen at the block's decision minute.

        The two black boxes know different amounts at that minute.  ADV is policy 6, whose plan for
        the whole horizon exists before minute 0, so the block may use ``x~_u`` for every slot of
        it.  ROB is policy 5, a feedback rule that has seen nothing past the decision minute, so
        ``s_u`` is frozen at the value it had just produced.  The phase test, the contraction proxy
        ``q`` and the running costs are frozen with it (``_decide`` reads them and mutates
        nothing), which is what makes the answer depend on the decision minute alone.
        """
        rob = _at(self._rob_seq, len(self._rob_seq) - 1)
        plan = self._adv.trajectory()
        level = _at(self.x_fluid, len(self.x_fluid) - 1)  # the slot already committed in this block
        for u in range(int(start_slot), max(int(end_slot), int(start_slot) + 1)):
            level = max(level, self._decide(_take(plan, u), rob)[0])
        return float(level)

    # --- diagnostics ----------------------------------------------------------------------------
    def phase_share(self) -> float:
        """Fraction of committed slots taken in the ADV phase (1.0 = the calendar was followed)."""
        return (sum(p == ADV_PHASE for p in self.phase) / len(self.phase)) if self.phase else 1.0


__all__ = ["ADV_PHASE", "ROB_PHASE", "BDInterpPolicy"]
