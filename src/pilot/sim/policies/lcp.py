"""Policy 5: lazy capacity provisioning (Lin et al. 2011), lagged by the lead time ``d``.

LCP keeps the current capacity while it stays inside a moving interval ``[x_L, x_U]`` and projects
onto that interval otherwise.  Lin, Wierman, Andrew and Thereska define the two bounds through the
*offline* problem on the revealed prefix ``1..t``:

    x_t^L  = last component of the optimum of  sum_tau [ f_tau(x_tau) + beta (x_tau - x_{tau-1})^+ ]
    x_t^U  = last component of the optimum of  sum_tau [ f_tau(x_tau) + beta (x_{tau-1} - x_tau)^+ ]

both started from ``x_0 = 0``.  The lower bound pays to switch *on*, so it is reluctant to grow;
the upper bound pays to switch *off*, so it is reluctant to shrink, and ``x_t^L <= x_t^U``.
LCP(0) is then ``x_t = clamp(x_{t-1}, x_t^L, x_t^U)``.

``LazyCapacityPolicy`` computes both bounds exactly by carrying the two prefix value functions over
integer replica counts forward one slot at a time.  A slot costs one pass over the state grid - the
same prefix-min / suffix-min recurrence ``offline.round_to_replicas`` uses - so a whole run is
``O(T * X)`` with no per-slot search, and a two-week institution-wide run stays inside a couple of
seconds.

``GridLazyCapacityPolicy`` is the approximation this policy replaced: the window start was
maximised over a geometric grid of lookbacks, the inner minimisation taken over a bounded grid of
demand breakpoints, and the capacity held constant inside the window.  It stays registered as
``lcp_grid`` because ``tests/test_sim_lcp_fidelity.py`` measures how far it lands from the paper's
algorithm, which is what motivated the replacement.
"""

import logging
from typing import Tuple

import numpy as np

from .. import register_policy
from .base import Policy, PolicyContext, SimState, hit_cost

logger = logging.getLogger(__name__)

_LOOKBACKS = (1, 2, 3, 5, 10, 20, 40, 60)
_MAX_STATES = 2000  # box bound on the state grid; CostConfig.x_max caps it further


@register_policy("lcp")
class LazyCapacityPolicy(Policy):
    """Exact LCP(0): lazy projection onto the two offline bounds of the revealed prefix."""

    name = "lcp"

    def reset(self) -> None:
        """Forget the current capacity and both prefix value functions."""
        self._x = 0.0
        self._settled = 0
        self._g: np.ndarray = np.empty(0)  # pay-to-switch-on prefix problem  -> x_L
        self._h: np.ndarray = np.empty(0)  # pay-to-switch-off prefix problem -> x_U
        n_x = int(max(1.0, min(self.ctx.cost.x_max, float(_MAX_STATES)))) + 1
        self._states = np.arange(n_x, dtype=float)

    def _fold(self, state: SimState, upto: int) -> None:
        """Advance both value functions over the slots revealed since the last call."""
        beta = self.ctx.cost.beta_start
        st = self._states
        for u in range(self._settled, max(self._settled, upto)):
            f = hit_cost(st, float(state.lam_in[u]), float(state.lam_off[u]), self.ctx.mu,
                         self.ctx.cost)
            if self._g.size == 0:
                self._g = f + beta * st  # from x_0 = 0, reaching x costs beta * x
                self._h = f.copy()  # switching off is free on the way up from x_0 = 0
                continue
            up = np.minimum.accumulate(self._g - beta * st) + beta * st
            down = np.minimum.accumulate(self._g[::-1])[::-1]
            self._g = f + np.minimum(up, down)
            keep = np.minimum.accumulate(self._h)
            shed = np.minimum.accumulate((self._h + beta * st)[::-1])[::-1] - beta * st
            self._h = f + np.minimum(keep, shed)
        self._settled = max(self._settled, upto)

    def _bounds(self) -> Tuple[float, float]:
        """``(x_L, x_U)`` of the revealed prefix; ties break low for ``x_L``, high for ``x_U``."""
        if self._g.size == 0:
            return 0.0, 0.0
        lo = float(np.argmin(self._g))
        hi = float(self._h.size - 1 - int(np.argmin(self._h[::-1])))
        return lo, max(lo, hi)

    def observe(self, t: int, state: SimState) -> float:
        """Project the current capacity onto the interval."""
        self._fold(state, t)
        lo, hi = self._bounds()
        self._x = float(np.clip(self._x if t > 0 else 0.0, lo, hi))
        return self.clip(self.replicas_for(self._x * self.ctx.mu))


@register_policy("lcp_grid")
class GridLazyCapacityPolicy(Policy):
    """The superseded approximation: constant capacity over a geometric grid of lookbacks."""

    name = "lcp_grid"

    def __init__(self, ctx: PolicyContext) -> None:
        super().__init__(ctx)
        self._windows = tuple(w for w in _LOOKBACKS if w <= max(1, ctx.policy.lcp_lookback_min))

    def reset(self) -> None:
        """Forget the current capacity."""
        self._x = 0.0

    def _bounds(self, state: SimState) -> tuple:
        """``(x_L, x_U)``: the interval justified by the recent revealed demand."""
        cost, mu = self.ctx.cost, self.ctx.mu
        span = max(self._windows)
        lam_in = state.past_in(span)
        lam_all = state.past_total(span)
        if lam_in.size == 0:
            return 0.0, 0.0
        cand = np.unique(np.concatenate([[0.0], lam_in / mu, lam_all / mu]))
        if cand.size > self.ctx.policy.lcp_candidates:
            idx = np.linspace(0, cand.size - 1, self.ctx.policy.lcp_candidates)
            cand = cand[np.round(idx).astype(int)]
        served = mu * cand[:, None]
        short_in = np.maximum(lam_in[None, :] - served, 0.0)
        short_all = np.maximum(lam_all[None, :] - served, 0.0)
        hit = (cost.c_run * cand[:, None] + (cost.p_in - cost.p_off) * short_in
               + cost.p_off * short_all)
        rev = np.cumsum(hit[:, ::-1], axis=1)[:, ::-1]
        lo, hi = 0.0, 0.0
        for w in self._windows:
            if w > lam_in.size:
                continue
            col = rev[:, lam_in.size - w]
            lo = max(lo, float(cand[int(np.argmin(col))]))
            hi = max(hi, float(cand[int(np.argmin(col - cost.beta_start * cand))]))
        return lo, max(lo, hi)

    def observe(self, t: int, state: SimState) -> float:
        """Project the current capacity onto the interval."""
        lo, hi = self._bounds(state)
        self._x = float(np.clip(self._x if t > 0 else 0.0, lo, hi))
        return self.clip(self.replicas_for(self._x * self.ctx.mu))


__all__ = ["GridLazyCapacityPolicy", "LazyCapacityPolicy"]
