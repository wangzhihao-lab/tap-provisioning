"""Policy 11: the bell-aware guard, the standing-guard witness S of lemma 1's upper bound.

S knows the *family* of the paper's analysis and nothing of the instance: the bell grid (the minutes
classes start on), the burst profile every bell shares (the normalised mean forecast shape after
a bell) and the family's largest amplitude ``A_max``.  It never reads the per-bell forecast, so it
cannot tell a large bell from an empty one.  At every bell it stands guard at ``A_max`` times the
profile for the minutes it cannot yet have observed - the lead time plus the one minute the
simulator takes to reveal a minute's arrivals - then corrects downward to the amplitude it has
observed for the rest of the profile window, and everywhere else tracks the revealed total demand
with the lead delay.  Lemma 1 (upper) charges S ``(c m + beta)(A_max - A_tau)/mu`` per bell plus
background terms, with ``m = min(ell, d)``; in the simulator the blind time is ``d + 1`` minutes.
"""

import bisect
import logging
import math
from typing import List, Optional

import numpy as np

from .. import register_policy
from .base import Policy, PolicyContext, SimState

logger = logging.getLogger(__name__)

PROFILE_MIN = 45  # minutes after a bell over which the family's burst profile is defined
BACKGROUND_MIN = 3  # revealed minutes the background level is taken over (their maximum)
PROFILE_FLOOR = 0.05  # profile values below this are not used to estimate an amplitude
MIN_OBSERVED = 2  # burst minutes that must be revealed before the guard may shrink


@register_policy("guard")
class GuardPolicy(Policy):
    """Stand guard at the family's largest amplitude on every bell; track the rest."""

    name = "guard"

    def __init__(self, ctx: PolicyContext) -> None:
        self.bells: List[int] = [int(b) for b in (ctx.bells if ctx.bells is not None else [])]
        self.profile, self.a_max = self._family(ctx)
        super().__init__(ctx)
        logger.debug("guard: %d bells, A_max %.1f req/min, profile peak at minute %d",
                     len(self.bells), self.a_max, int(np.argmax(self.profile)))

    @staticmethod
    def _family(ctx: PolicyContext) -> "tuple[np.ndarray, float]":
        """Normalised mean burst profile and the largest forecast peak over the bell grid."""
        shape = np.zeros(PROFILE_MIN)
        count, a_max = 0, 0.0
        for tau in (ctx.bells if ctx.bells is not None else []):
            seg = np.asarray(ctx.lam_hat_in[int(tau): int(tau) + PROFILE_MIN], dtype=float)
            if seg.size < PROFILE_MIN or seg.max() <= 0:
                continue
            shape += seg / seg.max()
            count += 1
            a_max = max(a_max, float(seg.max()))
        profile = shape / count if count else np.zeros(PROFILE_MIN)
        if profile.max() > 0:  # triggers peak on different minutes; the guard is A_max at the peak
            profile = profile / profile.max()
        return profile, a_max

    def reset(self) -> None:
        """Forget the per-bell background and amplitude estimates."""
        self._bg: dict = {}

    def _bell_before(self, u: int) -> Optional[int]:
        """The last bell at or before minute ``u`` within the profile window, else ``None``."""
        i = bisect.bisect_right(self.bells, u) - 1
        if i < 0:
            return None
        tau = self.bells[i]
        return tau if u - tau < PROFILE_MIN else None

    def _background(self, state: SimState) -> float:
        """Largest revealed total demand over the last few minutes (the model's known ``b_t``)."""
        past = state.past_total(BACKGROUND_MIN)
        return float(past.max()) if past.size else 0.0

    def _observed_amplitude(self, tau: int, state: SimState) -> Optional[float]:
        """Cumulative ``sum lambda_in / sum profile`` over the revealed minutes of the window.

        ``None`` until ``MIN_OBSERVED`` informative minutes are in: one minute of a jittered
        burst under-sizes it, and the lemma charges the guard for over-provisioning only.
        """
        hi = min(state.t, tau + PROFILE_MIN)
        num, den, n = 0.0, 0.0, 0
        for j in range(tau, hi):
            k = j - tau
            if self.profile[k] < PROFILE_FLOOR:
                continue
            num += float(state.lam_in[j])
            den += float(self.profile[k])
            n += 1
        return num / den if n >= MIN_OBSERVED and den > 0 else None

    def _guard_level(self, demand_per_min: float) -> float:
        """Whole replicas covering ``demand_per_min``: the guard commits fully."""
        return self.clip(math.ceil(demand_per_min / max(1e-9, self.ctx.mu) - 1e-9))

    def _track_level(self, demand_per_min: float) -> float:
        """Penalty-aware replicas for a level the policy has observed rather than guarded."""
        return self.clip(self.replicas_for(demand_per_min))

    def _in_window(self, tau: int, k: int, state: SimState) -> float:
        """Target ``k`` minutes after bell ``tau``: guard, or the observed amplitude, if known."""
        if tau not in self._bg:  # the background in force just before the bell
            self._bg[tau] = self._background(state)
        bg = self._bg[tau]
        guard = self._guard_level(bg + self.a_max * float(self.profile[k]))
        if k < self.ctx.lead_min + 1:
            return guard
        amp = self._observed_amplitude(tau, state)
        if amp is None:
            return guard
        # Shrink towards the observed burst, but never below what the last revealed minutes
        # already showed: the realised decay is fatter than the family profile, and the lemma
        # charges the guard for over-provisioning only.
        seen = self._track_level(self._background(state))
        return min(guard, max(seen, self._track_level(bg + amp * float(self.profile[k]))))

    def observe(self, t: int, state: SimState) -> float:
        """Target for minute ``t + d``: guard, observed-amplitude tracking, or background."""
        u = t + self.ctx.lead_min
        tau = self._bell_before(u)
        if tau is None:
            return self._track_level(self._background(state))
        return self._in_window(tau, u - tau, state)

    def block_target(self, start_slot: int, end_slot: int) -> float:
        """Level held over a commitment block: the largest guard any bell in it calls for.

        Decided ``lead_min`` minutes before the block starts, so a bell inside the block is
        still unobserved and gets the family maximum over its overlap with the block; a bell
        whose window reaches into the block from before it uses its observed amplitude when
        enough of it has been revealed.  Between windows the block holds the background level.
        """
        state = self._block_state
        level = self._track_level(self._background(state)) if state is not None else self.initial_level()
        for tau in self.bells:
            lo, hi = max(start_slot, tau), min(end_slot, tau + PROFILE_MIN)
            if hi <= lo:
                continue
            ks = np.arange(lo - tau, hi - tau)
            peak = float(self.profile[ks].max())
            if state is None:
                level = max(level, self._guard_level(self.a_max * peak))
                continue
            bg = self._bg.get(tau, self._background(state))
            amp = self._observed_amplitude(tau, state) if tau < state.t else None
            if amp is None:
                level = max(level, self._guard_level(bg + self.a_max * peak))
            else:
                level = max(level, self._track_level(bg + amp * peak))
        return level

    _block_state: Optional[SimState] = None

    def decide(self, t: int, state: SimState) -> float:
        """``observe`` plus the state, which ``block_target`` needs for its bell windows."""
        self._block_state = state
        return super().decide(t, state)


__all__ = ["GuardPolicy"]
