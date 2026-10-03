"""Closed forms of the paper's analysis: the calibration, the timetable inputs and the lemma bounds.

Everything here is arithmetic on the paper's analysis and nothing here reads a file or
draws anything, so the numbers the paper quotes can be checked one expression at a time.  Symbols
follow section 1 of the sketch:

``c``        USD per replica-minute; ``beta`` USD per replica start; ``mu`` requests per minute
             served by one replica.
``r``        the dimensionless price ratio ``p_in * mu / c``.  ``p_in`` is a dual variable rather
             than a measured price (R4), so every bound is reported as a function of ``r``.
``kappa``    ``p_in * mu - c = c (r - 1)``; ``kappa_off`` ``p_off * mu - c``.
``ell``      burst length ``2 / sigma_2`` in minutes, from the rectangular burst abstraction
             ``A_k * ell = n_k * rho``; ``ell_int`` is it rounded up to whole decision slots.
``m``        ``min(ell, d)``: how much of the burst a lead time of ``d`` minutes can still cover.
``A``        the per-bell amplitude of the R1 family, given in seat-starts and converted to
             requests per minute as ``n rho_window / ell`` (``rho_window = 1``: the first turn).

The two normalisations are the ones section 2 of the sketch locks: ``Q_max = (c ell + beta) |B| A
/ mu`` (every bell served at full amplitude) and ``Q_typ = Q_max / P2M_bell`` (the typical week).
``lower_over_qmax`` and ``upper_over_qmax`` are dimensionless in ``(r, m, ell, beta/c)`` alone,
which is why the headline interval is insensitive to the absolute calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

DEFAULT_R: Tuple[float, ...] = (6.0, 10.0, 30.0, 100.0)
DEFAULT_SIGMA2: Tuple[float, ...] = (0.13, 0.70, 0.90)
DEFAULT_D_MIN: Tuple[float, ...] = (0.6, 3.8, 8.0, 60.0)
DEFAULT_P_OFF_RATIO = 1.0 / 6.0
# H1 and H2 are stated with ">=", so an instance sitting exactly on the threshold satisfies them.
# The threshold is a ratio of calibrated floats, so it is compared with a slack of 1e-9
# replica-minutes; without it ``h2_min_r`` would hand back a value its own premise rejects.
SLACK = 1e-9


@dataclass(frozen=True)
class Calibration:
    """The six numbers section 7 of the paper's analysis asks for, plus the grids it reports over.

    The defaults are the PLACEHOLDER values of the sketch: ``c`` an A100 hour divided by 60,
    ``beta`` the idle time of a replica start plus a fixed term, ``mu`` the fluid capacity of one
    replica under the placeholder ``ServiceConfig``, and ``d`` the four provisioning tiers
    (``d_rep`` warm and first-run from results/q2, ``d_inst``, ``d_res``).
    """

    c: float = 0.05  # USD per replica-minute
    beta: float = 0.10  # USD per replica start
    mu: float = 296.0  # requests per minute per replica
    r: Tuple[float, ...] = DEFAULT_R  # p_in * mu / c
    p_off_ratio: float = DEFAULT_P_OFF_RATIO  # p_off / p_in
    sigma2: Tuple[float, ...] = DEFAULT_SIGMA2
    d_min: Tuple[float, ...] = DEFAULT_D_MIN
    s_min: float = 100.0  # minimum bell spacing, minutes
    rho_window: float = 1.0  # requests per seat inside the burst window: the first turn only
    placeholder: bool = True  # False once batch B calibration replaces the numbers above

    @property
    def beta_over_c(self) -> float:
        """``beta / c``, the switching cost in replica-minutes: the only place beta enters."""
        return self.beta / self.c

    def p_in(self, r: float) -> float:
        """``p_in`` implied by a price ratio: ``r * c / mu``."""
        return r * self.c / self.mu

    def p_off(self, r: float) -> float:
        """``p_off = (p_off / p_in) * p_in``; the graded ratio is held fixed across the scan."""
        return self.p_off_ratio * self.p_in(r)


@dataclass(frozen=True)
class Timetable:
    """The timetable quantities of the paper's analysis, as measured on the T2 tables."""

    bells: int  # |B|
    a_max: float  # A: max over bells of the week 5-16 maximum
    sum_median: float  # sum_tau of the per-bell medians (the typical week)
    sum_range: float  # sum_tau Delta_tau (R2 swing)
    sum_abs_dev: float  # the |S_w - median| variant of the swing, reported alongside
    step_sum: float  # sum_tau delta_tau of the R1.5 step (week 5 against week 4)
    week4_total: float  # week-4 seat-starts, the R1.5 baseline

    @property
    def p2m_bell(self) -> float:
        """``P2M_bell = |B| A / sum_tau median``: the full-amplitude week over the typical one."""
        return self.bells * self.a_max / self.sum_median

    @property
    def r2_share(self) -> float:
        """``sum_tau Delta_tau / sum_tau median``: the R2 swing as a share of the typical week."""
        return self.sum_range / self.sum_median

    @property
    def r15_share_typ(self) -> float:
        """``sum_tau delta_tau / sum_tau median``: the R1.5 step against the typical week."""
        return self.step_sum / self.sum_median

    @property
    def r15_share_week4(self) -> float:
        """``sum_tau delta_tau / week-4 total``: the R1.5 step against its own baseline."""
        return self.step_sum / self.week4_total


@dataclass(frozen=True)
class Inputs:
    """Everything one run of the instantiation depends on, with its provenance.

    The two CSV paths and the two week numbers are carried along so that the generated
    markdown can say where each timetable quantity came from without re-deriving it.
    """

    cal: Calibration
    tt: Timetable
    source: str
    dispersion_csv: str
    week_errors_csv: str
    step_week: int
    baseline_week: int


# --------------------------------------------------------------------------- primitives
def ell_of(sigma2: float) -> float:
    """Burst length ``ell = 2 / sigma_2`` in minutes (the rectangle of the analysis)."""
    if sigma2 <= 0.0:
        raise ValueError("sigma_2 must be positive")
    return 2.0 / sigma2


def ell_int_of(sigma2: float) -> int:
    """``ell`` in whole decision slots: ``ceil(2 / sigma_2)``."""
    return int(math.ceil(ell_of(sigma2) - 1e-12))


def m_of(ell: float, d: float) -> float:
    """``m = min(ell, d)``: the part of the burst a lead time of ``d`` can still cover."""
    return min(float(ell), float(d))


def kappa_of(c: float, r: float) -> float:
    """``kappa = p_in mu - c = c (r - 1)``."""
    return c * (r - 1.0)


def kappa_off_of(c: float, r: float, p_off_ratio: float) -> float:
    """``kappa'' = p_off mu - c = c (p_off_ratio * r - 1)``."""
    return c * (p_off_ratio * r - 1.0)


def lipschitz_x(c: float, r: float) -> float:
    """``L_x = max(c, p_in mu - c)``, the Lipschitz constant in the capacity."""
    return max(c, kappa_of(c, r))


# --------------------------------------------------------------------------- lemma 1
def amplitude_of(seats: float, ell: float, rho_window: float) -> float:
    """``A = n rho_window sigma_2 / 2 = n rho_window / ell`` requests per minute for ``n`` seat-starts.

    The rectangle of the paper's analysis holds ``rho_window`` requests per seat over ``ell = 2 / sigma_2``
    minutes.  ``rho_window = 1`` counts the first turn of each conversation; the second turn arrives
    170 s later on average (median 118 s) and belongs to the in-session background ``b^in``.
    """
    if ell <= 0.0:
        raise ValueError("ell must be positive")
    return float(seats) * float(rho_window) / float(ell)


def lemma1_lower_bell(cal: Calibration, r: float, ell: float, d: float, a_max: float) -> float:
    """Per-bell lower bound ``(c A / mu)(kappa m - beta)/(c + kappa)`` at ``b_off = 0``.

    ``a_max`` is in seat-starts; it is converted to requests per minute with ``amplitude_of``.

    This is the R1 branch of lemma 1: no history to lean on (``Abar = 0``), a deterministic
    calendar-oblivious algorithm, an adaptive adversary.  The value is the excess over OPT that the
    adversary can force at one bell; it is only a bound when A2 holds, so callers report
    ``a2_holds`` next to it rather than clipping a negative number.
    """
    kappa = kappa_of(cal.c, r)
    a_rate = amplitude_of(a_max, ell, cal.rho_window)
    return (cal.c * a_rate / cal.mu) * (kappa * m_of(ell, d) - cal.beta) / (cal.c + kappa)


def lower_over_qmax(beta_over_c: float, r: float, ell: float, d: float) -> float:
    """``(kappa m - beta)/((c + kappa)(ell + beta/c))``, dimensionless in ``(r, m, ell, beta/c)``."""
    m = m_of(ell, d)
    return ((r - 1.0) * m - beta_over_c) / (r * (ell + beta_over_c))


def upper_over_qmax(beta_over_c: float, ell: float, d: float) -> float:
    """``(m + beta/c)/(ell + beta/c)``: the standing-guard upper bound, and the R2 / R1.5 factor."""
    return (m_of(ell, d) + beta_over_c) / (ell + beta_over_c)


def q_max(cal: Calibration, ell: float, tt: Timetable) -> float:
    """``Q_max = (c ell + beta) |B| A / mu``: the calendar-attributable cost at full amplitude.

    ``A`` is the amplitude in requests per minute, ``amplitude_of(tt.a_max, ell, rho_window)``.
    """
    return (cal.c * ell + cal.beta) * tt.bells * amplitude_of(tt.a_max, ell, cal.rho_window) / cal.mu


def q_typ(cal: Calibration, ell: float, tt: Timetable) -> float:
    """``Q_typ = (c ell + beta) sum_tau median / mu = Q_max / P2M_bell`` (medians converted alike)."""
    return ((cal.c * ell + cal.beta) * amplitude_of(tt.sum_median, ell, cal.rho_window)
            / cal.mu)


def a2_holds(cal: Calibration, r: float, ell: float, d: float) -> bool:
    """A2: ``kappa m > beta``.  Without it the R1 lower bound is vacuous (theta* <= 0)."""
    return kappa_of(cal.c, r) * m_of(ell, d) > cal.beta


# --------------------------------------------------------------------------- lemma H
def a1_holds(r: float, p_off_ratio: float) -> bool:
    """A1: ``p_off mu > c``, i.e. ``r > 1 / (p_off / p_in)`` - at the graded 1/6, ``r > 6``."""
    return p_off_ratio * r > 1.0


def h1_holds(cal: Calibration, ell: float) -> bool:
    """H1: ``c (s - ell) >= beta`` - holding a unit through a gap beats restarting for it."""
    return cal.c * (cal.s_min - ell) >= cal.beta - SLACK * cal.c


def h2_holds(cal: Calibration, r: float, ell: float) -> bool:
    """H2: ``(p_off mu - c) ell >= beta`` - the off-class background is worth a start.

    Inclusive at the threshold ``h2_min_r`` returns, up to ``SLACK`` replica-minutes.
    """
    return kappa_off_of(cal.c, r, cal.p_off_ratio) * ell >= cal.beta - SLACK * cal.c


def h2_min_r(beta_over_c: float, ell: float, p_off_ratio: float = DEFAULT_P_OFF_RATIO) -> float:
    """Smallest ``r`` satisfying H2: ``(1 + (beta/c)/ell) / (p_off / p_in)``.

    At the graded ratio 1/6 this is the ``6 (1 + (beta/c)/ell)`` of the analysis: with
    ``beta/c = 1`` and ``ell = 3`` slots it is 8, with ``beta/c = 10`` it is 26.
    """
    return (1.0 + beta_over_c / ell) / p_off_ratio


# --------------------------------------------------------------------------- lemma A
def lemma_a_constant(cal: Calibration, r: float, d: float) -> float:
    """Lemma A's competitive constant ``3 + 2 L_x d / beta`` for ROB(d) = LCP lagged by ``d``."""
    return 3.0 + 2.0 * lipschitz_x(cal.c, r) * d / cal.beta


__all__ = [
    "amplitude_of","Calibration", "DEFAULT_D_MIN", "DEFAULT_P_OFF_RATIO", "DEFAULT_R", "DEFAULT_SIGMA2",
           "SLACK",
           "Inputs", "Timetable", "a1_holds", "a2_holds", "ell_int_of", "ell_of", "h1_holds", "h2_holds",
           "h2_min_r", "kappa_off_of", "kappa_of", "lemma1_lower_bell", "lemma_a_constant",
           "lipschitz_x", "lower_over_qmax", "m_of", "q_max", "q_typ", "upper_over_qmax"]
