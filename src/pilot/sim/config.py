"""Frozen configuration objects for the provisioning simulator (batch C).

Four families, all immutable:

``WorkloadConfig``   calendar slice -> arrival stream, and the calendar forecast lambda-hat.
``ServiceConfig``    one vLLM replica: concurrency limit, decode rate, TTFT curve, lead times.
``PolicyConfig``     thresholds and horizons shared by the provisioning policies.
``SweepConfig``      the sigma / calendar-error x price-ratio / d / trust (BDINTERP eps) / margin /
                     adoption grids.

``RunConfig`` bundles the four plus the master seed; the engine derives three RNG streams from it
(workload / service / policy) so that a comparison is paired: the same arrival stream is replayed
for every policy.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

# hour-of-day weights for off-class self-paced traffic (same shape as pilot.sources.mock)
DIURNAL: Tuple[float, ...] = (
    0.2, 0.1, 0.05, 0.02, 0.02, 0.05, 0.3, 0.8, 1.2, 1.5, 1.6, 1.4,
    1.0, 1.3, 1.6, 1.7, 1.5, 1.2, 1.4, 2.0, 2.4, 2.2, 1.5, 0.7,
)

# the two pairs of empirical arrival-offset pools ``WorkloadConfig.offset_source`` chooses between
OFFSET_SOURCES: Tuple[str, ...] = ("signin", "ai")


@dataclass(frozen=True)
class LeadTier:
    """One provisioning tier: availability delay, commitment granularity, billing granularity.

    These are three different constraints and the simulator keeps them apart.  ``lead_min`` is how
    long after a decision the capacity becomes usable.  ``commit_block_min`` is the length of the
    blocks the capacity has to be piecewise-constant on, each decided ``lead_min`` before it
    starts.  ``billing_block_min`` is the smallest chunk a replica is charged for.  A rented
    instance has a real lead time but per-minute commitment and billing; a reserved hour has the
    same lead time as the instance behind it and an hour of commitment *and* billing on top.
    """

    lead_min: float  # minutes between a decision and the capacity being usable
    commit_block_min: int = 1  # capacity is piecewise-constant on blocks of this length
    billing_block_min: int = 1  # minimum billed duration of one replica-block


# provisioning tiers.  d_rep is the replica start (batch B median, 25-40 s for
# a 7B-14B model) and d_inst the cloud instance acquisition; both are decided and billed per
# minute.  d_res is the *reserved* tier: the same 8 min of instance lead time, but the pool size is
# committed one hour ahead in whole hours and each committed replica-hour is paid in full.  The
# lead times are a PLACEHOLDER until batch B lands; they are the T3 dry-run medians rounded up
# (7B weights 12 s, server listening 25 s, first completion 27 s -> 0.5 min).
LEAD_TIERS: Dict[str, LeadTier] = {
    "d_rep": LeadTier(lead_min=0.5),
    "d_inst": LeadTier(lead_min=8.0),
    "d_res": LeadTier(lead_min=8.0, commit_block_min=60, billing_block_min=60),
}


@dataclass(frozen=True)
class WorkloadConfig:
    """Calendar slice, arrival mixture and the error sources that separate lambda from lambda-hat."""

    # --- calendar slice -------------------------------------------------------------------
    slots_csv: str = "data/timetable/slots_2026-2027-1.csv"
    stale_slots_csv: str = "data/timetable/slots_2025-2026-2.csv"  # policy 9: last semester
    college: Optional[str] = None
    campus: Optional[str] = None
    weeks: Tuple[int, ...] = (1, 2, 3, 4)
    week1_monday: str = "2026-09-07"  # a Monday; teaching week 1
    adoption: float = 0.10  # fraction of sections that use the TA service
    max_sections: int = 0  # 0 = no cap (cap is applied before adoption sampling)

    # --- triggers inside one session -----------------------------------------------------
    triggers_per_session: int = 2
    trigger_jitter_min: float = 3.0  # t_k = tau_k + U(0, jitter)
    period_minutes: float = 45.0

    # --- arrival mixture ------------------------------------------------------------------
    # ``sigma`` is the *mixing weight* w, not the measured synchrony: it is the probability that a
    # student is drawn from the prompted (sign-in) offset pool rather than the self-paced
    # (ASSISTments in-class) one.  The quantity the paper plots, sigma_2 - the share of per-student
    # first arrivals within 2 min of the session's first arrival - is measured on the generated
    # stream by ``metrics.measure_synchrony`` and recorded as ``sigma2_measured`` in every sweep
    # row.  The field keeps its name so that older CLIs, sweeps and CSVs stay valid;
    # ``sync_share`` is the canonical name for it.
    sigma: float = 0.70  # w: share of students drawn from the prompted (sign-in) pool
    pool_cap_min: Optional[float] = None  # offset cap; None = the session length ``period_minutes``
    # Which pair of empirical offset pools the mixture samples.  ``"signin"`` (the default) is the
    # measured pair the pilot quotes sigma_2 in: classroom sign-ins for the prompted side and
    # ASSISTments in-class for the self-paced one.  ``"ai"`` switches to the AI-assistant pools of
    # batch A (genuine platform exports of 2026-09-17; a 2-minute share of 0.84 against 0.70 on
    # the prompted side and 0.20 against 0.24 on the self-paced one).  The sign-in pair stays the
    # default so that every sweep of batch C is on one pair; the ``pool`` sweep compares them.
    offset_source: str = "signin"  # "signin" | "ai"
    signin_offsets_csv: str = "data/offsets/signin_offsets_min.csv"
    assistments_offsets_csv: str = "data/offsets/assistments_inclass_offsets_min.csv"
    prompted_offsets_csv: str = "data/offsets/prompted_ai_offsets_min.csv"
    selfpaced_offsets_csv: str = "data/offsets/selfpaced_ai_offsets_min.csv"

    # --- requests per arriving student ----------------------------------------------------
    # Calibrated on the genuine Xuexitong AI-assistant exports of 2026-09-17
    # (results/q1/assistant/calibration.json): 2.01 student turns per session in the prompted
    # theory class; the gap between two questions of one student has median 118 s in class, and an
    # exponential with mean 170 s has that median (mean = median / ln 2).
    rho_mean: float = 2.0  # geometric, mean rho (measured 2.01)
    inter_request_mean_s: float = 170.0  # exponential gap inside one conversation (median 118 s)

    # --- off-class background --------------------------------------------------------------
    # Off-class theory class B (biweekly homework deadline): 0.165 student turns per enrolled
    # student per day outside the 72 h pre-deadline windows, and 2.5 extra turns per student per
    # deadline; the simulator books one deadline per teaching week, so the extra is halved to keep
    # the weekly volume (0.34 turns per student-day in total) at the measured level.
    r_off_per_student_day: float = 0.17
    diurnal: Tuple[float, ...] = DIURNAL
    deadline_weekday: int = 0  # 0 = Monday; one deadline per teaching week
    deadline_hour: int = 23
    deadline_ramp_h: float = 48.0
    deadline_extra_per_student: float = 1.25

    # --- epsilon: what makes the realised stream differ from the calendar forecast --------
    roster_noise: float = 0.10  # forecast roster = n_k * (1 + U(-e, +e))
    # share of the roster that shows up and asks on a class day: 0.70-0.95 over the 17 class days
    # of the prompted theory class (median 0.87)
    attendance_lo: float = 0.70
    attendance_hi: float = 0.95
    p_cancel: float = 0.02

    # --- forecast perturbation: the three independent knobs of the F6 error axis ----------
    # ``forecast_scale`` biases the size of every booked session, ``bell_shift_frac`` mis-places a
    # fraction of them in time, and ``p_cancel`` above removes sessions the plan still books.  Each
    # is exact at its no-error value (1.0 / 0.0 / 0.0) so that one can be swept while the others
    # contribute nothing; the realised error of a run is measured, never assumed (``eta_norm``).
    forecast_scale: float = 1.0  # s: multiplies the expected requests of every booked session
    bell_shift_frac: float = 0.0  # fraction of sessions the forecast places one period off
    bell_shift_periods: int = 1  # size of that shift, in periods (the sign is drawn +/-)

    # --- nominal parameters the forecast is allowed to use --------------------------------
    forecast_sigma: Optional[float] = None  # None = use ``sigma`` (the planner knows the regime)
    forecast_attendance: float = 1.0  # planner books the whole roster
    forecast_rho: Optional[float] = None  # None = use ``rho_mean``
    # Trigger timing: the prompted burst of a session starts ``trigger_offset`` minutes after the
    # nominal trigger (negative = before the bell).  Measured on the genuine exports (batch A,
    # results/q1/assistant/trigger_offsets.md): mean -8.7 min, per-session sd 5.3 min, 96% before
    # the bell.  At the default 0 no random number is drawn, so older runs reproduce bit for bit.
    trigger_offset_mean_min: float = 0.0
    trigger_offset_sd_min: float = 0.0  # per-session spread within a course
    trigger_offset_course_sd_min: float = 0.0  # per-section (teacher habit) component, drawn once
    trigger_offset_common_sd_min: float = 0.0  # per-bell-occurrence component shared by every section at that bell
    trigger_offset_clip_min: Tuple[float, float] = (-25.0, 10.0)
    # What the planners assume: the shift they apply to the nominal trigger.  0 = timetable only
    # (the burst is expected at the bell); the measured mean = a learned course offset; earlier
    # than the mean = a learned offset plus a safety margin.  Every bell-aware policy sees it
    # (the calendar forecasts, the stale timetable, the guard's trigger grid).
    forecast_offset_min: float = 0.0
    # ... and the spread they assume around it: the forecast profile is convolved with a Gaussian
    # of this sd (minutes), the planner's model of the per-session timing randomness it cannot
    # predict.  0 = a point forecast at the assumed offset.
    forecast_offset_sd_min: float = 0.0
    # Cold-start prior of the learned-timing planner (policy 12a, ``calendar_learn_prior``): the
    # timetable of an earlier semester whose realised starts give a pooled (offset, sd) for every
    # session that has no history yet.  None = no prior, week 1 sits on the bell (policy 12).
    learn_prior_slots_csv: Optional[str] = None

    # --- request token lengths -------------------------------------------------------------
    azure_trace_csv: str = "data/tokens/azure_conv_tokens.csv"
    shared_prefix_tokens: int = 0  # course-material prefix on in-class prompts; 0 = off

    def __post_init__(self) -> None:
        """Reject an unknown ``offset_source`` at construction rather than at load time."""
        if self.offset_source not in OFFSET_SOURCES:
            raise ValueError(f"unknown offset_source '{self.offset_source}'; "
                             f"known: {sorted(OFFSET_SOURCES)}")

    @property
    def sync_share(self) -> float:
        """Canonical name of ``sigma``: the share of students drawn from the prompted pool."""
        return self.sigma

    @property
    def sync_offsets_csv(self) -> str:
        """Prompted-side offset pool of the selected source."""
        return (self.prompted_offsets_csv if self.offset_source == "ai"
                else self.signin_offsets_csv)

    @property
    def self_offsets_csv(self) -> str:
        """Self-paced-side offset pool of the selected source."""
        return (self.selfpaced_offsets_csv if self.offset_source == "ai"
                else self.assistments_offsets_csv)

    @property
    def pool_cap(self) -> float:
        """Offset cap in minutes; the session length unless ``pool_cap_min`` overrides it."""
        return self.period_minutes if self.pool_cap_min is None else float(self.pool_cap_min)

    @property
    def n_weeks(self) -> int:
        """Number of teaching weeks in the run."""
        return len(self.weeks)

    @property
    def horizon_min(self) -> int:
        """Run length in minutes (whole weeks from Monday 00:00 of the first week)."""
        return self.n_weeks * 7 * 1440


@dataclass(frozen=True)
class ServiceConfig:
    """One replica: concurrency limit, decode rate and the concurrency -> TTFT p95 curve.

    PLACEHOLDER until batch B (T3) runs.  ``load_service_config`` in ``service.py`` replaces every
    field marked below with the measured value from a T3 calibration CSV.
    """

    source: str = "PLACEHOLDER"  # set to the T3 csv path once real numbers are loaded
    model: str = "placeholder-7B"
    max_concurrency: int = 32  # PLACEHOLDER K: largest concurrency with TTFT p95 <= slo_in_s
    decode_tokens_per_s: float = 1500.0  # PLACEHOLDER aggregate decode rate of one replica
    per_stream_tokens_per_s: float = 80.0  # PLACEHOLDER single-stream ceiling
    ttft_floor_s: float = 0.30  # PLACEHOLDER TTFT p95 at concurrency 1
    ttft_at_k_s: float = 2.00  # PLACEHOLDER TTFT p95 at concurrency K
    prefill_tokens_per_s: float = 0.0  # 0 = TTFT does not depend on prompt length

    # --- lead time and commitment / billing granularity (minutes) ---------------------------
    # three separate constraints; ``LEAD_TIERS`` names the combinations the paper reports and
    # ``with_lead_tier`` applies one of them.  The defaults are the rented-instance tier: a real
    # lead time, but a decision and a bill every minute.
    lead_min: float = LEAD_TIERS["d_inst"].lead_min  # availability delay only
    commit_block_min: int = 1  # capacity is constant on blocks of this length (1 = unconstrained)
    billing_block_min: int = 1  # minimum billed duration per replica-block (1 = per minute)

    # --- queueing -----------------------------------------------------------------------------
    in_class_priority: bool = True  # False: one first-come queue across both classes (ablation)
    abandon_in_s: float = 60.0  # in-class request gives up after this queue wait -> timeout
    abandon_off_s: float = 300.0
    max_queue_in: int = 20000  # beyond this the request is rejected at the door
    max_queue_off: int = 20000

    def ttft_p95_s(self, concurrency: int) -> float:
        """TTFT p95 at ``concurrency`` active streams on one replica (linear between 1 and K)."""
        n = max(1, min(int(concurrency), self.max_concurrency))
        if self.max_concurrency <= 1:
            return self.ttft_at_k_s
        frac = (n - 1) / (self.max_concurrency - 1)
        return self.ttft_floor_s + frac * (self.ttft_at_k_s - self.ttft_floor_s)

    def stream_tokens_per_s(self, concurrency: int) -> float:
        """Per-stream decode rate at ``concurrency``: fair share, capped by the single-stream rate."""
        n = max(1, min(int(concurrency), self.max_concurrency))
        return min(self.per_stream_tokens_per_s, self.decode_tokens_per_s / n)

    def requests_per_min(self, mean_output_tokens: float) -> float:
        """Fluid capacity mu of one replica: requests per minute at the SLO-defining concurrency K."""
        service_s = self.ttft_p95_s(self.max_concurrency) + mean_output_tokens / max(
            1e-9, self.stream_tokens_per_s(self.max_concurrency))
        return 60.0 * self.max_concurrency / max(1e-9, service_s)


def with_lead_tier(service: ServiceConfig, tier: str) -> ServiceConfig:
    """``service`` moved onto the named tier: its lead time, commit block and billing block."""
    try:
        spec = LEAD_TIERS[tier]
    except KeyError as exc:
        raise KeyError(f"unknown lead tier '{tier}'; known: {sorted(LEAD_TIERS)}") from exc
    return replace(service, lead_min=spec.lead_min, commit_block_min=spec.commit_block_min,
                   billing_block_min=spec.billing_block_min)


@dataclass(frozen=True)
class CostConfig:
    """Cost of a trajectory, in the soft-constraint form of the paper's cost model."""

    c_run: float = 0.05  # USD per replica-minute (one A100 at about 3 USD/h)
    beta_start: float = 0.10  # USD per replica start: idle during d_rep plus a fixed term
    p_in: float = 0.30  # USD-equivalent weight of one unserved in-class request (a policy weight)
    p_off: float = 0.05  # USD-equivalent weight of one unserved off-class request
    slo_in_s: float = 2.0  # a
    slo_off_s: float = 10.0  # b
    first_minutes: int = 5  # the "first N minutes after the bell" violation rate is reported apart
    x_max: float = 2000.0  # bounded diameter D of the paper's analysis


@dataclass(frozen=True)
class PolicyConfig:
    """Thresholds, horizons and windows shared by the provisioning policies."""

    # 2 reactive / HPA
    hpa_up_util: float = 0.70
    hpa_down_util: float = 0.30
    hpa_up_hold_min: int = 1
    hpa_down_hold_min: int = 5
    hpa_cooldown_min: int = 3
    hpa_mode: str = "proportional"  # "proportional" (Kubernetes HPA) or "step" (+/- 1 replica)
    hpa_step: int = 1
    # 3 token velocity
    velocity_window_min: int = 2
    velocity_clip: float = 4.0  # cap on the extrapolation factor
    velocity_hysteresis: bool = True  # False restores the slope-every-minute baseline
    velocity_persist_min: int = 2  # act only on a slope that held this many consecutive slots
    velocity_dwell_min: int = 3  # minimum slots between two moves
    # 4 history forecast + ILP
    ilp_season_days: int = 7
    ilp_replan_min: int = 1440
    ilp_horizon_min: int = 1440
    forecast_warm_start: Optional[str] = None  # None, or "stale_calendar" for the week-1 plan
    # 5 LCP
    lcp_lookback_min: int = 60
    lcp_candidates: int = 24
    # 6/7 calendar / BDINTERP-d
    trust: float = 0.5  # policy 7's epsilon: gamma = delta = eps/4
    plan_margin: float = 1.0  # kappa: policies 6 and 7 plan for kappa * lambda_hat
    min_replicas: int = 0
    headroom: float = 1.0  # multiplies every target before rounding up


@dataclass(frozen=True)
class SweepConfig:
    """The parameter grids of the sweeps."""

    # the synchrony grid is in the *mixing weight* w (``WorkloadConfig.sigma``); F5 puts the
    # measured sigma_2 of each cell on its x axis, so the grid only has to span w from 0 to 1
    sigma: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    # which empirical offset pools the arrival mixture is drawn from (the ``pool`` sweep): the
    # measured sign-in / ASSISTments pair, then the sharper AI-assistant pair of batch A
    offset_sources: Tuple[str, ...] = ("signin", "ai")
    lead_tiers: Tuple[str, ...] = ("d_rep", "d_inst", "d_res")
    trust: Tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)  # BDINTERP-d epsilon
    # adoption a: with the calibrated 7B throughput (~1,067 req/min) and one always-on base
    # replica, anything below ~0.25 is absorbed by that replica and every policy coincides, so
    # the grid spans the range where provisioning is a decision at all (2026-09-17, batch C)
    adoption: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)  # 0.10 shows the absorbed regime
    rho: Tuple[float, ...] = (1.0, 2.0, 4.0)
    plan_margin: Tuple[float, ...] = (1.0, 1.2, 1.5)  # kappa, the planning margin of policies 6/7
    # calendar error (F6, first axis): one knob at a time, the other two held at their no-error
    # value.  The scale grid is log-spaced and runs to 20x because the robust branch of policy 7 is
    # insurance against a *catastrophic* over-forecast: it engages only once the advice is about
    # 4/eps times worse than LCP, which a 2x error never is (results/sim/f6_regime_probe.md).
    forecast_scale: Tuple[float, ...] = (0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0)
    bell_shift_frac: Tuple[float, ...] = (0.0, 0.1, 0.3)
    p_cancel: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.08)
    # F6, second axis: the dimensionless price ratio p_in * mu / c_run.  The epsilon sweep sets
    # p_in = ratio * c_run / mu per cell (mu depends on the stream, so it is known only once the
    # stream exists), which makes the facets differ in how expensive an unserved in-class request
    # is relative to a replica-minute and nothing else.  Empty = axis off, i.e. the base
    # ``CostConfig`` prices, whose ratio is about 1776 under the placeholder calibration.
    price_ratio: Tuple[float, ...] = (6.0, 18.0, 180.0)
    # trigger timing (the ``offset`` sweep): (realised mean, per-session sd, forecast offset);
    # the realised numbers are batch A's, the forecast offsets are timetable-only, the learned
    # mean, and the learned mean with 5 and 10 minutes of safety margin
    # Trigger timing cells: (realised mean, per-session sd, per-section sd, per-bell common sd,
    # forecast offset, forecast sd).  The three sd's are the correlation structure of the timing
    # error across the sections of one bell: independent (session + section terms), a common shock
    # (every section at that bell shifts together, so the institution stays synchronised but its
    # timing is uncertain), and an even mix; each is faced by a timetable-only planner (forecast
    # at the bell) and by one that learned the offset distribution.  Measured: mean -8.7, sd 5.3
    # within one course (batch A); the section term 4.0 is the spread of course means between two
    # teachers (sign-in launches); the mixed cell splits the 5.3 evenly.
    trigger_offset: Tuple[Tuple[float, float, float, float, float, float], ...] = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-8.7, 5.3, 4.0, 0.0, 0.0, 0.0), (-8.7, 5.3, 4.0, 0.0, -8.7, 6.6),
        (-8.7, 0.0, 0.0, 5.3, 0.0, 0.0), (-8.7, 0.0, 0.0, 5.3, -8.7, 5.3),
        (-8.7, 3.7, 0.0, 3.7, -8.7, 5.3),
        # sensitivity round (2026-09-21).  G-J: the share of the timing variance that is common to a
        # bell (0.25, 0.5, 0.75, 1.0) at the total spread of cell C (6.64 min), the planner knowing
        # the distribution; K: the mean offset known with no spread; L: a wide independent spread
        # (session sd 12 min, the widest sign-in course), the planner knowing the distribution.
        (-8.7, 4.59, 3.46, 3.32, -8.7, 6.6), (-8.7, 3.75, 2.83, 4.70, -8.7, 6.6),
        (-8.7, 2.65, 2.00, 5.75, -8.7, 6.6), (-8.7, 0.0, 0.0, 6.64, -8.7, 6.6),
        (-8.7, 5.3, 4.0, 0.0, -8.7, 0.0), (-8.7, 12.0, 4.0, 0.0, -8.7, 12.6))
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class RunConfig:
    """One simulation run: the four configs plus the master seed."""

    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    seed: int = 0
    root: str = "."  # every relative path in the configs is resolved against this


__all__ = [
    "DIURNAL",
    "LEAD_TIERS",
    "OFFSET_SOURCES",
    "CostConfig",
    "LeadTier",
    "PolicyConfig",
    "RunConfig",
    "ServiceConfig",
    "SweepConfig",
    "WorkloadConfig",
    "with_lead_tier",
]
