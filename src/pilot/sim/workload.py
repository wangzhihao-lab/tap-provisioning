"""Calendar -> request stream, and the calendar forecast lambda-hat.

One run covers whole teaching weeks starting at Monday 00:00 of the first week in
``WorkloadConfig.weeks``.  Time is minutes (fluid demand) or seconds (the request stream) from
that origin.

The realised stream and the forecast come from the same generative model; they differ only in the
error sources of the simulator design: roster noise, attendance, cancellation, trigger jitter and
sampling noise.  That difference is the epsilon of the sweeps.
"""

import logging
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..timetable.parse import BellSchedule, parse_week_part
from .config import WorkloadConfig
from .stale import stale_forecast

logger = logging.getLogger(__name__)

IN_CLASS = 0
OFF_CLASS = 1
_MIN_PER_WEEK = 7 * 1440
_KERNEL_MIN = 90  # longest lag (minutes) kept in the analytic in-class profile


@dataclass(frozen=True)
class ArrivalStream:
    """Realised requests plus the fluid demand series the cost model and the policies see."""

    t_arrive_s: np.ndarray
    kind: np.ndarray
    session_id: np.ndarray
    prompt_tokens: np.ndarray
    output_tokens: np.ndarray
    lam_in: np.ndarray
    lam_off: np.ndarray
    lam_hat_in: np.ndarray
    lam_hat_off: np.ndarray
    sessions: pd.DataFrame
    horizon_min: int
    lam_stale_in: Optional[np.ndarray] = None  # policy 9's forecast (last semester's timetable)
    student_id: Optional[np.ndarray] = None  # global student index per request, -1 off-class
    lam_hat_learn_in: Optional[np.ndarray] = None  # policy 12's forecast: timing learned from past weeks
    lam_hat_learn_prior_in: Optional[np.ndarray] = None  # policy 12a: the same, warm-started from an earlier semester

    @property
    def students(self) -> np.ndarray:
        """Per-request student index, ``-1`` where the generator did not record one."""
        if self.student_id is None:
            return np.full(self.n_requests, -1, dtype=np.int64)
        return self.student_id

    @property
    def n_requests(self) -> int:
        """Number of requests in the stream."""
        return int(self.t_arrive_s.size)

    @property
    def lam(self) -> np.ndarray:
        """Total realised demand per minute."""
        return self.lam_in + self.lam_off

    @property
    def lam_hat(self) -> np.ndarray:
        """Total forecast demand per minute."""
        return self.lam_hat_in + self.lam_hat_off

    @property
    def stale_in(self) -> np.ndarray:
        """Policy 9's in-class forecast; zeros when last semester's table is not on disk."""
        return (self.lam_hat_in * 0.0 if self.lam_stale_in is None else self.lam_stale_in)


def _resolve(root: str, path: str) -> Path:
    """Resolve ``path`` against ``root`` unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def load_offset_pool(path: Path, cap_min: float) -> np.ndarray:
    """Empirical arrival offsets (minutes after the session's first arrival), clipped at the cap.

    The cap is the session length, not a burst window: an offset above it is *clipped* to the cap
    and kept, never dropped.  Dropping the tail - which is what the old 2 min / 15 min caps did -
    renormalises the distribution and inflates its 2-minute share (the ASSISTments pool goes from
    0.24 to 0.42 at a 15 min cap and to 0.34 at a 45 min one), so the mixture would no longer
    reproduce the measured sigma_2 of either data source.  Clipping leaves the 2-minute share at
    its pooled raw value and turns the late arrivals into end-of-session, non-burst traffic.
    """
    col = pd.read_csv(path)["offset_min"].to_numpy(dtype=float)
    pool = col[(col >= 0.0) & np.isfinite(col)]
    if pool.size == 0:
        raise ValueError(f"empty offset pool: {path}")
    late = int((pool > cap_min).sum())
    pool = np.minimum(pool, float(cap_min))
    logger.debug("offset pool %s: %d rows, %d clipped to %.1f min", path.name, pool.size, late,
                 cap_min)
    return pool


def load_token_pool(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Prompt / output token lengths from the Azure 2023 conversation trace."""
    df = pd.read_csv(path, usecols=["ContextTokens", "GeneratedTokens"])
    ctx = df["ContextTokens"].to_numpy(dtype=np.int64)
    gen = np.maximum(df["GeneratedTokens"].to_numpy(dtype=np.int64), 1)
    return ctx, gen


def _expand_weeks(text: str) -> Tuple[int, ...]:
    """``"1-9,11-18"`` -> the tuple of week numbers."""
    out: List[int] = []
    for part in str(text).split(","):
        out.extend(parse_week_part(part))
    return tuple(sorted(set(out)))


def load_sessions(cfg: WorkloadConfig, rng: np.random.Generator, root: str = ".") -> pd.DataFrame:
    """Expand the timetable slice into one row per adopting section x teaching week.

    Columns: ``session_id, section_id, college, week, weekday, bell_min, duration_min, roster``.
    ``bell_min`` is the minute of the block start (the bell the students hear) from the run origin.
    """
    slots = pd.read_csv(_resolve(root, cfg.slots_csv))
    if cfg.college:
        slots = slots[slots["college"] == cfg.college]
    if cfg.campus:
        slots = slots[slots["campus"] == cfg.campus]
    slots = slots[slots["enrollment"].fillna(0) > 0].reset_index(drop=True)
    if cfg.max_sections and len(slots) > cfg.max_sections:
        slots = slots.iloc[: cfg.max_sections].reset_index(drop=True)
    if slots.empty:
        logger.warning("timetable slice is empty (college=%s campus=%s)", cfg.college, cfg.campus)
        return pd.DataFrame(columns=["session_id", "section_id", "college", "week", "weekday",
                                     "bell_min", "duration_min", "roster"])
    adopt = rng.random(len(slots)) < cfg.adoption
    slots = slots[adopt].reset_index(drop=True)
    logger.info("adoption %.2f keeps %d timetable slots", cfg.adoption, len(slots))

    bells = BellSchedule()
    rows = []
    for _, row in slots.iterrows():
        weeks = set(_expand_weeks(row["weeks"]))
        p0, p1 = int(row["period_start"]), int(row["period_end"])
        start_hm = bells.block_start(p0)
        end_hm = bells.window(min(p1, len(bells.windows)))[1]
        start_of_day = _hm_to_min(start_hm)
        duration = max(cfg.period_minutes, _hm_to_min(end_hm) - start_of_day)
        for i, week in enumerate(cfg.weeks):
            if week not in weeks:
                continue
            bell = i * _MIN_PER_WEEK + (int(row["weekday"]) - 1) * 1440 + start_of_day
            rows.append((0, int(row["section_id"]), row["college"], int(week),
                         int(row["weekday"]), int(bell), float(duration),
                         int(row["enrollment"])))
    out = pd.DataFrame(rows, columns=["session_id", "section_id", "college", "week", "weekday",
                                      "bell_min", "duration_min", "roster"])
    out = out.sort_values("bell_min").reset_index(drop=True)
    out["session_id"] = np.arange(len(out))
    return out


def _hm_to_min(hm: str) -> int:
    """``"08:30"`` -> 510."""
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


def _bell_shift(cfg: WorkloadConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """Per-session timing error of the forecast, in minutes.

    A fraction ``bell_shift_frac`` of the booked sessions is placed ``bell_shift_periods`` periods
    away from the bell the students actually hear; the sign is drawn uniformly so the knob is a
    pure timing error and does not move the forecast mass.  At the no-error value 0 no random
    number is drawn at all, so a run with the knob off is bit-identical to one before it existed.
    """
    if n <= 0 or cfg.bell_shift_frac <= 0.0:
        return np.zeros(max(0, n), dtype=np.int64)
    hit = rng.random(n) < cfg.bell_shift_frac
    sign = rng.choice(np.array([-1, 1]), size=n)
    step = int(round(cfg.period_minutes)) * max(1, int(cfg.bell_shift_periods))
    return (hit * sign * step).astype(np.int64)


def _trigger_minutes(cfg: WorkloadConfig, bell_min: int, duration_min: float) -> np.ndarray:
    """Nominal trigger instants inside one session (before the U(0, jitter) shift)."""
    m = max(1, cfg.triggers_per_session)
    step = duration_min / m
    return bell_min + step * np.arange(m)


def trigger_offsets(cfg: WorkloadConfig, sessions: pd.DataFrame,
                    rng: np.random.Generator) -> np.ndarray:
    """Realised start offset of every session's triggers, in minutes (negative = early).

    ``mean + section term + session term + common term``: the section term is one draw per
    section (a teacher's habit, ``trigger_offset_course_sd_min``), the session term one draw per
    session (``trigger_offset_sd_min``), the common term one draw per bell occurrence shared by
    every section that starts on it (``trigger_offset_common_sd_min``, the correlated part of the
    timing error); each is skipped, consuming no random numbers, when its sd is 0, so the default
    configuration reproduces streams generated before the knobs existed.
    """
    n = len(sessions)
    offsets = np.full(n, float(cfg.trigger_offset_mean_min))
    if n == 0:
        return offsets
    if cfg.trigger_offset_course_sd_min > 0.0:
        sections = sessions["section_id"].to_numpy()
        uniq, inverse = np.unique(sections, return_inverse=True)
        offsets += rng.normal(0.0, cfg.trigger_offset_course_sd_min, size=uniq.size)[inverse]
    if cfg.trigger_offset_sd_min > 0.0:
        offsets += rng.normal(0.0, cfg.trigger_offset_sd_min, size=n)
    if cfg.trigger_offset_common_sd_min > 0.0:  # one draw per bell occurrence, shared by its sections
        bells = sessions["bell_min"].to_numpy()
        uniq, inverse = np.unique(bells, return_inverse=True)
        offsets += rng.normal(0.0, cfg.trigger_offset_common_sd_min, size=uniq.size)[inverse]
    lo, hi = cfg.trigger_offset_clip_min
    return np.clip(offsets, lo, hi)


def _emit_offsets(rng: np.random.Generator, n_first: int, rho_mean: float,
                  gap_mean_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Expand ``n_first`` student arrivals into requests; return (student index, offset seconds)."""
    if n_first == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    p = 1.0 / max(1.0, rho_mean)
    rho = rng.geometric(p, size=n_first).astype(np.int64)
    idx = np.repeat(np.arange(n_first), rho)
    gaps = rng.exponential(gap_mean_s, size=int(rho.sum()))
    starts = np.cumsum(rho) - rho
    gaps[starts] = 0.0
    cum = np.cumsum(gaps)
    return idx, cum - np.repeat(cum[starts], rho)


def _in_class_arrivals(cfg: WorkloadConfig, sessions: pd.DataFrame, rng: np.random.Generator,
                       sync_pool: np.ndarray, self_pool: np.ndarray,
                       offsets: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Realised in-class requests: attendance, cancellation, the w mixture and rho requests.

    Also returns, per session, the minute of its first realised arrival (NaN when the session
    was cancelled or nobody came): what a deployment observes about when a class actually started
    using the assistant, the raw material of the learned-timing forecast.

    ``cfg.sigma`` is the mixing weight w - the probability that a student is drawn from the
    prompted pool - and every attending student is given a run-wide index so that the *measured*
    synchrony sigma_2 can be read off the stream afterwards (``metrics.measure_synchrony``).  A
    student keeps the same index across the triggers of their own session.
    """
    times: List[np.ndarray] = []
    ids: List[np.ndarray] = []
    students: List[np.ndarray] = []
    next_student = 0
    shift = np.zeros(len(sessions)) if offsets is None else np.asarray(offsets, dtype=float)
    starts = np.full(len(sessions), np.nan)  # realised first arrival of the first trigger, minutes
    for k, (row, offset_min) in enumerate(zip(sessions.itertuples(index=False), shift)):
        if rng.random() < cfg.p_cancel:
            continue
        attendance = rng.uniform(cfg.attendance_lo, cfg.attendance_hi)
        n_attend = int(rng.binomial(int(row.roster), attendance))
        if n_attend == 0:
            continue
        base = next_student
        next_student += n_attend
        for j, trig in enumerate(_trigger_minutes(cfg, int(row.bell_min), float(row.duration_min))):
            jitter_s = rng.uniform(0.0, cfg.trigger_jitter_min) * 60.0
            sync = rng.random(n_attend) < cfg.sigma
            delay = np.empty(n_attend, dtype=float)
            n_sync = int(sync.sum())
            if n_sync:
                delay[sync] = rng.choice(sync_pool, size=n_sync, replace=True)
            if n_attend - n_sync:
                delay[~sync] = rng.choice(self_pool, size=n_attend - n_sync, replace=True)
            first_s = (trig + offset_min) * 60.0 + jitter_s + delay * 60.0
            if j == 0:
                starts[k] = float(first_s.min()) / 60.0
            idx, off = _emit_offsets(rng, n_attend, cfg.rho_mean, cfg.inter_request_mean_s)
            times.append(first_s[idx] + off)
            ids.append(np.full(off.size, int(row.session_id), dtype=np.int64))
            students.append(base + idx.astype(np.int64))
    if not times:
        return pd.DataFrame({"t": np.empty(0), "session_id": np.empty(0, dtype=np.int64),
                             "student_id": np.empty(0, dtype=np.int64)}), starts
    return pd.DataFrame({"t": np.concatenate(times), "session_id": np.concatenate(ids),
                         "student_id": np.concatenate(students)}), starts


def off_class_intensity(cfg: WorkloadConfig, sessions: pd.DataFrame) -> np.ndarray:
    """Off-class requests per minute: diurnal background plus the 48 h pre-deadline ramp."""
    horizon = cfg.horizon_min
    lam = np.zeros(horizon, dtype=float)
    if sessions.empty:
        return lam
    students = float(sessions.drop_duplicates("section_id")["roster"].sum())
    diurnal = np.asarray(cfg.diurnal, dtype=float)
    diurnal = diurnal / diurnal.sum()
    hours = np.arange(horizon) // 60
    lam += students * cfg.r_off_per_student_day * diurnal[hours % 24] / 60.0
    ramp = np.asarray(cfg.deadline_ramp_h, dtype=float) * 60.0
    for week in range(cfg.n_weeks):
        due = week * _MIN_PER_WEEK + cfg.deadline_weekday * 1440 + cfg.deadline_hour * 60
        lo = int(max(0, due - ramp))
        hi = int(min(horizon, due))
        if hi <= lo:
            continue
        shape = np.exp(np.linspace(-3.0, 0.0, hi - lo))
        lam[lo:hi] += students * cfg.deadline_extra_per_student * shape / shape.sum()
    return lam


def _delay_pmf(cfg: WorkloadConfig, sync_pool: np.ndarray, self_pool: np.ndarray,
               sigma: float) -> np.ndarray:
    """Minute-binned pmf of the sigma mixture of the two empirical offset pools."""
    n_bins = int(np.ceil(max(float(sync_pool.max()), float(self_pool.max())))) + 1
    out = np.zeros(n_bins, dtype=float)
    for pool, weight in ((sync_pool, sigma), (self_pool, 1.0 - sigma)):
        if weight <= 0.0:
            continue
        hist = np.bincount(np.floor(pool).astype(int), minlength=n_bins)[:n_bins]
        out += weight * hist / hist.sum()
    return out


def _emit_kernel(cfg: WorkloadConfig, n_bins: int) -> np.ndarray:
    """Expected requests per arriving student, per minute of lag (closed form)."""
    rho = max(1.0, cfg.forecast_rho if cfg.forecast_rho is not None else cfg.rho_mean)
    q = 1.0 - 1.0 / rho
    rate = 1.0 / cfg.inter_request_mean_s
    edges = np.arange(n_bins + 1, dtype=float) * 60.0
    decay = rate / rho
    cdf = 1.0 - np.exp(-decay * edges)
    kernel = (q / (1.0 - q + 1e-12)) * np.diff(cdf) if q > 0 else np.zeros(n_bins)
    kernel[0] += 1.0
    return kernel


def _spread_profile(profile: np.ndarray, sd_min: float) -> Tuple[np.ndarray, int]:
    """``profile`` convolved with a Gaussian of ``sd_min`` minutes; returns it and its half width."""
    if sd_min <= 0.0:
        return profile, 0
    half = int(np.ceil(3.0 * sd_min))
    grid = np.arange(-half, half + 1, dtype=float)
    spread = np.exp(-0.5 * (grid / sd_min) ** 2)
    return np.convolve(profile, spread / spread.sum()), half


def calendar_forecast(cfg: WorkloadConfig, sessions: pd.DataFrame, rng: np.random.Generator,
                      sync_pool: np.ndarray, self_pool: np.ndarray,
                      offsets_min: Optional[np.ndarray] = None,
                      offset_sd_min: Optional[np.ndarray] = None) -> np.ndarray:
    """lambda-hat for the in-class component: rosters, nominal sigma / rho, analytic profile.

    The profile of every trigger is placed ``forecast_offset_min`` minutes from the nominal
    trigger, what the planner believes about when the burst starts (0 = at the bell), and spread
    with a Gaussian of ``forecast_offset_sd_min`` minutes, its model of the timing randomness.
    ``offsets_min`` / ``offset_sd_min`` override both per session (the learned-timing forecast).
    """
    horizon = cfg.horizon_min
    lam = np.zeros(horizon, dtype=float)
    if sessions.empty:
        return lam
    sigma = cfg.forecast_sigma if cfg.forecast_sigma is not None else cfg.sigma
    delay = _delay_pmf(cfg, sync_pool, self_pool, sigma)
    jitter_bins = max(1, int(np.ceil(cfg.trigger_jitter_min)))
    jitter = np.full(jitter_bins, 1.0 / jitter_bins)
    base = np.convolve(np.convolve(delay, jitter), _emit_kernel(cfg, _KERNEL_MIN))
    n = len(sessions)
    offs = (np.full(n, float(cfg.forecast_offset_min)) if offsets_min is None
            else np.asarray(offsets_min, dtype=float))
    sds = (np.full(n, float(cfg.forecast_offset_sd_min)) if offset_sd_min is None
           else np.asarray(offset_sd_min, dtype=float))
    sds = np.round(np.maximum(sds, 0.0) * 2.0) / 2.0  # half-minute bins keep the kernel cache small
    cache: Dict[float, Tuple[np.ndarray, int]] = {}
    noise = 1.0 + rng.uniform(-cfg.roster_noise, cfg.roster_noise, size=n)
    shift = _bell_shift(cfg, rng, n)
    for row, eps, off, o_min, sd in zip(sessions.itertuples(index=False), noise, shift, offs, sds):
        if float(sd) not in cache:
            cache[float(sd)] = _spread_profile(base, float(sd))
        profile, half = cache[float(sd)]
        scale = float(row.roster) * eps * cfg.forecast_attendance * cfg.forecast_scale
        for trig in _trigger_minutes(cfg, int(row.bell_min), float(row.duration_min)):
            t0 = int(round(trig)) + int(round(o_min)) + int(off) - half
            hi = min(horizon, t0 + profile.size)
            lo = max(0, t0)
            if t0 >= horizon or hi <= lo:
                continue
            lam[lo:hi] += scale * profile[lo - t0: hi - t0]
    return lam


LEARN_MIN_OBS = 2  # a section needs this many past observations before its own sd is used


def learned_timing(sessions: pd.DataFrame, realised_start_min: np.ndarray,
                   prior: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-session (offset, sd) a planner can learn from strictly earlier weeks' realised starts.

    The observation of a session is ``realised first arrival - bell`` in minutes.  For the
    sessions of week ``w`` the planner uses the section's own past observations (mean, and its own
    sd once it has ``LEARN_MIN_OBS`` of them), else the pooled mean and sd over every section seen
    so far; in the first week nothing has been observed and the forecast sits on the bell with no
    spread - unless a ``prior`` (pooled offset, sd) from an earlier semester is given, which then
    stands in for the pooled estimate wherever nothing has been observed yet.  Only weeks strictly
    before ``w`` are used, so weeks 2.. are held-out tests of what the timing history predicts.
    """
    n = len(sessions)
    offs = np.zeros(n)
    sds = np.zeros(n)
    if prior is not None:
        offs[:] = float(prior[0])
        sds[:] = max(0.0, float(prior[1]))
    obs = realised_start_min - sessions["bell_min"].to_numpy(dtype=float)
    weeks = sessions["week"].to_numpy()
    sections = sessions["section_id"].to_numpy()
    for w in np.unique(weeks):
        past = (weeks < w) & np.isfinite(obs)
        cur = weeks == w
        if not past.any():
            continue  # week 1: the bell only, or the prior
        pooled_mean = float(obs[past].mean())
        pooled_sd = float(obs[past].std(ddof=1)) if past.sum() >= 2 else 0.0
        by_sec: Dict[int, np.ndarray] = {}
        for sec in np.unique(sections[past]):
            by_sec[int(sec)] = obs[past & (sections == sec)]
        for i in np.where(cur)[0]:
            own = by_sec.get(int(sections[i]))
            if own is None or own.size == 0:
                offs[i], sds[i] = pooled_mean, pooled_sd
            else:
                offs[i] = float(own.mean())
                sds[i] = float(own.std(ddof=1)) if own.size >= LEARN_MIN_OBS else pooled_sd
    return offs, sds


def timing_prior(cfg: WorkloadConfig, root: str, seed: int) -> Tuple[float, float]:
    """Pooled (offset, sd) of ``realised start - bell`` over the semester of ``learn_prior_slots_csv``.

    The earlier semester is generated with the same workload parameters as the run (its timing
    structure is what the institution's habits are) and its own generator, so the run's stream is
    untouched; the estimate is what a deployment would have logged the semester before.
    """
    if not cfg.learn_prior_slots_csv:
        raise ValueError("timing_prior needs WorkloadConfig.learn_prior_slots_csv")
    prior_cfg = replace(cfg, slots_csv=cfg.learn_prior_slots_csv, learn_prior_slots_csv=None)
    rng = np.random.default_rng(seed)
    sync_pool = load_offset_pool(_resolve(root, prior_cfg.sync_offsets_csv), prior_cfg.pool_cap)
    self_pool = load_offset_pool(_resolve(root, prior_cfg.self_offsets_csv), prior_cfg.pool_cap)
    sessions = load_sessions(prior_cfg, rng, root)
    offsets = trigger_offsets(prior_cfg, sessions, rng)
    _, starts = _in_class_arrivals(prior_cfg, sessions, rng, sync_pool, self_pool, offsets)
    obs = starts - sessions["bell_min"].to_numpy(dtype=float)
    obs = obs[np.isfinite(obs)]
    if obs.size < 2:
        logger.warning("timing_prior: %d observations in %s; prior is the bell", obs.size,
                       cfg.learn_prior_slots_csv)
        return 0.0, 0.0
    prior = float(obs.mean()), float(obs.std(ddof=1))
    logger.info("timing prior from %s: %d sessions, offset %.2f min, sd %.2f", cfg.learn_prior_slots_csv,
                obs.size, *prior)
    return prior


# The learned-timing generator is keyed on a CRC-32 of the slots file name.  The three released
# timetables keep the keys of the runs reported in the paper, so every learned-timing stream is
# bit-identical to those runs; any other file is keyed on its own name.
_SLOTS_KEYS: Dict[str, int] = {
    "data/timetable/slots_2026-2027-1.csv": 1773360777,
    "data/timetable/slots_2025-2026-2.csv": 1868587266,
    "data/timetable/slots_2025-2026-1.csv": 4135716611,
}


def _slots_key(slots_csv: str) -> int:
    """The CRC-32 the learned-timing key starts from (see ``_SLOTS_KEYS``)."""
    return _SLOTS_KEYS.get(slots_csv, zlib.crc32(slots_csv.encode("utf-8")))


def generate(cfg: WorkloadConfig, rng: np.random.Generator, root: str = ".",
             sessions: Optional[pd.DataFrame] = None) -> ArrivalStream:
    """Build the realised request stream and both demand series for one workload instance."""
    horizon = cfg.horizon_min
    sync_pool = load_offset_pool(_resolve(root, cfg.sync_offsets_csv), cfg.pool_cap)
    self_pool = load_offset_pool(_resolve(root, cfg.self_offsets_csv), cfg.pool_cap)
    if sessions is None:
        sessions = load_sessions(cfg, rng, root)
    hat_in = calendar_forecast(cfg, sessions, rng, sync_pool, self_pool)
    hat_off = off_class_intensity(cfg, sessions)
    stale_in = stale_forecast(cfg, root, sync_pool, self_pool)

    offsets = trigger_offsets(cfg, sessions, rng)
    in_df, starts = _in_class_arrivals(cfg, sessions, rng, sync_pool, self_pool, offsets)
    sessions = sessions.assign(trigger_offset_min=offsets, realised_start_min=starts)
    counts = rng.poisson(hat_off)
    off_min = np.repeat(np.arange(horizon), counts)
    off_t = off_min * 60.0 + rng.uniform(0.0, 60.0, size=off_min.size)

    t = np.concatenate([in_df["t"].to_numpy(dtype=float), off_t])
    kind = np.concatenate([np.full(len(in_df), IN_CLASS, dtype=np.int8),
                           np.full(off_t.size, OFF_CLASS, dtype=np.int8)])
    sid = np.concatenate([in_df["session_id"].to_numpy(dtype=np.int64),
                          np.full(off_t.size, -1, dtype=np.int64)])
    stu = np.concatenate([in_df["student_id"].to_numpy(dtype=np.int64),
                          np.full(off_t.size, -1, dtype=np.int64)])
    keep = (t >= 0.0) & (t < horizon * 60.0)
    t, kind, sid, stu = t[keep], kind[keep], sid[keep], stu[keep]
    order = np.argsort(t, kind="stable")
    t, kind, sid, stu = t[order], kind[order], sid[order], stu[order]

    ctx, gen = load_token_pool(_resolve(root, cfg.azure_trace_csv))
    pick = rng.integers(0, ctx.size, size=t.size)
    prompt = ctx[pick] + (cfg.shared_prefix_tokens * (kind == IN_CLASS))
    minute = np.minimum((t // 60.0).astype(np.int64), horizon - 1)
    lam_in = np.bincount(minute[kind == IN_CLASS], minlength=horizon).astype(float)
    lam_off = np.bincount(minute[kind == OFF_CLASS], minlength=horizon).astype(float)
    logger.info("workload: %d sessions, %d requests (%.0f in-class)", len(sessions), t.size,
                lam_in.sum())
    # the learned-timing forecast is built last, from a generator of its own, so that the realised
    # stream and every other forecast are bit-identical to runs made before it existed
    learn_off, learn_sd = learned_timing(sessions, starts)
    # a process-independent key: ``hash()`` of a str is salted per interpreter, which made the
    # learned forecast's roster-noise draws differ between runs of the same configuration
    learn_key = (_slots_key(cfg.slots_csv) + 7919 * len(sessions)
                 + horizon) % (2 ** 32)
    rng_learn = np.random.default_rng(learn_key)
    hat_learn_in = calendar_forecast(cfg, sessions, rng_learn, sync_pool, self_pool, learn_off, learn_sd)
    hat_learn_prior_in = None
    if cfg.learn_prior_slots_csv:  # policy 12a: the same learner, warm-started from an earlier semester
        prior = timing_prior(cfg, root, seed=(learn_key + 1) % (2 ** 32))
        prior_off, prior_sd = learned_timing(sessions, starts, prior=prior)
        rng_prior = np.random.default_rng((learn_key + 2) % (2 ** 32))
        hat_learn_prior_in = calendar_forecast(cfg, sessions, rng_prior, sync_pool, self_pool,
                                               prior_off, prior_sd)
    return ArrivalStream(t, kind, sid, prompt, gen[pick], lam_in, lam_off, hat_in, hat_off,
                         sessions, horizon, stale_in, stu, hat_learn_in, hat_learn_prior_in)


__all__ = [
    "learned_timing", "timing_prior",
    "trigger_offsets","IN_CLASS", "OFF_CLASS", "ArrivalStream", "calendar_forecast", "forecast_error",
           "generate", "load_offset_pool", "load_sessions", "load_token_pool",
           "off_class_intensity"]


def forecast_error(lam_in: np.ndarray, lam_off: np.ndarray, hat_in: np.ndarray,
                   hat_off: np.ndarray) -> Tuple[float, float]:
    """``(eta, eta_norm)``: the L1 calendar error and the same error per realised request.

    ``eta = sum_t (|lam_in - hat_in| + |lam_off - hat_off|)`` is the quantity the regret bound of
    the paper's analysis is stated in; ``eta_norm = eta / sum_t (lam_in + lam_off)`` divides it by the realised
    demand so that runs of different size sit on one axis.  Both are measured on the forecast the
    run actually used, so every perturbation knob is reflected in them.
    """
    eta = float(np.abs(lam_in - hat_in).sum() + np.abs(lam_off - hat_off).sum())
    total = float(np.sum(lam_in) + np.sum(lam_off))
    return eta, (eta / total if total > 0.0 else float("nan"))
