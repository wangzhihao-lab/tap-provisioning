"""Frozen configuration objects for the pilot."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SessionSpec:
    """One recurring class session on the timetable."""

    weekday: int  # 0 = Monday
    start_hour: int
    start_minute: int
    duration_min: int
    n_students: int
    group: str


@dataclass(frozen=True)
class MockConfig:
    """Synthetic calendar-driven workload."""

    seed: int = 42
    weeks: int = 8
    start_date: str = "2026-09-07"  # a Monday
    sessions: Tuple[SessionSpec, ...] = field(
        default_factory=lambda: (
            SessionSpec(2, 10, 10, 180, 118, "class-A"),
            SessionSpec(2, 14, 0, 180, 69, "class-B"),
        )
    )
    # in-class synchronized burst: fraction of students who query within the burst,
    # and the log-normal delay (minutes) after the trigger
    sync_fraction: float = 0.8
    sync_delay_median_min: float = 0.6
    sync_delay_sigma: float = 0.7
    triggers_per_session: int = 2
    # background self-paced traffic: per-student daily rate, diurnal profile
    background_rate_per_student_day: float = 0.15
    # deadline ramp: extra arrivals in the 48 h before each weekly deadline
    deadline_weekday: int = 1
    deadline_hour: int = 23
    deadline_extra_per_student: float = 0.6
    deadline_ramp_hours: float = 48.0


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters of the Q1 metrics."""

    bin_minutes: int = 1
    bin_minutes_coarse: int = 5
    ramp_lo: float = 0.1
    ramp_hi: float = 0.9
    burst_window_min: int = 15
    sync_k_minutes: Tuple[int, ...] = (1, 2, 5)
    forecast_horizon_min: int = 15
    triggers_per_session: int = 1
    # Q1 gate thresholds (pre-registered before the data were read)
    gate_ramp_min: float = 3.0
    gate_error_reduction: float = 0.25


@dataclass(frozen=True)
class SessionDetectConfig:
    """How class sessions (education logs) and load onsets (generic traces) are detected."""

    # gap-based clustering of a class log into sessions
    gap_min: int = 30
    min_students: int = 5
    min_events: int = 10
    # threshold-based onset detection for traces that have no timetable
    rise_factor: float = 1.5
    baseline_window_min: int = 120
    quiet_min: int = 30
    min_rate_per_bin: float = 1.0
    max_onsets: int = 500


@dataclass(frozen=True)
class PredictConfig:
    """Per-bin arrival forecasting at horizon ``horizon_min``."""

    bin_minutes: int = 1
    horizon_min: int = 15
    season_days: int = 7
    season_weeks: int = 4
    ewma_alpha: float = 0.5
    # per-session usage profile: arrivals per student per minute since session start
    profile_len_min: int = 180
    # cross-course prior used when no previous session exists (semester first week)
    prior_sync_fraction: float = 0.5
    prior_decay_min: float = 2.0
    # bins inside this many days of the series start are scored as the "first period"
    first_period_days: int = 7
    min_denominator: float = 1e-9


@dataclass(frozen=True)
class LoaderConfig:
    """Reading real exports: chunk size, row caps, trace epochs."""

    chunk_rows: int = 500_000
    max_rows: int = 0  # 0 = read everything
    # BurstGPT timestamps are seconds from 0:00:00 of an undisclosed first day; this anchor is
    # nominal (a Monday) so that weekday slots are self-consistent, not calendar-true.
    burstgpt_origin: str = "2023-01-02"


@dataclass(frozen=True)
class FigureConfig:
    """Figure style: Times-like serif, >= 8 pt, fixed colour cycle, no titles inside the axes."""

    width_in: float = 3.5
    height_in: float = 2.4
    dpi: int = 300
    base_font_pt: float = 9.0
    small_font_pt: float = 8.0
    palette: Tuple[str, ...] = (
        "#332288",
        "#117733",
        "#44AA99",
        "#88CCEE",
        "#DDCC77",
        "#CC6677",
        "#AA4499",
        "#882255",
    )
    markers: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
    linestyles: Tuple[str, ...] = ("-", "--", "-.", ":")
    ramp_xlim_min: float = 20.0
    sigma_bins: int = 20
    max_bar_width: float = 0.22
    bar_headroom: float = 1.55
