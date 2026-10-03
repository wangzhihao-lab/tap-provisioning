"""Anonymised Xuexitong AI-assistant session logs: parsing, loading and workload characterisation.

``parse``    one 会话详情 cell -> (turn, timestamp, role, character count); the text stops here.
``load``     the anonymised tables -> one frame pair per teaching class (``DatasetSpec``).
``metrics``  per-student first arrivals, sigma_k, 10-90% ramp, peak-to-mean, gaps, lengths.
``bursts``   a prompted class day -> one burst per bell, and the offset pool they form.
"""

from .bursts import (
    MIXED_BURST,
    PROMPTED_BURST,
    BurstConfig,
    bell_windows,
    burst_offsets,
    split_bursts,
)
from .load import (
    OFF_CLASS,
    PROMPTED,
    SELF_PACED,
    AssistantDataset,
    DatasetSpec,
    hm_to_min,
    load_dataset,
)
from .metrics import (
    KS,
    binned_counts,
    deadline_ramp,
    first_arrivals,
    gap_stats,
    length_stats,
    peak_to_mean,
    questions_per_student,
    ramp_10_90,
    relative_offsets,
    sigma_shares,
    turn_gaps,
    weekly_profile,
)
from .parse import ASSISTANT, STUDENT, MessageRow, detail_frame, parse_detail, speaker_role

__all__ = [
    "ASSISTANT",
    "KS",
    "MIXED_BURST",
    "OFF_CLASS",
    "PROMPTED",
    "PROMPTED_BURST",
    "SELF_PACED",
    "STUDENT",
    "AssistantDataset",
    "BurstConfig",
    "DatasetSpec",
    "MessageRow",
    "bell_windows",
    "binned_counts",
    "burst_offsets",
    "deadline_ramp",
    "detail_frame",
    "first_arrivals",
    "gap_stats",
    "hm_to_min",
    "length_stats",
    "load_dataset",
    "parse_detail",
    "peak_to_mean",
    "questions_per_student",
    "ramp_10_90",
    "relative_offsets",
    "sigma_shares",
    "speaker_role",
    "split_bursts",
    "turn_gaps",
    "weekly_profile",
]
