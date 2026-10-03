"""Build the released derived tables under ``data/`` from the non-public raw exports.

The raw inputs are not distributed (see README, "Data").  With them on disk the pipeline is

    uv run python scripts/t2/run_t2.py --input <timetable export>.txt --out outputs/t2_<semester>
    uv run python scripts/q1/signin_sessions.py ...        # sign-in offset pool
    uv run python scripts/q1/student_arrivals.py ...       # ASSISTments in-class offset pool
    uv run python scripts/q1/assistant_workload.py ...     # AI-assistant offset pools
    uv run python scripts/data/make_public_tables.py \\
        --slots 2026-2027-1=outputs/t2_2026-2027-1/slots.csv \\
        --slots 2025-2026-2=outputs/t2_2025-2026-2/slots.csv \\
        --slots 2025-2026-1=outputs/t2_2025-2026-1/slots.csv \\
        --azure <AzureLLMInferenceTrace_conv.csv>

The timetable slots keep only the section number (a running index assigned by the parser),
campus and college codes, roster size, weekday, periods and teaching weeks.  Campus and college
names are replaced by ``campus_<k>`` and ``college_<kk>``, numbered in the sort order of the
original names over all semesters given, so that every filter and group-by of the simulator
behaves as on the original table.  Course names and codes, class labels, rooms and teachers are
dropped.  The token pool keeps the two length columns of the public Azure trace.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

KEEP_SLOT_COLS: Tuple[str, ...] = ("section_id", "campus", "college", "enrollment", "slot_index",
                                   "weekday", "periods", "period_start", "period_end", "n_periods",
                                   "weeks", "n_weeks")
TOKEN_COLS: Tuple[str, ...] = ("ContextTokens", "GeneratedTokens")


def codebook(frames: Sequence[pd.DataFrame], column: str, prefix: str, width: int) -> Dict[str, str]:
    """Original value -> code, numbered in the sort order of the originals."""
    values = sorted({v for f in frames for v in f[column].dropna().unique()})
    return {v: f"{prefix}_{i + 1:0{width}d}" for i, v in enumerate(values)}


def public_slots(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """The released slot table of every semester."""
    campus = codebook(list(frames.values()), "campus", "campus", 1)
    college = codebook(list(frames.values()), "college", "college", 2)
    out: Dict[str, pd.DataFrame] = {}
    for sem, frame in frames.items():
        missing = sorted(set(KEEP_SLOT_COLS) - set(frame.columns))
        if missing:
            raise ValueError(f"{sem}: slots table lacks {missing}")
        pub = frame[list(KEEP_SLOT_COLS)].copy()
        pub["campus"] = pub["campus"].map(campus)
        pub["college"] = pub["college"].map(college)
        out[sem] = pub
    return out


def _pairs(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        sem, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--slots expects SEMESTER=PATH, got {item!r}")
        out[sem] = Path(path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slots", action="append", default=[], help="SEMESTER=PATH of a run_t2 slots.csv")
    ap.add_argument("--azure", type=Path, default=None, help="Azure 2023 conversation trace CSV")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    sources = _pairs(args.slots)
    if sources:
        frames = {sem: pd.read_csv(path) for sem, path in sources.items()}
        target = args.out / "timetable"
        target.mkdir(parents=True, exist_ok=True)
        for sem, pub in public_slots(frames).items():
            pub.to_csv(target / f"slots_{sem}.csv", index=False)
            logger.info("wrote %s (%d slots)", target / f"slots_{sem}.csv", len(pub))
    if args.azure is not None:
        target = args.out / "tokens"
        target.mkdir(parents=True, exist_ok=True)
        pd.read_csv(args.azure, usecols=list(TOKEN_COLS)).to_csv(target / "azure_conv_tokens.csv",
                                                                 index=False)
        logger.info("wrote %s", target / "azure_conv_tokens.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
