"""Write the sign-in and ASSISTments arrival-offset pools of ``data/offsets`` from the raw data.

    uv run python scripts/q1/offset_pools.py signin --paths "<sign-in exports>/*.csv"
    uv run python scripts/q1/offset_pools.py assistments --path <2012-2013 ASSISTments csv or zip>

An offset is a student's first arrival in a session minus the session's first arrival, in
minutes, rounded to 3 decimals; a pool is written in ascending order.  ``signin``: one session per
sign-in export (one class meeting), every check-in kept.  ``assistments``: sessions as ``student_arrivals.py`` splits them, kept when
they hold at least ``--min-students`` students and start on a weekday between 07:00 and 15:00
(the in-class proxy); the pool is a uniform sample without replacement of ``--size`` offsets
drawn with ``--seed``.  The AI-assistant pools are written by ``assistant_workload.py``.
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from pilot.sources.assistments import AssistmentsSource
from pilot.sources.xuexitong import SignInExport
from student_arrivals import SplitConfig, session_stats, student_arrivals

logger = logging.getLogger(__name__)


def signin_pool(paths: Sequence[Path]) -> np.ndarray:
    """Every check-in's offset from the first check-in of its export, in ascending order."""
    out: List[np.ndarray] = []
    for path in sorted(paths):
        ts = SignInExport([path]).load()["ts"].dropna().sort_values().to_numpy()
        if ts.size:
            out.append((ts - ts[0]) / np.timedelta64(1, "m"))
    return np.sort(np.round(np.concatenate(out), 3)) if out else np.array([])


def assistments_pool(path: Path, min_students: int, size: int, seed: int) -> np.ndarray:
    """First-arrival offsets of the in-class sessions, subsampled to ``size``."""
    src = AssistmentsSource(path)
    first = student_arrivals(src._read(), src.detect)  # noqa: SLF001
    stats = session_stats(first, min_students, SplitConfig())
    keep = stats.loc[stats["in_class_proxy"], ["group", "session"]]
    sub = first.merge(keep, on=["group", "session"])
    t0 = sub.groupby(["group", "session"], observed=True)["arrival"].transform("min")
    rel = ((sub["arrival"] - t0) / pd.Timedelta(minutes=1)).to_numpy(dtype=float)
    logger.info("%d in-class sessions, %d first arrivals", len(keep), rel.size)
    if 0 < size < rel.size:
        rel = np.random.default_rng(seed).choice(rel, size=size, replace=False)
    return np.sort(np.round(rel, 3))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="pool", required=True)
    s = sub.add_parser("signin")
    s.add_argument("--paths", nargs="+", required=True, help="sign-in exports (globs allowed)")
    s.add_argument("--out", type=Path, default=Path("data/offsets/signin_offsets_min.csv"))
    a = sub.add_parser("assistments")
    a.add_argument("--path", type=Path, required=True)
    a.add_argument("--min-students", type=int, default=10)
    a.add_argument("--size", type=int, default=200_000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", type=Path, default=Path("data/offsets/assistments_inclass_offsets_min.csv"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.pool == "signin":
        paths = [Path(p) for pat in args.paths for p in (glob.glob(pat) or [pat])]
        pool = signin_pool(paths)
    else:
        pool = assistments_pool(args.path, args.min_students, args.size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"offset_min": pool}).to_csv(args.out, index=False)
    logger.info("wrote %s (%d offsets)", args.out, pool.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
