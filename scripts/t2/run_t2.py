"""Build the T2 timetable artefacts from one export.

    uv run python scripts/t2/run_t2.py --input data/raw/timetable/<timetable export 2026-2027-1>.txt \
        --out outputs/t2

Writes sections.csv, slots.csv, slot_load.csv, session_starts.csv, same_bell.csv, T2_table.csv,
T2_summary.md and the three figures.  Teacher names and staff ids are dropped at read time and
never reach any output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:  # allow `python scripts/t2/run_t2.py` without an install
    sys.path.insert(0, str(REPO_SRC))

import pandas as pd  # noqa: E402

from pilot.timetable import parse, stats  # noqa: E402

logger = logging.getLogger("run_t2")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    ap = argparse.ArgumentParser(description="Timetable T2 statistics and figures")
    ap.add_argument("--input", required=True, type=Path, help="timetable export (.txt/.csv/.xlsx)")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--week1-monday", default=parse.TimetableConfig().week1_monday,
                    help="Monday of teaching week 1 (YYYY-MM-DD)")
    ap.add_argument("--no-figures", action="store_true", help="write the tables only")
    return ap


def _write_slots(slots: pd.DataFrame, path: Path) -> None:
    out = slots.copy()
    out["weeks"] = out["weeks"].map(parse.format_weeks)
    out["periods"] = out["periods"].map(lambda ps: parse.format_weeks(list(ps)))
    out.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline; returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        logger.error("input not found: %s", args.input)
        return 2
    cfg = parse.TimetableConfig(week1_monday=args.week1_monday)
    bells = parse.BellSchedule()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = parse.load_timetable(args.input, cfg)
    if sections.empty:
        logger.error("no sections parsed from %s", args.input)
        return 3
    slots = parse.explode_slots(sections, cfg)
    weeks = parse.expand_weeks(slots)
    result = stats.compute(sections, slots, weeks, stats.StatsConfig(), bells)

    sections.to_csv(out_dir / "sections.csv", index=False)
    _write_slots(slots, out_dir / "slots.csv")
    result.slot_load.to_csv(out_dir / "slot_load.csv", index=False)
    result.typical_week.to_csv(out_dir / "typical_week.csv", index=False)
    result.session_starts.to_csv(out_dir / "session_starts.csv", index=False)
    result.same_bell.to_csv(out_dir / "same_bell.csv", index=False)
    stats.write_t2(result, out_dir, cfg, stats.StatsConfig())

    if not args.no_figures:
        from pilot.timetable import plots

        plots.make_all(sections, result.typical_week, result.session_starts, out_dir, bells=bells)
    logger.info("T2 artefacts in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
