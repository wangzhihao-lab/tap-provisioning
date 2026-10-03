"""Q1 re-analysis: per-student first arrivals per class session (the sign-in analogue).

The package's burst metrics run on every problem event; here each student contributes only
their first event of a session, so the ramp measures how fast a class *shows up*, not how long
it keeps working. Sessions are split into weekday school hours (in-class proxy) and the rest.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pilot.config import SessionDetectConfig
from pilot.sources.assistments import AssistmentsSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """What counts as an in-class session (local wall clock of the log)."""

    school_start_hour: int = 7
    school_end_hour: int = 15
    window_min: int = 15
    ks: tuple = (1, 2, 5)


def student_arrivals(events: pd.DataFrame, detect: SessionDetectConfig) -> pd.DataFrame:
    """One row per (group, session, student): the student's first event in that session."""
    work = events.loc[:, ["ts", "group", "uid"]].sort_values(["group", "ts"], kind="mergesort")
    gap = work.groupby("group", sort=False, observed=True)["ts"].diff()
    is_new = gap.isna() | (gap >= pd.Timedelta(minutes=detect.gap_min))
    work["session"] = is_new.groupby(work["group"], sort=False, observed=True).cumsum()
    first = (work.groupby(["group", "session", "uid"], sort=False, observed=True)["ts"]
             .min().reset_index(name="arrival"))
    return first


def session_stats(first: pd.DataFrame, min_students: int, cfg: SplitConfig) -> pd.DataFrame:
    """Ramp and synchronisation of student arrivals for every session with enough students."""
    rows = []
    for (group, sess), g in first.groupby(["group", "session"], sort=False, observed=True):
        arr = g["arrival"].sort_values().to_numpy()
        n = len(arr)
        if n < min_students:
            continue
        t0 = arr[0]
        rel = (arr - t0) / np.timedelta64(1, "m")
        row = {"group": group, "session": sess, "t0": pd.Timestamp(t0), "n_students": n,
               "ramp_10_90_min": float(np.quantile(rel, 0.9) - np.quantile(rel, 0.1)),
               "median_arrival_min": float(np.median(rel)),
               "span_min": float(rel[-1])}
        for k in cfg.ks:
            row[f"sync_{k}min"] = float((rel < k).mean())
        row["share_within_window"] = float((rel < cfg.window_min).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    hour = out["t0"].dt.hour + out["t0"].dt.minute / 60.0
    weekday = out["t0"].dt.weekday < 5
    out["in_class_proxy"] = weekday & (hour >= cfg.school_start_hour) & (hour < cfg.school_end_hour)
    return out


def summarise(stats: pd.DataFrame, label: str) -> dict:
    q = lambda s, p: float(np.nanquantile(s, p))  # noqa: E731
    return {"subset": label, "sessions": int(len(stats)),
            "students_median": float(stats["n_students"].median()),
            "ramp_p25": q(stats["ramp_10_90_min"], .25), "ramp_median": q(stats["ramp_10_90_min"], .5),
            "ramp_p75": q(stats["ramp_10_90_min"], .75),
            "median_arrival_median": q(stats["median_arrival_min"], .5),
            "sync_1min_median": q(stats["sync_1min"], .5), "sync_2min_median": q(stats["sync_2min"], .5),
            "sync_5min_median": q(stats["sync_5min"], .5),
            "share_within_15min_median": q(stats["share_within_window"], .5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", default="outputs/q1_assistments")
    ap.add_argument("--min-students", type=int, nargs="+", default=[5, 10, 20])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = SplitConfig()
    src = AssistmentsSource(Path(args.path))
    events = src._read()  # noqa: SLF001 (same package, read-once cache)
    first = student_arrivals(events, src.detect)
    logger.info("%d student-session arrivals", len(first))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for m in args.min_students:
        stats = session_stats(first, m, cfg)
        stats.to_csv(out_dir / f"student_arrival_sessions_min{m}.csv", index=False)
        summaries.append(summarise(stats, f"all, >={m} students"))
        summaries.append(summarise(stats[stats["in_class_proxy"]], f"weekday 07-15h, >={m} students"))
        summaries.append(summarise(stats[~stats["in_class_proxy"]], f"other hours, >={m} students"))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "student_arrivals_summary.csv", index=False)
    with open(out_dir / "student_arrivals_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# Q1 re-analysis: per-student first arrivals per session\n\n")
        cols = list(summary.columns)
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in summary.iterrows():
            fh.write("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
        fh.write("\n")
    logger.info("\n%s", summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
