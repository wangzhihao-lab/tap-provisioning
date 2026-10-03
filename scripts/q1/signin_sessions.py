"""Q1 evidence: per-session sign-in ramps from Xuexitong (Chaoxing) sign-in detail exports.

One export is one sign-in activity, i.e. one class meeting: every row is a student's check-in
time. Each file is reduced to a single row of ramp and synchronisation metrics measured from
the first check-in of that file (t0), so a classroom "everyone at once" burst can be put next
to the ASSISTments in-class sessions written by ``student_arrivals.py``. When the export also
carries the moment the teacher started the activity (发起时间/开始时间) the same metrics are
computed from it as ``*_from_start``; the t0-relative numbers stay primary because they are the
ones that are comparable across sources.

Only timestamps and counts are written. Names and student ids stay inside the loader.
"""

import argparse
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pilot.config import FigureConfig
from pilot.figures import apply_style, save  # imported first: it pins the Agg backend
from pilot.sources.xuexitong import SignInExport

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Columns that carry the activity start; the check-in column itself is found by the loader.
START_COLS = ("发起时间", "开始时间", "活动时间")


@dataclass(frozen=True)
class MetricConfig:
    """Windows used by the per-session metrics (minutes)."""

    ks: Tuple[int, ...] = (1, 2, 5)
    window_min: int = 15
    ramp_lo: float = 0.1
    ramp_hi: float = 0.9


def collect_paths(patterns: Sequence[str], directory: Optional[str] = None,
                  pattern: str = "*.xls*") -> List[Path]:
    """Expand --paths (globs, in case the shell did not) and --dir into a sorted unique list."""
    found: List[Path] = []
    for pat in patterns or []:
        hits = sorted(glob.glob(pat))
        found.extend(Path(h) for h in hits)
        if not hits and Path(pat).exists():
            found.append(Path(pat))
    if directory:
        found.extend(Path(directory).glob(pattern))
    uniq = sorted({p.resolve() for p in found if not p.name.startswith("~$")})
    if not uniq:
        raise FileNotFoundError("no sign-in export matched --paths/--dir")
    return uniq


def read_start_time(path: Path) -> Optional[pd.Timestamp]:
    """Earliest 发起时间/开始时间 in the export, or None when the layout has no such column.

    The file is read a second time on purpose: the shared loader deliberately returns check-in
    times only, and it is not this script's place to widen it.
    """
    try:
        raw = (pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"}
               else pd.read_csv(path))
    except (ValueError, OSError) as exc:
        logger.warning("%s: no start time read (%s)", path.name, exc)
        return None
    for cand in START_COLS:
        for col in raw.columns:
            if cand in str(col):
                vals = pd.to_datetime(raw[col], errors="coerce").dropna()
                if not vals.empty:
                    return pd.Timestamp(vals.min())
    return None


def _relative(rel: np.ndarray, cfg: MetricConfig, suffix: str = "") -> Dict[str, float]:
    """Metrics of sign-in delays ``rel`` (minutes after some reference instant)."""
    out = {f"median_arrival_min{suffix}": float(np.median(rel)),
           f"span_min{suffix}": float(rel.max())}
    for k in cfg.ks:
        out[f"sync_{k}min{suffix}"] = float((rel < k).mean())
    out[f"share_within_{cfg.window_min}min{suffix}"] = float((rel < cfg.window_min).mean())
    return out


def session_metrics(ts: pd.Series, cfg: MetricConfig,
                    t_start: Optional[pd.Timestamp] = None) -> Dict[str, object]:
    """Calendar position and ramp/synchronisation metrics of one sign-in activity."""
    arr = np.sort(pd.to_datetime(ts).dropna().to_numpy(dtype="datetime64[ns]"))
    t0 = pd.Timestamp(arr[0])
    rel = (arr - arr[0]) / np.timedelta64(1, "m")
    row: Dict[str, object] = {
        "date": t0.date().isoformat(), "weekday": int(t0.weekday()), "hour": int(t0.hour),
        "n": int(arr.size),
        "ramp_10_90_min": float(np.quantile(rel, cfg.ramp_hi) - np.quantile(rel, cfg.ramp_lo)),
    }
    row.update(_relative(rel, cfg))
    if t_start is not None:
        row["t_start"] = t_start.isoformat(sep=" ")
        from_start = (arr - t_start.to_datetime64()) / np.timedelta64(1, "m")
        row.update(_relative(from_start, cfg, "_from_start"))
    return row


def session_table(paths: Sequence[Path], cfg: MetricConfig) -> pd.DataFrame:
    """One row per export; files the loader cannot read or that hold no time are skipped."""
    rows: List[Dict[str, object]] = []
    for path in paths:
        try:
            frame = SignInExport([path]).load()
        except (KeyError, ValueError, OSError) as exc:
            logger.warning("%s: skipped (%s)", path.name, exc)
            continue
        if frame.empty:
            logger.warning("%s: no parsable sign-in time", path.name)
            continue
        row: Dict[str, object] = {"session": path.stem}
        row.update(session_metrics(frame["ts"], cfg, read_start_time(path)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(stats: pd.DataFrame, counts: Dict[str, int]) -> pd.DataFrame:
    """Median and quartiles of every numeric metric across the kept sessions."""
    skip = {"weekday", "hour"}
    rows = []
    for col in stats.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(stats[col]):
            continue
        values = pd.to_numeric(stats[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        row = {"metric": col, "p25": float(np.nanquantile(values, 0.25)),
               "median": float(np.nanquantile(values, 0.5)),
               "p75": float(np.nanquantile(values, 0.75))}
        row.update(counts)
        rows.append(row)
    cols = ["metric", "p25", "median", "p75", *counts]
    return pd.DataFrame(rows, columns=cols)


def write_markdown(summary: pd.DataFrame, counts: Dict[str, int], path: Path) -> None:
    """Markdown twin of the summary CSV, written by hand so no extra dependency is needed."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Q1: Xuexitong sign-in sessions\n\n")
        fh.write(f"{counts['n_sessions']} session(s), {counts['n_signins']} sign-ins; "
                 f"{counts['n_sessions_dropped']} session(s) dropped for "
                 f"n < {counts['min_students']}. Minutes run from the first sign-in of the "
                 "file (t0); `*_from_start` runs from 发起时间 where the export has it.\n\n")
        cols = ["metric", "p25", "median", "p75"]
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in summary.iterrows():
            cells = [str(r["metric"])] + [f"{float(r[c]):.3f}" for c in cols[1:]]
            fh.write("| " + " | ".join(cells) + " |\n")
        fh.write("\n")


def load_compare(path: Path) -> Optional[pd.DataFrame]:
    """In-class rows of a student_arrivals session CSV, used as the reference curve."""
    frame = pd.read_csv(path)
    if "ramp_10_90_min" not in frame.columns:
        logger.warning("%s: no ramp_10_90_min column, comparison skipped", path.name)
        return None
    if "in_class_proxy" in frame.columns:
        mask = frame["in_class_proxy"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame.loc[mask]
    logger.info("%s: %d reference session(s)", path.name, len(frame))
    return frame if not frame.empty else None


def ramp_cdf(stats: pd.DataFrame, compare: Optional[pd.DataFrame], out_dir: Path,
             cfg: FigureConfig, xmax: Optional[float] = None,
             name: str = "signin_ramp_cdf") -> None:
    """Empirical CDF of the 10-90% sign-in ramp, with an optional reference curve.

    Without ``xmax`` the axis stops just past the widest 98th percentile of the curves drawn.
    """
    curves = [("Chaoxing sign-in", stats)]
    if compare is not None:
        curves.append(("ASSISTments in-class", compare))
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    upper = 1.0
    for i, (label, frame) in enumerate(curves):
        vals = np.sort(pd.to_numeric(frame.get("ramp_10_90_min"), errors="coerce")
                       .dropna().to_numpy(dtype=float))
        if vals.size == 0:
            logger.warning("%s: no ramp times to plot", label)
            continue
        ax.plot(vals, np.arange(1, vals.size + 1) / vals.size, label=f"{label} (n={vals.size})",
                linewidth=1.2, markevery=max(1, vals.size // 10), markersize=3.5,
                color=cfg.palette[i % len(cfg.palette)],
                linestyle=cfg.linestyles[i % len(cfg.linestyles)],
                marker=cfg.markers[i % len(cfg.markers)])
        upper = max(upper, float(np.quantile(vals, 0.98)))
    ax.set_xlim(0, xmax if xmax else upper * 1.05)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Sign-in ramp, 10-90% (min)")
    ax.set_ylabel("Fraction of sessions")
    ax.legend(loc="lower right")
    save(fig, out_dir, name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-session sign-in ramps of sign-in detail exports.")
    ap.add_argument("--paths", nargs="*", default=[], help="exports; globs allowed")
    ap.add_argument("--dir", help="directory of exports, filtered by --pattern")
    ap.add_argument("--pattern", default="*.xls*", help="glob used inside --dir")
    ap.add_argument("--out", default="outputs/q1_signin")
    ap.add_argument("--min-students", type=int, default=10)
    ap.add_argument("--compare", help="student_arrivals CSV to overlay (its in-class rows)")
    ap.add_argument("--xmax", type=float, help="cap the ramp axis of the figure (min)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("fontTools", "matplotlib"):  # font subsetting chatter at INFO level
        logging.getLogger(noisy).setLevel(logging.WARNING)
    paths = collect_paths(args.paths, args.dir, args.pattern)
    logger.info("%d sign-in export(s)", len(paths))
    cfg = MetricConfig()
    table = session_table(paths, cfg)
    if table.empty:
        raise SystemExit("no export yielded a parsable sign-in time")
    kept = table.loc[table["n"] >= args.min_students].reset_index(drop=True)
    counts = {"n_sessions": int(len(kept)), "n_signins": int(kept["n"].sum()) if len(kept) else 0,
              "n_sessions_dropped": int(len(table) - len(kept)), "min_students": args.min_students}
    logger.info("%d session(s) kept, %d dropped for n < %d", counts["n_sessions"],
                counts["n_sessions_dropped"], args.min_students)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_dir / "signin_sessions.csv", index=False)
    summary = summarise(kept, counts)
    summary.to_csv(out_dir / "signin_summary.csv", index=False)
    write_markdown(summary, counts, out_dir / "signin_summary.md")
    compare = load_compare(Path(args.compare)) if args.compare else None
    ramp_cdf(kept, compare, out_dir, apply_style(), args.xmax)
    logger.info("\n%s", summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
