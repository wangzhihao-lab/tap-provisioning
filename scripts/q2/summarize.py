#!/usr/bin/env python3
"""Q2: aggregate every measurement into the two CSVs and the lead-time figure.

Reads whatever the four measurement scripts left under ``--results``
(``coldstart/``, ``concurrency/``, ``prefix/``, ``instance/``) and writes:

  <out>/q2_summary.csv        long table: tier x model x metric, median/min/max
  <out>/T3_calibration.csv    the T3 table of the paper plan (model x metric)
  <out>/coldstart_vs_ramp.pdf, .png

The figure overlays the measured lead times with the classroom burst ramp band
(10-90 %% ramp, ``--ramp-band LO HI`` seconds, from the education data) and the
published vLLM no-prewarm TTFT values.

    python summarize.py --results results/b1 --out results/b1
    python summarize.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

logger = logging.getLogger("q2.summarize")

PALETTE = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]

COLDSTART_METRICS = (
    ("weights_loaded_s", "s"), ("graph_capture_s", "s"), ("cuda_graph_done_s", "s"),
    ("server_listening_s", "s"), ("health_ok_s", "s"), ("first_completion_done_s", "s"),
    ("first_completion_ttft_s", "s"), ("steady_ttft_s", "s"),
)
CONCURRENCY_METRICS = (
    ("ttft_p50_s", "s"), ("ttft_p95_s", "s"), ("ttft_p99_s", "s"), ("e2e_p95_s", "s"),
    ("throughput_req_s", "req/s"), ("throughput_tokens_s", "tok/s"), ("error_rate", "fraction"),
)
INSTANCE_METRICS = tuple(
    (f"request_to_{k}_s", "s")
    for k in ("boot", "ssh_ready", "deps_ready", "weights_present", "server_ready")
)
ROW_COLUMNS = ("tier", "source", "model", "subject", "metric", "unit", "n", "median", "min", "max")


@dataclass(frozen=True)
class SummaryConfig:
    results_dir: Path = Path("results")
    out_dir: Path = Path("results")
    ramp_lo_s: float = 60.0
    ramp_hi_s: float = 120.0
    ramp_label: str = "classroom burst ramp (10-90%)"
    literature: tuple[tuple[str, float], ...] = (
        ("WarmServe 7B", 25.3), ("WarmServe 13B", 28.0), ("WarmServe 70B", 40.3),
    )
    literature_source: str = "WarmServe, arXiv 2512.09472 (vLLM, no pre-warm TTFT)"
    figure: bool = True
    dpi: int = 300


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.info("missing %s", path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping %s:%d (not JSON)", path, line_no)
    logger.info("read %d records from %s", len(rows), path)
    return rows


def _row(
    tier: str, source: str, model: str, subject: str, metric: str, unit: str,
    values: Sequence[float],
) -> dict[str, Any] | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=float)
    return {
        "tier": tier, "source": source, "model": model, "subject": subject,
        "metric": metric, "unit": unit, "n": int(arr.size),
        "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()),
    }


def _group(records: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        out.setdefault(str(rec.get(key, "unknown")), []).append(rec)
    return out


def rows_from_coldstart(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model, recs in _group(records, "model").items():
        for metric, unit in COLDSTART_METRICS:
            row = _row("replica_start", "measured", model, model, metric, unit,
                       [r.get(metric) for r in recs])
            if row:
                rows.append(row)
    return rows


def rows_from_concurrency(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        model, level = str(rec.get("model", "unknown")), rec.get("level")
        for metric, unit in CONCURRENCY_METRICS:
            row = _row("concurrency", "measured", model, f"{model} @N={level}",
                       f"{metric}@N={level}", unit, [rec.get(metric)])
            if row:
                rows.append(row)
    return rows


def rows_from_prefix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    arms = [r for r in records if r.get("schema") == "q2.prefix.arm.v1"]
    deltas = [r for r in records if r.get("schema") == "q2.prefix.delta.v1"]
    for model, recs in _group(deltas, "model").items():
        for metric in ("ttft_first_gain_s", "ttft_mean_gain_s", "ttft_p95_gain_s"):
            row = _row("prefix_warmup", "measured", model, model, metric, "s",
                       [r.get(metric) for r in recs])
            if row:
                rows.append(row)
    for model, recs in _group(arms, "model").items():
        for arm in ("cold", "warm"):
            subset = [r for r in recs if r.get("arm") == arm]
            for metric in ("ttft_first_s", "ttft_mean_s", "ttft_p95_s"):
                row = _row("prefix_warmup", "measured", model, f"{model} [{arm}]",
                           f"{arm}_{metric}", "s", [r.get(metric) for r in subset])
                if row:
                    rows.append(row)
    return rows


def rows_from_instance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    label = "cloud GPU instance"
    for metric, unit in INSTANCE_METRICS:
        row = _row("instance_acquire", "measured", label, label, metric, unit,
                   [r.get(metric) for r in records])
        if row:
            rows.append(row)
    for key in ("gap_submitted_to_boot_s", "gap_boot_to_ssh_ready_s",
                "gap_ssh_ready_to_deps_ready_s", "gap_deps_ready_to_weights_present_s",
                "gap_weights_present_to_server_ready_s"):
        row = _row("instance_acquire", "measured", label, label, key, "s",
                   [r.get(key) for r in records])
        if row:
            rows.append(row)
    return rows


def rows_from_literature(cfg: SummaryConfig) -> list[dict[str, Any]]:
    rows = []
    for name, value in cfg.literature:
        row = _row("literature", cfg.literature_source, name, name,
                   "ttft_no_prewarm_s", "s", [value])
        if row:
            rows.append(row)
    return rows


def collect(cfg: SummaryConfig) -> pd.DataFrame:
    base = cfg.results_dir
    rows: list[dict[str, Any]] = []
    rows += rows_from_coldstart(read_jsonl(base / "coldstart" / "coldstart.jsonl"))
    rows += rows_from_concurrency(read_jsonl(base / "concurrency" / "concurrency_summary.jsonl"))
    rows += rows_from_prefix(read_jsonl(base / "prefix" / "prefix_summary.jsonl"))
    rows += rows_from_instance(read_jsonl(base / "instance" / "instance_acquire_summary.jsonl"))
    rows += rows_from_literature(cfg)
    return pd.DataFrame(rows, columns=list(ROW_COLUMNS))


def t3_table(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame["tier"].isin(("replica_start", "concurrency", "prefix_warmup"))
    t3 = frame.loc[keep, ["model", "metric", "unit", "n", "median", "min", "max"]].copy()
    return t3.sort_values(["model", "metric"]).reset_index(drop=True)


def _apply_style() -> None:
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def _series(frame: pd.DataFrame, tier: str, metric: str) -> list[tuple[str, float, float, float]]:
    sel = frame[(frame["tier"] == tier) & (frame["metric"] == metric)]
    return [
        (str(r.model), float(r.median), float(r.min), float(r.max))
        for r in sel.itertuples() if np.isfinite(r.median)
    ]


def _log_ticks(ax: plt.Axes, lows: list[float], highs: list[float]) -> None:
    """Readable decade-and-a-half ticks; a pure log axis labels only the decades."""
    if not lows or not highs:
        return
    lo, hi = min(lows), max(highs)
    candidates = (5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000)
    ticks = [t for t in candidates if lo / 1.7 <= t <= hi * 1.7]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()


def _short(name: str) -> str:
    return name.split("/")[-1].replace("-Instruct", "").replace("Qwen2.5-", "Qwen2.5 ")


def make_figure(frame: pd.DataFrame, cfg: SummaryConfig) -> list[Path]:
    """Measured lead times vs the burst ramp band and the published values."""
    _apply_style()
    health = _series(frame, "replica_start", "health_ok_s")
    answer = _series(frame, "replica_start", "first_completion_done_s")
    acquire = _series(frame, "instance_acquire", "request_to_server_ready_s")
    lit = _series(frame, "literature", "ttft_no_prewarm_s")
    labels = [_short(n) for n, *_ in answer or health]
    labels += [_short(n) for n, *_ in acquire] + [_short(n) for n, *_ in lit]
    if not labels:
        logger.warning("nothing to plot")
        return []
    y_all = np.arange(len(labels))[::-1]
    n_replica = len(answer or health)
    n_acquire = len(acquire)
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(labels) + 1.5))
    ax.axvspan(cfg.ramp_lo_s, cfg.ramp_hi_s, color=PALETTE[3], alpha=0.30, lw=0, zorder=0)
    specs = (
        (health, y_all[:n_replica] + 0.16, PALETTE[0], "o", "replica: /health OK"),
        (answer, y_all[:n_replica] - 0.16, PALETTE[1], "s", "replica: first answer returned"),
        (acquire, y_all[n_replica:n_replica + n_acquire], PALETTE[2], "^",
         "cloud instance: create to serving"),
        (lit, y_all[n_replica + n_acquire:], PALETTE[4], "D",
         "vLLM no pre-warm TTFT (WarmServe)"),
    )
    for data, ys, colour, marker, label in specs:
        if not data:
            continue
        med = np.array([d[1] for d in data], dtype=float)
        lo = np.maximum(med - np.array([d[2] for d in data]), 0.0)
        hi = np.maximum(np.array([d[3] for d in data]) - med, 0.0)
        ax.errorbar(med, ys[: len(data)], xerr=np.vstack([lo, hi]), fmt=marker, color=colour,
                    markersize=5, capsize=2.5, lw=1.1, linestyle="none", label=label, zorder=3)
    ax.set_yticks(y_all)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("provisioning lead time (s, log scale)")
    _log_ticks(ax, [d[2] for spec in specs for d in spec[0]] + [cfg.ramp_lo_s],
               [d[3] for spec in specs for d in spec[0]] + [cfg.ramp_hi_s])
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.grid(axis="x", which="major", color="0.85", lw=0.6, zorder=0)
    handles, lbls = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=PALETTE[3], alpha=0.30,
                         label=f"{cfg.ramp_label}: {cfg.ramp_lo_s:g}-{cfg.ramp_hi_s:g} s"))
    lbls.append(handles[-1].get_label())
    ax.legend(handles, lbls, loc="lower right", frameon=False, handletextpad=0.5)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".pdf", ".png"):
        path = cfg.out_dir / f"coldstart_vs_ramp{suffix}"
        fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _log_headline(frame: pd.DataFrame, cfg: SummaryConfig) -> None:
    ramp = f"{cfg.ramp_lo_s:g}-{cfg.ramp_hi_s:g} s"
    for tier, metric, what in (
        ("replica_start", "first_completion_done_s", "replica start (to first answer)"),
        ("instance_acquire", "request_to_server_ready_s", "cloud instance acquisition"),
    ):
        for name, med, lo, hi in _series(frame, tier, metric):
            if med > cfg.ramp_hi_s:
                verdict = "above"
            else:
                verdict = "inside" if med >= cfg.ramp_lo_s else "below"
            logger.info("%s | %s: median %.1f s (%.1f-%.1f) -> %s ramp band %s",
                        what, _short(name), med, lo, hi, verdict, ramp)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = SummaryConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=d.results_dir)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ramp-band", type=float, nargs=2, metavar=("LO", "HI"),
                   default=[d.ramp_lo_s, d.ramp_hi_s], help="10-90%% burst ramp, seconds")
    p.add_argument("--ramp-label", default=d.ramp_label)
    p.add_argument("--literature", default=",".join(f"{k}={v}" for k, v in d.literature),
                   help="NAME=SECONDS pairs, comma separated")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--dpi", type=int, default=d.dpi)
    p.add_argument("--dry-run", action="store_true", help="read and write results/dryrun")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    logging.basicConfig(
        level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    results = Path("results/dryrun") if a.dry_run else a.results
    out = a.out if a.out is not None else results
    lit = []
    for item in str(a.literature).split(","):
        if "=" in item:
            name, value = item.rsplit("=", 1)
            lit.append((name.strip(), float(value)))
    cfg = SummaryConfig(
        results_dir=results, out_dir=out, ramp_lo_s=float(a.ramp_band[0]),
        ramp_hi_s=float(a.ramp_band[1]), ramp_label=a.ramp_label,
        literature=tuple(lit), figure=not a.no_figure, dpi=a.dpi,
    )
    frame = collect(cfg)
    if frame.empty:
        logger.error("no measurement records found under %s", cfg.results_dir)
        return 2
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.out_dir / "q2_summary.csv"
    t3_path = cfg.out_dir / "T3_calibration.csv"
    frame.to_csv(summary_path, index=False)
    t3_table(frame).to_csv(t3_path, index=False)
    logger.info("wrote %d rows to %s", len(frame), summary_path)
    logger.info("wrote %d rows to %s", len(t3_table(frame)), t3_path)
    if cfg.figure:
        for path in make_figure(frame, cfg):
            logger.info("wrote %s", path)
    _log_headline(frame, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
