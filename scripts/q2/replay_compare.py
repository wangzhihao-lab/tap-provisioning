#!/usr/bin/env python3
"""Batch B' (F10): the simulator's service model against the real-machine burst replay.

For every replay label the box produced (``<results>/b2/replay/<label>.jsonl`` plus, for the
late-replica runs, ``<label>_server.json``), rebuild the identical arrival stream (the scheduled
offsets, prompt and output lengths the client actually used), push it through
``pilot.sim.service.replay`` with the calibrated 7B service model and the replica trajectory the
run realised (a warm replica: one replica throughout; a late one: none until the measured
readiness instant, rounded up to the simulator's minute), and put the two per-request TTFT
distributions side by side.  The measured side is also scored under the simulator's abandonment
rule (an in-class request that waits longer than ``abandon_in_s`` is a timeout), so that the
"within 2 s" shares compare like with like.

Writes ``replay_compare.csv`` (one row per label), ``replay_bins.csv`` (10-s bins) and the F10
figure (``f10_replay_validation``) under ``--out``.
Run from the repository root: ``uv run python scripts/q2/replay_compare.py --results scripts/q2/results/b2_remote``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import replace
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator,  # noqa: E402
                               MultipleLocator, NullLocator)

from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style, save  # noqa: E402
from pilot.policy_style import GREY  # noqa: E402
from pilot.sim.config import ServiceConfig, with_lead_tier  # noqa: E402
from pilot.sim.service import load_service_config, replay  # noqa: E402
from pilot.sim.workload import IN_CLASS, ArrivalStream  # noqa: E402

logger = logging.getLogger("replay_compare")

PILOT = Path(__file__).resolve().parents[2]
T3_DEFAULT = PILOT / "scripts/q2/results/b1_remote/b1/T3_calibration.csv"
MODEL_ROW = "qwen2.5-7b"
CHARS_PER_TOKEN = 1.45
PROMPT_TEMPLATE_TOKENS = 12  # "[idx] 课程问题：" and the chat template, roughly
BIN_S = 10.0
SLO_S = 2.0

# F10 is printed one column wide (\includegraphics[width=\columnwidth]): two panels of about
# 4.3 x 4.0 cm.  Every text is set at FONT_PT and the log axes are labelled in plain digits,
# because a 10^k tick carries its exponent at 0.7 x the font size.
CM_PER_IN = 2.54
FONT_PT = 8.0
PRINT_DPI = 600
F10_SIZE_CM: Tuple[float, float] = (8.8, 4.0)
PAD_IN = 0.012
LINE_PT = 1.2    # data lines; the light palette colours need at least 1.2 pt to hold at print
MARKER_PT = 3.2
REF_PT = 1.0     # the grey SLO line; a ready instant takes its replay's colour, so LINE_PT
WARM_LABELS = ("measured", "sim., replay card", "sim., ladder card", "sim., re-fitted")
WARM_X_TICKS = (1, 10, 100)
WARM_Y_TICKS = (0.1, 1, 10, 100)
# every late-replica share is 1 from the 40-s bin to the end of the replay: the axis stops two
# bins after the last change
LATE_XLIM_S: Tuple[float, float] = (-3.0, 63.0)
LATE_XTICK_S = 20.0
# most shares sit at exactly 0 or 1: the headroom keeps those markers clear of the spines
LATE_YLIM: Tuple[float, float] = (-0.06, 1.06)


def load_records(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows).sort_values("offset_s").reset_index(drop=True)


def ready_offset_s(server_json: Optional[Path]) -> float:
    """Seconds after the burst onset at which the replica answered ``/health`` (<= 0: before)."""
    if server_json is None or not server_json.exists():
        return -math.inf
    meta = json.loads(server_json.read_text(encoding="utf-8"))
    return -float(meta["realised_lead_s"])


def build_stream(rec: pd.DataFrame, tail_min: int = 10) -> ArrivalStream:
    t = rec["offset_s"].to_numpy(dtype=float)
    horizon = int(math.ceil(t.max() / 60.0)) + tail_min
    minute = np.minimum((t // 60.0).astype(np.int64), horizon - 1)
    lam_in = np.bincount(minute, minlength=horizon).astype(float)
    zeros = np.zeros(horizon, dtype=float)
    prompt = (rec["prompt_chars"].to_numpy(dtype=float) / CHARS_PER_TOKEN
              + PROMPT_TEMPLATE_TOKENS).round().astype(np.int64)
    return ArrivalStream(t_arrive_s=t, kind=np.full(t.size, IN_CLASS),
                         session_id=rec["student"].to_numpy(dtype=np.int64),
                         prompt_tokens=prompt, output_tokens=rec["max_tokens"].to_numpy(np.int64),
                         lam_in=lam_in, lam_off=zeros, lam_hat_in=lam_in.copy(),
                         lam_hat_off=zeros, sessions=pd.DataFrame(), horizon_min=horizon)


def replicas_for(stream: ArrivalStream, ready_s: float) -> np.ndarray:
    x = np.ones(stream.horizon_min, dtype=float)
    if ready_s > 0:
        x[: min(stream.horizon_min, int(math.ceil(ready_s / 60.0)))] = 0.0
    return x


def _q(values: np.ndarray, q: float) -> float:
    v = values[np.isfinite(values)]
    return float(np.quantile(v, q)) if v.size else float("nan")


def compare_label(label: str, rec: pd.DataFrame, cfg: ServiceConfig, ready_s: float,
                  second_resolution: bool = True) -> tuple[Dict[str, object], pd.DataFrame]:
    """Measured against simulated TTFT for one replay label.

    A late replica is emulated at second resolution by default: every request that arrives before
    the measured readiness instant is held until it (the hold counts towards its TTFT) and the
    pool then holds one replica throughout.  With ``second_resolution=False`` the simulator's own
    minute granularity is used instead (no replica until the minute after readiness), which is
    what batch C does and is conservative by up to one minute on the replica tier.
    """
    t = rec["offset_s"].to_numpy(dtype=float)
    hold = np.zeros_like(t)
    if ready_s > 0 and second_resolution:
        hold = np.maximum(ready_s - t, 0.0)
        shifted = rec.assign(offset_s=t + hold)
        stream = build_stream(shifted)
        result = replay(stream, np.ones(stream.horizon_min), cfg)
    else:
        stream = build_stream(rec)
        result = replay(stream, replicas_for(stream, ready_s), cfg)
    sim_ttft = np.asarray(result.ttft_s, dtype=float) + hold
    sim_ttft[~np.asarray(result.served)] = np.nan
    meas = rec["ttft_s"].to_numpy(dtype=float)  # NaN = gave up (client limit)
    meas_aligned = np.where(meas > cfg.abandon_in_s, np.nan, meas)  # the simulator's 60 s rule
    n = len(rec)
    row: Dict[str, object] = {
        "label": label, "n_requests": n, "realised_ready_after_onset_s": ready_s,
        "meas_p50_s": _q(meas, 0.5), "meas_p95_s": _q(meas, 0.95), "meas_max_s": _q(meas, 1.0),
        "meas_share_slo": float(np.nanmean(meas <= SLO_S)) if n else float("nan"),
        "meas_gave_up": int(np.isnan(meas).sum()),
        "meas_timeouts_60s": int((meas > cfg.abandon_in_s).sum()),
        "meas_p95_aligned_s": _q(meas_aligned, 0.95),
        "meas_share_slo_aligned": float(np.sum(meas_aligned <= SLO_S) / n) if n else float("nan"),
        "sim_p50_s": _q(sim_ttft, 0.5), "sim_p95_s": _q(sim_ttft, 0.95),
        "sim_max_s": _q(sim_ttft, 1.0),
        "sim_share_slo": float(np.sum(sim_ttft <= SLO_S) / n) if n else float("nan"),
        "sim_timeouts": int(np.asarray(result.timed_out).sum()),
        "sim_rejected": int(np.asarray(result.rejected).sum()),
    }
    bins = []
    for b in range(int(rec["offset_s"].max() // BIN_S) + 1):
        lo, hi = b * BIN_S, (b + 1) * BIN_S
        sel = ((rec["offset_s"] >= lo) & (rec["offset_s"] < hi)).to_numpy()
        if not sel.any():
            continue
        m, sm = meas[sel], sim_ttft[sel]
        bins.append({"label": label, "bin_start_s": lo, "n": int(sel.sum()),
                     "meas_p95_s": _q(m, 0.95), "sim_p95_s": _q(sm, 0.95),
                     "meas_share_slo": float(np.nanmean(m <= SLO_S)),
                     "sim_share_slo": float(np.mean(sm <= SLO_S))})
    return row, pd.DataFrame(bins)


def printed_style(size_cm: Sequence[float], font_pt: float = FONT_PT) -> FigureConfig:
    """House style at the printed size; ``savefig.bbox`` standard keeps the page at ``size_cm``."""
    cfg = apply_style(FigureConfig(width_in=size_cm[0] / CM_PER_IN,
                                   height_in=size_cm[1] / CM_PER_IN,
                                   base_font_pt=font_pt, small_font_pt=font_pt, dpi=PRINT_DPI))
    plt.rcParams["savefig.bbox"] = "standard"
    return cfg


def _plain_log(ax, axis: str, ticks: Sequence[float]) -> None:
    """Plain-digit labels on a log axis; the default 10^k tick sets its exponent at 5.6 pt."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(list(ticks)))
    target.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    target.set_minor_locator(NullLocator())


def late_name(label: str) -> str:
    """Plain-language name of a late-replica replay: ``late30_x10`` -> "×10, started 30 s early".

    ``late<lead>`` launches the server ``lead`` seconds before the burst onset and ``x<k>`` replays
    the recorded burst at k times its intensity (``measurements.REPLAY_LATE``).
    """
    match = re.fullmatch(r"late(\d+)_x(\d+)", label)
    if match is None:
        return label.replace("_", " ")
    lead, scale = int(match.group(1)), int(match.group(2))
    start = "started at onset" if lead == 0 else f"started {lead} s early"
    return f"×{scale}, {start}"


def figure(table: pd.DataFrame, bins: pd.DataFrame, out: Path,
           size_cm: Sequence[float] = F10_SIZE_CM, font_pt: float = FONT_PT) -> None:
    """F10 at its printed size: warm replica p95 against scale (measured, simulator calibrated on
    the replay card, on the ladder card, re-fitted); late replica share-within-SLO over time.

    The late-replica legend has no room inside a 4.3 cm panel, so it sits under the figure: one
    colour per replay on the first row, and the measured (solid, circles) and simulated (dashed)
    line styles in neutral grey on the second.  The late-replica axis ends shortly after the last
    bin whose share moves (``LATE_XLIM_S``).
    """
    cfg = printed_style(size_cm, font_pt)
    warm = table[table["label"].str.startswith("warm")].copy()
    warm["scale"] = warm["label"].str.extract(r"x(\d+)").astype(int)
    warm = warm.sort_values("scale")
    late = [lab for lab in table["label"] if lab.startswith("late")]
    fig, axes = plt.subplots(1, 2, figsize=(cfg.width_in, cfg.height_in), layout="constrained")
    fig.get_layout_engine().set(w_pad=PAD_IN, h_pad=PAD_IN, wspace=0.03, hspace=0.0)
    ax = axes[0]
    if len(warm):
        # the four warm series take the palette in order: measured, then the three simulators
        ax.plot(warm["scale"], warm["meas_p95_s"], marker="o", markersize=MARKER_PT,
                linewidth=LINE_PT, color=cfg.palette[0], label=WARM_LABELS[0])
        ax.plot(warm["scale"], warm["sim_p95_s"], marker="s", markersize=MARKER_PT,
                linewidth=LINE_PT, linestyle="--", color=cfg.palette[1], label=WARM_LABELS[1])
        if "simref_p95_s" in warm:
            ax.plot(warm["scale"], warm["simref_p95_s"], marker="^", markersize=MARKER_PT,
                    linewidth=LINE_PT, linestyle=":", color=cfg.palette[2], label=WARM_LABELS[2])
        if "simfit_p95_s" in warm:
            ax.plot(warm["scale"], warm["simfit_p95_s"], marker="D", markersize=MARKER_PT,
                    linewidth=LINE_PT, linestyle="-.", color=cfg.palette[3], label=WARM_LABELS[3])
        ax.axhline(SLO_S, color=GREY, linewidth=REF_PT, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log(ax, "x", WARM_X_TICKS)
        _plain_log(ax, "y", WARM_Y_TICKS)
        ax.set_xlabel("Burst scale (× real class)")
        ax.set_ylabel("TTFT p95 (s)")
        # above the figure: four entries at 8 pt inside a 4.3 cm panel would sit on the curves
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=4,
                   frameon=False, fontsize=font_pt, handlelength=1.0, handletextpad=0.25,
                   columnspacing=0.5, borderpad=0.0, borderaxespad=0.1)
    ax = axes[1]
    swatches: List[Patch] = []
    for i, lab in enumerate(late[:4]):
        part = bins[bins["label"] == lab]
        # the scenarios continue down the palette after the four warm series of the left panel
        color = cfg.palette[(i + 4) % len(cfg.palette)]
        ax.plot(part["bin_start_s"], part["meas_share_slo"], color=color, marker="o",
                markersize=MARKER_PT, linewidth=LINE_PT)
        ax.plot(part["bin_start_s"], part["sim_share_slo"], color=color, linestyle="--",
                linewidth=LINE_PT)
        swatches.append(Patch(facecolor=color, edgecolor="none", label=late_name(lab)))
        ready = float(table.loc[table["label"] == lab, "realised_ready_after_onset_s"].iloc[0])
        if np.isfinite(ready):
            ax.axvline(ready, color=color, linewidth=LINE_PT, linestyle=":", zorder=1)
    ax.set_xlabel("Seconds after onset")
    ax.set_ylabel("Share within 2 s")
    ax.set_ylim(*LATE_YLIM)
    ax.set_xlim(*LATE_XLIM_S)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_locator(MultipleLocator(LATE_XTICK_S))
    for a in axes:
        a.tick_params(length=2.0, width=0.6, pad=1.5)
        for spine in a.spines.values():
            spine.set_linewidth(0.7)
    # one colour swatch per replay on the first row, the two line styles in neutral grey under the
    # first two: matplotlib fills a legend column by column, hence the interleaving
    samples = [Line2D([], [], color=GREY, marker="o", markersize=MARKER_PT, linewidth=LINE_PT,
                      label="measured"),
               Line2D([], [], color=GREY, linestyle="--", linewidth=LINE_PT, label="simulated")]
    handles = [h for pair in zip_longest(swatches, samples) for h in pair if h is not None]
    fig.legend(handles=handles, loc="outside lower center", ncol=max(1, len(swatches)),
               frameon=False, fontsize=font_pt, labelspacing=0.15, handlelength=1.1,
               handletextpad=0.3, columnspacing=0.6, borderpad=0.0, borderaxespad=0.0)
    save(fig, out, "f10_replay_validation")


PROBE_KEYS = ("ttft_p50_s", "ttft_p95_s", "e2e_p95_s", "throughput_req_s", "throughput_tokens_s",
              "error_rate", "requests_total")


def _summary_rows(path: Path, model_slug: str) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        return {}
    out: Dict[int, Dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if model_slug in str(row.get("model", "")) and not row.get("dry_run", False):
            out[int(row["level"])] = row
    return out


def probe_table(results: Path, out_name: str, reference: Path, model_slug: str = "qwen2.5-7b"
                ) -> pd.DataFrame:
    """The replay box's concurrency probe next to the batch-B (T3) curve, level by level."""
    probe = _summary_rows(results / out_name / "concurrency" / "concurrency_summary.jsonl",
                          model_slug)
    ref = _summary_rows(reference, model_slug)
    rows = []
    for level in sorted(set(probe) | set(ref)):
        row: Dict[str, object] = {"level": level, "in_probe": level in probe, "in_t3": level in ref}
        for key in PROBE_KEYS:
            row[f"probe_{key}"] = probe.get(level, {}).get(key, float("nan"))
            row[f"t3_{key}"] = ref.get(level, {}).get(key, float("nan"))
        for key in ("ttft_p95_s", "throughput_tokens_s"):
            a, b = row[f"probe_{key}"], row[f"t3_{key}"]
            row[f"ratio_{key}"] = (float(a) / float(b) if isinstance(a, (int, float))
                                   and isinstance(b, (int, float)) and float(b) > 0 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=PILOT / "scripts/q2/results/b2_remote")
    ap.add_argument("--out-name", default="b2")
    ap.add_argument("--t3", type=Path, default=None,
                    help="T3 CSV of the replay card (default: <results>/<out_name>/T3_calibration.csv)")
    ap.add_argument("--t3-ref", type=Path, default=T3_DEFAULT,
                    help="T3 CSV of the batch-B card, for the reference curve")
    ap.add_argument("--model-row", default=MODEL_ROW)
    ap.add_argument("--out", type=Path, default=PILOT / "results/q2/replay")
    ap.add_argument("--t3-summary", type=Path,
                    default=PILOT / "scripts/q2/results/b1_remote/b1/concurrency/concurrency_summary.jsonl")
    ap.add_argument("--decode-scale", type=float, default=1.0,
                    help="multiply the calibrated aggregate decode rate (a capacity re-fit to the "
                         "replay's own saturated drain rate; labelled as fitted)")
    ap.add_argument("--k-scale", type=float, default=1.0,
                    help="multiply the calibrated concurrency knee K (same caveat)")
    ap.add_argument("--draw-only", action="store_true",
                    help="redraw F10 from replay_compare.csv and replay_bins.csv under --out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.draw_only:
        figure(pd.read_csv(args.out / "replay_compare.csv"),
               pd.read_csv(args.out / "replay_bins.csv"), args.out)
        logger.info("redrew F10 from the tables in %s", args.out)
        return 0
    replay_dir = args.results / args.out_name / "replay"
    t3_card = args.t3 or (args.results / args.out_name / "T3_calibration.csv")
    base = with_lead_tier(ServiceConfig(), "d_rep")
    cfg_card = load_service_config(t3_card, args.model_row, base=base, lead_stat="median")
    cfg_ref = load_service_config(args.t3_ref, args.model_row, base=base, lead_stat="median")
    fitted = args.decode_scale != 1.0 or args.k_scale != 1.0
    cfg_fit = replace(cfg_card, decode_tokens_per_s=cfg_card.decode_tokens_per_s * args.decode_scale,
                      max_concurrency=int(round(cfg_card.max_concurrency * args.k_scale)))
    if fitted:
        logger.info("fitted capacity variant: K=%d decode=%.0f tok/s", cfg_fit.max_concurrency,
                    cfg_fit.decode_tokens_per_s)
    logger.info("service model (this card): K=%d decode=%.0f tok/s | (batch-B card): K=%d "
                "decode=%.0f tok/s", cfg_card.max_concurrency, cfg_card.decode_tokens_per_s,
                cfg_ref.max_concurrency, cfg_ref.decode_tokens_per_s)
    rows: List[Dict[str, object]] = []
    bins: List[pd.DataFrame] = []
    for path in sorted(replay_dir.glob("*.jsonl")):
        label = path.stem
        rec = load_records(path)
        if rec.empty:
            continue
        late = label.startswith("late")
        ready = ready_offset_s(replay_dir / f"{label}_server.json") if late else -math.inf
        row, part = compare_label(label, rec, cfg_card, ready)
        ref, _ = compare_label(label, rec, cfg_ref, ready)
        row.update({f"simref_{k[4:]}": v for k, v in ref.items() if k.startswith("sim_")})
        if fitted:
            fit, part = compare_label(label, rec, cfg_fit, ready)  # bins follow the fitted variant
            row.update({f"simfit_{k[4:]}": v for k, v in fit.items() if k.startswith("sim_")})
        if late:
            minute, _ = compare_label(label, rec, cfg_card, ready, second_resolution=False)
            row.update({f"simmin_{k[4:]}": v for k, v in minute.items() if k.startswith("sim_")})
        rows.append(row)
        bins.append(part)
        logger.info("%s: n=%d measured p95 %.2fs (slo %.3f) | sim(card) p95 %.2fs (slo %.3f) | "
                    "sim(ref) p95 %.2fs (slo %.3f)", label, row["n_requests"], row["meas_p95_s"],
                    row["meas_share_slo"], row["sim_p95_s"], row["sim_share_slo"],
                    row["simref_p95_s"], row["simref_share_slo"])
    if not rows:
        raise SystemExit(f"no replay records under {replay_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "replay_compare.csv", index=False)
    allbins = pd.concat(bins, ignore_index=True)
    allbins.to_csv(args.out / "replay_bins.csv", index=False)
    figure(table, allbins, args.out)
    probe = probe_table(args.results, args.out_name, args.t3_summary)
    if len(probe):
        probe.to_csv(args.out / "probe_vs_t3.csv", index=False)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
