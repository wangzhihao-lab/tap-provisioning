"""One bell, minute by minute: arrivals, replicas and in-class TTFT under a few policies (F1, right).

    uv run python scripts/sim/single_bell_replay.py                       # c3, instance tier, cell C
    uv run python scripts/sim/single_bell_replay.py --week 3 --window -20 40

Builds the validation round's central 72B case (shared 4k context, r = 30, the measured
independent timing dispersion with a distribution-aware planner = cell C) at the instance tier
(8 min lead), runs the offline optimum, the free-tuned token-velocity baseline, the SLO-floor-tuned
reactive baseline and the timetable planner with learned timing (previous-semester prior), and
writes, for the busiest bell of the chosen week, one row per minute of the window:
in-class arrivals, replicas of every policy, and the in-class TTFT p95 of every policy.
Output: ``results/sim/validation/f1_right_bell.csv`` (+ a draft figure next to the other
validation figures).  Numbers come from the same engine and calibration as the sweeps.
``--with-tap`` also runs TAP (``tap``) on the same stream and adds its columns to the table and its
whole-run numbers to the JSON; the draft figure keeps the four policies above.
``--tuned-disp`` names the ``best.json`` of the SLO-floor-tuned reactive baseline (default: the
validation round's ``tuning8_r30_slo95disp``); the JSON records that file and its reactive knobs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator,  # noqa: E402
                               MultipleLocator, NullLocator)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style, save  # noqa: E402
from pilot.sim import CostConfig, PolicyConfig, RunConfig, ServiceConfig, WorkloadConfig, engine  # noqa: E402
from pilot.sim.config import with_lead_tier  # noqa: E402
from pilot.sim.service import load_service_config  # noqa: E402
from pilot.sim.sweeps import with_price_ratio  # noqa: E402
from pilot.sim.workload import IN_CLASS  # noqa: E402
from tuned_config import apply_policy_values, policy_overrides  # noqa: E402

logger = logging.getLogger("single_bell_replay")

LADDER = PILOT / "scripts/q2/results/b4_remote/b4/ladder/rag_shared_4000/T3_calibration.csv"
MODEL = "qwen2.5-72b-awq"
CASE = "c3_72b_shared4k"
MIRROR = PILOT / "outputs/validation_remote/box2"
TUNED_FREE = MIRROR / "validation" / CASE / "tuning8_r30" / "best.json"
TUNED_DISP = MIRROR / "validation2" / CASE / "tuning8_r30_slo95disp" / "best.json"
PRIOR_SLOTS = "data/timetable/slots_2025-2026-2.csv"
POLICIES = ("offline", "velocity", "reactive", "calendar_learn_prior")
LABELS = {"offline": "Offline optimum", "velocity": "Token velocity (free-tuned)",
          "reactive": "Reactive (HPA), SLO-floor tuned", "calendar_learn_prior": "Timetable + learned timing"}
# Printed size of F1(right): one of the two panels of a one-column figure, included with
# \includegraphics[height=4.6cm].  The legend uses the short names because five entries at 8 pt
# have 4.1 cm of width to fit into; the caption carries the full names of ``LABELS``.
CM_PER_IN = 2.54
FONT_PT = 8.0
PRINT_DPI = 600
F1_RIGHT_CM: Tuple[float, float] = (4.1, 4.6)
HEIGHT_RATIOS: Tuple[float, float] = (1.0, 1.15)
PAD_IN = 0.012
SHORT = {"offline": "Offline", "velocity": "Velocity", "reactive": "Reactive",
         "calendar_learn_prior": "Timetable"}
DEMAND_LABEL = "Demand"


def base_config(seed: int, weeks: int, lead_min: float, tuned: Optional[Path]) -> RunConfig:
    """Cell C of the validation round at the instance tier, with the given baseline tuning."""
    workload = WorkloadConfig(weeks=tuple(range(1, weeks + 1)), adoption=1.0, sigma=0.7,
                              trigger_offset_mean_min=-8.7, trigger_offset_sd_min=5.3,
                              trigger_offset_course_sd_min=4.0, trigger_offset_common_sd_min=0.0,
                              forecast_offset_min=-8.7, forecast_offset_sd_min=6.6,
                              learn_prior_slots_csv=PRIOR_SLOTS)
    service = with_lead_tier(ServiceConfig(), "d_inst")
    loaded = load_service_config(LADDER, MODEL, base=service, lead_stat="median", slo_in_s=2.0)
    service = replace(with_lead_tier(loaded, "d_inst"), lead_min=float(lead_min))
    cost = replace(CostConfig(), c_run=0.0333, beta_start=0.0333, slo_in_s=2.0)
    policy = PolicyConfig(min_replicas=1)
    overrides = policy_overrides(tuned, None) if tuned and tuned.exists() else {}
    if overrides:
        policy = apply_policy_values(policy, overrides)
    return RunConfig(workload=workload, service=service, cost=cost, policy=policy, seed=seed, root=str(REPO))


def tuning_record(path: Path) -> Dict[str, object]:
    """The reactive tuning file (repo-relative where possible) and the reactive knobs it sets."""
    try:
        name = str(path.resolve().relative_to(REPO))
    except ValueError:
        name = str(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"path": name, "reactive": payload["policies"]["reactive"]["config"]}


def busiest_bell(stream, week: int) -> int:
    """The bell minute of ``week`` with the most booked seats."""
    s = stream.sessions
    part = s[s["week"] == week]
    seats = part.groupby("bell_min")["roster"].sum()
    return int(seats.idxmax())


def minute_ttft_p95(result, stream, lo: int, hi: int) -> np.ndarray:
    """In-class TTFT p95 per minute of the window; NaN where no in-class request arrived."""
    minute = np.minimum((stream.t_arrive_s / 60.0).astype(np.int64), stream.horizon_min - 1)
    out = np.full(hi - lo, np.nan)
    for m in range(lo, hi):
        sel = (minute == m) & (stream.kind == IN_CLASS)
        if not sel.any():
            continue
        t = result.ttft_s[sel]
        t = np.where(np.isfinite(t), t, 1e6)  # never-started requests count as unbounded waits
        out[m - lo] = float(np.percentile(t, 95))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Single-bell replay for F1 (right)")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--weeks", type=int, default=3)
    ap.add_argument("--week", type=int, default=3, help="week whose busiest bell is replayed")
    ap.add_argument("--lead-min", type=float, default=8.0)
    ap.add_argument("--ratio", type=float, default=30.0)
    ap.add_argument("--window", type=int, nargs=2, default=(-20, 40), help="minutes around the bell")
    ap.add_argument("--out", type=Path, default=REPO / "results/sim/validation")
    ap.add_argument("--draw-only", action="store_true", help="redraw from the CSV/JSON written earlier")
    ap.add_argument("--plot-window", type=int, nargs=2, default=(-20, 25), help="minutes shown in the figure")
    ap.add_argument("--ttft-cap-s", type=float, default=120.0, help="TTFT values above this are drawn at the cap")
    ap.add_argument("--with-tap", action="store_true", help="also replay TAP and add its columns")
    ap.add_argument("--tuned-disp", type=Path, default=TUNED_DISP,
                    help="best.json of the SLO-floor-tuned reactive baseline")
    args = ap.parse_args(argv)
    if not args.draw_only and not args.tuned_disp.is_file():  # base_config would ignore it silently
        ap.error(f"--tuned-disp {args.tuned_disp}: no such file")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("pilot.sim").setLevel(logging.WARNING)
    if args.draw_only:
        table = pd.read_csv(args.out / "f1_right_bell.csv")
        meta = json.loads((args.out / "f1_right_bell.json").read_text())
        draw(table, meta, args.out / "figures", args.plot_window, args.ttft_cap_s)
        return 0

    cfg_free = base_config(args.seed, args.weeks, args.lead_min, TUNED_FREE)
    cfg_disp = base_config(args.seed, args.weeks, args.lead_min, args.tuned_disp)
    stream = engine.build_stream(cfg_free)  # the stream does not depend on the policy knobs
    mu = engine.fluid_mu(cfg_free, stream)
    cfg_free = with_price_ratio(cfg_free, mu, args.ratio)
    cfg_disp = with_price_ratio(cfg_disp, mu, args.ratio)
    ctx_free = engine.make_context(cfg_free, stream)
    ctx_disp = engine.make_context(cfg_disp, stream)
    runs = {"offline": engine.run_policy("offline", cfg_free, stream, ctx_free),
            "velocity": engine.run_policy("velocity", cfg_free, stream, ctx_free),
            "reactive": engine.run_policy("reactive", cfg_disp, stream, ctx_disp),
            "calendar_learn_prior": engine.run_policy("calendar_learn_prior", cfg_free, stream, ctx_free)}
    if args.with_tap:  # TAP reads no tuned knob, so either config gives the same run
        runs["tap"] = engine.run_policy("tap", cfg_free, stream, ctx_free)
    bell = busiest_bell(stream, args.week)
    lo, hi = bell + args.window[0], bell + args.window[1]
    minutes = np.arange(lo, hi)
    table = pd.DataFrame({"minute": minutes, "minute_from_bell": minutes - bell,
                          "arrivals_in": stream.lam_in[lo:hi], "arrivals_off": stream.lam_off[lo:hi],
                          "forecast_in": stream.lam_hat_in[lo:hi]})
    for name, run in runs.items():
        table[f"replicas_{name}"] = run.x[lo:hi]
        table[f"ttft_p95_{name}"] = minute_ttft_p95(run.service, stream, lo, hi)
    meta = {"case": CASE, "seed": args.seed, "week": args.week, "bell_min": bell, "lead_min": args.lead_min,
            "ratio": args.ratio, "mu_req_min": mu, "seats_at_bell": int(stream.sessions[(stream.sessions["week"] == args.week)
                                                                           & (stream.sessions["bell_min"] == bell)]["roster"].sum()),
            "whole_run_slo_in": {n: float(r.metrics.slo_in) for n, r in runs.items()},
            "whole_run_gpu_hours": {n: float(r.metrics.gpu_hours) for n, r in runs.items()},
            "tuned_disp": tuning_record(args.tuned_disp)}
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "f1_right_bell.csv", index=False)
    (args.out / "f1_right_bell.json").write_text(json.dumps(meta, indent=2))
    logger.info("bell at minute %d (week %d, %d seats); wrote %s", bell, args.week, meta["seats_at_bell"],
                args.out / "f1_right_bell.csv")
    draw(table, meta, args.out / "figures", args.plot_window, args.ttft_cap_s)
    return 0


def printed_style(size_cm: Sequence[float], font_pt: float = FONT_PT) -> FigureConfig:
    """House style at the printed size: the PDF page is ``size_cm`` and every text ``font_pt``.

    ``savefig.bbox`` is pinned to ``standard``: a tight bounding box would change the page size,
    LaTeX would rescale the panel, and the text would print at some other point size.
    """
    cfg = apply_style(FigureConfig(width_in=size_cm[0] / CM_PER_IN, height_in=size_cm[1] / CM_PER_IN,
                                   base_font_pt=font_pt, small_font_pt=font_pt, dpi=PRINT_DPI))
    plt.rcParams["savefig.bbox"] = "standard"
    return cfg


def draw(table: pd.DataFrame, meta: Dict[str, object], out: Path, window: Sequence[int] = (-20, 25),
         cap_s: float = 120.0, size_cm: Sequence[float] = F1_RIGHT_CM,
         font_pt: float = FONT_PT) -> None:
    """F1 (right) at its printed size: arrivals and replicas on top, in-class TTFT p95 below."""
    cfg = printed_style(size_cm, font_pt)
    part = table[(table["minute_from_bell"] >= window[0]) & (table["minute_from_bell"] < window[1])]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(cfg.width_in, cfg.height_in), sharex=True,
                                   height_ratios=list(HEIGHT_RATIOS), layout="constrained")
    fig.get_layout_engine().set(w_pad=PAD_IN, h_pad=PAD_IN, wspace=0.0, hspace=0.02)
    t = part["minute_from_bell"].to_numpy()
    ax0.fill_between(t, 0, part["arrivals_in"] / float(meta["mu_req_min"]), step="mid", color="0.85",
                     label=DEMAND_LABEL)
    for i, name in enumerate(POLICIES):
        ax0.step(t, part[f"replicas_{name}"], where="mid", color=cfg.palette[i], linewidth=1.0,
                 linestyle=cfg.linestyles[i % 4], label=SHORT[name])
        ax1.plot(t, np.minimum(part[f"ttft_p95_{name}"], cap_s), color=cfg.palette[i], linewidth=1.0,
                 linestyle=cfg.linestyles[i % 4], marker=cfg.markers[i], markersize=2.0,
                 label=SHORT[name])
    ax1.axhline(2.0, color="0.5", linewidth=0.7, linestyle=(0, (2, 2)))
    # both notes sit in clear space: the SLO one at the right, where every curve is at ~1 s, the
    # cap one in the headroom above the 120 s ceiling of the data
    ax1.text(window[1] - 0.5, 2.3, "SLO 2 s", va="bottom", ha="right", fontsize=font_pt,
             color="0.35")
    ax1.text(window[0] + 0.5, cap_s * 6.5, f">={cap_s:.0f} s: not served", va="top", ha="left",
             fontsize=font_pt, color="0.35")
    ax1.set_yscale("log")
    ax1.set_ylim(0.7, cap_s * 8.0)
    ax1.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax1.yaxis.set_major_formatter(FixedFormatter(["1", "10", "100"]))
    ax1.yaxis.set_minor_locator(NullLocator())  # 10^k ticks would carry a 5.6 pt exponent
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    for ax in (ax0, ax1):
        ax.axvline(0, color="0.6", linewidth=0.7)
        ax.tick_params(length=2.0, width=0.6, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    ax0.set_ylabel("Replicas")
    ax1.set_ylabel("TTFT p95 (s)")
    ax1.set_xlabel("Minutes from the bell")
    ax0.set_ylim(0, part[[f"replicas_{n}" for n in POLICIES]].to_numpy().max() * 1.15)
    ax0.yaxis.set_major_locator(MultipleLocator(5))
    fig.legend(*ax0.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False,
               fontsize=font_pt, labelspacing=0.15, handlelength=1.3, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0.0, borderpad=0.0)
    save(fig, out, "f1_right_single_bell")


if __name__ == "__main__":
    raise SystemExit(main())
