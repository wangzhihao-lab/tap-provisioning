#!/usr/bin/env python3
"""Measure Lemma H's inequality on the calibrated batch-C workload.

Lemma H pins the offline optimum above the lower envelope by one start per
bell:  C(X) >= sum_t c max(x_min, lambda_t / mu) + beta sum_tau A_tau / mu - beta TV(b) / mu.
This script evaluates both sides per teaching week on the batch-C r = 30, d_rep workload (seeds
10-14): the left side from the offline DP's realised trajectory, the right side from the realised
per-minute demand.  ``A_tau`` is the window-mean in-class rate over the first ``ell_int`` minutes of
each trigger (bell or later in-session prompt) minus the in-session background (mean rate 15-45 min later); ``b_t`` is the
demand outside those windows, smoothed over 15 minutes, and its total variation is an upper
estimate of the model's ``TV(b)`` because Poisson noise survives the smoothing.

Writes ``results/theory/t5_lemma_h_check.csv``.  Run from the repository root.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pilot.sim import PolicyConfig, RunConfig, ServiceConfig, WorkloadConfig, engine
from pilot.sim.config import CostConfig, with_lead_tier
from pilot.sim.metrics import fluid_cost
from pilot.sim.service import load_service_config
from pilot.sim.sweeps import with_price_ratio

logger = logging.getLogger("t5_lemma_h_check")
PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
MIN_PER_WEEK = 7 * 1440
ELL_INT = 3  # sigma_2 = 0.70 -> ell = 2.86 min, integer slot 3 (instantiation.csv)
BG_FROM, BG_TO = 15, 45  # in-session background window, minutes after the bell
SMOOTH_MIN = 15




def burst_amplitudes(lam_in: np.ndarray, bells: np.ndarray) -> np.ndarray:
    """``A_tau``: window-mean in-class rate above the in-session background, per bell."""
    amps = []
    for tau in bells:
        win = lam_in[tau: tau + ELL_INT]
        bg = lam_in[tau + BG_FROM: tau + BG_TO]
        if win.size < ELL_INT or bg.size == 0:
            amps.append(0.0)
            continue
        amps.append(max(0.0, float(win.mean()) - float(bg.mean())))
    return np.asarray(amps)


def background_tv(lam: np.ndarray, bells: np.ndarray, lo: int, hi: int) -> float:
    """Total variation of the smoothed non-burst demand over ``[lo, hi)``."""
    b = lam[lo:hi].astype(float).copy()
    for tau in bells:
        a, z = max(lo, tau), min(hi, tau + ELL_INT + 2)
        if z > a:
            b[a - lo: z - lo] = np.nan
    idx = np.arange(b.size)
    ok = ~np.isnan(b)
    if ok.sum() < 2:
        return 0.0
    b = np.interp(idx, idx[ok], b[ok])
    kernel = np.ones(SMOOTH_MIN) / SMOOTH_MIN
    sm = np.convolve(b, kernel, mode="same")
    return float(np.abs(np.diff(sm)).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[10, 11, 12, 13, 14])
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=REPO / "results/theory/t5_lemma_h_check.csv")
    ap.add_argument("--calibration", type=Path, default=PILOT / "scripts/q2/results/b1_remote/b1/T3_calibration.csv",
                    help="T3 calibration CSV (default: batch B, 7B)")
    ap.add_argument("--model", default="qwen2.5-7b", help="model row of the calibration CSV")
    ap.add_argument("--slo-in-s", type=float, default=None, help="in-class TTFT SLO seconds (default 2)")
    ap.add_argument("--lead-rep-min", type=float, default=None, help="replica-tier lead override (measured cold start)")
    ap.add_argument("--lead-inst-min", type=float, default=None, help="instance-tier lead override (e.g. 8)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    workload = WorkloadConfig(weeks=tuple(range(1, args.weeks + 1)), adoption=1.0, sigma=0.7)
    service = with_lead_tier(ServiceConfig(), "d_rep")
    loaded = load_service_config(args.calibration, args.model, base=service, lead_stat="median",
                                 **({"slo_in_s": args.slo_in_s} if args.slo_in_s else {}))
    service = replace(with_lead_tier(loaded, "d_rep"), lead_min=loaded.lead_min)
    if args.lead_rep_min is not None:
        service = replace(service, lead_min=float(args.lead_rep_min))
    cost = replace(CostConfig(), c_run=0.0333, beta_start=0.0333)
    if args.slo_in_s:
        cost = replace(cost, slo_in_s=float(args.slo_in_s))
    policy = PolicyConfig(min_replicas=1)
    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        cfg = RunConfig(workload=workload, service=service, cost=cost, policy=policy, seed=seed,
                        root=str(REPO))
        stream = engine.build_stream(cfg)
        mu = engine.fluid_mu(cfg, stream)
        cfg = with_price_ratio(cfg, mu, args.ratio)
        ctx = engine.make_context(cfg, stream)
        opt = engine.run_policy("offline", cfg, stream, ctx)
        x = np.asarray(opt.x, dtype=float)
        lam_in, lam_off = stream.lam_in, stream.lam_off
        bells = engine.trigger_grid(cfg, stream)  # bells and the later in-session triggers
        amps = burst_amplitudes(lam_in, bells)
        x_min = float(policy.min_replicas)
        for w in range(args.weeks):
            lo, hi = w * MIN_PER_WEEK, min((w + 1) * MIN_PER_WEEK, x.size)
            prev = float(x[lo - 1]) if lo > 0 else 0.0
            brk = fluid_cost(x[lo:hi], lam_in[lo:hi], lam_off[lo:hi], mu, cfg.cost,
                             cfg.service.billing_block_min, x_prev=prev)
            envelope = float(cfg.cost.c_run * np.maximum(x_min, (lam_in[lo:hi] + lam_off[lo:hi]) / mu).sum())
            in_week = (bells >= lo) & (bells < hi)
            sum_a = float(amps[in_week].sum())
            tv = background_tv(lam_in + lam_off, bells, lo, hi)
            starts_usd = cfg.cost.beta_start * sum_a / mu
            tv_usd = cfg.cost.beta_start * tv / mu
            lhs = brk.total_cost - envelope
            rows.append({"seed": seed, "week": w + 1, "n_bells": int(in_week.sum()),
                         "offline_total_usd": brk.total_cost, "envelope_usd": envelope,
                         "lhs_excess_usd": lhs, "starts_bound_usd": starts_usd,
                         "tv_slack_usd": tv_usd, "rhs_usd": starts_usd - tv_usd,
                         "offline_switch_usd": brk.switch_cost, "offline_penalty_usd": brk.penalty_cost,
                         "holds": bool(lhs >= starts_usd - tv_usd),
                         "holds_without_tv_slack": bool(lhs >= starts_usd)})
        logger.info("seed %d: %d bells, mu %.1f req/min", seed, bells.size, mu)
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("week")[["lhs_excess_usd", "starts_bound_usd", "tv_slack_usd",
                                     "rhs_usd", "offline_switch_usd", "offline_penalty_usd"]].mean()
    summary["holds_all"] = frame.groupby("week")["holds"].all()
    summary["holds_no_slack_all"] = frame.groupby("week")["holds_without_tv_slack"].all()
    logger.info("wrote %s\n%s", args.out, summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
