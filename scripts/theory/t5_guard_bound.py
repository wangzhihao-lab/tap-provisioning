#!/usr/bin/env python3
"""Lemma 1's upper bound on the standing guard S, evaluated on the simulated family itself.

Lemma 1 (upper) charges S, per bell, ``(c m + beta)(A_max - A_tau) / mu`` - the guard it buys and
does not need - plus background terms it shares with the optimum.  This script evaluates that sum
per teaching week with the simulator's own family parameters: the bell grid and the per-bell
forecast peaks ``A_tau`` of ``lam_hat_in``, ``A_max`` their maximum, and two blind times ``m``: the
theory's ``min(ell, d)`` and the simulator's ``d + 1`` minutes (a minute's arrivals are revealed
one minute late).  Writes ``results/theory/t5_guard_bound.csv``.  Run from the repository root.
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
from pilot.sim.config import LEAD_TIERS, CostConfig, with_lead_tier
from pilot.sim.policies.guard import PROFILE_MIN
from pilot.sim.service import load_service_config

logger = logging.getLogger("t5_guard_bound")
PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
MIN_PER_WEEK = 7 * 1440
ELL = 2.0 / 0.70  # rectangular burst length at sigma_2 = 0.70


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--out", type=Path, default=REPO / "results/theory/t5_guard_bound.csv")
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
    cost = replace(CostConfig(), c_run=0.0333, beta_start=0.0333)
    if args.slo_in_s:
        cost = replace(cost, slo_in_s=float(args.slo_in_s))
    cfg = RunConfig(workload=workload, service=with_lead_tier(loaded, "d_rep"), cost=cost,
                    policy=PolicyConfig(min_replicas=1), seed=args.seed, root=str(REPO))
    lead_override = {"d_rep": args.lead_rep_min if args.lead_rep_min is not None else loaded.lead_min,
                     "d_inst": args.lead_inst_min}
    stream = engine.build_stream(cfg)
    mu = engine.fluid_mu(cfg, stream)
    bells = engine.trigger_grid(cfg, stream)  # bells and the later in-session triggers
    peaks = np.array([float(stream.lam_hat_in[b: b + PROFILE_MIN].max()) for b in bells])
    a_max = float(peaks.max())
    rows: List[Dict[str, object]] = []
    for tier, spec in LEAD_TIERS.items():
        d = float(spec.lead_min)
        if lead_override.get(tier) is not None:
            d = float(lead_override[tier])
        blind = float(int(np.ceil(d - 1e-9)) + 1)
        for w in range(args.weeks):
            lo, hi = w * MIN_PER_WEEK, (w + 1) * MIN_PER_WEEK
            sel = (bells >= lo) & (bells < hi)
            unused = float((a_max - peaks[sel]).sum())
            for label, m in (("theory m = min(ell, d)", min(ELL, d)), ("simulator blind d + 1", blind)):
                rows.append({"tier": tier, "d_min": d, "week": w + 1, "n_bells": int(sel.sum()),
                             "a_max_req_min": a_max, "a_max_replicas": a_max / mu,
                             "sum_unused_replicas": unused / mu, "m_label": label, "m_min": m,
                             "bound_usd": (cost.c_run * m + cost.beta_start) * unused / mu,
                             "full_guard_usd": (cost.c_run * m + cost.beta_start) * a_max * int(sel.sum()) / mu})
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    logger.info("seed %d: %d bells, A_max %.0f req/min = %.1f replicas, mu %.1f", args.seed,
                bells.size, a_max, a_max / mu, mu)
    logger.info("wrote %s\n%s", args.out, frame.groupby(["tier", "m_label"])[["bound_usd", "full_guard_usd"]]
                .mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
