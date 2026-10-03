#!/usr/bin/env python3
"""Split the standing guard's excess over the optimum into the phases lemma 1 does and does not charge.

Lemma 1 (upper) charges S only for the guard it buys and does not need during the blind minutes
after each trigger, ``(c m + beta)(A_max - A_tau)/mu``.  In the simulator S also pays for what
the rectangular model has no room for: the shrink towards a burst whose decay is fatter than the
family profile (the rest of the profile window) and the delayed tracking of the in-session and
off-class background (outside every window).  This script replays the guard and the optimum on the
batch-C workload and attributes each minute's cost difference (running, penalty and the start
cost at the minute the level rises) to ``blind``, ``window`` or ``outside``.  Writes
``results/theory/t5_guard_decompose.csv``.  Run from the repository root.
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
from pilot.sim.policies.guard import PROFILE_MIN
from pilot.sim.service import load_service_config
from pilot.sim.sweeps import with_price_ratio

logger = logging.getLogger("t5_guard_decompose")
PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
MIN_PER_WEEK = 7 * 1440
PHASES = ("blind", "window", "outside")


def minute_cost(x: np.ndarray, lam_in: np.ndarray, lam_off: np.ndarray, mu: float,
                cost: CostConfig) -> np.ndarray:
    """Per-minute cost of a trajectory: running, penalties and the start cost where it rises."""
    x = np.asarray(x, dtype=float)
    served = mu * x
    short_in = np.maximum(lam_in - served, 0.0)
    short_off = np.maximum(lam_in + lam_off - served, 0.0) - short_in
    ups = np.maximum(np.diff(x, prepend=0.0), 0.0)
    return cost.c_run * x + cost.p_in * short_in + cost.p_off * short_off + cost.beta_start * ups


def phase_of(horizon: int, bells: np.ndarray, blind: int) -> np.ndarray:
    """Phase index per minute: 0 blind, 1 rest of the profile window, 2 outside every window."""
    phase = np.full(horizon, 2, dtype=int)
    for tau in bells:  # later triggers overwrite earlier windows, as the guard itself resolves them
        hi = min(horizon, tau + PROFILE_MIN)
        phase[tau:hi] = 1
        phase[tau:min(hi, tau + blind)] = 0
    return phase


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[10])
    ap.add_argument("--tiers", nargs="*", default=["d_rep", "d_inst"])
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=REPO / "results/theory/t5_guard_decompose.csv")
    ap.add_argument("--calibration", type=Path, default=PILOT / "scripts/q2/results/b1_remote/b1/T3_calibration.csv",
                    help="T3 calibration CSV (default: batch B, 7B)")
    ap.add_argument("--model", default="qwen2.5-7b", help="model row of the calibration CSV")
    ap.add_argument("--slo-in-s", type=float, default=None, help="in-class TTFT SLO seconds (default 2)")
    ap.add_argument("--lead-rep-min", type=float, default=None, help="replica-tier lead override (measured cold start)")
    ap.add_argument("--lead-inst-min", type=float, default=None, help="instance-tier lead override (e.g. 8)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    workload = WorkloadConfig(weeks=tuple(range(1, args.weeks + 1)), adoption=1.0, sigma=0.7)
    loaded = load_service_config(args.calibration, args.model,
                                 base=with_lead_tier(ServiceConfig(), "d_rep"), lead_stat="median",
                                 **({"slo_in_s": args.slo_in_s} if args.slo_in_s else {}))
    cost = replace(CostConfig(), c_run=0.0333, beta_start=0.0333)
    if args.slo_in_s:
        cost = replace(cost, slo_in_s=float(args.slo_in_s))
    overrides = {"d_rep": args.lead_rep_min, "d_inst": args.lead_inst_min}
    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        for tier in args.tiers:
            service = with_lead_tier(loaded, tier)
            if tier == "d_rep":
                service = replace(service, lead_min=loaded.lead_min)  # the measured start-up
            if overrides.get(tier) is not None:
                service = replace(service, lead_min=float(overrides[tier]))
            cfg = RunConfig(workload=workload, service=service, cost=cost,
                            policy=PolicyConfig(min_replicas=1), seed=seed, root=str(REPO))
            stream = engine.build_stream(cfg)
            mu = engine.fluid_mu(cfg, stream)
            cfg = with_price_ratio(cfg, mu, args.ratio)
            ctx = engine.make_context(cfg, stream)
            opt = engine.run_policy("offline", cfg, stream, ctx)
            guard = engine.run_policy("guard", cfg, stream, ctx)
            c_opt = minute_cost(opt.x, stream.lam_in, stream.lam_off, mu, cfg.cost)
            c_guard = minute_cost(guard.x, stream.lam_in, stream.lam_off, mu, cfg.cost)
            phase = phase_of(stream.horizon_min, ctx.bells, ctx.lead_min + 1)
            for w in range(args.weeks):
                lo, hi = w * MIN_PER_WEEK, min((w + 1) * MIN_PER_WEEK, stream.horizon_min)
                for i, name in enumerate(PHASES):
                    sel = np.zeros(stream.horizon_min, dtype=bool)
                    sel[lo:hi] = phase[lo:hi] == i
                    rows.append({"seed": seed, "tier": tier, "week": w + 1, "phase": name,
                                 "minutes": int(sel.sum()), "guard_cost_usd": float(c_guard[sel].sum()),
                                 "opt_cost_usd": float(c_opt[sel].sum()),
                                 "excess_usd": float((c_guard[sel] - c_opt[sel]).sum()),
                                 "guard_extra_replica_min": float(np.maximum(guard.x[sel] - opt.x[sel], 0).sum())})
            logger.info("seed %d %s: guard %.1f vs offline %.1f USD, guard slo_in %.3f", seed, tier,
                        guard.metrics.total_cost, opt.metrics.total_cost, guard.metrics.slo_in)
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    logger.info("wrote %s\n%s", args.out, frame.pivot_table(index=["tier", "week"], columns="phase",
                                                            values="excess_usd").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
