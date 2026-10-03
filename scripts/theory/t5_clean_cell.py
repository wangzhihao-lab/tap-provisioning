#!/usr/bin/env python3
"""The T5 'clean' cell: the calendar planner with an unbiased size forecast.

Lemma H bounds the excess a ramp-or-hold calendar policy pays for the *timetable drift* of repeat
weeks (R2) and of the freshman step (R1.5); the batch-C calendar policy also pays for its size
forecast (it books the whole roster while 70-95% attend) and for integer rounding, which the lemma
does not cover.  This cell removes the size bias - the planner books the mean attendance and the
roster is known exactly - so that the per-week excess it still pays is drift plus rounding plus the
residual attendance randomness, the closest the simulator gets to the lemma's quantity.

Writes ``results/theory/t5_<cell>_cell.csv`` (week, excess_usd_mean, excess_usd_sd, n_seeds,
slo_in_mean) for ``--cell unbiased`` (attendance still random) or ``--cell exact`` (session sizes
known exactly: attendance fixed at its mean, no cancellations).
Run from the repository root: ``uv run python scripts/theory/t5_clean_cell.py``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pilot.sim import PolicyConfig, RunConfig, ServiceConfig, WorkloadConfig, engine
from pilot.sim.config import CostConfig, with_lead_tier
from pilot.sim.metrics import weekly_rows
from pilot.sim.service import load_service_config
from pilot.sim.sweeps import with_price_ratio

logger = logging.getLogger("t5_clean_cell")
PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[10, 11, 12, 13, 14])
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=30.0)
    ap.add_argument("--cell", choices=("unbiased", "exact"), default="unbiased",
                    help="unbiased: roster known, mean attendance booked, attendance still random; "
                         "exact: attendance fixed at its mean and no cancellations, so the planner's "
                         "session sizes are exact and only arrival randomness and rounding remain")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    out = args.out or REPO / f"results/theory/t5_{args.cell}_cell.csv"
    base_wl = WorkloadConfig(weeks=tuple(range(1, args.weeks + 1)), adoption=1.0, sigma=0.7,
                             roster_noise=0.0, forecast_attendance=0.825)  # mean of U(0.70, 0.95)
    if args.cell == "exact":
        base_wl = replace(base_wl, attendance_lo=0.825, attendance_hi=0.825, p_cancel=0.0)
    # Same construction as the batch-C lead sweep's d_rep cell (the source of main_r30.csv): the
    # calibrated 7B service moved onto the replica tier, whose nominal lead is 0.5 min.
    service = with_lead_tier(ServiceConfig(), "d_rep")
    loaded = load_service_config(PILOT / "scripts/q2/results/b1_remote/b1/T3_calibration.csv",
                                 "qwen2.5-7b", base=service,
                                 lead_stat="median")
    service = with_lead_tier(loaded, "d_rep")
    cost = replace(CostConfig(), c_run=0.0333, beta_start=0.0333)
    rows = []
    for seed in args.seeds:
        cfg = RunConfig(workload=base_wl, service=service, cost=cost,
                        policy=PolicyConfig(min_replicas=1), seed=seed, root=str(REPO))
        stream = engine.build_stream(cfg)
        cfg = with_price_ratio(cfg, engine.fluid_mu(cfg, stream), args.ratio)
        runs = engine.run_all(cfg, policies=("calendar", "offline"), stream=stream)
        ctx = engine.make_context(cfg, stream)
        weeks = cfg.workload.weeks
        opt = weekly_rows("offline", seed, runs["offline"].x, stream, runs["offline"].service,
                          ctx.mu, cfg.cost, weeks, billing_block_min=cfg.service.billing_block_min)
        cal = weekly_rows("calendar", seed, runs["calendar"].x, stream, runs["calendar"].service,
                          ctx.mu, cfg.cost, weeks, billing_block_min=cfg.service.billing_block_min)
        for o, c in zip(opt, cal):
            rows.append({"seed": seed, "week": int(c["week"]),
                         "excess_usd": float(c["total_cost"]) - float(o["total_cost"]),
                         "slo_in": float(c["slo_in"])})
        logger.info("seed %d: whole-run calendar %.1f vs offline %.1f USD", seed,
                    runs["calendar"].metrics.total_cost, runs["offline"].metrics.total_cost)
    frame = pd.DataFrame(rows)
    table = (frame.groupby("week")["excess_usd"].agg(["mean", "std", "count"]).reset_index()
             .rename(columns={"mean": "excess_usd_mean", "std": "excess_usd_sd", "count": "n_seeds"}))
    table["slo_in_mean"] = frame.groupby("week")["slo_in"].mean().to_numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logger.info("wrote %s\n%s", out, table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
