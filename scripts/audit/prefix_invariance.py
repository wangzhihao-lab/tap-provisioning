#!/usr/bin/env python3
"""Causality check for the learned-timing plan.

The learned forecast of week ``w`` is built from the realised starts of weeks before ``w``, while
the planner solves one plan over the whole run at minute 0.  The plan is causal if its decisions up
to the end of every week are unchanged when the forecast is cut at the end of that week: no
decision then uses a forecast that would not yet exist when the decision is taken.

Run from the repository root (one case, one seed, cell C of the central runs)::

    uv run python scripts/audit/prefix_invariance.py --case c2 --seed 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np

PILOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PILOT / "scripts" / "sim"))
sys.path.insert(0, str(PILOT / "src"))
import run_sweep  # noqa: E402
from pilot.sim.engine import build_stream, make_context  # noqa: E402
from pilot.sim.policies.base import PolicyContext  # noqa: E402
from pilot.sim.policies.offline import round_to_replicas, solve_offline_lp  # noqa: E402

logger = logging.getLogger(__name__)

WEEK_MIN = 7 * 1440
CASES: Dict[str, Dict[str, str]] = {  # the command lines of the central learned-prior runs
    "c2": {"model": "qwen2.5-32b-awq", "p_in": "0.002500756"},
    "c3": {"model": "qwen2.5-72b-awq", "p_in": "0.003758436"},
}
CELL_C = dict(trigger_offset_mean_min=-8.7, trigger_offset_sd_min=5.3, trigger_offset_course_sd_min=4.0,
              trigger_offset_common_sd_min=0.0, forecast_offset_min=-8.7, forecast_offset_sd_min=6.6)


def sweep_args(case: str) -> list:
    """Arguments of ``run_sweep.py`` for the central run of ``case`` (instance tier, r = 30)."""
    spec = CASES[case]
    return ("--sweep offset --weeks 6 --seeds 10 --adoption 1.0 --min-replicas 1 --c-run 0.0333 "
            "--beta-start 0.0333 --lead-stat median --policies calendar_learn_prior "
            "--calibration scripts/q2/results/b4_remote/b4/ladder/rag_shared_4000/T3_calibration.csv "
            f"--model {spec['model']} --slo-in-s 2 --lead-tier d_inst --lead-min 8 --p-in {spec['p_in']} "
            "--learn-prior-from data/timetable/slots_2025-2026-2.csv").split()


def plan(hat_in: np.ndarray, hat_off: np.ndarray, ctx: PolicyContext) -> np.ndarray:
    """The timetable planner's plan for a forecast: the calls of ``CalendarFtPPolicy.trajectory``."""
    x_min = float(ctx.policy.min_replicas)
    z = solve_offline_lp(hat_in, hat_off, ctx.mu, ctx.cost, 1, x_min=x_min)
    return round_to_replicas(z, hat_in, hat_off, ctx.mu, ctx.cost, x_min=x_min)


def check(case: str, seed: int) -> int:
    """Print, per week boundary, how many minutes of the plan change when the forecast is cut there."""
    cfg = run_sweep.base_config(run_sweep.build_parser().parse_args(sweep_args(case)))
    cfg = replace(cfg, workload=replace(cfg.workload, **CELL_C), seed=seed)
    stream = build_stream(cfg)
    ctx = make_context(cfg, stream)
    hat_in, hat_off = stream.lam_hat_learn_prior_in, stream.lam_hat_off
    if hat_in is None:
        raise ValueError("the stream carries no learned-prior forecast; check --learn-prior-from")
    full = plan(hat_in, hat_off, ctx)
    print(f"{case} seed {seed}: plan {full.sum() / 60.0 / (hat_in.size / WEEK_MIN):.1f} GPU-h per week")
    changed = 0
    for w in range(1, hat_in.size // WEEK_MIN):
        cut = w * WEEK_MIN
        diff = np.flatnonzero(plan(hat_in[:cut], hat_off[:cut], ctx) != full[:cut])
        changed += diff.size
        print(f"  cut after week {w}: {diff.size} of {cut} minutes differ")
    return changed


def main() -> None:
    """Parse arguments and run the check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES), default="c2")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    changed = check(args.case, args.seed)
    print("prefix-invariant" if changed == 0 else f"NOT prefix-invariant: {changed} minutes differ")


if __name__ == "__main__":
    main()
