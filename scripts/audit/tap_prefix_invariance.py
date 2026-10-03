#!/usr/bin/env python3
"""Causality check for TAP (``tap``) and its advice, the way ``prefix_invariance.py`` checks the plan.

TAP's advice plans on the learned-timing forecast warm-started from an earlier semester; the part
of that forecast for week ``w`` is built from realised starts of weeks before ``w``, and the plan
is solved over the whole horizon at minute 0 and re-solved over the rest of it after every size
correction.  A committed decision could therefore depend on a later week's forecast.  It does not
if the decisions for the slots of weeks ``1..w`` come out the same when every forecast array ends
at week ``w`` (the realised demand is cut there too; a policy never reads it ahead anyway, see
``tests/test_sim_tap.py``).  For each seed and week boundary this counts the minutes where TAP's
committed decision or its advice sequence differs, and checks that the advice run on its own
produces the sequence it produces inside TAP (BDINTERP-d needs the two black boxes to be
sequences of their own).

Run from the repository root (central runs: cell C, instance tier, r = 30)::

    uv run python scripts/audit/tap_prefix_invariance.py --case c2 --seeds 10 11 --out audit.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PILOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PILOT / "scripts" / "sim"))
sys.path.insert(0, str(PILOT / "scripts" / "audit"))
sys.path.insert(0, str(PILOT / "src"))
import run_sweep  # noqa: E402
from prefix_invariance import CASES, CELL_C, WEEK_MIN  # noqa: E402
from pilot.sim.engine import build_stream, make_context  # noqa: E402
from pilot.sim.policies.base import Policy, PolicyContext, SimState  # noqa: E402
from pilot.sim.policies.tap import LearnedPriorAdaptPolicy, TAPPolicy  # noqa: E402

logger = logging.getLogger(__name__)


def sweep_args(case: str) -> List[str]:
    """``run_sweep.py`` arguments of the central TAP run of ``case`` (the R1 instance-tier job)."""
    spec = CASES[case]
    return ("--sweep offset --weeks 6 --seeds 10 --adoption 1.0 --min-replicas 1 --c-run 0.0333 "
            "--beta-start 0.0333 --lead-stat median --policies tap "
            "--calibration scripts/q2/results/b4_remote/b4/ladder/rag_shared_4000/T3_calibration.csv "
            f"--model {spec['model']} --slo-in-s 2 --lead-tier d_inst --lead-min 8 --p-in {spec['p_in']} "
            "--learn-prior-from data/timetable/slots_2025-2026-2.csv").split()


def cut_context(ctx: PolicyContext, cut: int) -> PolicyContext:
    """``ctx`` with every forecast (and the trigger grid) ending at minute ``cut``."""
    def head(arr):
        return None if arr is None else arr[:cut]
    bells = None if ctx.bells is None else ctx.bells[ctx.bells < cut]
    return replace(ctx, horizon_min=cut, lam_hat_in=ctx.lam_hat_in[:cut], lam_hat_off=ctx.lam_hat_off[:cut],
                   lam_stale_in=head(ctx.lam_stale_in), lam_hat_learn_in=head(ctx.lam_hat_learn_in),
                   lam_hat_learn_prior_in=head(ctx.lam_hat_learn_prior_in), bells=bells)


def decisions(policy: Policy, lam_in: np.ndarray, lam_off: np.ndarray, lead: int) -> np.ndarray:
    """The engine's decision loop (per-minute tier): target committed for every slot ``< horizon``."""
    horizon = lam_in.size
    plan = np.zeros(horizon + lead + 1, dtype=float)
    policy.reset()
    policy.start_run()
    prefix = np.asarray(policy.prefix(lead), dtype=float)
    plan[:lead] = prefix[:lead]
    state = SimState(lam_in, lam_off, np.zeros(horizon))
    for t in range(horizon):
        state.t = t
        state.x_now = float(plan[t])
        plan[t + lead] = policy.decide(t, state)
    return plan[:horizon]


def check_seed(case: str, seed: int) -> Dict[str, object]:
    """Full run, the advice alone, and one cut per week boundary for one seed."""
    cfg = run_sweep.base_config(run_sweep.build_parser().parse_args(sweep_args(case)))
    cfg = replace(cfg, workload=replace(cfg.workload, **CELL_C), seed=seed)
    stream = build_stream(cfg)
    ctx = make_context(cfg, stream)
    lead, weeks = ctx.lead_min, stream.horizon_min // WEEK_MIN
    t0 = time.perf_counter()
    tap = TAPPolicy(ctx)
    full = decisions(tap, stream.lam_in, stream.lam_off, lead)
    adv_full = np.asarray(tap._adv_seq[:stream.horizon_min], dtype=float)
    alone = decisions(LearnedPriorAdaptPolicy(ctx), stream.lam_in, stream.lam_off, lead)
    out: Dict[str, object] = {"eps": tap.eps, "phase_share": tap.phase_share(),
                              "n_replans": tap._adv.n_replans,
                              "advice_alone_equals_advice_in_tap": bool(np.array_equal(alone, adv_full)),
                              "cuts": {}}
    for w in range(1, weeks):
        cut = w * WEEK_MIN
        part = TAPPolicy(cut_context(ctx, cut))
        x_cut = decisions(part, stream.lam_in[:cut], stream.lam_off[:cut], lead)
        adv_cut = np.asarray(part._adv_seq[:cut], dtype=float)
        out["cuts"][str(w)] = {"minutes": cut,
                               "tap_minutes_differing": int((x_cut != full[:cut]).sum()),
                               "advice_minutes_differing": int((adv_cut != adv_full[:cut]).sum())}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main() -> None:
    """Parse arguments, run every seed, print and write the verdict."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=sorted(CASES), default="c2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[10])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    results: Dict[str, object] = {}
    changed: List[Tuple[int, str]] = []
    for seed in args.seeds:
        res = check_seed(args.case, seed)
        results[str(seed)] = res
        for w, c in res["cuts"].items():
            print(f"{args.case} seed {seed} cut after week {w}: TAP {c['tap_minutes_differing']}, advice "
                  f"{c['advice_minutes_differing']} of {c['minutes']} minutes differ")
            if c["tap_minutes_differing"] or c["advice_minutes_differing"]:
                changed.append((seed, w))
        print(f"  advice alone == advice inside TAP: {res['advice_alone_equals_advice_in_tap']}; "
              f"phase_share {res['phase_share']:.4f}, re-solves {res['n_replans']}", flush=True)
    ok = not changed and all(r["advice_alone_equals_advice_in_tap"] for r in results.values())
    verdict = "prefix-invariant" if ok else f"NOT prefix-invariant at {changed}"
    print(verdict)
    if args.out:
        target = args.out if args.out.is_absolute() else PILOT / args.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"case": args.case, "verdict": verdict, "seeds": results},
                                     indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
