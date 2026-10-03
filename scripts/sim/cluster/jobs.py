#!/usr/bin/env python3
"""Job lists of the validation round, split across machines by capacity case.

Every job is one ``run_sweep.py`` / ``tune_baselines.py`` command line with paths relative to the
repository root.  Cases = a calibration file, a model row, the SLO in
seconds and the replica-tier lead (the measured cold start); each case gets its own tuning (r = 30,
plus r = 100 for the two central cases, plus an SLO-floor tuning at r = 30), then the six-cell
offset sweep on both tiers, SLO-matched reactive/velocity runs, and a held-out run on the other
semester's timetable for the central cases.  ``python jobs.py --boxes 3 --out <dir>`` writes
``box<k>/phase1.txt`` and ``box<k>/phase2.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

LADDER = "scripts/q2/results/b4_remote/b4/ladder"
COMMON = "--weeks 6 --seeds 10 11 12 13 14 --adoption 1.0 --min-replicas 1 --c-run 0.0333 --beta-start 0.0333 --lead-stat median"
POL = "--policies fixed reactive velocity forecast_ilp lcp calendar calendar_adapt bdinterp@0.1 guard offline informed stale_calendar"
TUNE_COMMON = "--college ALL --weeks 1 --max-sections 0 --adoption 1.0 --min-replicas 1 --lead-tier d_rep --c-run 0.0333 --beta-start 0.0333 --lead-stat median"
STALE = "--slots data/timetable/slots_2025-2026-2.csv"
C = 0.0333

# name -> (calibration csv, model row, slo seconds, measured replica lead in minutes, central?)
CASES: Dict[str, Tuple[str, str, float, float, bool]] = {
    "c1_7b_shared4k": (f"{LADDER}/rag_shared_4000/T3_calibration.csv", "qwen2.5-7b", 2.0, 0.6, False),
    "c2_32b_shared4k": (f"{LADDER}/rag_shared_4000/T3_calibration.csv", "qwen2.5-32b-awq", 2.0, 1.0, True),
    "c3_72b_shared4k": (f"{LADDER}/rag_shared_4000/T3_calibration.csv", "qwen2.5-72b-awq", 2.0, 1.5, True),
    "c4_32b_unique2k": (f"{LADDER}/rag_unique_2000/T3_calibration.csv", "qwen2.5-32b-awq", 5.0, 1.0, False),
    "c5_72b_unique4k": (f"{LADDER}/rag_unique_4000/T3_calibration.csv", "qwen2.5-72b-awq", 5.0, 1.5, False),
}
TIERS = {"d_rep": None, "d_inst": 8.0}  # lead override: None = the case's measured replica lead


def mu_of(cal: str, model: str, slo: float) -> float:
    """Fluid capacity of the case, so that --p-in gives the intended price ratio."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from pilot.sim import ServiceConfig
    from pilot.sim.service import load_service_config
    root = Path(__file__).resolve().parents[3]
    svc = load_service_config(root / cal, model, slo_in_s=slo, base=ServiceConfig())
    return svc.requests_per_min(211.0)


def jobs_for(case: str) -> Tuple[List[str], List[str]]:
    cal, model, slo, lead_rep, central = CASES[case]
    mu = mu_of(cal, model, slo)
    ratios = [30, 100] if central else [30]
    cal_args = f"--calibration {cal} --model {model} --slo-in-s {slo:g}"
    out = f"outputs/validation/{case}"
    p1: List[str] = []
    p2: List[str] = []
    for r in ratios:
        p1.append(f"uv run python scripts/sim/tune_baselines.py {TUNE_COMMON} {cal_args} --lead-min {lead_rep:g} --price-ratio {r} --out {out}/tuning_r{r}")
    p1.append(f"uv run python scripts/sim/tune_baselines.py {TUNE_COMMON} {cal_args} --lead-min {lead_rep:g} --price-ratio 30 --slo-floor 0.95 --out {out}/tuning_r30_slo95")
    for r in ratios:
        pin = round(r * C / mu, 9)
        for tier, lead in TIERS.items():
            lead_arg = f"--lead-min {lead_rep:g}" if lead is None else f"--lead-min {lead:g}"
            p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {POL} {cal_args} --lead-tier {tier} {lead_arg} --p-in {pin} --tuned {out}/tuning_r{r}/best.json --out {out}/r{r}_{tier}")
            if r == 30:
                p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} --policies reactive velocity offline {cal_args} --lead-tier {tier} {lead_arg} --p-in {pin} --tuned {out}/tuning_r30_slo95/best.json --out {out}/r30_{tier}_slo95")
            if r == 30 and central:
                p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {POL} {cal_args} {STALE} --lead-tier {tier} {lead_arg} --p-in {pin} --tuned {out}/tuning_r30/best.json --out {out}/r30_{tier}_heldout_sem2")
    return p1, p2


POL_LEARN = POL + " calendar_learn"


def jobs_phase3(case: str) -> Tuple[List[str], List[str]]:
    """Follow-up round: baselines re-tuned at the instance tier, the learned-timing
    planner at both tiers, and the held-out semester at the re-tuned instance tier."""
    cal, model, slo, lead_rep, central = CASES[case]
    mu = mu_of(cal, model, slo)
    ratios = [30, 100] if central else [30]
    cal_args = f"--calibration {cal} --model {model} --slo-in-s {slo:g}"
    out = f"outputs/validation/{case}"
    tune8 = TUNE_COMMON.replace("--lead-tier d_rep", "--lead-tier d_inst")
    p1: List[str] = []
    p2: List[str] = []
    for r in ratios:
        p1.append(f"uv run python scripts/sim/tune_baselines.py {tune8} {cal_args} --lead-min 8 --price-ratio {r} --out {out}/tuning8_r{r}")
    p1.append(f"uv run python scripts/sim/tune_baselines.py {tune8} {cal_args} --lead-min 8 --price-ratio 30 --slo-floor 0.95 --out {out}/tuning8_r30_slo95")
    for r in ratios:
        pin = round(r * C / mu, 9)
        p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {POL_LEARN} {cal_args} --lead-tier d_inst --lead-min 8 --p-in {pin} --tuned {out}/tuning8_r{r}/best.json --out {out}/r{r}_d_inst_t8")
        p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} --policies calendar calendar_adapt calendar_learn offline {cal_args} --lead-tier d_rep --lead-min {lead_rep:g} --p-in {pin} --tuned {out}/tuning_r{r}/best.json --out {out}/r{r}_d_rep_learn")
        if r == 30:
            p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} --policies reactive velocity offline {cal_args} --lead-tier d_inst --lead-min 8 --p-in {pin} --tuned {out}/tuning8_r30_slo95/best.json --out {out}/r30_d_inst_t8_slo95")
        if r == 30 and central:
            p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {POL_LEARN} {cal_args} {STALE} --lead-tier d_inst --lead-min 8 --p-in {pin} --tuned {out}/tuning8_r30/best.json --out {out}/r30_d_inst_t8_heldout_sem2")
    return p1, p2


OUT2 = "outputs/validation2"  # the re-run tree: one code version for every table
PHASE4_BOXES: Dict[int, List[str]] = {1: ["c2_32b_shared4k"], 2: ["c3_72b_shared4k"],
                                      3: ["c1_7b_shared4k", "c4_32b_unique2k", "c5_72b_unique4k"]}


def jobs_phase4(case: str) -> Tuple[List[str], List[str]]:
    """Re-run of the replica-tier base sweeps under the follow-up round's code (2026-09-20).

    The follow-up round rounds the planner's spread kernel to half minutes, which moved the
    calendar-family rows of cells C/E/F by up to 2%; the replica-tier tables must come from the
    same code as the instance-tier ones, so the full policy list (learned timing included) is
    re-run at d_rep with the round-1 tuning.  Output goes to ``outputs/validation2``; the
    feedback-only ``_slo95`` runs and the tuning are unaffected and stay where they are.  The
    c5 tuning directory must be copied onto box 3 first (``launch.py`` does this).
    """
    cal, model, slo, lead_rep, central = CASES[case]
    mu = mu_of(cal, model, slo)
    ratios = [30, 100] if central else [30]
    cal_args = f"--calibration {cal} --model {model} --slo-in-s {slo:g}"
    out = f"outputs/validation/{case}"
    p2: List[str] = []
    for r in ratios:
        pin = round(r * C / mu, 9)
        p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {POL_LEARN} {cal_args} --lead-tier d_rep --lead-min {lead_rep:g} --p-in {pin} --tuned {out}/tuning_r{r}/best.json --out {OUT2}/{case}/r{r}_d_rep")
        if r == 30 and central:
            p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {POL_LEARN} {cal_args} {STALE} --lead-tier d_rep --lead-min {lead_rep:g} --p-in {pin} --tuned {out}/tuning_r30/best.json --out {OUT2}/{case}/r30_d_rep_heldout_sem2")
    return [], p2


DISP = "--trigger-offset -8.7 5.3 4.0 0"  # the measured independent dispersion (batch A)
PHASE6_BOXES: Dict[int, List[str]] = {2: ["c2_32b_shared4k", "c3_72b_shared4k", "c5_72b_unique4k"],
                                      3: ["c1_7b_shared4k", "c4_32b_unique2k"]}


def jobs_phase6(case: str) -> Tuple[List[str], List[str]]:
    """SLO-matched feedback baselines tuned under the measured dispersion (2026-09-21).

    The round's SLO-floor tuning ran on the synchronous slice, where no grid point reaches 0.95 at
    the instance tier, so its "matched" baselines equal the free ones.  Here the floor tuning runs
    on a slice with the measured independent dispersion (cells B/C), at the instance tier for every
    case and at the replica tier for the two central cases, and the two baselines are then swept
    at those settings (``*_slo95disp``).  Same CPU family as the case's base runs.
    """
    cal, model, slo, lead_rep, central = CASES[case]
    mu = mu_of(cal, model, slo)
    cal_args = f"--calibration {cal} --model {model} --slo-in-s {slo:g}"
    pin = round(30 * C / mu, 9)
    out = f"{OUT2}/{case}"
    tune8 = TUNE_COMMON.replace("--lead-tier d_rep", "--lead-tier d_inst")
    p1 = [f"uv run python scripts/sim/tune_baselines.py {tune8} {cal_args} --lead-min 8 --price-ratio 30 --slo-floor 0.95 {DISP} --out {out}/tuning8_r30_slo95disp"]
    p2 = [f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} --policies reactive velocity offline {cal_args} --lead-tier d_inst --lead-min 8 --p-in {pin} --tuned {out}/tuning8_r30_slo95disp/best.json --out {out}/r30_d_inst_t8_slo95disp"]
    if central:
        p1.append(f"uv run python scripts/sim/tune_baselines.py {TUNE_COMMON} {cal_args} --lead-min {lead_rep:g} --price-ratio 30 --slo-floor 0.95 {DISP} --out {out}/tuning_r30_slo95disp")
        p2.append(f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} --policies reactive velocity offline {cal_args} --lead-tier d_rep --lead-min {lead_rep:g} --p-in {pin} --tuned {out}/tuning_r30_slo95disp/best.json --out {out}/r30_d_rep_slo95disp")
    return p1, p2


PRIOR_SLOTS = {"main": "data/timetable/slots_2025-2026-2.csv", "heldout": "data/timetable/slots_2026-2027-1.csv"}
PHASE7_BOXES: Dict[int, List[str]] = {1: ["c2_32b_shared4k"], 2: ["c3_72b_shared4k"]}


def jobs_phase7(case: str) -> Tuple[List[str], List[str]]:
    """Policy 12a (learned timing warm-started from the other semester) for the central cases.

    Four planners only - offline, calendar, calendar_learn, calendar_learn_prior - at both tiers
    with the round's own tuning, plus the held-out semester at the instance tier (whose prior then
    comes from 2026-2027-1).  Same box as the case's base runs.
    """
    cal, model, slo, lead_rep, _ = CASES[case]
    mu = mu_of(cal, model, slo)
    cal_args = f"--calibration {cal} --model {model} --slo-in-s {slo:g}"
    pin = round(30 * C / mu, 9)
    out = f"outputs/validation/{case}"
    pol = "--policies offline calendar calendar_learn calendar_learn_prior"
    p2 = [
        f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {pol} {cal_args} --lead-tier d_inst --lead-min 8 --p-in {pin} --tuned {out}/tuning8_r30/best.json --learn-prior-from {PRIOR_SLOTS['main']} --out {OUT2}/{case}/r30_d_inst_t8_learnprior",
        f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {pol} {cal_args} --lead-tier d_rep --lead-min {lead_rep:g} --p-in {pin} --tuned {out}/tuning_r30/best.json --learn-prior-from {PRIOR_SLOTS['main']} --out {OUT2}/{case}/r30_d_rep_learnprior",
        f"uv run python scripts/sim/run_sweep.py --sweep offset {COMMON} {pol} {cal_args} {STALE} --lead-tier d_inst --lead-min 8 --p-in {pin} --tuned {out}/tuning8_r30/best.json --learn-prior-from {PRIOR_SLOTS['heldout']} --out {OUT2}/{case}/r30_d_inst_t8_heldout_sem2_learnprior",
    ]
    return [], p2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boxes", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "generated")
    ap.add_argument("--phase3", action="store_true", help="write the follow-up round instead (same box assignment)")
    ap.add_argument("--phase4", action="store_true", help="write the replica-tier re-run (fixed box assignment)")
    ap.add_argument("--phase6", action="store_true", help="write the dispersion-tuned SLO-matched round (fixed box assignment)")
    ap.add_argument("--phase7", action="store_true", help="write the learned-timing-with-prior round (central cases)")
    a = ap.parse_args()
    if a.phase4 or a.phase6 or a.phase7:
        tag, boxes, maker = (("p4", PHASE4_BOXES, jobs_phase4) if a.phase4 else
                             ("p6", PHASE6_BOXES, jobs_phase6) if a.phase6 else
                             ("p7", PHASE7_BOXES, jobs_phase7))
        a.out.mkdir(parents=True, exist_ok=True)
        for k, cases in boxes.items():
            d = a.out / f"box{k}_{tag}"
            d.mkdir(exist_ok=True)
            plans = [maker(case) for case in cases]
            p1_all = [cmd for p1, _ in plans for cmd in p1]
            p2_all = [cmd for _, p2 in plans for cmd in p2]
            (d / "phase1.txt").write_text("\n".join(p1_all) + ("\n" if p1_all else ""))
            (d / "phase2.txt").write_text("\n".join(p2_all) + "\n")
            (d / "CASES").write_text("\n".join(cases) + "\n")
            print(f"box{k}: cases {cases}: {len(p1_all)} tuning jobs, {len(p2_all)} sweeps")
        return 0
    # greedy balance by sweep count, heaviest cases first (each case stays on one box: its sweeps
    # read the tuning it produced there)
    plans = {case: (jobs_phase3(case) if a.phase3 else jobs_for(case)) for case in CASES}
    weights = {case: len(jobs_for(case)[1]) for case in CASES}  # keep the round-1 assignment
    order = sorted(CASES, key=lambda c: -weights[c])
    assign: Dict[int, List[str]] = {k: [] for k in range(a.boxes)}
    load = {k: 0 for k in range(a.boxes)}
    for case in order:
        k = min(load, key=load.get)
        assign[k].append(case)
        load[k] += weights[case]
    a.out.mkdir(parents=True, exist_ok=True)
    for k, cases in assign.items():
        d = a.out / (f"box{k + 1}_p3" if a.phase3 else f"box{k + 1}")
        d.mkdir(exist_ok=True)
        p1_all: List[str] = []
        p2_all: List[str] = []
        for case in cases:
            p1, p2 = plans[case]
            p1_all += p1
            p2_all += p2
        (d / "phase1.txt").write_text("\n".join(p1_all) + "\n")
        (d / "phase2.txt").write_text("\n".join(p2_all) + "\n")
        (d / "CASES").write_text("\n".join(cases) + "\n")
        print(f"box{k + 1}: cases {cases}: {len(p1_all)} tuning jobs, {len(p2_all)} sweeps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
