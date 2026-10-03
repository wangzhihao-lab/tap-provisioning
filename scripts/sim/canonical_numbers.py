"""The canonical numbers of the validation round, one table the paper text is audited against.

    uv run python scripts/sim/canonical_numbers.py      # tables/ -> results/paper_numbers.md

About 2.5 minutes, most of it section 29c rebuilding the laptop streams of five seeds.

Every headline number the manuscript may quote is written here with the CSV it came from, so
that the text-vs-data audit has a single source and no number is ever typed by hand.  Rounding
is fixed here (GPU-hours per week to 1 decimal, SLO to 3 decimals, percentages to the nearest
integer) and the manuscript must use the same rounding.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
sys.path.insert(0, str(PILOT / "src"))
sys.path.insert(0, str(PILOT / "scripts" / "sim" / "cluster"))
sys.path.insert(0, str(PILOT / "scripts" / "sim"))
from canonical_extra import (  # noqa: E402
    f6_curve_table, f6_setting_table, f6_switch_table, forecast_error_table, lemma1_sanity_table,
    lemma1_candidates_table, fig1_facts_table, pool_facts_table, pool_table, sensitivity_facts_table, seed_spread_table,
    laptop_cell_c_table, laptop_weekly_table, mass_ratio_table, strict_floor_table, tap_diagnostics_table,
    floor_config_table, floor_pick_table, tap_prefix_table)
from canonical_laptop import (  # noqa: E402
    components_table, laptop_seed_spread_tables, resolve_time_table, supplementary_facts_table)
from canonical_billing import billing_tables  # noqa: E402
from canonical_sensitivity import (  # noqa: E402
    ADOPTION_RUNS, HELDOUT_RUNS, RESERVATION_RUNS, ablation_blocks, cells_blocks, check_paired,
    lead_blocks, sweep_blocks)
from canonical_sensitivity_tables import (  # noqa: E402
    ablation_table, adoption_gap_table, floor_table, lead_crossing_table, lead_sweep_table,
    sweep_table)
from canonical_cells_laptop import CELLS_CSV, cells_tables, cells_weekly_table, six_cell_tables  # noqa: E402
from runs_cells_table import check_paired as check_cells_paired  # noqa: E402
from canonical_r100_laptop import ratio100_tables  # noqa: E402
from canonical_platform import platform_tables  # noqa: E402
logger = logging.getLogger("canonical_numbers")

CASES = {"c1_7b_shared4k": "c1 7B shared-4k", "c2_32b_shared4k": "c2 32B shared-4k",
         "c3_72b_shared4k": "c3 72B shared-4k", "c4_32b_unique2k": "c4 32B unique-2k",
         "c5_72b_unique4k": "c5 72B unique-4k"}
KEY_POLICIES = ("offline", "calendar", "calendar_adapt", "calendar_learn", "calendar_learn_prior",
                "stale_calendar", "forecast_ilp", "guard", "lcp", "velocity", "velocity (SLO>=0.95, disp)",
                "reactive", "reactive (SLO>=0.95, disp)", "fixed")
LADDER = "scripts/q2/results/b4_remote/b4/ladder"  # relative to the repository root, as in scripts/sim/cluster/jobs.py
LADDER_REGIMES = ("short", "rag_shared_2000", "rag_shared_4000", "rag_shared_8000",
                  "rag_unique_2000", "rag_unique_4000", "rag_unique_8000")
LADDER_MODELS = ("qwen2.5-7b", "qwen2.5-32b-awq", "qwen2.5-72b-awq")


def cell_c_table(cells: pd.DataFrame, ratio: float = 30) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    sel = cells[(cells["ratio"] == ratio) & (cells["tier"] == cells["tuned_at"]) & (cells["cell"] == "C")]
    for (case, tier, slots), part in sel.groupby(["case", "tier", "slots"]):
        part = part.set_index("policy")
        for policy in KEY_POLICIES:
            if policy not in part.index:
                continue
            r = part.loc[policy]
            rows.append({"id": f"{case}|{tier}|{slots}|C|{policy}", "case": CASES.get(case, case), "tier": tier,
                         "timetable": "2026-2027-1" if slots == "main" else "2025-2026-2 (held out)",
                         "policy": policy, "gpu_h_wk": round(float(r["gpu_h_wk_mean"]), 1),
                         "slo_in": round(float(r["slo_in_mean"]), 3), "usd_wk": round(float(r["usd_wk_mean"]), 0),
                         "source": "results/sim/validation/tables/cells_long.csv"})
    return pd.DataFrame(rows)


def weekly_table(weekly: pd.DataFrame) -> pd.DataFrame:
    sel = weekly[(weekly["ratio"] == 30) & (weekly["tier"] == "d_inst") & (weekly["metric"] == "slo_in")
                 & (weekly["policy"].isin(["calendar", "calendar_learn", "calendar_learn_prior", "forecast_ilp", "stale_calendar", "velocity", "reactive"]))]
    out = sel[["case", "policy", "w1", "w2", "w3", "w4", "w5", "w6"]].copy()
    for w in ("w1", "w2", "w3", "w4", "w5", "w6"):
        out[w] = out[w].round(3)
    out["case"] = out["case"].map(lambda c: CASES.get(c, c))
    out["source"] = "results/sim/validation/tables/weekly_learn.csv"
    return out.reset_index(drop=True)


# simulator-consistent instantiation: (setting, nominal lead d in minutes) per case; see theory_table
SIM_LEADS: Dict[str, Sequence[Tuple[str, float]]] = {
    "c2_32b_shared4k": (("replica tier (= lead-time sweep, d = 1)", 1.0), ("instance tier", 8.0)),
    "c3_72b_shared4k": (("lead-time sweep, d = 1", 1.0), ("replica tier", 1.5), ("instance tier", 8.0)),
}


def executed_slots(lead_min: float) -> int:
    """One-minute slots of lead the engine executes for ``lead_min`` (``engine.make_context``)."""
    return max(0, int(math.ceil(lead_min - 1e-9)))


def _theory_row(r: pd.Series, path: Path) -> Dict[str, object]:
    return {"sigma2": f"{float(r['sigma2']):.2f}", "ell_min": f"{float(r['ell_min']):.2f}",
            "blind_min": f"{float(r['d_min']):g}", "m_min": f"{float(r['m_min']):.2f}",
            "q_max_usd_wk": f"{float(r['q_max_usd']):.1f}",
            "lemma1_lower_usd_wk": f"{float(r['lower_week_usd']):.1f}",
            "lower_over_qmax": f"{float(r['lower_over_qmax']):.2f}",
            "upper_over_qmax": f"{float(r['upper_over_qmax']):.2f}", "source": str(path.relative_to(REPO))}


def theory_table(theory: Path, simulator: bool = False) -> pd.DataFrame:
    """Lemma 1 / Proposition 1 instantiation at r = 30.

    ``simulator=False``: the model instantiation (sigma_2 = 0.70 of the sign-in launches, m = d, one
    burst per bell block).  ``simulator=True``: the simulator-consistent one (sigma_2 = 0.57 of the
    simulated streams, two bursts per block, blind time ``ceil(d) + 1``: the engine executes a lead
    of ``ceil(d)`` one-minute slots and reveals a slot after it ends, so 72B's replica tier, 1.5 min,
    is blind for 3 min), the values the text and T5 quote, plus 72B at d = 1 for the lead-time sweep.
    """
    rows: List[Dict[str, object]] = []
    for case in ("c2_32b_shared4k", "c3_72b_shared4k"):
        path = theory / f"validation_{case}" / ("sim/instantiation.csv" if simulator else "instantiation.csv")
        if not path.exists():
            continue
        t = pd.read_csv(path)
        if simulator:
            t = t[(t["r"] == 30) & (t["sigma2"].round(2) == 0.57)]
            for setting, lead in SIM_LEADS[case]:
                slots = executed_slots(lead)
                hit = t[t["d_min"].round(2) == float(slots + 1)]
                if hit.empty:
                    raise ValueError(f"{path}: no row at blind time {slots + 1} min; rerun "
                                     "scripts/theory/instantiate.py as its docstring says")
                rows.append({"case": CASES[case], "setting": setting, "lead_min": f"{lead:g}",
                             "executed_slots": slots, **_theory_row(hit.iloc[0], path)})
            continue
        lead = {"c2_32b_shared4k": 1.0, "c3_72b_shared4k": 1.5}[case]
        t = t[(t["r"] == 30) & (t["sigma2"].round(2) == 0.70) & (t["d_min"].round(2).isin([lead, 8.0]))]
        for _, r in t.iterrows():
            rows.append({"case": CASES[case], **_theory_row(r, path)})
    return pd.DataFrame(rows)


def _row(group: str, name: str, value: object, unit: str, source: str) -> Dict[str, object]:
    return {"group": group, "name": name, "value": value, "unit": unit, "source": source}


Q2 = PILOT / "scripts" / "q2"


def _serve_options(tokens: Sequence[str]) -> List[str]:
    """The options of a ``vllm serve`` command, without the binary, the model path, host and port."""
    rest, out = list(tokens)[list(tokens).index("serve") + 2:], []
    while rest:
        flag = rest.pop(0)
        if flag in ("--host", "--port"):
            rest.pop(0)
        else:
            out.append(flag)
    return out


def software_rows() -> List[Dict[str, object]]:
    """Software and serving configuration of the GPU measurements, each value read from its record.

    Versions: the environment records of the capacity ladder (B4) and of the cold-start box (B1).
    Checkpoint ids: the download calls of the box bootstrap.  Ladder launch options: the measurement
    code, checked against the launch line every ladder server logged.  B5: every round's pool_meta.
    """
    envs = {"B4": Q2 / "results/b4_remote/env.json", "B1": Q2 / "results/b1_remote/env.json"}
    env = {tag: json.loads(path.read_text(encoding="utf-8")) for tag, path in envs.items()}
    env_src = "; ".join(str(path.relative_to(REPO)) for path in envs.values())

    def version(key: str) -> str:  # one value when both records agree, else both
        seen = {tag: str(e[key]).split("+")[0] for tag, e in env.items()}
        return seen["B4"] if len(set(seen.values())) == 1 else " / ".join(f"{t} {v}" for t, v in seen.items())

    boot, code_path = Q2 / "remote/bootstrap.sh", Q2 / "remote/measurements.py"
    hf = {slug: hub for hub, slug in re.findall(r'download_model "([^"]+)"\s+"\$MODELS/([^"]+)"',
                                                 boot.read_text(encoding="utf-8"))}
    code = code_path.read_text(encoding="utf-8")
    slugs = re.findall(r'"([^"]+)"', re.search(r"^LADDER_MODELS = \(([^)]*)\)", code, re.M).group(1))
    dtype = dict(re.findall(r'ModelSpec\("([^"]+)", "([^"]+)"\)', code))
    max_len = re.search(r"^LADDER_MAX_MODEL_LEN = (\d+)", code, re.M).group(1)
    common = (f"--max-model-len {max_len} --gpu-memory-utilization "
              f"{env['B4']['gpu_tier']['gpu_memory_utilization']:g} --enable-prefix-caching")
    quant: Dict[str, str] = {}
    for slug in slugs:  # what each ladder server was started with, and the quantization vLLM found
        log = Q2 / f"results/b4_remote/logs/server_ladder_{slug}.log"
        text = log.read_text(encoding="utf-8", errors="replace")
        line = next(ln for ln in text.splitlines() if ln.startswith("$ ") and " serve " in ln)
        if " ".join(_serve_options(line.split())) != f"--dtype {dtype[slug]} {common}":
            raise ValueError(f"{log}: launch line differs from {code_path.name}: {line}")
        quant[slug] = re.search(r"quantization=(\w+)", text).group(1)
    awq = sorted({q for slug, q in quant.items() if slug.endswith("-awq")})
    rows = [_row("software", "vLLM (capacity ladder B4 / cold start B1)", version("vllm_version"), "", env_src),
            _row("software", "PyTorch with CUDA",
                 f"{version('torch_version')} with CUDA {version('torch_cuda_version')}", "", env_src),
            _row("software", "Python", version("python_version"), "", env_src),
            _row("software", "checkpoints of the capacity ladder", ", ".join(hf[s] for s in slugs), "",
                 f"{boot.relative_to(REPO)} (download_model); {code_path.relative_to(REPO)} LADDER_MODELS"),
            _row("software", "ladder launch options (vllm serve)",
                 f"{common}; --dtype " + ", ".join(f"{dtype[s]} ({s})" for s in slugs)
                 + f"; no --quantization flag, AWQ read from the checkpoint (engine quantization={'/'.join(awq)})", "",
                 f"{code_path.relative_to(REPO)} (LADDER_MAX_MODEL_LEN, _server, DEFAULT_MODELS); "
                 f"{envs['B4'].relative_to(REPO)} gpu_tier; scripts/q2/results/b4_remote/logs/server_ladder_*.log")]
    runs = sorted((Q2 / "results/b5_remote/b5/pool/runs").glob("*/pool_meta.json"))
    engines = set()
    for path in runs:
        meta = json.loads(path.read_text(encoding="utf-8"))
        engines.add((Path(meta["args"]["model"]).name, meta["engine"]["mode"],
                     " ".join(_serve_options(meta["engine"]["args"])), int(meta["replica_concurrency"])))
    if len(engines) != 1:
        raise ValueError(f"the B5 rounds ran different engines: {sorted(engines)}")
    slug, mode, opts, k = engines.pop()
    src = f"{(Q2 / 'results/b5_remote/b5/pool/runs').relative_to(REPO)}/*/pool_meta.json ({len(runs)} rounds)"
    rows.append(_row("software", "B5 pool replay: the one vLLM engine behind every replica (checkpoint, options)",
                     f"{hf[slug]}, {opts}", "", src))
    rows.append(_row("software", f"B5 pool replay: admission limit K per replica, enforced by the router (engine mode {mode})",
                     k, "concurrent streams", src))
    return rows


def constants_table() -> pd.DataFrame:
    """Calibration constants of the model section, each read from the file that produced it.

    Capacity (mu, K, replica cold start) comes from the batch-B4 ladder through the same loader the
    sweeps use; the lead tiers and timing cells from the simulator configuration; the prices from
    the validation job list; the timetable, sign-in and trigger-offset statistics from the T2 / Q1
    outputs.  The instance-acquisition trial is the one hand-timed clone in
    results/q2/b3_shortprompt_calibration.md (two clock readings, transcribed here).
    """
    import jobs  # scripts/sim/cluster/jobs.py: CASES, COMMON, C
    from pilot.sim import ServiceConfig
    from pilot.sim.config import LEAD_TIERS, SweepConfig
    from pilot.sim.service import load_service_config

    rows: List[Dict[str, object]] = []
    for case, (cal, model, slo, lead_rep, _central) in jobs.CASES.items():
        svc = load_service_config(PILOT / cal, model, slo_in_s=slo, base=ServiceConfig())
        label = CASES[case]
        rows.append(_row("capacity", f"{label}: mu (211-token output)", round(svc.requests_per_min(211.0), 1), "req/min", cal))
        rows.append(_row("capacity", f"{label}: K (largest concurrency with TTFT p95 <= target)", svc.max_concurrency, "streams", cal))
        rows.append(_row("capacity", f"{label}: in-class TTFT target", slo, "s", "scripts/sim/cluster/jobs.py CASES"))
        rows.append(_row("lead", f"{label}: replica-tier lead d_rep (measured cold start, median)", lead_rep, "min", "scripts/sim/cluster/jobs.py CASES"))
    for tier, spec in LEAD_TIERS.items():
        rows.append(_row("lead", f"tier {tier}: lead / commit block / billing block",
                         f"{spec.lead_min:g} / {spec.commit_block_min} / {spec.billing_block_min}", "min", "src/pilot/sim/config.py LEAD_TIERS"))
    cold = [json.loads(line) for line in (PILOT / "scripts/q2/results/b1_remote/b1/coldstart/coldstart.jsonl").read_text().splitlines() if line.strip()]
    by_model: Dict[str, List[float]] = {}
    for rec in cold:
        by_model.setdefault(rec["model"].rsplit("/", 1)[-1], []).append(float(rec["first_completion_done_s"]))
    for model, vals in by_model.items():
        rows.append(_row("lead", f"cold start to first completion, {model}: fresh node / warm cache",
                         f"{vals[0]:.0f} / {min(vals[1:]):.0f}", "s", "scripts/q2/results/b1_remote/b1/coldstart/coldstart.jsonl"))
    t_press, t_ssh, t_start, t_srv = "04:09:06", "04:18:41", "04:20:34", 159.0  # b3 trial clock readings
    def _mins(a: str, b: str) -> float:
        ha, ma, sa = map(int, a.split(":"))
        hb, mb, sb = map(int, b.split(":"))
        return ((hb * 60 + mb) * 60 + sb - (ha * 60 + ma) * 60 - sa) / 60.0
    rows.append(_row("lead", "instance acquisition, one hand-timed clone: power-on to ssh / to server start / to 7B ready",
                     f"{_mins(t_press, t_ssh):.1f} / {_mins(t_press, t_start):.1f} / {_mins(t_press, t_start) + t_srv / 60:.1f}", "min",
                     "results/q2/b3_shortprompt_calibration.md (single trial, about 1 min timing error)"))
    m = re.search(r"--c-run (\S+) --beta-start (\S+)", jobs.COMMON)
    rows.append(_row("price", "run cost c per replica-minute", float(m.group(1)), "USD", "scripts/sim/cluster/jobs.py COMMON"))
    rows.append(_row("price", "start cost beta per replica start", float(m.group(2)), "USD", "scripts/sim/cluster/jobs.py COMMON"))
    rows.append(_row("price", "always-on replicas (fixed floor)", 1, "replica", "scripts/sim/cluster/jobs.py COMMON (--min-replicas)"))
    rows.append(_row("price", "price ratio r = p_in mu / c reported", "30 (100 for c2/c3)", "", "scripts/sim/cluster/jobs.py"))
    rows.append(_row("price", "off-class penalty p_off / p_in", "1/6", "", "the paper's analysis"))
    cells = SweepConfig().trigger_offset
    names = ["A sync", "B indep / bell", "C indep / dist", "D common / bell", "E common / dist", "F mixed / dist"]
    for name, cell in zip(names, cells):
        rows.append(_row("timing", f"cell {name}: realised mean, session sd, section sd, common sd, forecast mean, forecast sd",
                         ", ".join(f"{v:g}" for v in cell), "min", "src/pilot/sim/config.py SweepConfig.trigger_offset"))
    t2 = pd.read_csv(PILOT / "outputs/t2/T2_table.csv").set_index("statistic")
    for stat in ("Course sections", "Meeting slots", "Teaching weeks", "Enrollment median", "Enrollment p90",
                 "Sections starting at the same bell (max)", "Seats starting at the same bell (max)", "Peak-to-mean of seats"):
        val = float(t2.loc[stat, "value"])
        rows.append(_row("timetable", stat, int(val) if val.is_integer() else val, str(t2.loc[stat, "unit"]), "outputs/t2/T2_table.csv"))
    sig = pd.read_csv(REPO / "results/q1/signin_summary.csv").set_index("metric")
    rows.append(_row("arrivals", "sigma_2 of sign-in launches (median over sessions, p25 / p75)",
                     f"{sig.loc['sync_2min', 'median']:.2f} ({sig.loc['sync_2min', 'p25']:.2f} / {sig.loc['sync_2min', 'p75']:.2f})",
                     "share", "results/q1/signin_summary.csv"))
    rows.append(_row("arrivals", "sign-in sessions / sign-ins behind sigma_2", f"{int(sig.loc['sync_2min', 'n_sessions'])} / {int(sig.loc['sync_2min', 'n_signins'])}", "", "results/q1/signin_summary.csv"))
    tr = pd.read_csv(REPO / "results/q1/assistant/trigger_offsets.csv")
    off = tr["first_arrival_offset_min"]
    rows.append(_row("arrivals", "prompted bursts: first arrival relative to the bell, mean / sd / median", f"{off.mean():.1f} / {off.std():.1f} / {off.median():.1f}", "min", "results/q1/assistant/trigger_offsets.csv"))
    rows.append(_row("arrivals", "prompted bursts: share before the bell, count before the bell, n", f"{(off < 0).mean():.3f}, {int((off < 0).sum())}, {len(off)}", "", "results/q1/assistant/trigger_offsets.csv"))
    rows.append(_row("arrivals", "prompted bursts: sigma_2 median, 10-90 ramp median", f"{tr['sigma_2'].median():.2f}, {tr['ramp_10_90_min'].median():.1f} min", "", "results/q1/assistant/trigger_offsets.csv"))
    per_day = pd.read_csv(REPO / "results/q1/assistant/per_day.csv")
    lab = per_day[per_day["dataset"] == "dsv_lab"]
    rows.append(_row("arrivals", "dsv_lab (self-paced): sigma_2 median over class days, n days", f"{lab['sigma_2'].median():.2f}, {len(lab)}", "share", "results/q1/assistant/per_day.csv"))
    bells = sorted(int(n.split("bell ")[1][:2]) * 60 + int(n.split("bell ")[1][3:5]) for n in t2.index if n.startswith("Seats starting at bell "))
    rows.append(_row("timetable", "daily bell grid: bells, minimum spacing", f"{len(bells)}, {min(b - a for a, b in zip(bells, bells[1:]))}", "bells, min", "outputs/t2/T2_table.csv"))
    rows.append(_row("timetable", "bell times", ", ".join(f"{b // 60:02d}:{b % 60:02d}" for b in bells), "HH:MM", "outputs/t2/T2_table.csv"))
    rows.append(_row("hardware", "GPU of the capacity ladder (B4) / of the cold-start measurement (B1)", "RTX PRO 6000 96 GB / RTX 6000D 84 GB", "",
                     "results/q2/b4_ladder.md / results/q2/b1_coldstart_rtx6000d.md"))
    rows.extend(software_rows())
    from pilot.sim.config import ServiceConfig as _Svc
    from pilot.sim.policies import calendar_adapt as _ca
    from pilot.sim import workload as _wl
    svc0 = _Svc()
    rows.append(_row("framework", "online scale correction: trailing window (days), re-solve tolerance, min re-solve interval (min), ratio clip",
                     f"{_ca.WINDOW_MIN // 1440}, {_ca.REPLAN_TOL:g}, {_ca.REPLAN_EVERY_MIN}, [{_ca.RATIO_LO:g}, {_ca.RATIO_HI:g}]", "",
                     "src/pilot/sim/policies/calendar_adapt.py"))
    rows.append(_row("framework", "timing learner: past observations before a section's own sd is used", _wl.LEARN_MIN_OBS, "sessions", "src/pilot/sim/workload.py LEARN_MIN_OBS"))
    rows.append(_row("framework", "queue abandonment: in-class / off-class", f"{svc0.abandon_in_s:g} / {svc0.abandon_off_s:g}", "s", "src/pilot/sim/config.py ServiceConfig"))
    rows.append(_row("framework", "BDINTERP-d: gamma = delta as a fraction of eps", "1/4", "", "src/pilot/sim/policies/bdinterp.py"))
    calib = json.loads((REPO / "results/q1/assistant/calibration.json").read_text())
    for ds in ("ds_prompted", "dsv_lab"):
        rows.append(_row("arrivals", f"{ds}: adoption median (roster)", f"{calib['adoption'][ds]['median']} ({calib['adoption'][ds]['roster']})", "share", "results/q1/assistant/calibration.json"))
        rows.append(_row("arrivals", f"{ds}: questions per student per class, median / mean", f"{calib['questions_per_student_per_class'][ds]['median']} / {calib['questions_per_student_per_class'][ds]['mean']}", "", "results/q1/assistant/calibration.json"))
        rows.append(_row("arrivals", f"{ds}: gap between a student's turns, median", calib["student_turn_gap_s"][ds]["median"], "s", "results/q1/assistant/calibration.json"))
    for tag, d in (("2026-2027-1 (main)", "t2"), ("2025-2026-2 (held out)", "t2_2025-2026-2"), ("2025-2026-1", "t2_2025-2026-1")):
        tt = pd.read_csv(PILOT / "outputs" / d / "T2_table.csv").set_index("statistic")
        vals = [int(float(tt.loc[k, "value"])) for k in ("Course sections", "Meeting slots", "Total enrolled seats", "Sections starting at the same bell (max)", "Seats starting at the same bell (max)")]
        rows.append(_row("datasets", f"timetable {tag}: sections, meeting slots, enrolled seats, sections / seats at the busiest bell",
                         ", ".join(f"{v:,}" for v in vals), "", f"outputs/{d}/T2_table.csv"))
    summary = (REPO / "results/q1/assistant/batch_a_summary.md").read_text(encoding="utf-8").splitlines()
    for line in summary:
        cells_ = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells_) >= 8 and cells_[0] in ("dsv_lab", "ds_prompted", "ds_offclass") and cells_[1] in ("self_paced_in_class", "prompted_in_class", "off_class_deadline"):
            rows.append(_row("datasets", f"assistant export {cells_[0]} ({cells_[1]}): roster, sessions, students, messages, active days",
                             ", ".join(cells_[2:7]), "", "results/q1/assistant/batch_a_summary.md section 1"))
    from pilot.sim.config import CostConfig as _Pol
    mm = re.search(r"--weeks ([0-9]+) --seeds ([0-9 ]+?) --adoption ([0-9.]+)", jobs.COMMON)
    rows.append(_row("workload", "validation runs: teaching weeks, seeds, adoption", f"{mm.group(1)}, {mm.group(2).strip().replace(' ', ', ')}, {mm.group(3)}", "", "scripts/sim/cluster/jobs.py COMMON"))
    rows.append(_row("workload", "TTFT targets: in-class (shared context) / off-class", f"{_Pol().slo_in_s:g} / {_Pol().slo_off_s:g}", "s", "src/pilot/sim/config.py CostConfig"))
    from pilot.sim.config import WorkloadConfig
    wl, wl_src = WorkloadConfig(), "src/pilot/sim/config.py WorkloadConfig"
    rows.append(_row("workload", "share of a session's roster that attends and asks, drawn uniformly per session",
                     f"{wl.attendance_lo:.2f} to {wl.attendance_hi:.2f}", "share", f"{wl_src}.attendance_lo, attendance_hi"))
    rows.append(_row("workload", "probability that a session is cancelled", f"{wl.p_cancel:g}", "", f"{wl_src}.p_cancel"))
    rows.append(_row("workload", "share of the roster the forecast books", f"{wl.forecast_attendance:.1f} (whole roster)",
                     "share", f"{wl_src}.forecast_attendance"))
    rows.append(_row("workload", "forecast roster noise e: forecast roster = roster x (1 + U(-e, e))",
                     f"{wl.roster_noise:.2f} (+/-{100 * wl.roster_noise:.0f}%)", "", f"{wl_src}.roster_noise"))
    rp = pd.read_csv(REPO / "results/q2/replay/replay_compare.csv").set_index("label")
    for lab in ("warm_x10", "warm_x40", "late0_x10", "late0_x40", "late30_x10"):
        if lab in rp.index:
            r_ = rp.loc[lab]
            ready = f"; ready {float(r_['realised_ready_after_onset_s']):.0f} s after onset" if float(r_["realised_ready_after_onset_s"]) > 0 else ""
            rows.append(_row("replay", f"single-replica replay {lab}: requests; share within 2 s measured / simulated (probe calibration) / simulated (refit capacity)",
                             f"{int(r_['n_requests'])}; {r_['meas_share_slo']:.3f} / {r_['sim_share_slo']:.3f} / {r_['simfit_share_slo']:.3f}{ready}", "", "results/q2/replay/replay_compare.csv"))
    import tune_baselines as _tb  # scripts/sim/tune_baselines.py
    rows.append(_row("tuning", "grid points: reactive, velocity", f"{len(_tb.reactive_grid())}, {len(_tb.velocity_grid())}", "", "scripts/sim/tune_baselines.py"))
    rows.append(_row("tuning", "reactive grid: up utilisation, down utilisation, cooldown (min), scale-down hold (min)",
                     f"{list(_tb.HPA_UP_UTIL)}, {list(_tb.HPA_DOWN_UTIL)}, {list(_tb.HPA_COOLDOWN)}, {list(_tb.HPA_DOWN_HOLD)}", "", "scripts/sim/tune_baselines.py"))
    rows.append(_row("tuning", "velocity grid: window (min), extrapolation clip, hysteresis bands",
                     f"{list(_tb.VELOCITY_WINDOWS)}, {list(_tb.VELOCITY_CLIPS)}, {len(_tb.VELOCITY_BANDS)}", "", "scripts/sim/tune_baselines.py"))
    rows.append(_row("tuning", "tuning seeds, held-out seed", f"{list(_tb.TUNING_SEEDS)}, {_tb.HELDOUT_SEED}", "", "scripts/sim/tune_baselines.py"))
    rows.append(_row("tuning", "scale-down hold cap (below the bell spacing)", _tb.HPA_HOLD_CAP_MIN, "min", "scripts/sim/tune_baselines.py"))
    rows.append(_row("tuning", "tuning slice: teaching weeks, sections", f"{re.search(r'--weeks ([0-9]+)', jobs.TUNE_COMMON).group(1)}, all (--max-sections 0)", "", "scripts/sim/cluster/jobs.py TUNE_COMMON"))
    rows.append(_row("tuning", "SLO floor of the matched variants; dispersed slice trigger offsets", f"0.95; {jobs.DISP if hasattr(jobs, 'DISP') else 'see jobs.py'}", "", "scripts/sim/cluster/jobs.py"))
    return pd.DataFrame(rows)


def ladder_table() -> pd.DataFrame:
    """The whole measured capacity ladder of batch B4: one row per model, prompt regime and target.

    K and mu are read through the same loader the sweeps use, so the five cases of section 5 reappear
    here unchanged (rag_shared_4000 at 2 s, rag_unique_2000 / rag_unique_4000 at 5 s).  The long-prompt
    targets are the two the paper uses: 2 s everywhere, plus 5 s for the unique-context regimes, where
    the prefill alone can exceed 2 s.  A combination the loader rejects because even concurrency 1
    misses the target is kept with K and mu marked unreachable, so the ladder stays complete.
    """
    from pilot.sim import ServiceConfig
    from pilot.sim.service import load_service_config

    paths = {regime: PILOT / f"{LADDER}/{regime}/T3_calibration.csv" for regime in LADDER_REGIMES}
    cals = {regime: pd.read_csv(path) for regime, path in paths.items()}
    rows: List[Dict[str, object]] = []
    for short in LADDER_MODELS:
        for regime in LADDER_REGIMES:
            cal, path = cals[regime], paths[regime]
            models = {str(m).rsplit("/", 1)[-1]: str(m) for m in cal["model"].unique()}
            if short not in models:
                continue
            model_row = models[short]
            ttft1 = cal[(cal["model"] == model_row) & (cal["metric"] == "ttft_p95_s@N=1")]["median"]
            for target in ((2.0, 5.0) if regime.startswith("rag_unique") else (2.0,)):
                note: str = ""
                try:
                    svc = load_service_config(path, model_row, slo_in_s=target, base=ServiceConfig())
                    k: object = svc.max_concurrency
                    mu: object = round(svc.requests_per_min(211.0), 1)
                except ValueError as exc:
                    logger.info("ladder %s / %s at %g s is unreachable: %s", short, regime, target, exc)
                    k = mu = "unreachable"
                    note = str(exc).splitlines()[0]
                rows.append({"model": short, "regime": regime, "target_s": target, "K": k,
                             "mu_req_per_min": mu, "ttft_p95_at_1_s": f"{float(ttft1.iloc[0]):.2f}",
                             "source": str(path.relative_to(REPO)), "note": note})
    return pd.DataFrame(rows)


def boundary_table(cells: pd.DataFrame) -> pd.DataFrame:
    """The fluid-model boundary cases: SLO-floor-tuned reactive GPU-hours as a multiple of the offline optimum."""
    rows: List[Dict[str, object]] = []
    sel = cells[(cells["ratio"] == 30) & (cells["tier"] == "d_inst") & (cells["tuned_at"] == "d_inst") & (cells["cell"] == "C") & (cells["slots"] == "main")]
    for case in ("c4_32b_unique2k", "c5_72b_unique4k"):
        part = sel[sel["case"] == case].set_index("policy")
        if "offline" not in part.index or "reactive (SLO>=0.95, disp)" not in part.index:
            continue
        off, rea = part.loc["offline"], part.loc["reactive (SLO>=0.95, disp)"]
        rows.append({"case": CASES[case], "offline_gpu_h_wk": round(float(off["gpu_h_wk_mean"]), 1), "offline_slo": round(float(off["slo_in_mean"]), 3),
                     "reactive_disp_gpu_h_wk": round(float(rea["gpu_h_wk_mean"]), 1), "reactive_disp_slo": round(float(rea["slo_in_mean"]), 3),
                     "reactive_over_offline": round(float(rea["gpu_h_wk_mean"] / off["gpu_h_wk_mean"]), 1), "source": "results/sim/validation/tables/cells_long.csv"})
    return pd.DataFrame(rows)


def theory_checks_table(theory: Path) -> pd.DataFrame:
    """Model-internal checks of the theory under the validation calibration (T5 addendum, section 3)."""
    rows: List[Dict[str, object]] = []
    for case in ("c2_32b_shared4k", "c3_72b_shared4k"):
        root = theory / f"validation_{case}"
        if not root.exists():
            continue
        gb = pd.read_csv(root / "t5_guard_bound.csv")
        gd = pd.read_csv(root / "t5_guard_decompose.csv")
        blind = gd[gd["phase"].astype(str).str.contains("blind", case=False)]
        for tier, part in blind.groupby("tier"):
            b = gb[(gb["tier"] == tier) & (gb["m_label"].astype(str).str.contains("simulator", case=False))]
            if b.empty:
                b = gb[gb["tier"] == tier]
            merged = part.merge(b[["week", "bound_usd"]], on="week", how="inner")
            ratio = merged["excess_usd"] / merged["bound_usd"]
            rows.append({"case": CASES[case], "check": f"guard blind-phase excess over the family upper bound, {tier} (weekly min / max)",
                         "value": f"{ratio.min():.2f} / {ratio.max():.2f}", "verdict": "holds" if ratio.max() <= 1.0 else "consistent (exceeds by up to {:.0f}%)".format(100 * (ratio.max() - 1)),
                         "source": str((root / "t5_guard_decompose.csv").relative_to(REPO))})
        lh = pd.read_csv(root / "t5_lemma_h_check.csv")
        rows.append({"case": CASES[case], "check": "Lemma 2 (ramp or hold): seed-weeks where OPT >= envelope + starts bound - TV slack",
                     "value": f"{int(lh['holds'].sum())} / {len(lh)} (without slack {int(lh['holds_without_tv_slack'].sum())} / {len(lh)})",
                     "verdict": "holds" if lh["holds"].all() else "fails", "source": str((root / "t5_lemma_h_check.csv").relative_to(REPO))})
    return pd.DataFrame(rows)


def to_md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(none)_"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


# sections 14 and 18-23 (canonical_sensitivity.py): where the rows come from, and the floor tables
SENSITIVITY_SOURCES = (
    "Laptop runs (arm64), seeds 10-14; the source column names the run of each row. offline and the calendar planners come from the original run "
    "(outputs/validation_local/sensitivity_sweeps or priority_ablation); reactive and velocity are the SLO-floor rules at the configurations the resource rule selected "
    "(section 28b), re-run with the original's arguments on its streams (outputs/validation_local/runs/s3_consistency/<run>; a rule whose configuration "
    "the resource rule kept reproduces the original's rows); tap comes from the TAP job of the same stream group "
    "(outputs/validation_local/runs/tap/<job>, results/runs/JOBS_README.md). A run is used only if both jobs' checks passed against the original "
    "(invariant.json), the TAP job ran with the original's learned-timing prior and the offline rows of the three are equal in every column; otherwise this file is not written.")
CLOUD_NOTE = "x86 validation round (cloud)."  # sections 1, 2, 8 and 10
FLOOR_NOTE = (
    "TAP's GPU-hours and attainment; the cheapest of the SLO-floor-tuned feedback rules of the table above whose unrounded mean attainment is >= 0.95 "
    "(none reaches 0.95 if neither does), and the SLO-floor-tuned reactive rule, always listed; extra GPU-hours in percent of TAP's "
    "(extra_over_tap_pct to 1 decimal, extra_over_tap_pct_int to the nearest integer) and of calendar_learn_prior's (1 decimal), from unrounded means; "
    "attainment unrounded to 4 decimals, the lowest seed's included.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Canonical numbers of the validation round")
    ap.add_argument("--tables", type=Path, default=REPO / "results/sim/validation/tables")
    ap.add_argument("--theory", type=Path, default=REPO / "results/theory")
    ap.add_argument("--out", type=Path, default=REPO / "results/paper_numbers.md")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cells = pd.read_csv(args.tables / "cells_long.csv")
    weekly = pd.read_csv(args.tables / "weekly_learn.csv")
    resolve, streams = resolve_time_table(REPO)  # first: it builds the streams section 29b reuses
    spread_rows, spread_max = laptop_seed_spread_tables(REPO)
    billing_rows, billing_floor = billing_tables(REPO)
    ablation, leads, timing = ablation_blocks(REPO), lead_blocks(REPO), cells_blocks(REPO)
    reservation, adoption = sweep_blocks(REPO, RESERVATION_RUNS), sweep_blocks(REPO, ADOPTION_RUNS)
    heldout = sweep_blocks(REPO, HELDOUT_RUNS)
    paired_rows = check_paired(REPO, leads)
    six = six_cell_tables(REPO, CASES)
    cells_paired = check_cells_paired(REPO, pd.read_csv(REPO / CELLS_CSV, float_precision="round_trip"))
    r100, r100_runs = ratio100_tables(REPO, CASES)
    platform = platform_tables(REPO, CASES)
    parts = ["# Canonical numbers (validation round, r = 30, cell C unless stated)", "",
             "Generated by `scripts/sim/canonical_numbers.py`; the manuscript quotes numbers only from here, "
             "with this rounding. Regenerate after any table change; the text audit compares against this file.", "",
             "## 1. Per policy, cell C, whole run (GPU-hours per week / in-class SLO / USD per week)", "",
             f"{CLOUD_NOTE} The c2/c3 main-timetable rows are superseded by the laptop sections 7a (cells A-F) and 27a (cell C). "
             "Of the c2/c3 held-out rows (2025-2026-2) only offline, calendar and calendar_learn_prior at the instance tier have laptop "
             "counterparts (section 22, with TAP and the floor-tuned rules); the other held-out rows (calendar_learn and the cost-tuned "
             "feedback rules among them) and the c1/c4/c5 rows have no laptop run.", "",
             to_md(cell_c_table(cells)), "",
             "## 2. Cheapest feedback baseline reaching SLO 0.95 (within rounding, i.e. >= 0.9495) vs the timetable planners (percent)", "",
             f"{CLOUD_NOTE} Superseded by section 28 (strict floor); not quoted.", "",
             "## 3. In-class SLO by teaching week, instance tier, cell C", "", to_md(weekly_table(weekly)), "",
             "## 4. Lemma 1 / Proposition 1 instantiation at r = 30", "",
             "4a. Simulator-consistent (the values the text and T5 quote): sigma_2 = 0.57 measured on the simulated streams, two bursts per bell block (one per 45-min period, 50 per week), "
             "blind time ceil(d) + 1 at the executed lead: the engine runs a lead of ceil(d) one-minute slots (engine.make_context) and reveals a slot after it ends, "
             "so the 72B replica tier (d = 1.5 min, 2 slots) is blind for 3 min. The 72B row at d = 1 (blind 2 min) is the bound at the lead-time sweep's first point; "
             "the 72B runs of section 18 start at d = 1.5, which executes as 2 slots like d = 2.", "",
             to_md(theory_table(args.theory, simulator=True)), "",
             "4b. Model instantiation at the campus sign-in anchor: sigma_2 = 0.70, m = d, one burst per bell block (25 per week).", "",
             to_md(theory_table(args.theory)), "",
             "4c. Week-1 excess of the best calendar-oblivious policy (cell C) against the Lemma 1 lower bound: a worst-case bound on a benign instance, reported for scale (T5 reading: instantiated).", "",
             to_md(lemma1_sanity_table(REPO)), "",
             "4d. Every calendar-oblivious candidate of 4c, cheapest first per case and tier: week-1 excess over the offline optimum (USD, cell C, mean over seeds 10-14), "
             "each run against its own offline rows (equal in the two runs). The cells-laptop run gives fixed, lcp, forecast_ilp and the cost-tuned reactive and velocity rules; "
             "tuning_resource/<case>_<tier>/eval gives the floor-tuned (dispersed) pair at the configurations the resource rule selected (section 28c).", "",
             to_md(lemma1_candidates_table(REPO)), "",
             "## 5. Calibration constants of the model section (each read from the file named in its source column)", "",
             to_md(constants_table()), "",
             "## 6. Measured capacity ladder (batch B4, one GPU, 211-token output)", "",
             "K is the largest concurrency whose measured TTFT p95 stays within the target and mu = 60 K / (TTFT "
             "at K + 211 / per-stream decode rate at K) is one replica's fluid capacity, both through the loader "
             "the sweeps use: the five cases of section 5 are the rag_shared_4000 rows at 2 s and the "
             "rag_unique_2000 / rag_unique_4000 rows at 5 s. Rows marked unreachable miss the target already at "
             "concurrency 1, so no replica count attains it.", "",
             to_md(ladder_table()), "",
             "## 7. Timing cells A-F on the laptop platform (arm64), GPU-hours per week / in-class SLO (r = 30, five seeds 10-14, six weeks, baselines tuned on their own tier)", "",
             "Cells: A sync; B independent dispersion, planner at the bell; C independent, planner knows the distribution; "
             "D common shock, planner at the bell; E common, planner knows; F mixed, planner knows (parameters in section 5).", "",
             "From results/runs/cells_laptop.csv (scripts/sim/runs_cells_table.py; `run` is the offset.csv under the repository root). Per case and tier the rows come from four "
             "laptop runs of one stream group, each with cells A-F: the 13-policy run and the learned-prior run of the cells-laptop queue (outputs/validation_local/runs/cells_laptop, "
             "the cloud's six-cell commands run on the laptop), the R1 TAP job (tap, offline) and section 3's six-week rerun of the resource-rule floor rules "
             "(runs/tuning_resource/<case>_<tier>/eval). Every job finished (DONE marker, exit 0), the other three passed their checks against the R1 job (invariant.json: verified), "
             "the offline rows of the four runs are equal in every column, and a policy two runs hold has the same rows in both; it is read from the first of R1 job, 13-policy run, "
             "learned-prior run and rerun. "
             "Rows named as in section 27a: cost-tuned (own tier) = the cost tuning of the tier's own slice (the job's --tuned); floor-tuned (dispersed) = the resource rule's SLO-floor "
             "configuration (section 28b). Whole run, means over the five seeds: GPU-hours per week to 1 decimal / in-class attainment to 3. "
             f"Cell C holds section 27a's rows of the same name (checked, {cells_paired['whole_run']} whole-run and {cells_paired['weekly']} weekly rows, every column, unrounded); "
             f"section 32 holds every row of every cell (checked, {six.cells_checked} rows: GPU-hours per week and attainment, unrounded and as printed). "
             "The cloud's six-cell runs (results/sim/validation/tables/cells_long.csv) draw other streams and are not listed; section 10 reads the cloud's boundary cases from that file.", "",
             "7a. Whole run (six weeks), one table per case and tier.", "",
             *[part for title, frame in cells_tables(REPO, CASES) for part in (f"{title}:", "", to_md(frame), "")],
             "7b. In-class attainment by teaching week (mean over seeds, 3 decimals), cell C, instance tier, every policy variant of 7a.", "",
             to_md(cells_weekly_table(REPO, CASES)), "",
             "## 8. Per policy, cell C, whole run at r = 100 (sensitivity)", "",
             f"{CLOUD_NOTE} Its c2/c3 rows are superseded by section 33 (laptop R7 jobs) for TAP and the offline optimum only; "
             "the other policies at r = 100 have no laptop run (section 33b).", "",
             to_md(cell_c_table(cells, ratio=100)), "",
             "## 9. Model-internal theory checks under the validation calibration (r = 30)", "", to_md(theory_checks_table(args.theory)), "",
             "## 10. Fluid-model boundary cases (instance tier, cell C, r = 30)", "",
             f"{CLOUD_NOTE} Not replaced: the laptop has no c4 or c5 run at r = 30 and its c1 runs hold only TAP and the offline optimum "
             "(section 32), so this section and section 1's c1/c4/c5 rows remain the only source for those cases' baselines.", "",
             to_md(boundary_table(cells)), "",
             "## 11. Forecast-error sweep at 32B (F6): setting of each chain (timing exact, size error only)", "",
             to_md(f6_setting_table(REPO)), "",
             "## 12. F6: cost ratio to the offline optimum per forecast scale (mean over seeds), plan net cost over LCP net cost, in-class SLO", "",
             "Net cost excludes the always-on replica (c * min_replicas * minutes), as in the phase test of BDINTERP-d.", "",
             to_md(f6_curve_table(REPO)), "",
             "## 13. F6: observed against predicted switch scale of BDINTERP-d (net of the always-on tier)", "",
             "Observed: the smallest scale above 1 at which bdinterp@eps costs more than 1% less than the plan. Predicted: the smallest scale at which the plan's net cost reaches 4/eps times LCP's (Remark on the switch point).", "",
             to_md(f6_switch_table(REPO)), "",
             "## 14. Admission-priority and penalty-grading ablation (cell C, instance tier, r = 30, five seeds; cell C re-drawn alone, so streams differ from the main run within the seed spread)", "",
             f"{SENSITIVITY_SOURCES} TAP jobs: r1_<case>_inst (cell C of its cells A-F) for the priority-on arm, whose streams are section 27a's at the instance tier; "
             "r8_<case>_off and r8_<case>_ungraded for the other two. gpu_h_wk_vs_main_pct: GPU-hours per week against the same policy in the main setting, from unrounded means.", "",
             to_md(ablation_table(ablation)), "",
             f"14b. The SLO floor against TAP per setting: {FLOOR_NOTE}", "",
             to_md(floor_table(ablation)), "",
             "## 15. Residual forecast error of the plan (eta normalised by the realised in-class demand), cell C, instance tier", "",
             to_md(forecast_error_table(REPO)), "",
             "## 16. Facts added in the sensitivity round (each read from the file in its source column)", "",
             to_md(sensitivity_facts_table(REPO)), "",
             "16b. TAP's causality check (scripts/audit/tap_prefix_invariance.py; the arguments of the R1 instance-tier job: cell C, d = 8, r = 30): "
             "for every seed and week boundary w, TAP is re-run with every forecast ending at w, and the minutes of weeks 1..w where its committed "
             "decision or its advice sequence differs from the full run's are counted (tap_minutes_differing, advice_minutes_differing: sums over "
             "seeds and boundaries); advice_alone_equals_advice_in_tap: for every seed the advice run on its own gives the sequence it gives inside TAP; "
             "n_replans: size re-solves in the full run. Read from the JSON when this file is generated.", "",
             to_md(tap_prefix_table(REPO)), "",
             "## 17. Seed spread of the six-week means, cell C, r = 30 (largest standard deviation over the policies of each group)", "",
             to_md(seed_spread_table(REPO)), "",
             "## 18. Lead-time curve (sensitivity round): cell C, r = 30, five seeds, feedback rules at the resource-rule SLO-floor configuration of the nearest measured tier (replica tuning for d <= 2 min, instance tuning above)", "",
             f"{SENSITIVITY_SOURCES} TAP jobs: r1_<case>_rep (cell C) at d = 1 (32B) and 1.5 (72B), r1_<case>_inst (cell C) at d = 8, r3_<case>_d<d> otherwise. "
             f"At d = 1, 1.5 and 8 every row is section 27a's row of the same name and tier (checked, {paired_rows} rows: GPU-hours per week and attainment, unrounded).", "",
             to_md(lead_sweep_table(leads)), "",
             f"18b. The SLO floor against TAP per lead time: {FLOOR_NOTE}", "",
             to_md(floor_table(leads)), "",
             "18c. Where each rule falls below the floor: the smallest lead time of the curve at which its unrounded mean attainment is below 0.95 "
             "(none: at least 0.95 at every lead time), the attainment there to 4 decimals, and whether it stays below at every longer lead time "
             "(the feedback rules change configuration between 2 and 3 min, see the title).", "",
             to_md(lead_crossing_table(leads)), "",
             "## 19. Timing cells G-L (sensitivity round): instance tier, r = 30, five seeds; G-J keep the total timing spread of cell C (6.64 min) and vary the share of its variance that is common to a bell", "",
             f"{SENSITIVITY_SOURCES} TAP job: r4_<case>_cellsGL.", "",
             to_md(sweep_table(timing)), "",
             f"19b. The SLO floor against TAP per timing cell: {FLOOR_NOTE}", "",
             to_md(floor_table(timing)), "",
             "## 20. Reservation tier (8 min lead, hourly commit and billing blocks), cell C, r = 30, five seeds", "",
             f"{SENSITIVITY_SOURCES} TAP job: r5_<case>_res.", "",
             to_md(sweep_table(reservation)), "",
             f"20b. The SLO floor against TAP at the reservation tier: {FLOOR_NOTE}", "",
             to_md(floor_table(reservation)), "",
             "## 21. Section-level adoption below the campus-wide case, cell C, instance tier, r = 30, five seeds", "",
             f"{SENSITIVITY_SOURCES} TAP jobs: r6_<case>_adopt<level>. Section 23 compares the feedback rules with TAP and the learned planner.", "",
             to_md(sweep_table(adoption)), "",
             "## 22. Held-out semester (2025-26 sem. 2) with the SLO-floor variants, instance tier, r = 30, five seeds; prior from the other semester (symmetric) or the earlier one (forward)", "",
             f"{SENSITIVITY_SOURCES} TAP jobs: r2_<case>_inst_sym (cell C of its cells A-F) and r2_<case>_inst_fwd, each with the prior of its rows "
             "(the offline rows do not depend on the prior, so the prior is checked on the jobs' command lines).", "",
             to_md(sweep_table(heldout)), "",
             f"22b. The SLO floor against TAP per prior: {FLOOR_NOTE}", "",
             to_md(floor_table(heldout)), "",
             "## 23. Adoption: GPU-hours of the SLO-floor feedback rules over the learned planner's and TAP's, per adoption level (percent, from unrounded means)", "",
             "The rows of section 21. feedback_extra_pct: over calendar_learn_prior, to the nearest integer; feedback_extra_over_tap_pct / _int: over TAP, "
             "to 1 decimal / the nearest integer; reaches_floor: the rule's unrounded mean in-class attainment (feedback_slo_in_4dp) is >= 0.95.", "",
             to_md(adoption_gap_table(adoption)), "",
             "## 24. Closed-loop replica-pool replay (batch B5): the busiest bell of week 3 replayed on the measurement GPU, one round per policy and tier", "",
             to_md(pool_facts_table(REPO)), "",
             "## 25. Batch B5 round by round (measured on the GPU against the simulator on the same stream)", "",
             to_md(pool_table(REPO)), "",
             "## 26. Numbers printed inside Fig. 1 (busiest bell of week 3, c3 72B shared-4k, instance tier, seed 10; window -20 to +24 min; TAP = tap, Reactive = floor-tuned (dispersed), selected by resource cost)", "",
             to_md(fig1_facts_table(REPO)), "",
             "## 27. Central cell C on the laptop platform (arm64): TAP and every baseline variant on the same streams, r = 30, five seeds", "",
             "From results/runs/paired_cell_c.csv (laptop runs only; every source's offline rows equal the TAP job's; `run` is the offset.csv under the repository root). "
             "The cloud rows of sections 1-3 are kept but are other streams (x86 and arm64 draw different arrivals from the same arguments). "
             "GPU-hours and starts per week to 1 decimal; USD per week is the resource cost (running + starts, without the unserved-request penalty that section 1's usd_wk includes) to 0 decimals; "
             "in-class attainment is the mean over seeds to 3 decimals, and to 4 where it lies within 0.002 of 0.95; lowest seed to 3 decimals. "
             "Tunings: cost-tuned (own tier) = tuned for cost on the tier's own slice (at the replica tier, the replica-tier tuning); cost-tuned (replica-tier tuning) = the replica tier's configuration run at the instance tier; "
             "floor-tuned (dispersed) = the SLO-floor variant of the dispersed slice selected by resource cost on the laptop's tuning week (among grid points whose mean in-class attainment over the tuning seeds is >= 0.95 unrounded, the lowest mean running + start cost; section 28b); "
             "floor-tuned (dispersed, previous selection) = the configuration the penalty-inclusive cost ratio selected on the cloud's tuning week, listed where the resource rule chose another; "
             "floor-tuned (synchronous) = the SLO-floor variant of the synchronous slice (grid points reaching 0.95 on the tuning slice). The note column names the run each row comes from.", "",
             "27a. Whole run (six weeks).", "", to_md(laptop_cell_c_table(REPO)), "",
             "27b. In-class attainment by teaching week (mean over seeds, 3 decimals).", "",
             "calendar_adapt (timing given plus the size correction) and calendar (timing given, no correction) are listed at the instance tier only.", "",
             to_md(laptop_weekly_table(REPO)), "",
             "27c. TAP against its components, whole run (the rows of 27a): calendar_learn_prior is TAP's learned-prior plan without the size correction and the robust fallback, "
             "calendar_adapt the plan on the given timing distribution (the cell C forecast) with the size correction, calendar the given timing without it, offline the offline optimum. "
             "GPU-hours per week to 1 decimal and relative to TAP's in percent from unrounded means (1 decimal); in-class attainment, mean over seeds and lowest seed, to 3 decimals.", "",
             to_md(components_table(REPO)), "",
             "27d. Seed spread of the rows of 27a: standard deviation over the five seeds (ddof = 1, as section 17) of the six-week GPU-hours per week (1 decimal) and in-class attainment (3 decimals), "
             "from the seeds of each row's run, which reproduce 27a's means. Groups: planners = tap, calendar_learn_prior, calendar_adapt, calendar; feedback rules = every reactive and velocity variant; "
             "other = the remaining rows. The second table is the largest standard deviation of each group over the four case-tiers, with the row that has it.", "",
             to_md(spread_rows), "", to_md(spread_max), "",
             "## 28. Strict SLO floor against TAP, laptop cell C: the cheapest feedback variant whose unrounded whole-run attainment is >= 0.95, and the SLO-floor-tuned reactive rule", "",
             "28a. Candidates: every reactive and velocity row of section 27a, the previous selection included. Extra GPU-hours in percent of TAP's and of calendar_learn_prior's, from unrounded means (1 decimal); attainment unrounded to 4 decimals.", "",
             "extra_over_tap_pct_int: the extra GPU-hours in percent of TAP's rounded to the nearest integer from the unrounded means (the percentages the text quotes).", "",
             to_md(strict_floor_table(REPO)), "",
             "28b. The floor picks the resource rule changed, on the laptop's tuning week (one teaching week, dispersed trigger offsets; means over the tuning seeds of the section-3 grid): "
             "the previous configuration next to the new one, with the previous one's attainment on the cloud's tuning week, where it was selected. "
             "Attainment to 4 decimals; GPU-hours and USD (running + starts) per tuning week to 2.", "",
             "prev_slo_in_tune_3dp / new_slo_in_tune_3dp: the tuning-week attainment of each configuration to 3 decimals.", "",
             to_md(floor_pick_table(REPO)), "",
             "28c. The SLO-floor configuration of every rule at c2 and c3, both tiers, as the supplementary's table of baseline tunings prints it (variant floor): "
             "the resource rule's selection on the laptop's tuning week (best.json; 28b lists the picks it changed), the best tuning-week attainment any point of the rule's grid reached "
             "(the largest mean over the tuning seeds, 3 decimals) and the grid points whose mean is >= 0.95 (unrounded). "
             "The floor rows of c1, c4 and c5 in that table are the cloud's dispersed-slice selection (results/sim/validation/tables/tuning.csv, instance tier only).", "",
             to_md(floor_config_table(REPO)), "",
             "## 29. TAP diagnostics, laptop cell C", "",
             "29a. Per R1 run (five seeds): phase_share = share of slots BDINTERP-d spends in the advice (ADV) phase, minimum over seeds (6 decimals); "
             "n_replans = size re-solves of the online correction; final_ratio = the size ratio in force at the end of the run (3 decimals).", "",
             to_md(tap_diagnostics_table(REPO)), "",
             "29b. Realised over forecast in-class request mass of the stream TAP plans on: sum of lam_in over sum of lam_hat_learn_prior_in, seed 10, cell C, "
             "six weeks and each week (3 decimals); the forecast books the whole roster, the stream draws attendance and cancellations (section 5, workload).", "",
             "ratio_whole_run_2dp: the six-week ratio to 2 decimals.", "",
             to_md(mass_ratio_table(REPO, streams)), "",
             "29c. When TAP's online size correction re-solves (R1 runs, cell C, seeds 10-14): the correction's rule (src/pilot/sim/policies/calendar_adapt.py: WINDOW_MIN trailing window, "
             "a check once REPLAN_EVERY_MIN minutes have passed since the last re-solve, a re-solve when the ratio moved by more than REPLAN_TOL, clip RATIO_LO / RATIO_HI, prior mass = the first teaching day's forecast mass), "
             "through the ratio() of TAP's advice (LearnedPriorAdaptPolicy, forecast lam_hat_learn_prior_in), replayed on each run's stream rebuilt from prefix_invariance.sweep_args and CELL_C at the run's tier and lead. "
             "The ratio reads only realised in-class arrivals and the forecast, so the replay solves no plan; it reproduces n_replans and final_ratio of every seed's row of the run's diagnostics.csv (section 29a). "
             "Minutes of the run count from Monday 00:00 of teaching week 1; the minute is the one the re-solve is decided at, and the rescaled plan applies from slot minute + d.", "",
             to_md(resolve), "",
             "## 30. Numbers the supplementary text quotes outside the tables above (each read from the file in its source column)", "",
             to_md(supplementary_facts_table(REPO)), "",
             "## 31. Billing sensitivity (s4 of the rerun round), laptop cell C, r = 30, five seeds (10-14): a replica start billed from the order against the main caliber", "",
             "From the offset.csv in each row's source column, through scripts/sim/runs_billing.py (block, measure), which also writes results/runs/billing_sensitivity.md (the same rows to 2 decimals). "
             "order = a replica is billed from the order on, so each start pays its whole lead, beta = c x d with d the one-minute slots the engine executes (8 at the instance tier; 2 at the 72B replica tier, whose 1.5 min lead runs as 2 slots); "
             "the planners re-solve and both baseline tunings are redone at that price (billing/<case>_<tier>/tune_cost from the tier's own cost tuning, tune_floor from the dispersed-slice floor tuning with the section-3 resource rule), and every row comes from billing/<case>_<tier>/eval_cost or eval_floor. "
             "main = the paper's caliber, billed from the first serving minute plus one replica-minute per start, beta = c: every row is section 27a's row of the same name "
             "(the cost-tuned rules and fixed rerun with the same configuration on the same streams in billing/<case>_<tier>/ref_main, with equal results). "
             "Billed GPU-hours per week = (gpu_hours + starts x d / 60) / weeks with d = 1 under main, so under main they exceed section 27a's GPU-hours (serving replica-minutes only) by the starts per week / 60; "
             "USD per week = (run_cost + switch_cost) / weeks, the resource cost without the unserved-request penalty.", "",
             "Means over the five seeds. Billed GPU-hours and USD per week to 1 decimal; in-class attainment (whole run) to 3 decimals, and where the mean lies within 0.01 of 0.95 also to 4 with the lowest seed; "
             "vs_tap_pct = billed GPU-hours in percent of TAP's in the same case, tier and caliber, from unrounded means, to 1 decimal, and vs_tap_pct_int the same to the nearest integer.", "",
             "31a. Every row, both calibers per case-tier (order first).", "", to_md(billing_rows), "",
             "31b. The SLO floor per case-tier and caliber: TAP's attainment and whether it reaches 0.95 (unrounded), and the cheapest feedback row of 31a (reactive or velocity, cost- or floor-tuned) "
             "whose unrounded attainment is >= 0.95, by billed GPU-hours, with its extra billed GPU-hours in percent of TAP's (unrounded means; 1 decimal and nearest integer), or none. "
             "Section 28a admits more variants (among them the previous floor selection) and ranks by serving GPU-hours, so the two can name different rows.", "",
             to_md(billing_floor), "",
             "## 32. TAP across the six timing cells A-F on the laptop platform (arm64), r = 30, five seeds (10-14), with every laptop baseline row on the same streams", "",
             "From scripts/sim/canonical_cells_laptop.py. TAP and the offline optimum come from the R1 jobs (outputs/validation_local/runs/tap/r1_<case>_<tier>, "
             "--policies offline tap, cells A-F), TAP's diagnostics from each job's diagnostics.csv. The baselines are every laptop run of the job's stream group whose offline rows "
             "equal the R1 job's on the same (timing cell, seed, week) keys in every column: in cells A-F the six-week rerun of section 3's resource-rule floor rules "
             "(runs/tuning_resource/<case>_<tier>/eval: reactive and velocity, floor-tuned (dispersed)) and the two runs of the cells-laptop queue behind section 7 "
             "(runs/cells_laptop/<case>_r30_d_inst_t8 or <case>_r30_d_rep: every other policy, the feedback rules cost-tuned on their own tier; the same with _learnprior: "
             "calendar_learn_prior); in cell C also the runs section 27a draws on and the billing round's "
             "main-caliber block (runs/billing/<case>_<tier>/ref_main). Cells are identified by their trigger-offset columns (SweepConfig.trigger_offset, parameters in section 5). "
             "Every source is a laptop run (platform column); the cloud rows of sections 1-3 draw other streams and are not compared here. "
             "Whole run (six weeks), means over the five seeds: GPU-hours per week to 1 decimal; in-class attainment to 3 decimals, and where the mean lies within 0.01 of 0.95 "
             "also to 4 with the lowest seed; reaches_0.95 compares the unrounded mean with 0.95.", "",
             "32a. TAP and the offline optimum per cell. phase_share_min: the lowest share over the seeds of the committed slots BDINTERP-d takes in the advice (ADV) phase, "
             "6 decimals (1 = the guard never leaves the plan); n_replans: size re-solves of the online correction; final_ratio: the size ratio in force at the end of the run "
             "(3 decimals); eps: TAP's trust parameter; laptop_baseline_rows: the rows of 32b in the cell.", "",
             to_md(six.tap), "",
             "32b. Every laptop baseline row that pairs, named as in section 27a (group as in 27d: planner, feedback rule, other). extra_over_tap_pct: GPU-hours per week in percent "
             "of TAP's in the same case, tier and cell, from unrounded means (1 decimal; _int to the nearest integer). A row two runs produce seed for seed in every column is listed once "
             f"with both sources. Cell C: every row, TAP's and the offline optimum's included, is section 27a's row of the same name (checked, {six.cell_c_checked} rows: "
             "GPU-hours per week and attainment, unrounded). Every cell: each row of section 7 is the row of the same cell and name here and each row here outside cell C is one of "
             f"section 7's (checked, {six.cells_checked} rows: GPU-hours per week and attainment, unrounded and as printed).", "",
             to_md(six.baselines), "",
             "32c. Rows no laptop run holds (no laptop row), per case and tier: the policy variants of 32b that some cells lack, with the cells without and with a laptop row; "
             "a case and tier without any laptop baseline run is one row (every baseline).", "",
             to_md(six.missing), "",
             "## 33. TAP at r = 100 against r = 30 on the laptop platform (arm64): the R7 jobs against the R1 jobs of the same streams, cells A-F, five seeds (10-14)", "",
             "33a. From scripts/sim/canonical_r100_laptop.py. TAP and the offline optimum of each R7 job (runs/tap/r7_<case>_<tier>) next to the R1 job of the same case "
             "and tier (the r = 30 rows are 32a's). The two jobs' run commands differ only in --p-in and --out, and their offline rows agree in every stream column "
             "(runs_check.STREAM_COLS without p_in, p_off and price_ratio_real) on every (timing cell, seed, week) key, so TAP runs on the same streams at both prices. "
             "r: the price ratio p_in mu / c of the rows. tap_gpu_h_wk_vs_r30_pct: TAP's GPU-hours per week at r = 100 in percent of its r = 30 value (unrounded means, 1 decimal); "
             "slo_in_vs_r30: the change of its unrounded mean attainment (4 decimals). Other columns as 32a.", "",
             to_md(r100), "",
             "33b. Laptop baselines at r = 100: none, so every case, tier and cell of 33a has no laptop row. The laptop sweeps at r = 100 besides the R7 jobs, each with the reason "
             "it does not pair (one that paired with an R7 job would stop this script until it is added as a source); the cloud runs at r = 100 (section 8) draw other streams.", "",
             to_md(r100_runs), "",
             "## 34. The two simulation platforms on the base cell: laptop (arm64) against cloud (x86) rows of the same policy and configuration, "
             "c2 and c3, both tiers, cell C, r = 30, five seeds (10-14), six weeks", "",
             "From scripts/sim/canonical_platform.py. The same simulator and arguments draw different arrival streams on the cloud boxes (x86) "
             "and on the laptop (arm64) (results/runs/JOBS_README.md, Platform); section 27a reports the laptop's streams, sections 1-3 the cloud's. "
             "Laptop rows: section 27a's (results/runs/paired_cell_c.csv). Cloud rows: results/sim/validation/tables/cells_long.csv at the main "
             "timetable, the rows of sections 1-3 and, at the instance tier, the reactive and velocity rows tuned at the replica tier (tuned_at d_rep, "
             "not in sections 1-3), which 27a also runs there; each is recomputed from its run's offset.csv and equals cells_long.csv "
             f"(checked, {platform.cloud_checked} rows, unrounded means). The replica-tuned rows come from the first round's r30_d_inst and r30_d_inst_slo95, "
             "which predate the follow-up round's planner-kernel change, so only their feedback rows enter, after two checks: the first-round and re-run "
             "r30_d_rep agree on offline, reactive and velocity, and r30_d_inst's offline rows equal r30_d_inst_t8's "
             f"(checked, {platform.round1_checked} rows, equal in GPU-hours, starts, every cost term, mean and peak replicas, requests, attainment and timeouts). "
             "A laptop and a cloud row pair when case, tier and policy agree and their runs' "
             "commands (laptop: the rebuilt command of a sensitivity-round or ablation original, checked against its CSV and log, or the START line of a "
             "rerun job's log; cloud: the phase file the box ran), parsed by run_sweep.py's parser, agree in everything the policy reads: sweep, "
             "workload, service, cost, seeds and policy configuration, with the tuned knobs compared only for the rule that reads them and the "
             "learned-timing prior only for calendar_learn_prior. Cloud rows of one configuration (an SLO-floor pick equal to the cost pick) are equal "
             "seed for seed and listed once (cloud_row joins their labels). Left out (34c): tap, which no cloud run has, and the floor-tuned (dispersed) "
             "rules whose resource-rule configuration (section 28b) no cloud run used; their previous selection is the cloud's configuration and pairs, "
             "and a floor-tuned (dispersed) rule whose configuration the resource rule kept pairs with the cloud's (SLO>=0.95, disp) row.", "",
             "34a. Every pair. GPU-hours per week to 2 decimals and in-class attainment (mean over seeds) to 4; d_gpu_h_wk and d_slo_in are cloud minus "
             "laptop and d_gpu_pct is d_gpu_h_wk in percent of the laptop value (2 decimals), all from unrounded means. Groups: offline = the offline "
             "optimum; planners = calendar, calendar_learn_prior, calendar_learn, calendar_adapt, stale_calendar, forecast_ilp; feedback rules = every "
             "reactive and velocity variant; other = the remaining policies. cloud_tuned_at: the tier the cloud row's baselines were tuned at.", "",
             to_md(platform.pairs), "",
             "34b. The largest absolute gap per case-tier and group, and over the four case-tiers (overall), each with the row of 34a that has it: "
             "GPU-hours per week (2 decimals), the same in percent of the laptop value (2 decimals), in-class attainment (4 decimals).", "",
             to_md(platform.summary), "",
             "34c. Rows with no counterpart of the same configuration on the other platform, and why.", "",
             to_md(platform.unpaired), ""]
    args.out.write_text("\n".join(parts), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
