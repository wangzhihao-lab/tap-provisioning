"""Canonical numbers, part 2: the forecast-error sweep (F6), the sensitivity-round facts and the central
cell C of the laptop platform (TAP next to every baseline).

Imported by ``canonical_numbers.py`` and kept apart so that both files stay within the size limit;
the admission-priority ablation and the sensitivity-round sweeps are in ``canonical_sensitivity.py``.
Each table names the CSV its rows were read from; rounding follows ``canonical_numbers.py``
(GPU-hours per week to 1 decimal, SLO to 3 decimals, cost ratios to 2).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("canonical_extra")

# (tier label, tables written by f6_validation.py, raw sweep CSV of the r = 30 chain)
F6_TIERS: Tuple[Tuple[str, str, str], ...] = (
    ("replica tier, d = 1 min", "results/sim/validation/tables",
     "outputs/validation_local/f6_32b/r30_d_rep/epsilon/epsilon.csv"),
    ("instance tier, d = 8 min", "results/sim/validation/f6_d_inst/tables",
     "outputs/validation_local/f6_32b/r30_d_inst/epsilon/epsilon.csv"),
)
F6_POLICIES: Tuple[str, ...] = ("calendar", "calendar_adapt", "bdinterp@0.1", "bdinterp@0.5",
                                "bdinterp@1.0", "lcp")
F6_COLUMNS: Dict[str, str] = {"calendar": "plan", "calendar_adapt": "plan_corrected",
                              "bdinterp@0.1": "bdinterp_eps0.1", "bdinterp@0.5": "bdinterp_eps0.5",
                              "bdinterp@1.0": "bdinterp_eps1.0", "lcp": "lcp"}
ABLATION_CASES: Dict[str, str] = {"c2_32b_shared4k": "c2 32B shared-4k",
                                  "c3_72b_shared4k": "c3 72B shared-4k"}
WEEKS = 6


def _num(value: object, fmt: str = "{:g}") -> str:
    """``value`` formatted with ``fmt``, or the text the source used for 'none'."""
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def f6_setting_table(repo: Path) -> pd.DataFrame:
    """The settings of each F6 chain, read from its raw sweep CSV (forecast-scale rows only)."""
    rows: List[Dict[str, object]] = []
    for tier, _tables, raw in F6_TIERS:
        path = repo / raw
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["error_knob"] == "forecast_scale"]
        whole = df[df["week"] == 0]
        exact = whole[whole["forecast_scale"] == 1.0]
        items = [
            ("lead time", _num(whole["lead_min"].iloc[0]), "min"),
            ("seeds", ", ".join(str(int(s)) for s in sorted(whole["seed"].unique())), ""),
            ("teaching weeks", str(int(df["week"].max())), ""),
            ("forecast scales", ", ".join(_num(v) for v in sorted(whole["forecast_scale"].unique())),
             "x booked demand"),
            ("price ratio p_in mu / c", ", ".join(_num(v) for v in sorted(whole["price_ratio"].unique())), ""),
            ("mu", _num(whole["mu"].iloc[0], "{:.0f}"), "requests per minute per replica"),
            ("measured sigma_2 of the streams", _num(whole["sigma2_measured"].mean(), "{:.2f}"), ""),
            ("residual forecast error eta_norm at scale 1", _num(exact["eta_norm"].mean(), "{:.3f}"), ""),
            ("policies", ", ".join(sorted(whole["policy"].unique())), ""),
        ]
        for name, value, unit in items:
            rows.append({"tier": tier, "setting": name, "value": value, "unit": unit, "source": raw})
    return pd.DataFrame(rows)


def f6_curve_table(repo: Path) -> pd.DataFrame:
    """Cost ratio to the offline optimum per forecast scale and policy (mean over seeds), the plan's
    net cost over LCP's, and the in-class attainment of the plan, of BDINTERP-d at eps = 1 and of LCP."""
    rows: List[Dict[str, object]] = []
    for tier, tables, _raw in F6_TIERS:
        path = repo / tables / "f6_forecast_scale.csv"
        if not path.exists():
            continue
        agg = pd.read_csv(path)
        for (ratio, scale), part in agg.groupby(["price_ratio", "forecast_scale"]):
            by = part.set_index("policy")
            row: Dict[str, object] = {"tier": tier, "price_ratio": int(ratio), "forecast_scale": _num(scale)}
            for policy in F6_POLICIES:
                row[F6_COLUMNS[policy]] = (f"{float(by.loc[policy, 'cost_ratio_mean']):.2f}"
                                           if policy in by.index else "")
            if "calendar" in by.index and "lcp" in by.index:
                row["plan_net_over_lcp_net"] = (
                    f"{float(by.loc['calendar', 'net_cost_mean'] / by.loc['lcp', 'net_cost_mean']):.2f}")
            for policy, name in (("calendar", "plan_slo_in"), ("bdinterp@1.0", "bdinterp_eps1.0_slo_in"),
                                 ("lcp", "lcp_slo_in")):
                row[name] = f"{float(by.loc[policy, 'slo_in_mean']):.3f}" if policy in by.index else ""
            row["source"] = f"{tables}/f6_forecast_scale.csv"
            rows.append(row)
    return pd.DataFrame(rows)


def f6_switch_table(repo: Path) -> pd.DataFrame:
    """Observed against predicted switch scale of BDINTERP-d, net of the always-on tier."""
    rows: List[Dict[str, object]] = []
    for tier, tables, _raw in F6_TIERS:
        path = repo / tables / "f6_switch.csv"
        if not path.exists():
            continue
        for _, r in pd.read_csv(path).iterrows():
            rows.append({"tier": tier, "price_ratio": int(r["price_ratio"]), "eps": _num(r["eps"]),
                         "observed_switch_scale": _num(r["observed_switch_scale_net"]),
                         "predicted_switch_scale": _num(r["predicted_switch_scale_net"]),
                         "required_net_ratio_4_over_eps": _num(r["required_factor"]),
                         "largest_net_ratio_in_grid": f"{float(r['max_plan_over_robust_net']):.2f}",
                         "switch_below_scale_1": _num(r["switch_below_1"]),
                         "source": f"{tables}/f6_switch.csv"})
    return pd.DataFrame(rows)


LEARNPRIOR_RUNS: Tuple[Tuple[str, str], ...] = (
    ("c2 32B shared-4k", "outputs/validation_remote/box1/validation2/c2_32b_shared4k/r30_d_inst_t8_learnprior/offset/offset.csv"),
    ("c3 72B shared-4k", "outputs/validation_remote/box2/validation2/c3_72b_shared4k/r30_d_inst_t8_learnprior/offset/offset.csv"),
)
SIGNIN_LAUNCHES = "results/q1/signin_launch_offsets.csv"


def forecast_error_table(repo: Path) -> pd.DataFrame:
    """Residual forecast error of the plan (eta normalised by the realised in-class demand), cell C, whole run."""
    rows: List[Dict[str, object]] = []
    for label, rel in LEARNPRIOR_RUNS:
        path = repo / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sel = df[(df["week"] == 0) & (df["policy"] == "calendar_learn_prior") & (df["trigger_offset_common_sd"] == 0)
                 & (df["trigger_offset_course_sd"] > 0) & (df["forecast_offset_sd"] > 0)]
        rows.append({"case": label, "cell": "C", "eta_norm_mean": f"{float(sel['eta_norm'].mean()):.2f}",
                     "eta_norm_sd": f"{float(sel['eta_norm'].std(ddof=1)):.3f}",
                     "sigma2_measured": f"{float(sel['sigma2_measured'].mean()):.2f}", "n_seeds": int(len(sel)), "source": rel})
    return pd.DataFrame(rows)


def sensitivity_facts_table(repo: Path) -> pd.DataFrame:
    """Facts the sensitivity round added to the text, each read from the file named in its source column."""
    rows: List[Dict[str, object]] = []

    def add(group: str, name: str, value: object, unit: str, source: str) -> None:
        rows.append({"group": group, "name": name, "value": value, "unit": unit, "source": source})

    launches = repo / SIGNIN_LAUNCHES
    if launches.exists():
        d = pd.read_csv(launches)
        g = d.groupby("course")["offset_min"].agg(["count", "mean", "std"])
        add("signin", "sign-in launches: courses / launches", f"{len(g)} / {int(g['count'].sum())}", "", SIGNIN_LAUNCHES)
        add("signin", "sign-in launches: sd of the course mean offsets", f"{float(g['mean'].std(ddof=1)):.1f}", "min", SIGNIN_LAUNCHES)
        within = g["std"].dropna()
        add("signin", "sign-in launches: within-course sd of the launch offset, min / max over courses",
            f"{float(within.min()):.0f} / {float(within.max()):.0f}", "min", SIGNIN_LAUNCHES)
    add("arrivals", "prompted course: share of the roster that asks (a_k), rounded", "0.87", "share", "results/q1/assistant/calibration.json (adoption median 0.872)")
    add("arrivals", "prompted course: questions per student per class, rounded", "3.1", "", "results/q1/assistant/calibration.json (mean 3.078)")
    add("workload", "simulator: questions per student per trigger (rho) x triggers per session", "2.0 x 2 = 4.0", "", "src/pilot/sim/config.py WorkloadConfig.rho_mean, triggers_per_session")
    add("workload", "simulator: mixing weight w of the synchronised pool", "0.70", "", "src/pilot/sim/config.py WorkloadConfig.sigma")
    add("workload", "simulator: off-class scaling basis, student-section pairs", "243,297", "", "results/t2/T2_summary.md (rosters summed over sections)")
    add("workload", "simulator: in-class abandonment / off-class abandonment / off-class TTFT target", "60 / 300 / 10", "s", "src/pilot/sim/config.py ServiceConfig, CostConfig")
    add("workload", "simulator: period length / trigger times per day / block spacing", "45 / 10 / 110", "min", "src/pilot/sim/config.py period_minutes; results/t2/T2_summary.md")
    add("workload", "size correction: trailing window / tolerance / min interval / ratio clip", "7 d / 0.1 / 60 min / [0.25, 4]", "", "src/pilot/sim/config.py PolicyConfig")
    add("timetable", "weekly seats booked, weeks 1-4 / weeks 5-6 (rounded)", "193,000 / 277,000", "seats", "results/t2/week_dynamics.md (193k; 277,302)")
    add("timetable", "cohort step, week 5 over week 4", "43", "%", "results/t2/week_dynamics.md")
    add("arrivals", "10-90% ramp at the median: prompted course / sign-in launches / ASSISTments", "1.8 / 7.1 / 10.0", "min", "results/q1/assistant/batch_a_summary.md; results/q1/signin_summary.md; results/q1/assistments_event_summary.md")
    add("datasets", "ASSISTments 2012-13: in-class sessions inferred from class ids", "10,362", "sessions", "the pilot study (Q1 result 2026-09-17)")
    add("lead", "instance acquisition, one hand-timed clone: reachable / first 7B request", "9.6 / 14.1", "min", "results/paper_numbers.md section 5 (power-on to ssh / to 7B ready)")
    if (repo / CELLS_LAPTOP_CSV).exists():
        for label, _resource, penalised, runs in _lcp_cost_ratios(repo):
            add("theory", f"measured LCP penalised cost over the offline optimum's (USD per week: running + starts + unserved-request penalty, "
                f"the cost the theorems bound), {label}, replica / instance tier", penalised, "",
                f"total_cost (main.tex eq. (3)) of the runs {CELLS_LAPTOP_CSV} names: " + "; ".join(runs))
    from canonical_numbers import executed_slots  # at call time: canonical_numbers imports this module
    delays = [(f"{label.split()[1]} replica tier", lead) for _, label, lead in LEMMA1_CASES] + [("instance tier", 8.0)]
    theorem1 = [3 + 2 * (THEORY_R - 1) * (executed_slots(d) + 1) for _, d in delays]  # 3 + 2 L_x d_exec / beta
    add("theory", f"Theorem 1 constant 3 + 2 L_x d_exec / beta at r = {THEORY_R} (L_x = {THEORY_R - 1} c, beta = c) at the executed delay ceil(d)+1: "
        + " / ".join(f"{name} (d = {d:g} -> {executed_slots(d) + 1})" for name, d in delays),
        " / ".join(f"{c:,}" for c in theorem1), "", THEORY_SOURCE)
    add("theory", f"Theorem 2 robustness constant 4(1+eps)/eps x (3 + 2 L_x d_exec / beta) at eps = {THEORY_EPS:g}, instance tier "
        f"(d = 8 -> {executed_slots(8.0) + 1}), at the executed delay ceil(d)+1",
        f"{4 * (1 + THEORY_EPS) / THEORY_EPS * theorem1[-1]:,.0f}", "", THEORY_SOURCE)
    return pd.DataFrame(rows)


CELLS_LAPTOP_CSV = "results/runs/cells_laptop.csv"
THEORY_R, THEORY_EPS = 30, 1.0  # price ratio p_in mu / c of the central runs; Theorem 2's trust parameter
THEORY_SOURCE = (f"formula: main.tex eq. (7) and (9) (eq:lcp, eq:robustness); L_x = {THEORY_R - 1}c at r = {THEORY_R}, beta = c; "
                 "d_exec = ceil(d)+1 (engine.make_context)")


def _lcp_cost_ratios(repo: Path) -> List[Tuple[str, str, str, List[str]]]:
    """(model, resource-cost ratios, penalised-cost ratios, runs read) of LCP over the offline optimum,
    replica / instance tier, laptop cell C, whole run, means over the seeds: the resource cost is
    ``usd_per_week`` of ``cells_laptop.csv``; the penalised cost is ``total_cost`` of the offset.csv each
    row names, whose running + start cost must reproduce that ``usd_per_week``."""
    cl = pd.read_csv(repo / CELLS_LAPTOP_CSV, float_precision="round_trip")
    cl = cl[(cl["cell"] == "C") & (cl["week"] == 0) & cl["policy"].isin(("lcp", "offline"))]
    out: List[Tuple[str, str, str, List[str]]] = []
    for case, label, _ in LEMMA1_CASES:
        resource, penalised, runs = [], [], []
        for tier in ("rep", "inst"):
            usd: Dict[str, Tuple[float, float]] = {}
            for policy in ("lcp", "offline"):
                row = cl[(cl["case"] == case) & (cl["tier"] == tier) & (cl["policy"] == policy)]
                if len(row) != 1:
                    raise ValueError(f"{CELLS_LAPTOP_CSV}: {len(row)} cell-C whole-run rows of {policy}, {case} {tier}")
                src = str(row["source"].iloc[0])
                runs.append(f"{src.removesuffix('/offset/offset.csv')} ({policy}, {tier})")
                run = _cell_c(pd.read_csv(repo / src))
                run = run[(run["week"] == 0) & (run["policy"] == policy)]
                own = float(((run["run_cost"] + run["switch_cost"]) / run["weeks"]).mean())
                if sorted(run["seed"]) != list(LEMMA1_SEEDS) or not np.isclose(own, float(row["usd_per_week"].iloc[0]), rtol=1e-12):
                    raise ValueError(f"{src}: the cell-C whole-run {policy} rows do not reproduce {CELLS_LAPTOP_CSV}")
                usd[policy] = (float(row["usd_per_week"].iloc[0]), float((run["total_cost"] / run["weeks"]).mean()))
            resource.append(f"{usd['lcp'][0] / usd['offline'][0]:.2f}")
            penalised.append(f"{usd['lcp'][1] / usd['offline'][1]:.2f}")
        out.append((label.split()[1], " / ".join(resource), " / ".join(penalised), runs))
    return out


def tap_prefix_table(repo: Path) -> pd.DataFrame:
    """Section 16b: TAP's causality check per case, read from the JSON of
    ``scripts/audit/tap_prefix_invariance.py`` as it stands when this file is generated."""
    rows: List[Dict[str, object]] = []
    for case in ABLATION_CASES:
        short = case.split("_")[0]
        rel = f"{RUNS}/audit/tap_prefix_{short}.json"
        if not (repo / rel).exists():
            logger.warning("TAP causality check: %s not found, row skipped", rel)
            continue
        report = json.loads((repo / rel).read_text(encoding="utf-8"))
        if report["case"] != short:
            raise ValueError(f"{rel} holds case {report['case']}, not {short}")
        seeds = report["seeds"]
        cuts = [cut for seed in seeds.values() for cut in seed["cuts"].values()]
        replans = [int(seed["n_replans"]) for seed in seeds.values()]
        rows.append({"case": ABLATION_CASES[case], "seeds": ", ".join(sorted(seeds, key=int)),
                     "week_boundaries_per_seed": " / ".join(str(n) for n in sorted({len(s["cuts"]) for s in seeds.values()})),
                     "cuts_checked": len(cuts),
                     "tap_minutes_differing": sum(int(cut["tap_minutes_differing"]) for cut in cuts),
                     "advice_minutes_differing": sum(int(cut["advice_minutes_differing"]) for cut in cuts),
                     "advice_alone_equals_advice_in_tap": "yes" if all(s["advice_alone_equals_advice_in_tap"] for s in seeds.values()) else "no",
                     "n_replans_min / max": f"{min(replans)} / {max(replans)}", "verdict": report["verdict"], "source": rel})
    return pd.DataFrame(rows)


OBLIVIOUS: Tuple[str, ...] = ("fixed", "reactive", "velocity", "lcp", "forecast_ilp")
FLOOR_RULES: Tuple[str, ...] = ("reactive", "velocity")  # the rules the resource rule floor-tunes
LEMMA1_CASES: Tuple[Tuple[str, str, float], ...] = (  # (case, label, replica-tier lead in minutes)
    ("c2_32b_shared4k", "c2 32B shared-4k", 1.0), ("c3_72b_shared4k", "c3 72B shared-4k", 1.5))
# tier -> (cells-laptop run suffix, tuning_resource folder suffix, the tuning of the cells-laptop run's
# reactive and velocity rules: its --tuned file tuning_r30 / tuning8_r30, as in canonical_cells_laptop)
LEMMA1_TIERS: Dict[str, Tuple[str, str, str]] = {"d_rep": ("d_rep", "rep", "cost (replica-tier tuning)"),
                                                 "d_inst": ("d_inst_t8", "inst", "cost (tier's own)")}
LEMMA1_SEEDS: Tuple[int, ...] = (10, 11, 12, 13, 14)
CELL_C = {"trigger_offset_sd": 5.3, "trigger_offset_course_sd": 4.0, "trigger_offset_common_sd": 0.0, "forecast_offset_sd": 6.6}
OFFLINE_ORDER: Tuple[str, ...] = ("trigger_offset_mean", "trigger_offset_sd", "trigger_offset_course_sd",
                                  "trigger_offset_common_sd", "forecast_offset", "forecast_offset_sd", "seed", "week")
Candidate = Tuple[str, float, str]  # (policy with its tuning in words, week-1 excess in USD, run folder)


def _cell_c(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in CELL_C.items():
        mask &= (df[col].round(2) == val)
    return df[mask]


def _oblivious_week1(repo: Path, case: str, tier: str, lead: float) -> List[Candidate]:
    """Week-1 excess over the offline optimum (cell C, mean over seeds 10-14) of every calendar-oblivious
    candidate on the laptop: the cells-laptop run's policies, reactive and velocity cost-tuned, and the
    resource rule's floor-tuned (dispersed) pair; each run against its own offline rows, which the two
    runs must share column for column."""
    cells, short, cost = LEMMA1_TIERS[tier]
    runs = ((f"{RUNS}/cells_laptop/{case}_r30_{cells}", {p: cost if p in FLOOR_RULES else "-" for p in OBLIVIOUS}),
            (f"{TUNING_RESOURCE}/{case}_{short}/eval", {p: FLOOR_DISPERSED for p in FLOOR_RULES}))
    out: List[Candidate] = []
    offline: List[pd.DataFrame] = []
    for folder, variants in runs:
        df = pd.read_csv(repo / folder / "offset" / "offset.csv")
        if not np.allclose(df["lead_min"], lead):
            raise ValueError(f"{folder}: lead {sorted(df['lead_min'].unique())} min, not {lead:g}")
        offline.append(df[df["policy"] == "offline"].sort_values(list(OFFLINE_ORDER)).reset_index(drop=True))
        wk = _cell_c(df)
        wk = wk[wk["week"] == 1]
        off = wk[wk["policy"] == "offline"].set_index("seed")["total_cost"]
        for policy, variant in variants.items():
            part = wk[wk["policy"] == policy].set_index("seed")["total_cost"]
            if sorted(part.index) != list(LEMMA1_SEEDS) or sorted(off.index) != list(LEMMA1_SEEDS):
                raise ValueError(f"{folder}: week-1 cell-C seeds of {policy} / offline are not {LEMMA1_SEEDS}")
            out.append((_variant_words(policy, variant, short), float((part - off.reindex(part.index)).mean()), folder))
    if not offline[0].equals(offline[1]):
        raise ValueError(f"{case} {tier}: the offline rows of {runs[0][0]} and {runs[1][0]} differ")
    return out


def lemma1_sanity_table(repo: Path) -> pd.DataFrame:
    """Week-1 excess over the offline optimum of the best calendar-oblivious policy, cell C, on the
    laptop (``_oblivious_week1``), against the Lemma 1 bound: simulator-consistent (sigma_2 0.57, two
    bursts per block, blind ceil(d) + 1 at the executed lead, as in section 4a) and model (0.70, m = d
    at the physical lead d)."""
    from canonical_numbers import executed_slots  # at call time: canonical_numbers imports this module

    rows: List[Dict[str, object]] = []
    for case, label, lead in LEMMA1_CASES:
        sim = repo / "results/theory" / f"validation_{case}" / "sim" / "instantiation.csv"
        model = repo / "results/theory" / f"validation_{case}" / "instantiation.csv"
        if not sim.exists() or not model.exists():
            continue
        ts, tm = pd.read_csv(sim), pd.read_csv(model)
        for tier, d_model in (("d_rep", lead), ("d_inst", 8.0)):
            blind_sim = float(executed_slots(d_model) + 1)
            candidates = _oblivious_week1(repo, case, tier, d_model)
            sources = list(dict.fromkeys(folder for _, _, folder in candidates))
            policy, measured, _ = min(candidates, key=lambda c: c[1])
            bs = ts[(ts["r"] == 30) & (ts["sigma2"].round(2) == 0.57) & (ts["d_min"].round(2) == round(blind_sim, 2))]["lower_week_usd"]
            bm = tm[(tm["r"] == 30) & (tm["sigma2"].round(2) == 0.70) & (tm["d_min"].round(2) == round(d_model, 2))]["lower_week_usd"]
            bound_sim, bound_model = float(bs.iloc[0]), float(bm.iloc[0])
            rows.append({"case": label, "tier": tier, "best_oblivious_policy_week1": policy,
                         "measured_week1_excess_usd": f"{measured:.1f}",
                         "bound_sim_usd": f"{bound_sim:.1f}", "measured_over_bound_sim": f"{measured / bound_sim:.2f}",
                         "bound_model_usd": f"{bound_model:.1f}", "measured_over_bound_model": f"{measured / bound_model:.2f}",
                         "n_runs": len(sources), "source": "; ".join(sources)})
    if rows:
        for label in ("c2 32B shared-4k", "c3 72B shared-4k"):
            pair = [r for r in rows if r["case"] == label]
            if len(pair) == 2:
                rep, inst = pair
                rows.append({"case": label, "tier": "instance over replica", "best_oblivious_policy_week1": "",
                             "measured_week1_excess_usd": f"{float(inst['measured_week1_excess_usd']) / float(rep['measured_week1_excess_usd']):.1f}x",
                             "bound_sim_usd": f"{float(inst['bound_sim_usd']) / float(rep['bound_sim_usd']):.1f}x", "measured_over_bound_sim": "",
                             "bound_model_usd": f"{float(inst['bound_model_usd']) / float(rep['bound_model_usd']):.1f}x", "measured_over_bound_model": "",
                             "n_runs": "", "source": "ratio of the two rows above"})
    return pd.DataFrame(rows)


def lemma1_candidates_table(repo: Path) -> pd.DataFrame:
    """Section 4d: every candidate section 4c takes the cheapest of, per case and tier, cheapest first
    (ties keep the runs' order)."""
    rows: List[Dict[str, object]] = []
    for case, label, lead in LEMMA1_CASES:
        for tier, d_model in (("d_rep", lead), ("d_inst", 8.0)):
            ranked = sorted(_oblivious_week1(repo, case, tier, d_model), key=lambda c: c[1])
            for rank, (policy, excess, folder) in enumerate(ranked, start=1):
                rows.append({"case": label, "tier": tier, "rank": rank, "policy": policy,
                             "measured_week1_excess_usd": f"{excess:.1f}", "source": folder})
    return pd.DataFrame(rows)


PLANNERS: Tuple[str, ...] = ("offline", "calendar", "calendar_adapt", "calendar_learn", "calendar_learn_prior", "stale_calendar", "bdinterp@0.1")
FEEDBACK: Tuple[str, ...] = ("reactive", "reactive (SLO>=0.95, disp)", "velocity", "velocity (SLO>=0.95, disp)", "lcp", "guard", "forecast_ilp", "fixed")


def seed_spread_table(repo: Path) -> pd.DataFrame:
    """Largest seed standard deviation of the six-week means in cell C (r = 30), planners against feedback rules."""
    path = repo / "results/sim/validation/tables/cells_long.csv"
    if not path.exists():
        return pd.DataFrame()
    cl = pd.read_csv(path)
    rows: List[Dict[str, object]] = []
    for case, label in (("c2_32b_shared4k", "c2 32B shared-4k"), ("c3_72b_shared4k", "c3 72B shared-4k")):
        for tier in ("d_rep", "d_inst"):
            sel = cl[(cl["case"] == case) & (cl["tier"] == tier) & (cl["cell"] == "C") & (cl["ratio"] == 30)
                     & (cl["slots"] == "main") & (cl["tuned_at"] == tier)]
            for group, names in (("planners", PLANNERS), ("feedback rules", FEEDBACK)):
                part = sel[sel["policy"].isin(names)]
                if part.empty:
                    continue
                rows.append({"case": label, "tier": tier, "group": group,
                             "max_gpu_h_wk_sd": f"{float(part['gpu_h_wk_sd'].max()):.1f}",
                             "max_slo_in_sd": f"{float(part['slo_in_sd'].max()):.3f}",
                             "policy_with_max_gpu_sd": str(part.loc[part["gpu_h_wk_sd"].idxmax(), "policy"]),
                             "n_seeds": int(part["n_seeds"].iloc[0]), "source": "results/sim/validation/tables/cells_long.csv"})
    return pd.DataFrame(rows)


POOL_DIR = "results/q2/pool"
POOL_LABELS: Dict[str, str] = {
    "plan_learnprior_d_inst": "planner, instance tier",
    "plan_learnprior_d_rep": "planner, replica tier",
    "plan_offline": "offline optimum, replica tier",
    "velocity_d_inst": "token velocity, instance tier",
    "velocity_d_rep": "token velocity, replica tier",
    "reactive_d_inst": "reactive, instance tier",
    "reactive_d_rep": "reactive, replica tier",
}


def pool_table(repo: Path) -> pd.DataFrame:
    """Batch B5: the closed-loop replica-pool replay against the simulator, one row per round."""
    df = pd.read_csv(repo / POOL_DIR / "pool_compare.csv")
    rows = []
    for label, name in POOL_LABELS.items():
        part = df[df["label"] == label]
        if part.empty:
            continue
        r = part.iloc[0]
        closed = str(r["policy"]) in ("velocity", "reactive")
        rows.append({
            "round": name,
            "requests": int(r["n_requests"]),
            "measured share within 2 s": f"{float(r['meas_share_slo']):.3f}",
            "simulated share": f"{float(r['simA_share_slo']):.3f}",
            "difference (points)": f"{100 * (float(r['meas_share_slo']) - float(r['simA_share_slo'])):.1f}",
            "measured replica-minutes": f"{float(r['meas_replica_minutes']):.0f}",
            "simulated replica-minutes": f"{float(r['simA_replica_minutes']):.0f}",
            "minutes with the same replica count": f"{int(r['x_equal_minutes'])} of 45",
            "decisions re-checked": int(r["targets_checked"]) if closed else 0,
            "decisions differing": int(r["targets_mismatched"]) if closed else 0,
            "measured TTFT p95 (s)": f"{float(r['meas_p95_s']):.1f}",
            "simulated TTFT p95 (s)": f"{float(r['simA_p95_s']):.1f}",
        })
    return pd.DataFrame(rows)


def pool_facts_table(repo: Path) -> pd.DataFrame:
    """The setting of batch B5: what the replay carries and what it emulates."""
    df = pd.read_csv(repo / POOL_DIR / "pool_compare.csv")
    meta = json.loads((repo / "scripts/q2/pool_inputs/bell_c3_w3.json").read_text(encoding="utf-8"))
    calib = pd.read_csv(repo / "scripts/q2/results/b5_remote/calib_single_short/T3_calibration.csv")
    calib = calib[calib["model"].astype(str).str.endswith("qwen2.5-7b")]

    def med(metric: str) -> float:
        row = calib[calib["metric"] == metric]
        return float(row["median"].iloc[0]) if not row.empty else float("nan")

    diffs = 100.0 * (df["meas_share_slo"] - df["simA_share_slo"])
    equal = df["x_equal_minutes"].astype(int)
    closed = df[df["policy"].isin(["velocity", "reactive"])]
    return pd.DataFrame([
        {"fact": "rounds", "value": f"{len(df)}"},
        {"fact": "requests replayed per round", "value": f"{int(df['n_requests'].iloc[0]):,}"},
        {"fact": "window", "value": f"bell {meta['bell_min']}, minutes {meta['window_from_bell_min'][0]} to "
                                   f"{meta['window_from_bell_min'][1]} ({meta['duration_min']} min)"},
        {"fact": "peak arrivals per minute", "value": f"{int(meta['peak_arrivals_per_min'])}"},
        {"fact": "request lengths", "value": "drawn from the recorded classroom burst (178 pairs), "
                                             f"median reply {int(meta['lengths'].get('reply_tokens_median', 198))} tokens"},
        {"fact": "replica unit", "value": "K = 10 concurrent streams of one vLLM engine (0.90 of the card, 256 seats)"},
        {"fact": "emulated cold start (s, median / max over rounds)",
         "value": f"{df['cold_start_median_s'].median():.1f} / {df['cold_start_max_s'].max():.1f}"},
        {"fact": "minutes with the same replica count", "value": f"{equal.sum()} of {45 * len(df)}"},
        {"fact": "rounds with all 45 minutes equal", "value": f"{int((equal == 45).sum())} of {len(df)}"},
        {"fact": "closed-loop decisions re-checked / differing",
         "value": f"{int(closed['targets_checked'].sum())} / {int(closed['targets_mismatched'].sum())}"},
        {"fact": "attainment difference, measured minus simulated (points)",
         "value": f"{diffs.min():.1f} to {diffs.max():.1f}"},
        {"fact": "co-residency check (5 replicas through the router, tokens/s at 8/16/24/32/40 streams)",
         "value": "696 / 1,382 / 1,916 / 2,521 / 3,141"},
        {"fact": "capped replica alone, classroom prompts (TTFT p95 at K, tokens/s at K)",
         "value": f"{med('ttft_p95_s@N=10'):.3f} s, {med('throughput_tokens_s@N=10'):.0f}"},
    ])


# single_bell_replay.py --with-tap --tuned-disp <resource-cost floor tuning, c3 instance tier>
F1_CSV = "results/runs/fig1/f1_right_bell.csv"
F1_JSON = "results/runs/fig1/f1_right_bell.json"
F1_WINDOW = (-20, 24)  # minutes from the bell, inclusive: the 45 minutes Fig. 1 draws
SLO_S = 2.0
UNSERVED_S = 60.0


def fig1_facts_table(repo: Path) -> pd.DataFrame:
    """Every number printed inside Fig. 1 (the busiest bell of week 3, 72B, instance tier)."""
    df = pd.read_csv(repo / F1_CSV)
    meta = json.loads((repo / F1_JSON).read_text(encoding="utf-8"))
    bell_meta = json.loads((repo / "scripts/q2/pool_inputs/bell_c3_w3.json").read_text(encoding="utf-8"))
    bursts = pd.read_csv(repo / "results/q1/assistant/bursts.csv")
    one = bursts[(bursts["dataset"] == "ds_prompted") & (bursts["date"] == "2026-03-24") & (bursts["bell"] == "08:30")].iloc[0]
    lo, hi = F1_WINDOW
    w = df[(df["minute_from_bell"] >= lo) & (df["minute_from_bell"] <= hi)]
    mu = float(meta["mu_req_min"])
    bell = int(meta["bell_min"])
    week, rem = divmod(bell, 7 * 1440)
    day, clock = divmod(rem, 1440)
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    rows = [
        {"fact": "bell", "value": f"week {week + 1}, {days[day]} {clock // 60:02d}:{clock % 60:02d}"},
        {"fact": "sections starting at the bell", "value": f"{int(bell_meta['n_sessions_at_bell'])}"},
        {"fact": "students (roster seats) at the bell", "value": f"{int(meta['seats_at_bell']):,}"},
        {"fact": "peak campus in-class arrivals per minute", "value": f"{int(w['arrivals_in'].max())}"},
        {"fact": "first minute the demand exceeds one replica", "value": f"{int(w.loc[w['arrivals_in'] > mu, 'minute_from_bell'].min())}"},
        {"fact": "lead time drawn (instance tier, min)", "value": f"{float(meta['lead_min']):.0f}"},
        {"fact": "recorded class: students, 10-90% ramp (min)", "value": f"{int(one['n_students'])}, {float(one['ramp_10_90_min']):.1f}"},
    ]
    for pol, name in (("tap", "TAP"), ("reactive", "Reactive")):
        t = w[f"ttft_p95_{pol}"]
        x = w[f"replicas_{pol}"]
        rows.append({"fact": f"{name}: minutes with p95 > 2 s / not served, replica-minutes, peak",
                     "value": f"{int((t > SLO_S).sum())} / {int((t > UNSERVED_S).sum())}, {x.sum():.0f}, {x.max():.0f}"})
    r = w["replicas_reactive"]
    at_peak = w.loc[r == r.max(), "minute_from_bell"]
    rows.append({"fact": "Reactive at its peak: minutes, first, last",
                 "value": f"{int((r == r.max()).sum())}, {int(at_peak.min())}, {int(at_peak.max())}"})
    return pd.DataFrame(rows)


# --- central cell C on the laptop's streams (rerun round) ---------------
PAIRED_CSV = "results/runs/paired_cell_c.csv"
RUNS = "outputs/validation_local/runs"
LAPTOP_TIERS: Dict[str, str] = {"inst": "d_inst", "rep": "d_rep"}
LAPTOP_POLICIES: Tuple[str, ...] = (
    "tap", "calendar_learn_prior", "calendar_adapt", "calendar", "calendar_learn", "stale_calendar",
    "bdinterp@0.1", "informed", "offline", "forecast_ilp", "lcp", "guard", "fixed", "velocity", "reactive")
VARIANT_WORDS: Dict[str, str] = {"cost (tier's own)": "cost-tuned (own tier)",
                                 "cost (replica-tier tuning)": "cost-tuned (replica-tier tuning)",
                                 "floor (dispersed slice)": "floor-tuned (dispersed)",
                                 "floor (dispersed slice, previous selection)":
                                     "floor-tuned (dispersed, previous selection)",
                                 "floor (synchronous slice)": "floor-tuned (synchronous)"}
FLOOR_DISPERSED = "floor (dispersed slice)"
TUNING_RESOURCE = f"{RUNS}/tuning_resource"  # section 3: <case>_<tier>/{compare.json, *_grid.csv}
WEEKLY_POLICIES: Tuple[str, ...] = (
    "tap", "calendar_learn_prior", "calendar_adapt", "calendar", "calendar_learn", "forecast_ilp",
    "stale_calendar", "offline", "reactive, cost-tuned (own tier)", "reactive, floor-tuned (dispersed)",
    "velocity, cost-tuned (own tier)", "velocity, floor-tuned (dispersed)")
WEEKLY_INSTANCE_ONLY: Tuple[str, ...] = ("calendar_adapt", "calendar")  # quoted at the instance tier
SLO_FLOOR = 0.95
MASS_SEED = 10
WEEK_MIN = 7 * 1440


def _variant_words(policy: str, variant: str, tier: str) -> str:
    """The policy with its tuning in words; at the replica tier its replica-tier tuning is its own."""
    if variant == "-":
        return policy
    if tier == "rep" and variant == "cost (replica-tier tuning)":
        variant = "cost (tier's own)"
    return f"{policy}, {VARIANT_WORDS.get(variant, variant)}"


def _paired(repo: Path) -> pd.DataFrame:
    """``paired_cell_c.csv`` with each variant named in words, ordered case, tier, policy, variant."""
    df = pd.read_csv(repo / PAIRED_CSV)
    df["name"] = [_variant_words(p, v, t) for p, v, t in zip(df["policy"], df["variant"], df["tier"])]
    if df.duplicated(["case", "tier", "name", "week"]).any():
        raise ValueError(f"{PAIRED_CSV}: two rows share a case, tier, policy name and week")
    keys = {"_tier": df["tier"].map(LAPTOP_TIERS),
            "_policy": df["policy"].map({p: i for i, p in enumerate(LAPTOP_POLICIES)}).fillna(99),
            "_variant": df["variant"].map({v: i for i, v in enumerate(VARIANT_WORDS)}).fillna(-1)}
    df = df.assign(**keys).sort_values(["case", *keys, "week"], kind="stable")
    return df.drop(columns=list(keys)).reset_index(drop=True)


def _where(case: str, tier: str) -> Dict[str, str]:
    return {"case": ABLATION_CASES.get(case, case), "tier": LAPTOP_TIERS.get(tier, tier)}


def laptop_cell_c_table(repo: Path) -> pd.DataFrame:
    """Central cell C on the laptop's streams, whole run: every policy variant of the paired table."""
    rows: List[Dict[str, object]] = []
    for r in _paired(repo).query("week == 0").itertuples(index=False):
        near = abs(float(r.slo_in) - SLO_FLOOR) <= 0.002
        rows.append({**_where(r.case, r.tier), "policy": r.name,
                     "gpu_h_wk": f"{r.gpu_h_per_week:.1f}", "starts_wk": f"{r.starts_per_week:.1f}",
                     "usd_wk": f"{r.usd_per_week:.0f}", "slo_in": f"{r.slo_in:.3f}",
                     "slo_in_4dp_if_within_0.002_of_0.95": f"{r.slo_in:.4f}" if near else "",
                     "slo_in_lowest_seed": f"{r.slo_in_min:.3f}", "seeds": int(r.seeds),
                     "run": r.source, "note": r.note, "source": PAIRED_CSV})
    return pd.DataFrame(rows)


def laptop_weekly_table(repo: Path) -> pd.DataFrame:
    """In-class attainment by teaching week, laptop cell C, the policies and variants the paper plots."""
    rows: List[Dict[str, object]] = []
    for (case, tier), part in _paired(repo).groupby(["case", "tier"], sort=False):
        for name in WEEKLY_POLICIES:
            if tier != "inst" and name in WEEKLY_INSTANCE_ONLY:
                continue
            weeks = part[(part["name"] == name) & (part["week"] > 0)].set_index("week")["slo_in"]
            if weeks.empty:
                continue
            rows.append({**_where(case, tier), "policy": name,
                         **{f"w{w}": f"{float(weeks.loc[w]):.3f}" for w in range(1, WEEKS + 1)},
                         "source": PAIRED_CSV})
    return pd.DataFrame(rows)


def strict_floor_table(repo: Path) -> pd.DataFrame:
    """Per case and tier: the cheapest feedback variant whose unrounded whole-run attainment is at
    least 0.95 against TAP and the learned-prior plan, and always the SLO-floor-tuned reactive rule."""
    rows: List[Dict[str, object]] = []
    whole = _paired(repo).query("week == 0")
    for (case, tier), part in whole.groupby(["case", "tier"], sort=False):
        gpu = part.set_index("name")["gpu_h_per_week"]
        tap, prior = float(gpu.loc["tap"]), float(gpu.loc["calendar_learn_prior"])
        feedback = part[part["policy"].isin(("reactive", "velocity"))]
        cheapest = feedback[feedback["slo_in"] >= SLO_FLOOR].sort_values("gpu_h_per_week").head(1)
        floor = feedback[(feedback["policy"] == "reactive") & (feedback["variant"] == FLOOR_DISPERSED)]
        for label, pick in (("cheapest feedback variant at >= 0.95 (unrounded)", cheapest),
                            ("SLO-floor-tuned reactive rule (always listed)", floor)):
            base = {**_where(case, tier), "row": label}
            if pick.empty:
                rows.append({**base, "feedback_variant": "none reaches 0.95", "source": PAIRED_CSV})
                continue
            r = pick.iloc[0]
            g, s = float(r["gpu_h_per_week"]), float(r["slo_in"])
            rows.append({**base, "feedback_variant": r["name"], "gpu_h_wk": f"{g:.1f}",
                         "slo_in": f"{s:.3f}", "slo_in_4dp": f"{s:.4f}",
                         "lowest_seed_4dp": f"{float(r['slo_in_min']):.4f}",
                         "reaches_0.95": "yes" if s >= SLO_FLOOR else "no",
                         "tap_gpu_h_wk": f"{tap:.1f}", "extra_over_tap_pct": f"{100 * (g / tap - 1):+.1f}",
                         "extra_over_tap_pct_int": f"{100 * (g / tap - 1):+.0f}",
                         "calendar_learn_prior_gpu_h_wk": f"{prior:.1f}",
                         "extra_over_calendar_learn_prior_pct": f"{100 * (g / prior - 1):+.1f}",
                         "note": r["note"], "source": PAIRED_CSV})
    return pd.DataFrame(rows)


def _grid_row(grid: pd.DataFrame, config: Dict[str, object]) -> pd.Series:
    """The one row of a tuning grid that carries ``config``."""
    hit = grid[np.logical_and.reduce([grid[k] == v for k, v in config.items()])]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} tuning-grid rows carry {config}")
    return hit.iloc[0]


FLOOR_PICK_COLUMNS: Tuple[Tuple[str, str, str], ...] = (  # (column, grid column, format)
    ("slo_in_tune", "slo_in_tune_mean", ".4f"), ("slo_in_tune_3dp", "slo_in_tune_mean", ".3f"),
    ("gpu_h_tune", "gpu_hours_tune_mean", ".2f"), ("usd_tune", "resource_cost_tune_mean", ".2f"))


def _pick_reason(grid: pd.DataFrame, old: pd.Series) -> str:
    """Why the resource rule moved away from the previous configuration."""
    if not (grid["slo_in_tune_mean"] >= SLO_FLOOR).any():
        return "no grid point reaches 0.95: the lowest cost ratio is kept"
    if old["slo_in_tune_mean"] < SLO_FLOOR:
        return "previous below 0.95 on the laptop's tuning week"
    return "both reach 0.95: new has the lower resource cost"


def floor_pick_table(repo: Path) -> pd.DataFrame:
    """Per floor pick section 3 changed: the previous and the new configuration on the laptop's
    tuning week (means over the tuning seeds of the resource-rule grid) and why the pick moved."""
    rows: List[Dict[str, object]] = []
    for case in ABLATION_CASES:
        for tier in LAPTOP_TIERS:
            base = repo / TUNING_RESOURCE / f"{case}_{tier}"
            if not (base / "compare.json").exists():
                continue
            decided = json.loads((base / "compare.json").read_text())
            cloud = json.loads((repo / decided["old"]).read_text())["policies"]
            for policy in decided["changed"]:
                old_cfg, new_cfg = decided["old_config"][policy], decided["new_config"][policy]
                grid = pd.read_csv(base / f"{policy}_grid.csv", float_precision="round_trip")
                picks = {"prev": _grid_row(grid, old_cfg), "new": _grid_row(grid, new_cfg)}
                knobs = [f"{k} {old_cfg[k]} -> {v}" for k, v in new_cfg.items() if old_cfg[k] != v]
                n_ok = int((grid["slo_in_tune_mean"] >= SLO_FLOOR).sum())
                rows.append({
                    **_where(case, tier), "policy": policy, "knobs (previous -> new)": ", ".join(knobs),
                    "prev_slo_in_tune_cloud": f"{cloud[policy]['tuning']['slo_in_mean']:.4f}",
                    **{f"{tag}_{col}": f"{row[src]:{fmt}}" for tag, row in picks.items()
                       for col, src, fmt in FLOOR_PICK_COLUMNS},
                    "grid_points_at_0.95": f"{n_ok} of {len(grid)}", "why": _pick_reason(grid, picks["prev"]),
                    "source": f"{TUNING_RESOURCE}/{case}_{tier}/{policy}_grid.csv"})
    return pd.DataFrame(rows)


def resource_floor_rows(repo: Path) -> pd.DataFrame:
    """The floor configuration the resource rule selected per case, tier and policy, in the columns
    of ``tuning.csv``: best.json's configuration and, over the rule's grid, the best mean attainment
    over the tuning seeds (``max_slo_tune``) and the grid points reaching the floor."""
    rows: List[Dict[str, object]] = []
    for case in ABLATION_CASES:
        for tier, tuned_at in LAPTOP_TIERS.items():
            base = repo / TUNING_RESOURCE / f"{case}_{tier}"
            best = json.loads((base / "best.json").read_text())
            decided = json.loads((base / "compare.json").read_text())
            if best["slice"]["slo_select"] != "resource" or best["slice"]["slo_floor"] != SLO_FLOOR:
                raise ValueError(f"{base}/best.json is not the resource rule at the {SLO_FLOOR} floor")
            for policy, block in best["policies"].items():
                if block["config"] != decided["new_config"][policy]:
                    raise ValueError(f"{base}: best.json and compare.json name different {policy} configurations")
                grid = pd.read_csv(base / f"{policy}_grid.csv", float_precision="round_trip")
                _grid_row(grid, block["config"])  # the selection is one point of the grid
                slo = grid["slo_in_tune_mean"]
                rows.append({"case": case, "tier": tier, "tuned_at": tuned_at, "policy": policy,
                             "config": ", ".join(f"{k}={v}" for k, v in block["config"].items()),
                             "max_slo_tune": float(slo.max()), "n_reach_floor": int((slo >= SLO_FLOOR).sum()),
                             "grid_size": len(grid),
                             "source": f"{TUNING_RESOURCE}/{case}_{tier}/best.json, {policy}_grid.csv"})
    return pd.DataFrame(rows)


def floor_config_table(repo: Path) -> pd.DataFrame:
    """Section 28c: the resource rule's floor configuration of every rule at c2 and c3, both tiers,
    with the best tuning-week attainment of its grid and the grid points at the floor."""
    return pd.DataFrame([{**_where(r.case, r.tier), "policy": r.policy, "floor_config": r.config,
                          "max_slo_in_tune": f"{r.max_slo_tune:.3f}",
                          "grid_points_at_0.95": f"{r.n_reach_floor} of {r.grid_size}", "source": r.source}
                         for r in resource_floor_rows(repo).itertuples(index=False)])


def tap_diagnostics_table(repo: Path) -> pd.DataFrame:
    """TAP's own report per R1 run, cell C, five seeds (runs without a DONE marker are skipped)."""
    rows: List[Dict[str, object]] = []
    root = repo / RUNS
    for path in sorted((root / "tap").glob("r1_*/diagnostics.csv")):
        if not (root / "logs" / f"{path.parent.name}.DONE").exists():
            logger.info("TAP diagnostics: %s is not finished, skipped", path.parent.name)
            continue
        d = pd.read_csv(path)
        for (case, tier), part in d[d["cell"] == "C"].groupby(["case", "tier"], sort=False):
            rows.append({"case": ABLATION_CASES.get(case, case), "tier": tier,
                         "lead_min": f"{float(part['lead_min'].iloc[0]):g}",
                         "eps": f"{float(part['trust'].iloc[0]):g}",
                         "phase_share_min": f"{float(part['phase_share'].min()):.6f}",
                         "n_replans_min / max": f"{int(part['n_replans'].min())} / {int(part['n_replans'].max())}",
                         "final_ratio_min / max": f"{float(part['final_ratio'].min()):.3f} / "
                                                  f"{float(part['final_ratio'].max()):.3f}",
                         "seeds": int(part["seed"].nunique()), "source": str(path.relative_to(repo))})
    return pd.DataFrame(rows)


StreamCache = Dict[Tuple[object, ...], Tuple[np.ndarray, np.ndarray]]  # (workload, seed, root) -> series


def mass_ratio_table(repo: Path, streams: Optional[StreamCache] = None) -> pd.DataFrame:
    """Realised over forecast in-class request mass of the stream TAP plans on: seed 10, cell C, six
    weeks, the stream built as ``scripts/audit/prefix_invariance.py`` builds it (about 15 s), unless
    ``streams`` already holds its (lam_in, lam_hat_learn_prior_in) (``canonical_laptop``)."""
    pilot = repo
    for sub in ("src", "scripts/sim", "scripts/audit"):
        if str(pilot / sub) not in sys.path:
            sys.path.insert(0, str(pilot / sub))
    import prefix_invariance as pi  # the command lines of the central learned-prior runs
    import run_sweep
    from pilot.sim.engine import build_stream

    rows: List[Dict[str, object]] = []
    streams = dict(streams or {})  # a local copy: the caller's mapping is only read
    for short, case in (("c2", "c2_32b_shared4k"), ("c3", "c3_72b_shared4k")):
        args = pi.sweep_args(short)
        at = args.index("--calibration") + 1
        args[at] = str(pilot / args[at])  # relative to the repository root in the runs; absolute here
        cfg = run_sweep.base_config(run_sweep.build_parser().parse_args(args))
        cfg = replace(cfg, workload=replace(cfg.workload, **pi.CELL_C), seed=MASS_SEED)
        key = (cfg.workload, cfg.seed, str(cfg.root))  # the stream depends on nothing else
        if key not in streams:
            stream = build_stream(cfg)
            if stream.lam_hat_learn_prior_in is None:
                raise ValueError("the stream carries no learned-prior forecast")
            streams[key] = (stream.lam_in, stream.lam_hat_learn_prior_in)
        lam, hat = streams[key]
        row: Dict[str, object] = {"case": ABLATION_CASES[case], "seed": MASS_SEED, "cell": "C",
                                  "realised_in_class_requests": f"{float(lam.sum()):,.0f}",
                                  "forecast_in_class_mass": f"{float(hat.sum()):,.0f}",
                                  "ratio_whole_run": f"{float(lam.sum() / hat.sum()):.3f}",
                                  "ratio_whole_run_2dp": f"{float(lam.sum() / hat.sum()):.2f}"}
        for w in range(lam.size // WEEK_MIN):
            part = slice(w * WEEK_MIN, (w + 1) * WEEK_MIN)
            row[f"w{w + 1}"] = f"{float(lam[part].sum() / hat[part].sum()):.3f}"
        row["source"] = ("stream of scripts/audit/prefix_invariance.py (sweep_args, CELL_C): lam_in / "
                         "lam_hat_learn_prior_in, prior data/timetable/slots_2025-2026-2.csv")
        rows.append(row)
    return pd.DataFrame(rows)
