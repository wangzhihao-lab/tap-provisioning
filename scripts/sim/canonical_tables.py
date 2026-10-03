"""LaTeX tables for the supplementary material, generated from the canonical tables.

    uv run python scripts/sim/canonical_tables.py --from-results   # results/tables: Table II and the CSV-only tables
    uv run python scripts/sim/canonical_tables.py                  # every table (needs the run trees under outputs/)

Every table here is read from the same CSV sources as ``canonical_numbers.py`` with the same
rounding, so the extended tables of the supplementary carry no hand-typed number: the six timing
cells and the weekly table from ``results/runs/cells_laptop.csv`` (the laptop's runs,
canonical section 7), the boundary cases from the cloud's cells_long and the tunings from
tuning.csv.  The c2 and c3 floor rows of ``ext_tuning.tex`` are canonical section 28c
(``canonical_extra.resource_floor_rows``).  ``ext_billing.tex`` prints canonical section 31a
(``canonical_billing``) as a whole float, caption and label included.  ``main_results.tex``, the
main text's results table, prints canonical section 27a (``paired_cell_c.csv``, whole run) as a
bare tabular; the main text wraps it in its float.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical_billing import billing_tables  # noqa: E402
from canonical_extra import (  # noqa: E402
    FLOOR_DISPERSED, LAPTOP_TIERS, PAIRED_CSV, pool_table, resource_floor_rows)
from canonical_sensitivity import (  # noqa: E402
    ADOPTION_RUNS, HELDOUT_RUNS, RESERVATION_RUNS, ablation_blocks, cells_blocks, sweep_blocks)
from canonical_sensitivity_tables import ablation_table, sweep_table  # noqa: E402
from canonical_cells_laptop import CELLS_CSV  # noqa: E402

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
logger = logging.getLogger("canonical_tables")

CASE_LABEL = {"c1_7b_shared4k": "c1", "c2_32b_shared4k": "c2", "c3_72b_shared4k": "c3",
              "c4_32b_unique2k": "c4", "c5_72b_unique4k": "c5"}
POLICY_LABEL = {"offline": "Offline optimum", "informed": "Informed optimum", "fixed": "Base tier only",
                "reactive": "Reactive (HPA)", "reactive (SLO>=0.95, disp)": "Reactive, SLO-floor tuned",
                "velocity": "Token velocity", "velocity (SLO>=0.95, disp)": "Token velocity, SLO-floor tuned",
                "lcp": "LCP", "forecast_ilp": "History + ILP", "stale_calendar": "Last semester's timetable",
                "calendar": "Timetable, known timing", "calendar_adapt": "Timetable + online size correction",
                "calendar_learn": "Timetable + learned timing", "calendar_learn_prior": "Timetable + learned timing (prior)",
                "tap": "TAP", "bdinterp@0.1": "BDINTERP-$d$, $\\varepsilon=0.1$", "guard": "Bell-aware guard"}
POLICY_ORDER = list(POLICY_LABEL)


def _tex_escape(text: str) -> str:
    return text.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


LAPTOP_KEYS: Dict[Tuple[str, str], str] = {  # (tier, variant) of cells_laptop.csv -> key suffix
    ("inst", "-"): "", ("rep", "-"): "", ("inst", "cost (tier's own)"): "",
    ("rep", "cost (replica-tier tuning)"): "", ("inst", FLOOR_DISPERSED): " (SLO>=0.95, disp)",
    ("rep", FLOOR_DISPERSED): " (SLO>=0.95, disp)"}


def laptop_keyed(laptop: pd.DataFrame) -> pd.DataFrame:
    """``cells_laptop.csv`` with the ``POLICY_LABEL`` key of each row (the tier's own cost tuning
    under the policy's name, the dispersed-slice floor tuning with its suffix) and the tier as
    ``d_inst`` / ``d_rep``; a variant without a key raises."""
    pairs = list(zip(laptop["tier"], laptop["variant"]))
    unknown = sorted(set(pairs) - set(LAPTOP_KEYS))
    if unknown:
        raise ValueError(f"{CELLS_CSV}: variants without a table key: {unknown}")
    keys = laptop["policy"] + pd.Series([LAPTOP_KEYS[k] for k in pairs], index=laptop.index)
    if not set(keys) <= set(POLICY_LABEL):
        raise ValueError(f"{CELLS_CSV}: policies without a label: {sorted(set(keys) - set(POLICY_LABEL))}")
    out = laptop.assign(key=keys, tier=laptop["tier"].map(LAPTOP_TIERS))
    if out.duplicated(["case", "tier", "cell", "key", "week"]).any():
        raise ValueError(f"{CELLS_CSV}: two rows share a case, tier, cell, table key and week")
    return out


def cells_table(laptop: pd.DataFrame, case: str, tier: str) -> str:
    """One case x tier of ``cells_laptop.csv`` (``laptop_keyed``), whole run: rows = policies,
    columns = cells A..F, entry = GPU-h / attainment."""
    sel = laptop[(laptop["case"] == case) & (laptop["tier"] == tier) & (laptop["week"] == 0)]
    cols = sorted(sel["cell"].unique())
    lines: List[str] = ["\\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", "\\toprule",
                        "Policy & " + " & ".join(cols) + " \\\\", "\\midrule"]
    for policy in POLICY_ORDER:
        part = sel[sel["key"] == policy].set_index("cell")
        if part.empty:
            continue
        vals = [f"{part.loc[c, 'gpu_h_per_week']:.1f} / {part.loc[c, 'slo_in']:.3f}" if c in part.index else "---" for c in cols]
        lines.append(_tex_escape(POLICY_LABEL[policy]) + " & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


TUNING_VARIANT: Dict[str, str] = {"tuning8_r30": "cost", "tuning_r30": "cost",  # tuning.csv dir -> variant
                                  "tuning8_r30_slo95disp": "floor", "tuning_r30_slo95disp": "floor"}


def tuning_table(tuning: pd.DataFrame, floor: pd.DataFrame) -> str:
    """Selected baseline configuration per case, tier and variant: cost (the cost-ratio tuning of
    ``tuning.csv``) and floor, the cloud's dispersed-slice selection except at the (case, tier)
    pairs of ``floor`` (the resource rule's, ``canonical_extra.resource_floor_rows``)."""
    sel = tuning[(tuning["ratio"] == 30) & tuning["tuning"].isin(TUNING_VARIANT)]
    sel = sel.assign(variant=sel["tuning"].map(TUNING_VARIANT))
    keys = set(zip(floor["case"], floor["tuned_at"]))
    replaced = pd.Series([k in keys for k in zip(sel["case"], sel["tuned_at"])], index=sel.index)
    sel = pd.concat([sel[~(replaced & (sel["variant"] == "floor"))], floor.assign(variant="floor")], ignore_index=True)
    lines: List[str] = ["\\begin{tabular}{@{}llllp{0.42\\textwidth}r@{}}", "\\toprule",
                        "Case & Tier & Variant & Baseline & Configuration & Best attainment on the tuning slice \\\\", "\\midrule"]
    for _, r in sel.sort_values(["case", "tuned_at", "variant", "policy"]).iterrows():
        cfg = _tex_escape(str(r["config"]).replace("hpa_", "").replace("velocity_", ""))
        lines.append(f"{CASE_LABEL.get(r['case'], r['case'])} & {r['tuned_at'].replace('d_', '')} & {_tex_escape(str(r['variant']))} & "
                     f"{_tex_escape(POLICY_LABEL.get(r['policy'], r['policy']))} & {cfg} & {float(r['max_slo_tune']):.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def weekly_table(laptop: pd.DataFrame) -> str:
    """In-class attainment by teaching week, cell C, instance tier, every policy of
    ``cells_laptop.csv`` (``laptop_keyed``) with weekly rows."""
    sel = laptop[(laptop["cell"] == "C") & (laptop["tier"] == "d_inst") & (laptop["week"] > 0)]
    weeks = list(range(1, 7))
    lines: List[str] = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule", "Case & Policy & " + " & ".join(f"week {w}" for w in weeks) + " \\\\", "\\midrule"]
    for case in ("c2_32b_shared4k", "c3_72b_shared4k"):
        part = sel[sel["case"] == case]
        for policy in POLICY_ORDER:
            by = part[part["key"] == policy].set_index("week")["slo_in"]
            if by.empty:
                continue
            vals = " & ".join(f"{float(by.loc[w]):.3f}" for w in weeks)
            lines.append(f"{CASE_LABEL[case]} & {_tex_escape(POLICY_LABEL[policy])} & {vals} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


ABLATION_CASE = {"c2 32B shared-4k": "32B (c2)", "c3 72B shared-4k": "72B (c3)"}
ABLATION_SETTING = {"graded penalty, in-class priority on (main setting)": "graded, priority on (main)",
                    "graded penalty, in-class priority off": "graded, priority off",
                    "p_off = p_in, in-class priority on": "flat penalty, priority on"}
ABLATION_POLICY = {"offline": "Offline optimum", "tap": "TAP", "calendar_learn_prior": "Timetable + learned timing (prior)",
                   "calendar_adapt": "Timetable + online size correction", "reactive": "Reactive, SLO-floor tuned"}


BOUNDARY_CASES = (("c1_7b_shared4k", "7B shared 4k (c1)"), ("c4_32b_unique2k", "32B private 2k (c4)"), ("c5_72b_unique4k", "72B private 4k (c5)"))


def boundary_cases_table(cells: pd.DataFrame) -> str:
    """The three non-central cases at cell C, r = 30, both tiers: rows = policies, entry = GPU-h / attainment."""
    sel = cells[(cells["cell"] == "C") & (cells["ratio"] == 30) & (cells["tier"] == cells["tuned_at"]) & (cells["slots"] == "main")]
    cols = [(case, tier) for case, _ in BOUNDARY_CASES for tier in ("d_rep", "d_inst")]
    lines: List[str] = ["\\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", "\\toprule",
                        "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}}" for _, label in BOUNDARY_CASES) + " \\\\",
                        "Policy & " + " & ".join("replica & instance" for _ in BOUNDARY_CASES) + " \\\\", "\\midrule"]
    for policy in POLICY_ORDER:
        vals = []
        for case, tier in cols:
            part = sel[(sel["case"] == case) & (sel["tier"] == tier) & (sel["policy"] == policy)]
            vals.append(f"{float(part['gpu_h_wk_mean'].iloc[0]):.1f} / {float(part['slo_in_mean'].iloc[0]):.3f}" if not part.empty else "---")
        if all(v == "---" for v in vals):
            continue
        lines.append(_tex_escape(POLICY_LABEL[policy]) + " & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


SENSITIVITY_POLICY = {"offline": "Offline optimum", "tap": "TAP", "calendar": "Timetable + known timing",
                 "calendar_learn_prior": "Timetable + learned timing (prior)", "reactive": "Reactive, SLO-floor tuned",
                 "velocity": "Token velocity, SLO-floor tuned"}


def sensitivity_table_tex(rows: pd.DataFrame, group_col: str, group_head: str) -> str:
    """Sensitivity-round sweeps: one block per group value and case, entry = GPU-h / attainment / excess USD per week."""
    lines: List[str] = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
                        f"{group_head} & Policy & GPU-h/wk & $S^{{\\mathrm{{in}}}}$ & $S^{{\\mathrm{{off}}}}$ & Excess USD/wk \\\\", "\\midrule"]
    first = True
    for case, block in rows.groupby("case", sort=False):
        if not first:
            lines.append("\\addlinespace")
        first = False
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{ABLATION_CASE.get(case, case)}}}}} \\\\")
        for _, r in block.iterrows():
            cells = [str(r[group_col]), SENSITIVITY_POLICY.get(r["policy"], r["policy"]), str(r["gpu_h_wk"]), str(r["slo_in"]),
                     str(r["slo_off"]), str(r["excess_usd_wk"])]
            lines.append(" & ".join(_tex_escape(c) for c in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def ablation_table_tex(rows: pd.DataFrame) -> str:
    """The admission-priority and penalty-grading ablation (canonical section 14), one block per case."""
    lines: List[str] = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
                        "Setting & Policy & GPU-h/wk & USD/wk & $S^{\\mathrm{in}}$ & $S^{\\mathrm{off}}$ & "
                        "Timeouts per $10^6$ \\\\", "\\midrule"]
    for i, (case, block) in enumerate(rows.groupby("case", sort=False)):
        if i:
            lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{ABLATION_CASE.get(case, case)}}}}} \\\\")
        for _, r in block.iterrows():
            cells = [ABLATION_SETTING.get(r["setting"], r["setting"]), ABLATION_POLICY.get(r["policy"], r["policy"]),
                     str(r["gpu_h_wk"]), str(r["usd_wk"]), str(r["slo_in"]), str(r["slo_off"]),
                     str(r["timeouts_per_million_requests"])]
            lines.append(" & ".join(_tex_escape(c) for c in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def pool_table_tex(rows: pd.DataFrame) -> str:
    """Batch B5: the closed-loop pool replay against the simulator, one row per round."""
    lines: List[str] = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
                        "Round & \\multicolumn{2}{c}{Share within 2\\,s} & \\multicolumn{2}{c}{Replica-minutes} & "
                        "\\multicolumn{2}{c}{TTFT p95 (s)} & Minutes \\\\",
                        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
                        " & GPU & sim. & GPU & sim. & GPU & sim. & equal \\\\", "\\midrule"]
    for _, r in rows.iterrows():
        cells = [str(r["round"]), str(r["measured share within 2 s"]), str(r["simulated share"]),
                 str(r["measured replica-minutes"]), str(r["simulated replica-minutes"]),
                 str(r["measured TTFT p95 (s)"]), str(r["simulated TTFT p95 (s)"]),
                 str(r["minutes with the same replica count"]).replace(" of 45", "")]
        lines.append(" & ".join(_tex_escape(c) for c in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


BILLING_LABEL: Dict[str, str] = {  # policy names of canonical section 31a -> labels of the table
    "tap": "TAP", "calendar_learn_prior": POLICY_LABEL["calendar_learn_prior"],
    "calendar": POLICY_LABEL["calendar"], "reactive, cost-tuned (own tier)": POLICY_LABEL["reactive"],
    "velocity, cost-tuned (own tier)": POLICY_LABEL["velocity"], "fixed": POLICY_LABEL["fixed"],
    "reactive, floor-tuned (dispersed)": POLICY_LABEL["reactive (SLO>=0.95, disp)"],
    "velocity, floor-tuned (dispersed)": POLICY_LABEL["velocity (SLO>=0.95, disp)"]}
BILLING_TIER: Dict[str, str] = {"d_inst": "instance tier", "d_rep": "replica tier"}
BILLING_CALIBERS: Tuple[Tuple[str, str], ...] = (("order", "From the order, $\\beta=c\\,d$"),
                                                 ("main", "Main runs, $\\beta=c$"))
BILLING_CAPTION = "Billing a replica start from its order."  # columns and prices: supplementary.tex, sup:billing


def _signed_pct(text: str) -> str:
    """A signed percentage of canonical section 31a in LaTeX; ``-`` (TAP's own row) as a dash."""
    return "---" if text == "-" else f"${text[0]}${text[1:]}\\%"


def billing_table_tex(rows: pd.DataFrame) -> str:
    """Canonical section 31a as a float: one block per case-tier, the two calibers side by side."""
    ncol = 1 + 3 * len(BILLING_CALIBERS)
    lines: List[str] = [
        "\\begin{table*}[t]", "  \\centering", f"  \\caption{{{BILLING_CAPTION}}}",
        "  \\label{tab:ext-billing}", "  \\footnotesize", "  \\setlength{\\tabcolsep}{5pt}",
        "  \\begin{tabular}{@{}l" + "r" * (ncol - 1) + "@{}}", "    \\toprule",
        "    & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{head}}}" for _, head in BILLING_CALIBERS)
        + " \\\\",
        "    " + "".join(f"\\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(BILLING_CALIBERS))),
        "    Policy" + " & GPU-h/wk & vs.\\ TAP & $S^{\\mathrm{in}}$" * len(BILLING_CALIBERS) + " \\\\",
        "    \\midrule"]
    for i, ((case, tier), block) in enumerate(rows.groupby(["case", "tier"], sort=False)):
        if i:
            lines.append("    \\addlinespace")
        title = f"{ABLATION_CASE.get(case, case)}, {BILLING_TIER[tier]}"
        lines.append(f"    \\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        by = {caliber: part.set_index("policy") for caliber, part in block.groupby("caliber")}
        policies = list(by[BILLING_CALIBERS[0][0]].index)
        if any(list(by[caliber].index) != policies for caliber, _ in BILLING_CALIBERS):
            raise ValueError(f"{case}, {tier}: the calibers list different policies")
        for policy in policies:
            cells = [_tex_escape(BILLING_LABEL[policy])]
            for caliber, _ in BILLING_CALIBERS:
                r = by[caliber].loc[policy]
                cells += [str(r["billed_gpu_h_wk"]), _signed_pct(str(r["vs_tap_pct"])), str(r["slo_in"])]
            lines.append("    " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


MAIN_COLUMNS: Tuple[Tuple[str, str, str], ...] = (  # (case, tier of paired_cell_c.csv, head) per column pair
    ("c2_32b_shared4k", "rep", "32B, replica"), ("c2_32b_shared4k", "inst", "32B, instance"),
    ("c3_72b_shared4k", "rep", "72B, replica"), ("c3_72b_shared4k", "inst", "72B, instance"))
OWN_COST: Dict[str, str] = {"inst": "cost (tier's own)",  # variant "cost": the tier's own cost tuning
                            "rep": "cost (replica-tier tuning)"}
MAIN_GROUPS: Tuple[Tuple[Tuple[str, str, str], ...], ...] = (  # (label, policy, variant); groups spaced
    (("Offline optimum", "offline", "-"), ("Informed optimum", "informed", "-")),
    (("TAP", "tap", "-"), ("Timetable, known timing", "calendar", "-"),
     ("+ size correction", "calendar_adapt", "-"), ("+ learned timing, prior", "calendar_learn_prior", "-"),
     ("+ learned timing, no prior", "calendar_learn", "-")),
    (("Last semester's timetable", "stale_calendar", "-"), ("History + ILP", "forecast_ilp", "-"),
     ("Bell-aware guard", "guard", "-"), ("LCP", "lcp", "-")),
    (("Reactive, cost-tuned", "reactive", "cost"), ("Reactive, floor-tuned", "reactive", FLOOR_DISPERSED),
     ("Token velocity, cost-tuned", "velocity", "cost"),
     ("Token velocity, floor-tuned", "velocity", FLOOR_DISPERSED)),
    (("Base tier only", "fixed", "-"),))


def _main_cell(whole: pd.DataFrame, case: str, tier: str, policy: str, variant: str) -> str:
    """GPU-hours per week (1 decimal) and attainment (3 decimals) of the one whole-run row of the
    case, tier, policy and variant; no row, several rows or a missing value raise."""
    want = OWN_COST[tier] if variant == "cost" else variant
    hit = whole[(whole["case"] == case) & (whole["tier"] == tier) & (whole["policy"] == policy)
                & (whole["variant"] == want)]
    if len(hit) != 1:
        raise ValueError(f"{PAIRED_CSV}: {len(hit)} whole-run rows for {case}, {tier}, {policy}, {want}")
    gpu, slo = float(hit["gpu_h_per_week"].iloc[0]), float(hit["slo_in"].iloc[0])
    if pd.isna(gpu) or pd.isna(slo):
        raise ValueError(f"{PAIRED_CSV}: {case}, {tier}, {policy}, {want}: GPU-hours or attainment missing")
    return f"{gpu:.1f} & {slo:.3f}"


def main_results_table_tex(paired: pd.DataFrame) -> str:
    """The main text's results table, canonical section 27a: ``paired_cell_c.csv`` at week 0 (the
    six weeks), one row per entry of ``MAIN_GROUPS``, per case and tier the GPU-hours per week and
    in-class attainment.  A bare tabular; the float and its caption are the main text's.  A label
    that begins with "+" changes one part of the known-timing plan above it, as in Fig. 4.  At
    ``\\scriptsize`` it fits the IEEEtran column (252 pt) with ``\\tabcolsep`` 1.75 pt; the last
    head closes with ``@{}`` like the column it spans, or it widens the last column."""
    whole = paired[paired["week"] == 0]
    heads = [f"\\multicolumn{{2}}{{c{'@{}' if k == len(MAIN_COLUMNS) - 1 else ''}}}{{{head}}}"
             for k, (_, _, head) in enumerate(MAIN_COLUMNS)]
    lines: List[str] = [
        "\\begin{tabular}{@{}l" + "rr" * len(MAIN_COLUMNS) + "@{}}", "\\toprule",
        "& " + " & ".join(heads) + " \\\\",
        "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(MAIN_COLUMNS))),
        "Policy" + " & GPU-h & $S^{\\mathrm{in}}$" * len(MAIN_COLUMNS) + " \\\\", "\\midrule"]
    for i, group in enumerate(MAIN_GROUPS):
        if i:
            lines.append("\\addlinespace[1pt]")
        for label, policy, variant in group:
            cells = [_main_cell(whole, case, tier, policy, variant) for case, tier, _ in MAIN_COLUMNS]
            lines.append(_tex_escape(label) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Supplementary and main-text tables from the canonical CSVs")
    ap.add_argument("--tables", type=Path, default=REPO / "results/sim/validation/tables")
    ap.add_argument("--out", type=Path, default=REPO / "results/tables")
    ap.add_argument("--from-results", action="store_true",
                    help="write only the tables that read the CSVs under results/ (Table II, the "
                         "timing-cell, weekly and boundary-case tables); the others need the run "
                         "trees under outputs/")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.tables / "cells_long.csv")
    tuning = pd.read_csv(args.tables / "tuning.csv")
    laptop = laptop_keyed(pd.read_csv(REPO / CELLS_CSV, float_precision="round_trip"))
    for case in ("c2_32b_shared4k", "c3_72b_shared4k"):
        for tier in ("d_rep", "d_inst"):
            (args.out / f"ext_cells_{CASE_LABEL[case]}_{tier}.tex").write_text(cells_table(laptop, case, tier) + "\n", encoding="utf-8")
    (args.out / "ext_weekly.tex").write_text(weekly_table(laptop) + "\n", encoding="utf-8")
    (args.out / "ext_cases_boundary.tex").write_text(boundary_cases_table(cells) + "\n", encoding="utf-8")
    paired = pd.read_csv(REPO / PAIRED_CSV, float_precision="round_trip")
    (args.out / "main_results.tex").write_text(main_results_table_tex(paired) + "\n", encoding="utf-8")
    if args.from_results:
        logger.info("wrote %s (tables from results/ only)", args.out)
        return 0
    (args.out / "ext_tuning.tex").write_text(tuning_table(tuning, resource_floor_rows(REPO)) + "\n", encoding="utf-8")
    pool = pool_table(REPO)
    if not pool.empty:
        (args.out / "ext_pool.tex").write_text(pool_table_tex(pool) + "\n", encoding="utf-8")
    for name, frame, col, head in (("ext_sensitivity_cells", sweep_table(cells_blocks(REPO)), "cell", "Timing cell"),
                                   ("ext_reservation", sweep_table(sweep_blocks(REPO, RESERVATION_RUNS)), "tier", "Tier"),
                                   ("ext_adoption", sweep_table(sweep_blocks(REPO, ADOPTION_RUNS)), "section_adoption", "Adoption"),
                                   ("ext_heldout", sweep_table(sweep_blocks(REPO, HELDOUT_RUNS)), "prior_from", "Prior from")):
        if not frame.empty:
            if name == "ext_reservation":
                frame = frame.assign(tier="reservation")
            (args.out / f"{name}.tex").write_text(sensitivity_table_tex(frame, col, head) + "\n", encoding="utf-8")
    ablation = ablation_table(ablation_blocks(REPO))
    if not ablation.empty:
        (args.out / "ext_ablation.tex").write_text(ablation_table_tex(ablation) + "\n", encoding="utf-8")
    billing = billing_tables(REPO)[0]
    (args.out / "ext_billing.tex").write_text(billing_table_tex(billing) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
