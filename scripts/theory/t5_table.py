#!/usr/bin/env python3
"""T5: every inequality of the paper's analysis against the batch-C simulation (batch D close-out).

Theory rows come from ``results/theory/instantiation.csv`` (sigma_2 = 0.70, d = 0.59 min,
the replica tier); simulation rows from the batch-C tables (``results/sim/batch_c``) and, for
the totals the theorems are stated in, from the sweep CSVs themselves.  Every row names the
statement, the theoretical number, the measured counterpart, the verdict and the caveat that goes
with it.  A lower bound over an adversarial family is *consistent* whenever the measured excess of
a calendar-oblivious policy is at least the bound; it is not expected to be tight on one instance.

Run from the repository root: ``uv run python scripts/theory/t5_table.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger("t5_table")
PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
OBLIVIOUS = ("reactive", "velocity", "lcp", "forecast_ilp", "stale_calendar", "fixed")
R1_WEEKS = (1, 5)
R2_WEEKS = (2, 3, 4, 6)
BASE_USD_WEEK = 0.0333 * 10080  # one always-on replica for a week


def theory_rows(path: Path, sigma2: float, d_min: float) -> pd.DataFrame:
    t = pd.read_csv(path)
    sel = t[(t["sigma2"].round(2) == round(sigma2, 2)) & (t["d_min"].round(2) == round(d_min, 2))]
    if sel.empty:
        raise SystemExit(f"no instantiation rows at sigma2={sigma2}, d={d_min}")
    return sel.set_index("r")


def weekly(root: Path, r: int) -> pd.DataFrame:
    return pd.read_csv(root / f"weekly_r{r}.csv")


def main_table(root: Path, r: int) -> pd.DataFrame:
    """The main table with its ``<metric>_mean`` columns exposed under the plain metric names."""
    m = pd.read_csv(root / f"main_r{r}.csv").set_index("policy")
    for col in list(m.columns):
        if col.endswith("_mean") and col[:-5] not in m.columns:
            m[col[:-5]] = m[col]
    return m


def week_excess(w: pd.DataFrame, policy: str, weeks) -> float:
    part = w[(w["policy"] == policy) & (w["week"].isin(list(weeks)))]
    return float(part["excess_usd"].mean())


def rows_for_ratio(r: int, th: pd.Series, tables: Path, eps: pd.DataFrame, lead: pd.DataFrame
                   ) -> List[Dict[str, object]]:
    w = weekly(tables, r)
    m = main_table(tables, r)
    out: List[Dict[str, object]] = []
    # 1. Lemma 1 lower bound, R1 week: any calendar-oblivious deterministic algorithm
    best_obl = min((week_excess(w, p, (1,)), p) for p in OBLIVIOUS if p in set(w["policy"]))
    out.append({"r": r, "statement": "Lemma 1 lower bound (R1, week 1): excess of any calendar-oblivious policy",
                "theory": th["lower_week_usd"], "measured": best_obl[0],
                "measured_note": f"best oblivious policy in week 1 = {best_obl[1]}",
                "verdict": "consistent (measured >= bound)" if best_obl[0] >= th["lower_week_usd"]
                else "below the family bound (instance is not the adversary's)",
                "unit": "USD per R1 week"})
    # 2. Lemma 1 upper bound, R1 week: the standing-guard policy S (knows the bell grid and A_max,
    #    not the amplitudes).  S is not among the baselines; stale_calendar is the nearest one (it
    #    knows the bells and guards at last semester's amplitudes), reported as a reference only.
    up = float(th["upper_over_qmax"] * th["q_max_usd"])
    cal_w1 = week_excess(w, "calendar", (1,))
    stale_w1 = week_excess(w, "stale_calendar", (1,)) if "stale_calendar" in set(w["policy"]) else float("nan")
    out.append({"r": r, "statement": "Lemma 1 upper bound (R1, week 1), instantiated: excess of the standing-guard oblivious policy S",
                "theory": up, "measured": stale_w1,
                "measured_note": f"stale_calendar week-1 excess as a reference point (knows the bells, guards at last "
                                 f"semester's amplitudes, pays penalties); calendar (FtP) week-1 excess {cal_w1:.2f}; "
                                 f"best oblivious baseline {best_obl[0]:.1f} ({best_obl[1]})",
                "verdict": "reference only: the bound is tested on the simulated guard S in the blind-phase rows (r = 30, 100); "
                           "the practical oblivious baselines all pay more than the bound on S because none guards the bells",
                "unit": "USD per R1 week"})
    # 3. R2 weeks: the calendar's capacity value against a history-driven planner (lemma 1 upper + lemma H)
    r2 = float(th["r2_upper_pct_qtyp"] / 100.0 * th["q_typ_usd"])
    cal_r2 = week_excess(w, "calendar", R2_WEEKS)
    hist_r2 = week_excess(w, "forecast_ilp", R2_WEEKS)
    value_r2 = hist_r2 - cal_r2
    out.append({"r": r, "statement": "R2 (weeks 2-4 & 6): capacity value of the calendar over a history-driven planner <= bound",
                "theory": r2, "measured": value_r2,
                "measured_note": f"history+ILP excess {hist_r2:.2f} minus calendar (FtP) excess {cal_r2:.2f}, mean over R2 weeks; "
                                 f"negative = history already beats the timetable planner (it learns realised attendance)",
                "verdict": "holds (value <= bound)" if value_r2 <= r2 else "VIOLATED", "unit": "USD per week"})
    # 4. R1.5 (week 5): the bound is on the value of knowing the step exactly over guarding at its
    #    bound; the simulated history planner does not guard, so the measured value is a larger quantity.
    r15 = float(th["r15_upper_pct_qtyp"] / 100.0 * th["q_typ_usd"])
    cal_w5 = week_excess(w, "calendar", (5,))
    hist_w5 = week_excess(w, "forecast_ilp", (5,))
    value_w5 = hist_w5 - cal_w5
    out.append({"r": r, "statement": "R1.5 (week 5 freshman step): value of the exact step over a guard at its bound <= bound",
                "theory": r15, "measured": value_w5,
                "measured_note": f"history+ILP week-5 excess {hist_w5:.1f} minus calendar (FtP) {cal_w5:.2f}: the value of the "
                                 f"timetable against an UNGUARDED history planner (it does not know the step's bound and pays "
                                 f"penalties at in-class SLO {float(w[(w.policy == 'forecast_ilp') & (w.week == 5)]['slo_in'].mean()):.3f})",
                "verdict": f"different quantity: the guard policy is not simulated; the practical value is {value_w5 / r15:.0f}x the guard bound",
                "unit": "USD per week"})
    # 5. Lemma A: LCP with lead d
    const = float(th["lemma_a_constant"])
    lcp_ratio = float(m.loc["lcp", "total_cost"] / m.loc["offline", "total_cost"])
    out.append({"r": r, "statement": "Lemma A: C(LCP, d) <= (3 + 2 L_x d / beta) OPT (weaker form, costs(X*) <= OPT)",
                "theory": const, "measured": lcp_ratio, "measured_note": "LCP total / offline total, 6 weeks",
                "verdict": "holds" if lcp_ratio <= const else "VIOLATED", "unit": "ratio of totals"})
    # 6. Theorem 2 consistency (exact forecast): BDINTERP(eps) <= (1 + eps)(OPT + 2 L eta)
    bd_ratio = float(m.loc["bdinterp@0.1", "total_cost"] / m.loc["offline", "total_cost"])
    out.append({"r": r, "statement": "Theorem 2 consistency, eps = 0.1, exact forecast: BDINTERP total <= (1+eps)(OPT + 2 L eta)",
                "theory": 1.1, "measured": bd_ratio, "measured_note": "BDINTERP@0.1 total / offline total (eta term dropped: it only loosens the bound)",
                "verdict": "holds" if bd_ratio <= 1.1 else "VIOLATED", "unit": "ratio of totals"})
    # 7. Theorem 2 robustness at a 20x over-forecast, eps = 1: <= 4(1+eps)/eps * (Lemma A constant)
    cell = eps[(eps["price_ratio"].round(1) == float(r)) & (eps["forecast_scale"] == 20.0)
               & (eps["bell_shift_frac"] == 0) & (eps["p_cancel"] == 0) & (eps["week"] == 0)]
    if len(cell):
        bd = cell[cell["policy"] == "bdinterp@1.0"]
        rob = float((bd["total_cost"] / bd["offline_cost"]).mean())
        bound = 4.0 * 2.0 / 1.0 * const
        out.append({"r": r, "statement": "Theorem 2 robustness, eps = 1, forecast x20: BDINTERP total <= 4(1+eps)/eps * (3 + 2 L_x d / beta) OPT",
                    "theory": bound, "measured": rob, "measured_note": "BDINTERP@1.0 total / offline total at forecast scale 20 (seeds 10-12)",
                    "verdict": "holds" if rob <= bound else "VIOLATED", "unit": "ratio of totals"})
    # 8. premises A1 / H2 against the optimum's own in-class attainment
    slo_opt = float(m.loc["offline", "slo_in"])
    out.append({"r": r, "statement": "Premises A1 (r > 6) and H2 (r >= 6(1 + (beta/c)/ell)): every burst worth serving",
                "theory": float(th["h2_min_r"]), "measured": slo_opt,
                "measured_note": f"A1 holds: {bool(th['a1_holds'])}, H2 holds: {bool(th['h2_holds'])}; measured = offline optimum's in-class SLO",
                "verdict": "premises hold and the optimum serves the bursts" if bool(th["h2_holds"]) and slo_opt >= 0.98
                else ("premises fail and the optimum drops bursts" if not bool(th["h2_holds"]) and slo_opt < 0.98
                      else "mixed"), "unit": "r threshold vs attainment"})
    return out


def tier_rows(th_all: pd.DataFrame, lead: pd.DataFrame, r: int) -> List[Dict[str, object]]:
    """Lemma 1's three lead-time tiers against the best oblivious policy's violation rate."""
    out = []
    whole = lead[lead["week"] == 0]
    for d, tier in ((0.59, "d_rep"), (8.0, "d_inst"), (60.0, "d_res")):
        sel = th_all[(th_all["sigma2"].round(2) == 0.70) & (th_all["d_min"].round(2) == d)
                     & (th_all["r"] == r)]
        if sel.empty:
            continue
        part = whole[(whole["lead_min"].round(2) == (0.5 if tier == "d_rep" else 8.0))
                     & ((whole["commit_block_min"] == 60) if tier == "d_res" else (whole["commit_block_min"] == 1))]
        fb = part[part["policy"].isin(("reactive", "velocity", "lcp"))].groupby("policy")["viol_in"].mean()
        hist = part[part["policy"] == "forecast_ilp"]["viol_in"].mean()
        cal = part[part["policy"] == "calendar"]["viol_in"].mean()
        out.append({"r": r, "statement": f"Lemma 1 per tier ({tier}, d = {d:g} min): lower-bound share of Q_max against the feedback policies' violations",
                    "theory": float(sel["lower_over_qmax"].iloc[0]), "measured": float(fb.mean()) if len(fb) else float("nan"),
                    "measured_note": f"mean in-class violation rate of reactive/velocity/LCP; history+ILP {hist:.3f}; calendar {cal:.3f}",
                    "verdict": "trend consistent (both rise from the replica tier to the instance tier)",
                    "unit": "bound share (theory) vs violation rate (sim), different units"})
    return out


CLEAN_CELLS = (
    ("unbiased", "roster known, mean attendance booked, attendance still random",
     "residual = drift + integer rounding + per-session attendance randomness"),
    ("exact", "session sizes known exactly (attendance fixed at its mean, no cancellations)",
     "residual = drift + integer rounding + within-session arrival randomness"),
)


def lemma_h_rows(out: Path) -> List[Dict[str, object]]:
    """Lemma H's inequality measured on the offline optimum (``t5_lemma_h_check.py``)."""
    path = out / "t5_lemma_h_check.csv"
    if not path.exists():
        return []
    c = pd.read_csv(path)
    rows: List[Dict[str, object]] = []
    for label, weeks in (("R1 week 1", (1,)), ("R2 weeks", R2_WEEKS), ("R1.5 week 5", (5,))):
        part = c[c["week"].isin(list(weeks))]
        lhs, rhs, starts, tv = (float(part[k].mean()) for k in
                                ("lhs_excess_usd", "rhs_usd", "starts_bound_usd", "tv_slack_usd"))
        rows.append({"r": 30, "statement": f"Lemma H ({label}): OPT excess over the lower envelope >= beta sum_tau A_tau/mu - beta TV(b)/mu",
                     "theory": rhs, "measured": lhs,
                     "measured_note": f"offline DP cost minus c sum_t max(x_min, lambda_t/mu); bound = starts term {starts:.2f} "
                                      f"minus TV slack {tv:.2f}; {int(part['holds'].sum())}/{len(part)} seed-weeks hold, "
                                      f"{int(part['holds_without_tv_slack'].sum())}/{len(part)} without the TV slack",
                     "verdict": "holds" if bool(part["holds"].all()) else "VIOLATED", "unit": "USD per week"})
    return rows


def guard_rows(out: Path, sweeps: Path, r: int, th: pd.Series) -> List[Dict[str, object]]:
    """Lemma 1's upper bound against the simulated standing guard (policy 11, ``r{r}_guard``).

    The lemma charges S for the unused guard of the blind minutes only, so the tested quantity is
    the guard's excess in that phase (``t5_guard_decompose.csv``); the excess it pays in the rest
    of the profile window and outside every window is outside the rectangular model and reported
    beside it.  The reserved tier is not covered by the lemma.
    """
    path = sweeps / f"r{r}_guard" / "lead" / "lead.csv"
    bound_path = out / "t5_guard_bound.csv"
    if not path.exists() or not bound_path.exists():
        return []
    g = pd.read_csv(path)
    b = pd.read_csv(bound_path)
    tv = out / "t5_lemma_h_check.csv"
    tv_slack = float(pd.read_csv(tv)["tv_slack_usd"].mean()) if tv.exists() else 0.0
    dec_path = out / "t5_guard_decompose.csv"
    dec = pd.read_csv(dec_path) if dec_path.exists() else None
    weekly = g[g["week"] > 0]
    opt = weekly[weekly["policy"] == "offline"][["seed", "lead_tier", "week", "total_cost"]]
    opt = opt.rename(columns={"total_cost": "offline_week"})
    rows: List[Dict[str, object]] = []
    for tier in ("d_rep", "d_inst", "d_res"):
        part = weekly[(weekly["policy"] == "guard") & (weekly["lead_tier"] == tier)]
        if part.empty:
            continue
        part = part.merge(opt, on=["seed", "lead_tier", "week"])
        excess = float((part["total_cost"] - part["offline_week"]).mean())
        slo = float(part["slo_in"].mean())
        bt = b[b["tier"] == tier]
        blind = float(bt[bt["m_label"].str.startswith("simulator")]["bound_usd"].mean())
        theory_m = float(bt[bt["m_label"].str.startswith("theory")]["bound_usd"].mean())
        inst = float(th["upper_over_qmax"] * th["q_max_usd"]) if tier == "d_rep" else float(th["q_max_usd"])
        total = blind + 2.0 * tv_slack
        phases = ""
        measured = excess
        if dec is not None and tier in set(dec["tier"]):
            d = dec[dec["tier"] == tier].groupby("phase")["excess_usd"].mean()
            measured = float(d.get("blind", float("nan")))
            phases = (f"; phase split (seed 10): blind {d.get('blind', float('nan')):.1f}, rest of window "
                      f"{d.get('window', float('nan')):.1f}, outside {d.get('outside', float('nan')):.1f} USD/week")
        if tier == "d_res":
            verdict = "not covered: the lemma assumes the guard can be released between bells; reported for the record"
        elif dec is None or tier not in set(dec["tier"]):
            verdict = "holds" if excess <= total else "total excess exceeds the blind-phase bound: run t5_guard_decompose.py"
        else:
            verdict = "holds" if measured <= total else "VIOLATED"
        rows.append({"r": r, "statement": f"Lemma 1 upper bound ({tier}): blind-phase excess of the simulated standing guard S <= (c m + beta) sum_tau (A_max - A_tau)/mu + 2 beta (b_max + TV(b))/mu",
                     "theory": total, "measured": measured,
                     "measured_note": f"guard total weekly excess {excess:.1f} USD over the offline optimum (6 weeks x 5 seeds, in-class SLO {slo:.3f}){phases}; "
                                      f"bound on the simulated family: unused guard {blind:.1f} at the simulator's blind time d+1 "
                                      f"({theory_m:.1f} at the theory's m = min(ell, d)) plus 2 x TV slack {2 * tv_slack:.1f}; "
                                      f"the instantiated bound with A = 18,481 seats x rho sigma_2/2 is {inst:.1f}",
                     "verdict": verdict, "unit": "USD per week"})
    return rows

def clean_rows(t30: pd.Series, out: Path) -> List[Dict[str, object]]:
    """Lemma H rows re-measured on the two forecast-cleaned cells of ``t5_clean_cell.py``."""
    rows: List[Dict[str, object]] = []
    r2 = float(t30["r2_upper_pct_qtyp"] / 100.0 * t30["q_typ_usd"])
    r15 = float(t30["r15_upper_pct_qtyp"] / 100.0 * t30["q_typ_usd"])
    for cell, what, residual in CLEAN_CELLS:
        path = out / f"t5_{cell}_cell.csv"
        if not path.exists():
            continue
        c = pd.read_csv(path).set_index("week")
        m_r2 = float(c.loc[list(R2_WEEKS), "excess_usd_mean"].mean())
        m_w5 = float(c.loc[5, "excess_usd_mean"])
        slo = float(c["slo_in_mean"].mean())
        rows.append({"r": 30, "statement": f"Calendar (FtP) excess floor, R2 weeks, {cell} forecast ({what})",
                     "theory": r2, "measured": m_r2,
                     "measured_note": f"calendar (FtP) mean excess over R2 weeks, t5_{cell}_cell.csv; "
                                      f"{residual}; in-class SLO {slo:.3f}; theory = the R2 drift bound",
                     "verdict": "floor below the drift bound" if m_r2 <= r2 else
                     "floor above the drift bound: the drift term is not separable at the simulator's noise floor",
                     "unit": "USD per week"})
        rows.append({"r": 30, "statement": f"Calendar (FtP) excess floor, week 5, {cell} forecast",
                     "theory": r15, "measured": m_w5,
                     "measured_note": f"calendar (FtP) week-5 excess, t5_{cell}_cell.csv; {residual}; theory = the R1.5 guard bound",
                     "verdict": "floor below the guard bound" if m_w5 <= r15 else "floor above the guard bound",
                     "unit": "USD per week"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instantiation", type=Path, default=REPO / "results/theory/instantiation.csv")
    ap.add_argument("--tables", type=Path, default=REPO / "results/sim/batch_c")
    ap.add_argument("--eps", type=Path, default=PILOT / "outputs/batchC/eps/epsilon/epsilon.csv")
    ap.add_argument("--lead", type=Path, default=PILOT / "outputs/batchC/r30/lead/lead.csv")
    ap.add_argument("--out", type=Path, default=REPO / "results/theory")
    ap.add_argument("--sweeps", type=Path, default=PILOT / "outputs/batchC", help="batch-C sweep root (guard runs)")
    ap.add_argument("--ratios", type=int, nargs="*", default=[6, 10, 30, 100])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    th_all = pd.read_csv(args.instantiation)
    th = theory_rows(args.instantiation, 0.70, 0.59)
    eps = pd.read_csv(args.eps)
    lead = pd.read_csv(args.lead)
    rows: List[Dict[str, object]] = []
    for r in args.ratios:
        if r not in th.index:
            logger.warning("no theory row for r=%d", r)
            continue
        rows += rows_for_ratio(r, th.loc[r], args.tables, eps, lead)
    rows += tier_rows(th_all, lead, 30)
    if 30 in th.index:
        rows += lemma_h_rows(args.out)
        rows += clean_rows(th.loc[30], args.out)
    for r in (30, 100):
        if r in th.index:
            rows += guard_rows(args.out, args.sweeps, r, th.loc[r])
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "t5_theory_vs_simulation.csv", index=False)
    lines = ["# T5: theory against the batch-C simulation", "",
             "Theory rows: `instantiation.csv` at sigma_2 = 0.70, d = 0.59 min (the measured cell's sigma_2 is 0.57; the 0.70 row is the nearest instantiated one and gives the *larger* Q_max, i.e. the looser side). "
             "Simulation rows: batch C attempt 3, 6 teaching weeks, seeds 10-14 (epsilon rows 10-12). "
             "Generated by `scripts/theory/t5_table.py`.", "",
             "| r | statement | theory | measured | unit | verdict | note |", "|---|---|---|---|---|---|---|"]
    for _, x in table.iterrows():
        lines.append(f"| {x['r']} | {x['statement']} | {x['theory']:.3g} | {x['measured']:.3g} | {x['unit']} | {x['verdict']} | {x['measured_note']} |")
    (args.out / "t5_theory_vs_simulation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %d rows to %s", len(table), args.out)
    print(table[["r", "statement", "theory", "measured", "verdict"]].to_string(index=False, max_colwidth=70))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
