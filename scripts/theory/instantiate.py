"""Numerical instantiation of the paper's analysis: the lemma bounds at a named calibration.

    uv run python scripts/theory/instantiate.py                       # placeholder calibration
    uv run python scripts/theory/instantiate.py --r 6 10 30 100 --d 0.6 3.8 8 60
    uv run python scripts/theory/instantiate.py --calibration batchB.json --source "batch B T3"

The sketch states every bound as a formula; this script is the one place the formulas meet numbers,
so that each figure in the paper has a single reproducible origin.  It takes the cost calibration
(``c``, ``beta``, ``mu``, the price-ratio scan ``r``, the graded ratio ``p_off/p_in``, the synchrony
levels ``sigma_2``, the lead tiers ``d`` and the bell spacing ``s``) from the command line or a JSON
file, reads the timetable quantities off the T2 tables

* ``results/t2/between_week_dispersion.csv``: ``|B|``, ``A`` (the largest per-bell week 5-16
  maximum), ``sum_tau`` of the per-bell medians (the typical week) and of the per-bell ranges (the
  R2 swing), from which ``P2M_bell`` follows;
* ``results/t2/week_to_week_error.csv``: the R1.5 step ``sum_tau delta_tau`` (week 5 against
  week 4) and the week-4 total it is measured against,

and writes ``results/theory/instantiation.csv`` (one row per ``sigma_2 x d x r``),
``results/theory/instantiation.md`` (the tables the paper quotes) and the section 7(b)
feasibility figure under ``results/theory/figures/``.

The default calibration is the PLACEHOLDER of the sketch and the markdown says so in its header;
re-run with the batch B numbers and ``--calibrated`` once T3 lands.

The simulator-consistent instantiation of the validation round, ``results/theory/
validation_<case>/sim/``, is this call per case.  The engine executes a lead of ``ceil(d)``
one-minute slots and reveals a slot after it ends, so the blind time of a lead ``d`` is
``ceil(d) + 1``: 2 and 9 min at 32B, 3 and 9 min at 72B (replica tier 1.5 min, instance tier
8 min), and 2 min for the lead-time sweep's d = 1.  The nominal leads (1, 1.5, 8) and 72B's
former 2.5 min stay in the list so that the earlier rows reproduce unchanged; the 72B blind times
2 and 3 min were added on 2026-09-23 with ``--no-figure`` (the feasibility figure does not
depend on ``--d``)::

    uv run python scripts/theory/instantiate.py --calibrated --c 0.0333 --beta 0.0333 --s 100 \\
        --rho-window 1 --p-off-ratio 0.16666666666666666 --sigma2 0.57 0.70 --r 30 100 \\
        --bells-multiplier 2 --mu 399.5 --d 1.0 2.0 8.0 9.0 \\
        --source "case c2_32b_shared4k, validation round 2026-09-21, ..." \\
        --out ../results/theory/validation_c2_32b_shared4k/sim
    #   72B: --mu 265.8 --d 1.5 2.0 2.5 3.0 8.0 9.0 --out ../results/theory/validation_c3_72b_shared4k/sim
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:  # sibling modules, not a package
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import draw_all  # noqa: E402
from lemmas import (  # noqa: E402
    Calibration,
    Inputs,
    Timetable,
    a1_holds,
    a2_holds,
    ell_int_of,
    ell_of,
    h1_holds,
    h2_holds,
    h2_min_r,
    kappa_off_of,
    kappa_of,
    lemma1_lower_bell,
    lemma_a_constant,
    lipschitz_x,
    lower_over_qmax,
    m_of,
    q_max,
    q_typ,
    upper_over_qmax,
)
from report import lemma1_tables, markdown  # noqa: E402

logger = logging.getLogger("instantiate")

DISPERSION_CSV = "results/t2/between_week_dispersion.csv"
WEEK_ERRORS_CSV = "results/t2/week_to_week_error.csv"
OUT_DIR = "results/theory"
STEP_WEEK = 5  # the R1.5 step: new-intake sections start here
BASELINE_WEEK = 4  # the profile the step is measured against
PLACEHOLDER_SOURCE = "PLACEHOLDER"


# --------------------------------------------------------------------------- inputs
def load_timetable(dispersion: Path, week_errors: Path, step_week: int = STEP_WEEK,
                   baseline_week: int = BASELINE_WEEK, bells_multiplier: int = 1) -> Timetable:
    """Read the section 1 timetable quantities off the two T2 tables.

    ``bells_multiplier`` counts more than one burst per bell block: the simulator prompts a
    session at the start of each of its two periods, so the simulator-consistent instantiation
    uses 2 (ten trigger times a day instead of five bells).
    """
    disp = pd.read_csv(dispersion)
    errs = pd.read_csv(week_errors).set_index("week")
    for column in ("max_w5_16", "median_w5_16", "range_w5_16", "max_abs_dev_w5_16"):
        if column not in disp.columns:
            raise KeyError(f"{dispersion}: missing column '{column}'")
    return Timetable(bells=int(len(disp)) * int(bells_multiplier), a_max=float(disp["max_w5_16"].max()),
                     sum_median=float(disp["median_w5_16"].sum()),
                     sum_range=float(disp["range_w5_16"].sum()),
                     sum_abs_dev=float(disp["max_abs_dev_w5_16"].sum()),
                     step_sum=float(errs.loc[step_week, "err_prev_week"]),
                     week4_total=float(errs.loc[baseline_week, "total_seat_starts"]))


def calibration_from(args: argparse.Namespace) -> Tuple[Calibration, str]:
    """``Calibration`` and its provenance line: JSON file first, then the explicit flags."""
    cal = Calibration()
    source = PLACEHOLDER_SOURCE
    if args.calibration:
        payload = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        source = str(payload.pop("source", args.calibration))
        tuples = {"r", "sigma2", "d_min"}
        cal = replace(cal, **{k: (tuple(float(x) for x in v) if k in tuples else v)
                              for k, v in payload.items()})
    explicit = {"c": args.c, "beta": args.beta, "mu": args.mu, "s_min": args.s,
                "p_off_ratio": args.p_off_ratio, "rho_window": args.rho_window}
    cal = replace(cal, **{k: v for k, v in explicit.items() if v is not None})
    for key, values in (("r", args.r), ("sigma2", args.sigma2), ("d_min", args.d)):
        if values:
            cal = replace(cal, **{key: tuple(float(v) for v in values)})
    if args.source:
        source = args.source
    return replace(cal, placeholder=not args.calibrated), source


# --------------------------------------------------------------------------- the table
def rows(inputs: Inputs) -> pd.DataFrame:
    """One row per ``(sigma_2, d, r)``: the lemma 1 interval, the premises and lemma A."""
    cal, tt = inputs.cal, inputs.tt
    b = cal.beta_over_c
    out: List[Dict[str, object]] = []
    for sigma2 in cal.sigma2:
        ell, ell_int = ell_of(sigma2), ell_int_of(sigma2)
        qmax, qtyp = q_max(cal, ell, tt), q_typ(cal, ell, tt)
        for d in cal.d_min:
            m = m_of(ell, d)
            factor = upper_over_qmax(b, ell, d)
            for r in cal.r:
                lower = lower_over_qmax(b, r, ell, d)
                out.append({
                    "sigma2": sigma2, "ell_min": ell, "ell_int": ell_int, "d_min": d, "m_min": m,
                    "r": r, "c": cal.c, "beta": cal.beta, "mu": cal.mu, "beta_over_c": b,
                    "p_off_ratio": cal.p_off_ratio, "s_min": cal.s_min, "rho_window": cal.rho_window,
                    "kappa": kappa_of(cal.c, r), "kappa_off": kappa_off_of(cal.c, r,
                                                                          cal.p_off_ratio),
                    "L_x": lipschitz_x(cal.c, r),
                    "a1_holds": a1_holds(r, cal.p_off_ratio),
                    "a2_holds": a2_holds(cal, r, ell, d),
                    "h1_holds": h1_holds(cal, ell), "h2_holds": h2_holds(cal, r, ell),
                    "h2_min_r": h2_min_r(b, ell, cal.p_off_ratio),
                    "h2_min_r_ell_int": h2_min_r(b, float(ell_int), cal.p_off_ratio),
                    "q_max_usd": qmax, "q_typ_usd": qtyp, "p2m_bell": tt.p2m_bell,
                    "lower_bell_usd": lemma1_lower_bell(cal, r, ell, d, tt.a_max),
                    "lower_week_usd": tt.bells * lemma1_lower_bell(cal, r, ell, d, tt.a_max),
                    "lower_over_qmax": lower, "upper_over_qmax": factor,
                    "lower_over_qtyp": lower * tt.p2m_bell,
                    "upper_over_qtyp": factor * tt.p2m_bell,
                    "r2_upper_pct_qtyp": 100.0 * tt.r2_share * factor,
                    "r15_upper_pct_qtyp": 100.0 * tt.r15_share_typ * factor,
                    "r15_upper_pct_week4": 100.0 * tt.r15_share_week4 * factor,
                    "lemma_a_constant": lemma_a_constant(cal, r, d),
                })
    frame = pd.DataFrame(out)
    logger.info("instantiated %d (sigma_2, d, r) combinations", len(frame))
    return frame


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    ap = argparse.ArgumentParser(description="Instantiate the analysis bounds numerically")
    ap.add_argument("--root", type=Path, default=REPO, help="root the input paths resolve against")
    ap.add_argument("--dispersion", default=DISPERSION_CSV)
    ap.add_argument("--week-errors", default=WEEK_ERRORS_CSV)
    ap.add_argument("--step-week", type=int, default=STEP_WEEK)
    ap.add_argument("--baseline-week", type=int, default=BASELINE_WEEK)
    ap.add_argument("--bells-multiplier", type=int, default=1,
                    help="bursts per bell block (2 = one per 45-min period, as the simulator triggers)")
    ap.add_argument("--out", type=Path, default=None, help=f"default <root>/{OUT_DIR}")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="JSON file of Calibration fields (plus an optional 'source')")
    ap.add_argument("--source", default=None, help="provenance line for the markdown header")
    ap.add_argument("--calibrated", action="store_true",
                    help="drop the PLACEHOLDER banner: the inputs are measured")
    ap.add_argument("--c", type=float, default=None, help="USD per replica-minute")
    ap.add_argument("--beta", type=float, default=None, help="USD per replica start")
    ap.add_argument("--mu", type=float, default=None, help="requests per minute per replica")
    ap.add_argument("--p-off-ratio", type=float, default=None, help="p_off / p_in")
    ap.add_argument("--r", type=float, nargs="*", default=None, help="price ratios p_in*mu/c")
    ap.add_argument("--sigma2", type=float, nargs="*", default=None, help="synchrony levels")
    ap.add_argument("--d", type=float, nargs="*", default=None, help="lead tiers, minutes")
    ap.add_argument("--s", type=float, default=None, help="minimum bell spacing, minutes")
    ap.add_argument("--rho-window", type=float, default=None,
                    help="requests per seat inside the burst window (default 1: the first turn)")
    ap.add_argument("--no-figure", action="store_true", help="skip the feasibility figure")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def inputs_from(args: argparse.Namespace) -> Inputs:
    """Assemble the calibration and the timetable quantities from the parsed arguments."""
    cal, source = calibration_from(args)
    tt = load_timetable(Path(args.root) / args.dispersion, Path(args.root) / args.week_errors,
                        step_week=args.step_week, baseline_week=args.baseline_week,
                        bells_multiplier=args.bells_multiplier)
    return Inputs(cal=cal, tt=tt, source=source, dispersion_csv=str(args.dispersion),
                  week_errors_csv=str(args.week_errors), step_week=args.step_week,
                  baseline_week=args.baseline_week)


def run(inputs: Inputs, out: Path, figure: bool = True) -> pd.DataFrame:
    """Write the CSV, the markdown and (unless suppressed) the figure; return the table."""
    out.mkdir(parents=True, exist_ok=True)
    frame = rows(inputs)
    frame.to_csv(out / "instantiation.csv", index=False)
    figure_dir = out / "figures"
    if figure:
        draw_all(inputs.cal, figure_dir)
    (out / "instantiation.md").write_text(
        markdown(inputs, frame, figure_dir, "instantiation.csv") + "\n", encoding="utf-8")
    logger.info("wrote %s and %s", out / "instantiation.csv", out / "instantiation.md")
    return frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    inputs = inputs_from(args)
    out = Path(args.out) if args.out else Path(args.root) / OUT_DIR
    frame = run(inputs, out, figure=not args.no_figure)
    head = frame[(frame["r"] == inputs.cal.r[min(1, len(inputs.cal.r) - 1)])]
    print(lemma1_tables(head, replace(inputs.cal, r=(head["r"].iloc[0],))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
