"""Render the instantiation tables of the paper's analysis as markdown.

Pure formatting on top of ``lemmas`` and the frame ``instantiate.rows`` builds: every table here
is a view of ``instantiation.csv``, so the markdown and the CSV cannot disagree.  Kept apart from
``instantiate`` because the arithmetic and the prose have no reason to share a file.

The blocks, in the order ``markdown`` assembles them: the calibration and timetable inputs with
their provenance, the two normalisations, lemma 1 against ``Q_max`` and against ``Q_typ``, the
three reading notes a table alone does not carry, the premise table, the R2 / R1.5 upper bounds,
lemma A's constant, and the pointer to the section 7(b) figure.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:  # sibling modules, not a package
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import NAME as FIGURE_NAME  # noqa: E402
from lemmas import (  # noqa: E402
    Calibration,
    Inputs,
    a1_holds,
    ell_of,
    h2_holds,
    lower_over_qmax,
    upper_over_qmax,
)


def _table(headers: Sequence[str], body: Sequence[Sequence[object]]) -> str:
    """A GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def _flag(value: object) -> str:
    """Premise flag as text, so a false premise is visible in a rendered table."""
    return "yes" if bool(value) else "**no**"


def lemma1_tables(frame: pd.DataFrame, cal: Calibration) -> str:
    """Lemma 1 normalised to ``Q_max`` and to ``Q_typ``, one block per normalisation."""
    ratios = list(cal.r)
    out = []
    for norm, low, high in (("Q_max", "lower_over_qmax", "upper_over_qmax"),
                            ("Q_typ", "lower_over_qtyp", "upper_over_qtyp")):
        headers = ["sigma_2", "ell (min)", "d (min)", "m (min)", f"upper/{norm}"]
        headers += [f"lower/{norm}, r={r:g}" for r in ratios]
        body = []
        for (sigma2, d), block in frame.groupby(["sigma2", "d_min"], sort=True):
            first = block.iloc[0]
            row = [f"{sigma2:.2f}", f"{first['ell_min']:.2f}", f"{d:g}", f"{first['m_min']:.2f}",
                   f"{first[high]:.3f}"]
            for r in ratios:
                cell = block[block["r"] == r].iloc[0]
                text = f"{cell[low]:.3f}"
                row.append(text if cell["a2_holds"] else f"{text} (A2 fails)")
            body.append(row)
        out.append(f"### Lemma 1, normalised to `{norm}`\n\n" + _table(headers, body))
    return "\n\n".join(out)


def premise_table(frame: pd.DataFrame, cal: Calibration) -> str:
    """Lemma H / A1 / A2 premises: the flags and the minimal ``r`` each one asks for."""
    headers = ["sigma_2", "ell (min)", "ell_int", "H1: c(s-ell) >= beta",
               "min r for A1", "min r for H2 (ell)", "min r for H2 (ell_int)",
               "min r for A2 at d >= ell", "r satisfying A1 & H2"]
    body = []
    for sigma2, block in frame.groupby("sigma2", sort=True):
        first = block.iloc[0]
        ell, ell_int = float(first["ell_min"]), int(first["ell_int"])
        a2_min = 1.0 + cal.beta_over_c / ell
        ok = [f"{r:g}" for r in cal.r
              if a1_holds(r, cal.p_off_ratio) and h2_holds(cal, r, ell)]
        body.append([f"{sigma2:.2f}", f"{ell:.2f}", ell_int, _flag(first["h1_holds"]),
                     f"{1.0 / cal.p_off_ratio:.2f}", f"{first['h2_min_r']:.2f}",
                     f"{first['h2_min_r_ell_int']:.2f}", f"{a2_min:.2f}",
                     ", ".join(ok) if ok else "none of the scan"])
    return "### Lemma H and the premises of lemma 1\n\n" + _table(headers, body)


def regime_table(frame: pd.DataFrame) -> str:
    """R2 and R1.5 upper bounds, as percentages of the two denominators."""
    headers = ["sigma_2", "d (min)", "(m + beta/c)/(ell + beta/c)", "R2 %Q_typ",
               "R1.5 %Q_typ", "R1.5 %Q_week4"]
    body = []
    for (sigma2, d), block in frame.groupby(["sigma2", "d_min"], sort=True):
        first = block.iloc[0]
        body.append([f"{sigma2:.2f}", f"{d:g}", f"{first['upper_over_qmax']:.3f}",
                     f"{first['r2_upper_pct_qtyp']:.1f}%",
                     f"{first['r15_upper_pct_qtyp']:.1f}%",
                     f"{first['r15_upper_pct_week4']:.1f}%"])
    return ("### R2 and R1.5: upper bounds only (lemma 1 upper + lemma H)\n\n"
            + _table(headers, body))


def lemma_a_table(frame: pd.DataFrame, cal: Calibration) -> str:
    """Lemma A's constant ``3 + 2 L_x d / beta``, which does not depend on ``sigma_2``."""
    headers = ["d (min)"] + [f"r={r:g}" for r in cal.r]
    body = []
    for d, block in frame.groupby("d_min", sort=True):
        row = [f"{d:g}"]
        for r in cal.r:
            row.append(f"{block[block['r'] == r].iloc[0]['lemma_a_constant']:.1f}")
        body.append(row)
    return ("### Lemma A: ROB(d) competitive constant `3 + 2 L_x d / beta`\n\n"
            + _table(headers, body))


def input_table(inputs: Inputs) -> str:
    """The calibration and timetable inputs, with where each number came from."""
    cal, tt = inputs.cal, inputs.tt
    body = [
        ["`c`", f"{cal.c:g} USD/replica-min", inputs.source],
        ["`beta`", f"{cal.beta:g} USD/start", inputs.source],
        ["`beta/c`", f"{cal.beta_over_c:g} replica-min", "derived"],
        ["`mu`", f"{cal.mu:g} req/min", inputs.source],
        ["`r = p_in mu / c`", ", ".join(f"{r:g}" for r in cal.r), "scan (p_in is a dual variable)"],
        ["`p_off/p_in`", f"{cal.p_off_ratio:.4g}", "graded SLO ratio, held fixed"],
        ["`sigma_2`", ", ".join(f"{s:g}" for s in cal.sigma2), "pilot Q1 arrival shapes"],
        ["`ell = 2/sigma_2`", ", ".join(f"{ell_of(s):.2f}" for s in cal.sigma2) + " min",
         "rectangular burst abstraction"],
        ["`d`", ", ".join(f"{d:g}" for d in cal.d_min) + " min", "provisioning tiers"],
        ["`s`", f"{cal.s_min:g} min", "minimum bell spacing"],
        ["`|B|`", f"{tt.bells}", f"`{inputs.dispersion_csv}`"],
        ["`A`", f"{tt.a_max:,.0f} seat-starts", f"max of `max_w5_16`, `{inputs.dispersion_csv}`"],
        ["`rho_window`", f"{cal.rho_window:g} request(s) per seat inside the window",
         "amplitude `A = n rho_window sigma_2 / 2` req/min; 1 = the first turn, the second turn "
         "(gap mean 170 s, median 118 s) is in-session background"],
        ["`sum_tau median`", f"{tt.sum_median:,.0f}", f"`{inputs.dispersion_csv}`"],
        ["`sum_tau Delta_tau`", f"{tt.sum_range:,.0f} ({100 * tt.r2_share:.1f}% of the typical "
         f"week)", f"sum of `range_w5_16`, `{inputs.dispersion_csv}`"],
        ["`P2M_bell`", f"{tt.p2m_bell:.3f}", "`|B| A / sum_tau median`"],
        ["`sum_tau delta_tau` (R1.5)", f"{tt.step_sum:,.0f}",
         f"week {inputs.step_week} `err_prev_week`, `{inputs.week_errors_csv}`"],
        ["week-{} total".format(inputs.baseline_week), f"{tt.week4_total:,.0f}",
         f"`{inputs.week_errors_csv}`"],
    ]
    return _table(["quantity", "value", "source"], body)


def notes(inputs: Inputs, frame: pd.DataFrame) -> str:
    """The three things a reader of the tables above has to be told, derived from the run."""
    cal = inputs.cal
    unsupported = sorted({float(r) for r in frame[~frame["h2_holds"]]["r"]})
    saturated = sorted({float(d) for d in cal.d_min if d >= min(ell_of(s) for s in cal.sigma2)})
    ref_b, ref_r = 1.0, 10.0
    ref_ell = ell_of(0.70) if 0.70 in cal.sigma2 else ell_of(cal.sigma2[0])
    sketch = (f"{lower_over_qmax(ref_b, ref_r, ref_ell, ref_ell):.2f}, "
              f"{upper_over_qmax(ref_b, ref_ell, ref_ell):.2f}")
    here = (f"{lower_over_qmax(cal.beta_over_c, ref_r, ref_ell, ref_ell):.2f}, "
            f"{upper_over_qmax(cal.beta_over_c, ref_ell, ref_ell):.2f}")
    return "\n\n".join([
        "### Reading the numbers",
        f"* **`beta/c` is what moved against the sketch.** Section 2 of the paper's analysis quotes the "
        f"headline interval `[{sketch}]` for `r = {ref_r:g}`, `m = ell` and `beta/c = 1`. This "
        f"calibration has `beta/c = {cal.beta_over_c:g}`, which gives `[{here}]` for the same "
        f"cell: a start costs twice as many replica-minutes here, and the lower bound pays for it "
        f"while the upper bound does not move (at `m = ell` it is 1 by construction). Running "
        f"this script with `--beta {cal.c:g}` reproduces the sketch's numbers exactly, which is "
        "the check `tests/test_theory_instantiation.py` makes.",
        f"* **The bound saturates in `d`.** `m = min(ell, d)`, so every tier with "
        f"`d >= ell` ({', '.join(f'{d:g}' for d in saturated)} min here) gets the same interval: "
        "lemma 1 cannot tell the reserved tier from the instance tier, because what makes the "
        "reserved tier worse is the *commitment block*, which the lemma does not model (remark "
        "(b) of section 2). Only the simulator separates them.",
        (f"* **Scan points without lemma H support.** `r = "
         f"{', '.join(f'{r:g}' for r in unsupported)}` fail H2 at some synchrony level in the "
         "table above, so the R2 / R1.5 upper bounds are not supported there and the figure draws "
         "those points open. The R1 lower bound needs only A1 and A2 and survives."
         if unsupported else
         "* **Every scan point satisfies H2** at every synchrony level, so the R2 / R1.5 upper "
         "bounds are supported throughout."),
    ])


def markdown(inputs: Inputs, frame: pd.DataFrame, figure_dir: Path, csv_name: str) -> str:
    """The whole ``instantiation.md``."""
    cal = inputs.cal
    banner = ("> **PLACEHOLDER calibration.** `c`, `beta` and `mu` are the plausible A100 numbers of\n"
              "> the paper's analysis, not measurements; `p_in` is a dual variable and is reported as\n"
              "> the scan `r` throughout. Re-run this script with the batch B (T3) calibration and\n"
              "> `--calibrated` to replace every absolute USD figure below. The dimensionless\n"
              "> columns depend only on `r`, `beta/c` and `ell`, so they survive the replacement\n"
              "> unless `beta/c` moves.\n" if cal.placeholder else
              f"> Calibration source: {inputs.source}.\n")
    qmax = frame.groupby("sigma2")[["q_max_usd", "q_typ_usd"]].first()
    scale = _table(["sigma_2", "Q_max (USD/week)", "Q_typ (USD/week)"],
                   [[f"{s:.2f}", f"{row['q_max_usd']:.1f}", f"{row['q_typ_usd']:.1f}"]
                    for s, row in qmax.iterrows()])
    return "\n\n".join([
        "# Numerical instantiation of the theory",
        f"Generated {date.today().isoformat()} by `scripts/theory/instantiate.py`; every "
        f"number below is a row of [`{csv_name}`](./{csv_name}).",
        banner.rstrip(),
        "## Inputs", input_table(inputs),
        "## Normalisation\n\n`Q_max = (c ell + beta) |B| A / mu` is the calendar-attributable cost "
        "of serving every bell at full amplitude, with `A = n rho_window sigma_2 / 2` requests per "
        "minute for `n` seat-starts (`rho_window` in the inputs table); `Q_typ = Q_max / P2M_bell` "
        "is the typical week. Both are fixed reference quantities, not properties of the "
        "adversary's instance.",
        scale,
        "## Lemma 1 (calendar-obliviousness penalty)\n\nThe interval is "
        "`[lower, upper]` of the *excess over OPT* that the R1 family forces on any deterministic "
        "calendar-oblivious algorithm, as a fraction of the reference cost. It is not "
        "`Excess/OPT`: the total cost also carries the background, which dilutes the ratio.",
        lemma1_tables(frame, cal),
        notes(inputs, frame),
        "## Premises", premise_table(frame, cal),
        "A1 and H2 are conditions on `r` alone at a fixed `beta/c` and `ell`; H2 implies A1 and is "
        "the binding one at the graded ratio "
        f"`p_off/p_in = {cal.p_off_ratio:.4g}`. The feasibility figure below draws all three over "
        "the `(beta/c, r)` plane.",
        regime_table(frame),
        "The R2 and R1.5 numbers are upper bounds on the *capacity* value of the timetable in "
        "those regimes, i.e. what a calendar can still be worth once history already carries the "
        "repeating profile. They hold where lemma H does, so read them together with the H2 column "
        "above.",
        "## Lemma A", lemma_a_table(frame, cal),
        "## Figure\n\n"
        f"![Feasibility of A1, A2 and H2](./{figure_dir.name}/{FIGURE_NAME}.png)\n\n"
        f"`{figure_dir.name}/{FIGURE_NAME}.{{svg,pdf,png}}` - section 7(b): the `(beta/c, r)` "
        "plane at each synchrony level, the three premise curves and the shaded region where all "
        "of them hold, with the calibration's `beta/c` and the `r` scan marked.",
        "## Reproduce\n\n```bash\nuv run python scripts/theory/instantiate.py\n```\n",
    ])


__all__ = ["input_table", "lemma1_tables", "lemma_a_table", "markdown", "notes", "premise_table",
           "regime_table"]
