"""Numerical instantiation of the paper's analysis (scripts/theory/).

The load-bearing checks are the ones that hold the script against the sketch's own arithmetic, so
that a formula transcribed wrongly fails here rather than in a paper table:

* the worked example of section 2 - ``sigma_2 = 0.70`` (``ell = 2.857`` min), ``r = 10``,
  ``beta/c = 1``, ``m = ell`` - has to come out at **0.64 / 1.00** relative to ``Q_max`` and
  **1.06 / 1.65** relative to ``Q_typ`` (``P2M_bell = 1.654``), within 1%;
* the H2 threshold of section 7(b) has to be ``6 (1 + (beta/c)/ell)``, i.e. ``r >= 8`` at
  ``beta/c = 1`` and ``ell = 3`` slots;
* the R2 and R1.5 percentages have to reproduce 6.6% / 42.7% / 29.7% at ``m = ell``, which is where
  the upper-bound factor is 1 and the numbers are pure timetable ratios.

The rest covers the shape of the outputs: the CSV, the markdown and the three figure formats.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "theory"
DISPERSION = REPO / "results" / "t2" / "between_week_dispersion.csv"
WEEK_ERRORS = REPO / "results" / "t2" / "week_to_week_error.csv"

# the paper's analysis, the worked example and the timetable quantities behind it
SKETCH_ELL = 2.857  # 2 / 0.70, as the sketch writes it
SKETCH_LOWER_QMAX = 0.64
SKETCH_UPPER_QMAX = 1.00
SKETCH_P2M = 1.654
SKETCH_LOWER_QTYP = 1.06
SKETCH_UPPER_QTYP = 1.65
SKETCH_R2_PCT = 6.6
SKETCH_R15_PCT_WEEK4 = 42.7
SKETCH_R15_PCT_TYP = 29.7
TOL = 0.01  # the sketch quotes two decimals; 1% relative is the contract


def _load(name: str) -> ModuleType:
    """Import ``scripts/theory/<name>.py``; the modules are CLIs, not an installed package."""
    key = f"_theory_{name}"
    spec = importlib.util.spec_from_file_location(key, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module  # dataclass annotations are strings (``from __future__``)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lemmas() -> ModuleType:
    """The closed forms."""
    return _load("lemmas")


@pytest.fixture(scope="module")
def instantiate() -> ModuleType:
    """The CLI that turns the closed forms into the CSV, the markdown and the figure."""
    return _load("instantiate")


@pytest.fixture(scope="module")
def timetable(instantiate: ModuleType):
    """The timetable quantities read off the two tracked T2 tables."""
    return instantiate.load_timetable(DISPERSION, WEEK_ERRORS)


# ------------------------------------------------------------------ against the sketch
def test_worked_example_reproduces_the_sketch(lemmas: ModuleType) -> None:
    """sigma_2 = 0.70, r = 10, beta/c = 1, m = ell: [0.64, 1.00] of Q_max, [1.06, 1.65] of Q_typ."""
    lower = lemmas.lower_over_qmax(1.0, 10.0, SKETCH_ELL, SKETCH_ELL)
    upper = lemmas.upper_over_qmax(1.0, SKETCH_ELL, SKETCH_ELL)
    assert lower == pytest.approx(SKETCH_LOWER_QMAX, rel=TOL)
    assert upper == pytest.approx(SKETCH_UPPER_QMAX, rel=TOL)
    assert lower * SKETCH_P2M == pytest.approx(SKETCH_LOWER_QTYP, rel=TOL)
    assert upper * SKETCH_P2M == pytest.approx(SKETCH_UPPER_QTYP, rel=TOL)
    # the same cell through the exact ell = 2 / sigma_2 rather than the sketch's rounding
    exact = lemmas.ell_of(0.70)
    assert lemmas.lower_over_qmax(1.0, 10.0, exact, exact) == pytest.approx(SKETCH_LOWER_QMAX,
                                                                           rel=TOL)


def test_h2_threshold_is_six_times_one_plus_beta_over_c_over_ell(lemmas: ModuleType) -> None:
    """H2's minimal r is ``6 (1 + (beta/c)/ell)``: 8 at beta/c = 1 and ell = 3, 26 at beta/c = 10."""
    assert lemmas.h2_min_r(1.0, 3.0) == pytest.approx(8.0, rel=TOL)
    assert lemmas.h2_min_r(10.0, 3.0) == pytest.approx(26.0, rel=TOL)
    for beta_over_c in (0.5, 1.0, 2.0, 10.0):
        for ell in (2.222, 3.0, 15.385):
            expected = 6.0 * (1.0 + beta_over_c / ell)
            assert lemmas.h2_min_r(beta_over_c, ell) == pytest.approx(expected)
    # the threshold is exactly where the premise flips
    cal = lemmas.Calibration(c=0.05, beta=0.05, sigma2=(0.70,))
    r_star = lemmas.h2_min_r(cal.beta_over_c, 3.0, cal.p_off_ratio)
    assert lemmas.h2_holds(cal, r_star, 3.0)
    assert not lemmas.h2_holds(cal, r_star * 0.99, 3.0)


def test_timetable_quantities_match_the_sketch(timetable, lemmas: ModuleType) -> None:
    """|B|, A, the two sums, P2M_bell and the R1.5 step are the numbers the paper's analysis locks."""
    assert timetable.bells == 25
    assert timetable.a_max == pytest.approx(18481.0)
    assert timetable.sum_median == pytest.approx(279276.0)
    assert timetable.sum_range == pytest.approx(18477.0)
    assert timetable.step_sum == pytest.approx(82926.0)
    assert timetable.week4_total == pytest.approx(194376.0)
    assert timetable.p2m_bell == pytest.approx(SKETCH_P2M, rel=TOL)
    assert 100.0 * timetable.r2_share == pytest.approx(SKETCH_R2_PCT, rel=TOL)
    assert 100.0 * timetable.r15_share_week4 == pytest.approx(SKETCH_R15_PCT_WEEK4, rel=TOL)
    assert 100.0 * timetable.r15_share_typ == pytest.approx(SKETCH_R15_PCT_TYP, rel=TOL)


def test_regime_percentages_at_m_equal_ell(instantiate: ModuleType, lemmas: ModuleType,
                                           timetable) -> None:
    """At m = ell the upper factor is 1, so R2 / R1.5 are the raw timetable ratios."""
    cal = lemmas.Calibration(sigma2=(0.70,), d_min=(8.0,), r=(10.0,))
    frame = instantiate.rows(lemmas.Inputs(cal=cal, tt=timetable, source="test",
                                           dispersion_csv="x", week_errors_csv="y",
                                           step_week=5, baseline_week=4))
    row = frame.iloc[0]
    assert row["m_min"] == pytest.approx(row["ell_min"])
    assert row["upper_over_qmax"] == pytest.approx(1.0)
    assert row["r2_upper_pct_qtyp"] == pytest.approx(SKETCH_R2_PCT, rel=TOL)
    assert row["r15_upper_pct_week4"] == pytest.approx(SKETCH_R15_PCT_WEEK4, rel=TOL)
    assert row["r15_upper_pct_qtyp"] == pytest.approx(SKETCH_R15_PCT_TYP, rel=TOL)


def test_normalisation_and_absolute_bound_agree(lemmas: ModuleType, timetable) -> None:
    """The per-bell USD bound divided by Q_max/|B| is the dimensionless lower bound."""
    cal = lemmas.Calibration()
    ell, r, d = lemmas.ell_of(0.70), 30.0, 8.0
    per_bell = lemmas.lemma1_lower_bell(cal, r, ell, d, timetable.a_max)
    qmax = lemmas.q_max(cal, ell, timetable)
    assert per_bell * timetable.bells / qmax == pytest.approx(
        lemmas.lower_over_qmax(cal.beta_over_c, r, ell, d))
    assert lemmas.q_typ(cal, ell, timetable) == pytest.approx(qmax / timetable.p2m_bell)


def test_premise_flags_are_consistent(lemmas: ModuleType) -> None:
    """A1 is r > 6 at the graded ratio, H2 implies A1, and A2 needs only kappa m > beta."""
    cal = lemmas.Calibration()
    assert not lemmas.a1_holds(6.0, cal.p_off_ratio) and lemmas.a1_holds(6.1, cal.p_off_ratio)
    for r in (7.0, 10.0, 30.0, 100.0):
        for sigma2 in cal.sigma2:
            ell = lemmas.ell_of(sigma2)
            if lemmas.h2_holds(cal, r, ell):
                assert lemmas.a1_holds(r, cal.p_off_ratio)  # H2 is strictly stronger than A1
    assert lemmas.h1_holds(cal, lemmas.ell_of(0.13))  # s = 100 min, ell = 15.4, beta/c = 2
    assert lemmas.a2_holds(cal, 10.0, lemmas.ell_of(0.70), 8.0)
    assert not lemmas.a2_holds(cal, 1.5, lemmas.ell_of(0.70), 0.6)  # kappa m < beta
    assert lemmas.lemma_a_constant(cal, 10.0, 0.0) == pytest.approx(3.0)


# ------------------------------------------------------------------ outputs
def test_end_to_end_writes_csv_markdown_and_three_figure_formats(instantiate: ModuleType,
                                                                 tmp_path: Path) -> None:
    """One full run: 3 x 4 x 4 rows, the markdown tables and the SVG / PDF / PNG figure."""
    args = instantiate.build_parser().parse_args(["--root", str(REPO), "--out", str(tmp_path)])
    inputs = instantiate.inputs_from(args)
    frame = instantiate.run(inputs, tmp_path, figure=True)
    assert len(frame) == len(inputs.cal.sigma2) * len(inputs.cal.d_min) * len(inputs.cal.r)
    on_disk = pd.read_csv(tmp_path / "instantiation.csv")
    assert len(on_disk) == len(frame)
    for column in ("lower_over_qmax", "upper_over_qmax", "lower_over_qtyp", "upper_over_qtyp",
                   "a1_holds", "a2_holds", "h1_holds", "h2_holds", "h2_min_r",
                   "r2_upper_pct_qtyp", "r15_upper_pct_qtyp", "r15_upper_pct_week4",
                   "lemma_a_constant", "lower_bell_usd", "q_max_usd", "q_typ_usd"):
        assert column in on_disk.columns
    assert on_disk["upper_over_qmax"].max() == pytest.approx(1.0)  # saturates at m = ell
    text = (tmp_path / "instantiation.md").read_text(encoding="utf-8")
    assert "PLACEHOLDER calibration" in text  # the default inputs are not measurements
    for heading in ("## Inputs", "Lemma 1, normalised to `Q_max`", "## Premises", "## Lemma A",
                    "## Figure"):
        assert heading in text
    for ext in ("svg", "pdf", "png"):
        figure = tmp_path / "figures" / f"instantiation_feasibility.{ext}"
        assert figure.exists() and figure.stat().st_size > 0
    svg = (tmp_path / "figures" / "instantiation_feasibility.svg").read_text(encoding="utf-8")
    assert "<text" in svg  # editable text rather than outlined glyphs


def test_cli_overrides_reach_the_calibration(instantiate: ModuleType, tmp_path: Path) -> None:
    """The flags move the calibration, and --calibrated drops the placeholder banner."""
    args = instantiate.build_parser().parse_args(
        ["--root", str(REPO), "--out", str(tmp_path), "--beta", "0.05", "--r", "10",
         "--sigma2", "0.70", "--d", "8", "--calibrated", "--source", "unit test", "--no-figure"])
    inputs = instantiate.inputs_from(args)
    assert inputs.cal.beta_over_c == pytest.approx(1.0)
    assert not inputs.cal.placeholder and inputs.source == "unit test"
    frame = instantiate.run(inputs, tmp_path, figure=False)
    assert len(frame) == 1
    # with beta/c = 1 the run has to land on the sketch's worked example
    assert frame.iloc[0]["lower_over_qmax"] == pytest.approx(SKETCH_LOWER_QMAX, rel=TOL)
    assert frame.iloc[0]["lower_over_qtyp"] == pytest.approx(SKETCH_LOWER_QTYP, rel=TOL)
    text = (tmp_path / "instantiation.md").read_text(encoding="utf-8")
    assert "PLACEHOLDER calibration" not in text and "unit test" in text
