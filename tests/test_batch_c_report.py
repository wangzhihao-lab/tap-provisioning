"""The batch C report: base-tier arithmetic, the invariant checks, the tables and the two figures.

The fixtures build a frame in the schema ``pilot.sim.sweeps.run_sweep`` writes - the lead sweep's
four cell-key columns first, then ``sweep`` and the shared block, then the metrics - so that the
cell key is discovered from the header here exactly as it is on a real CSV.  Two cells, two seeds,
three policies and weeks 0 (whole run) / 1 / 2, with ``weeks = 2``.

The numbers are chosen so that every derived quantity can be written out by hand: the reference
cell's optimum is 400 USD over the run, and the second cell's optimum sits *exactly* on the base
tier, which is the case that has to come out as a NaN attributable ratio rather than an infinity.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pilot.batchc import (  # noqa: E402
    SLO_MATCHED_SUFFIX,
    FloorSpec,
    cell_key_columns,
    cell_table,
    derive,
    main_table,
    ratio_table,
    reference_cell,
    verification_row,
    violation_count,
    weekly_table,
)
from pilot.batchc.discover import SweepFile  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "sim"

MIN_REPLICAS = 1
C_RUN = 0.01
N_WEEKS = 2
WEEK_MINUTES = 10080
WHOLE_MINUTES = WEEK_MINUTES * N_WEEKS  # 20160
BASE_WHOLE_USD = MIN_REPLICAS * C_RUN * WHOLE_MINUTES  # 201.6
BASE_WEEK_USD = MIN_REPLICAS * C_RUN * WEEK_MINUTES  # 100.8
BASE_WHOLE_HOURS = MIN_REPLICAS * WHOLE_MINUTES / 60.0  # 336.0
BASE_WEEK_HOURS = MIN_REPLICAS * WEEK_MINUTES / 60.0  # 168.0

CELL_COLUMNS: Tuple[str, ...] = ("lead_tier", "lead_min", "commit_block_min", "billing_block_min")
REFERENCE_CELL: Tuple[object, ...] = ("d_rep", 0.5, 1.0, 1.0)
FLOOR_CELL: Tuple[object, ...] = ("d_inst", 8.0, 60.0, 60.0)
SEEDS: Tuple[int, ...] = (0, 1)
POLICIES: Tuple[str, ...] = ("offline", "reactive", "calendar")
# total_cost of the reference cell at seed 0: (whole run, week 1, week 2)
REF_TOTALS: Dict[str, Tuple[float, float, float]] = {"offline": (400.0, 180.0, 220.0),
                                                     "calendar": (500.0, 230.0, 270.0),
                                                     "reactive": (700.0, 330.0, 370.0)}
# the second cell's optimum is the base tier itself, so nothing is attributable to the scheduler
FLOOR_TOTALS: Dict[str, Tuple[float, float, float]] = {
    "offline": (BASE_WHOLE_USD, BASE_WEEK_USD, BASE_WEEK_USD),
    "calendar": (BASE_WHOLE_USD + 100.0, BASE_WEEK_USD + 50.0, BASE_WEEK_USD + 50.0),
    "reactive": (BASE_WHOLE_USD + 200.0, BASE_WEEK_USD + 100.0, BASE_WEEK_USD + 100.0)}
GPU_HOURS: Dict[str, Tuple[float, float, float]] = {"offline": (400.0, 200.0, 200.0),
                                                    "calendar": (500.0, 250.0, 250.0),
                                                    "reactive": (600.0, 300.0, 300.0)}
SEED_BUMP: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0), 1: (20.0, 10.0)}
SLO_IN: Dict[str, float] = {"offline": 0.94, "calendar": 0.96, "reactive": 0.80}
X_PEAK: Dict[str, float] = {"offline": 3.0, "calendar": 4.0, "reactive": 5.0}
STARTS: Dict[str, float] = {"offline": 30.0, "calendar": 40.0, "reactive": 90.0}
# the SLO-matched retuning of the two feedback baselines (``outputs/batchC/r30_slo95``)
MATCHED_POLICIES: Tuple[str, ...] = ("reactive", "velocity")
MATCHED_SLO_IN: Dict[str, float] = {"reactive": 0.96, "velocity": 0.97}
MATCHED_HOURS: Dict[str, float] = {"reactive": 700.0, "velocity": 800.0}
MATCHED_TOTAL: Dict[str, float] = {"reactive": 620.0, "velocity": 680.0}


def _row(cell: Tuple[object, ...], seed: int, policy: str, week: int, total: float, hours: float,
         offline_whole: float, offline_week: float) -> Dict[str, object]:
    """One row of the synthetic sweep, in the column order ``run_sweep`` writes."""
    tier, lead_min, commit, billing = cell
    whole = week == 0
    denominator = offline_whole if whole else offline_week
    return {
        "lead_tier": tier, "lead_min": lead_min, "commit_block_min": commit,
        "billing_block_min": billing,
        "sweep": "lead", "seed": seed, "trust": 0.1, "sigma": 0.7, "sigma2_measured": 0.68,
        "adoption": 1.0, "rho": 2.0, "plan_margin": 1.0, "college": "ALL", "weeks": N_WEEKS,
        "mu": 1000.0, "eta": 100.0, "eta_norm": 0.3, "offline_cost": offline_whole,
        "p_in": 0.001,
        "week": week, "policy": policy,
        "gpu_hours": hours, "starts": STARTS[policy], "run_cost": total * 0.6,
        "switch_cost": total * 0.1, "penalty_cost": total * 0.3, "total_cost": total,
        "cost_ratio": total / denominator, "unserved_in": 0.0, "unserved_off": 0.0,
        "x_mean": 1.5, "x_peak": X_PEAK[policy], "n_in": 1000, "n_off": 200,
        "slo_in": SLO_IN[policy], "viol_in": 1.0 - SLO_IN[policy],
        "ttft_p95_in_s": 1.2 if whole else float("nan"),
        "slo_off": 0.90 if whole else float("nan"),
    }


def synthetic_frame() -> pd.DataFrame:
    """2 cells x 2 seeds x 3 policies x weeks 0/1/2 in the real schema."""
    rows: List[Dict[str, object]] = []
    for cell, totals in ((REFERENCE_CELL, REF_TOTALS), (FLOOR_CELL, FLOOR_TOTALS)):
        for seed in SEEDS:
            bump_whole, bump_week = SEED_BUMP[seed]
            for policy in POLICIES:
                extra = (0.0, 0.0) if policy == "offline" else (bump_whole, bump_week)
                for week in (0, 1, 2):
                    total = totals[policy][week] + (extra[0] if week == 0 else extra[1])
                    rows.append(_row(cell, seed, policy, week, total,
                                     GPU_HOURS[policy][week], totals["offline"][0],
                                     totals["offline"][week] if week else totals["offline"][0]))
    return pd.DataFrame(rows)


def _matched_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """``r30_slo95/lead/lead.csv``: the two feedback baselines retuned to ``slo_in >= 0.95``.

    ``batch_c.sh`` runs that sweep with ``--policies reactive velocity``, so the file carries no
    offline rows at all - only the whole-run optimum of its own stream, in ``offline_cost``, which
    is the fallback ``derive`` has to use for the whole-run rows.
    """
    source = frame[frame["policy"] == "reactive"]
    parts: List[pd.DataFrame] = []
    for policy in MATCHED_POLICIES:
        part = source.copy()
        whole = part["week"] == 0
        part["policy"] = policy
        part["slo_in"] = MATCHED_SLO_IN[policy]
        part["viol_in"] = 1.0 - MATCHED_SLO_IN[policy]
        part["gpu_hours"] = np.where(whole, MATCHED_HOURS[policy],
                                     MATCHED_HOURS[policy] / N_WEEKS)
        part["total_cost"] = np.where(whole, MATCHED_TOTAL[policy],
                                      MATCHED_TOTAL[policy] / N_WEEKS)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture()
def frame() -> pd.DataFrame:
    """The synthetic sweep frame."""
    return synthetic_frame()


@pytest.fixture()
def derived(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frame with the base-tier columns of ``derive``."""
    return derive(frame, MIN_REPLICAS, C_RUN)


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/sim/<name>.py`` as a module; they are CLIs, not an installed package."""
    spec = importlib.util.spec_from_file_location(f"_sim_script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick(frame: pd.DataFrame, policy: str, week: int, seed: int = 0,
          cell: Tuple[object, ...] = REFERENCE_CELL) -> pd.Series:
    """One row of a frame, by cell, seed, policy and week."""
    sel = frame[(frame["lead_tier"] == cell[0]) & (frame["seed"] == seed)
                & (frame["policy"] == policy) & (frame["week"] == week)]
    assert len(sel) == 1, f"{policy} week {week} seed {seed}: {len(sel)} rows"
    return sel.iloc[0]


# --------------------------------------------------------------------------- the cell key
def test_cell_key_is_the_columns_before_sweep(frame: pd.DataFrame) -> None:
    """The lead sweep's four grid parameters, read off the header rather than hard-coded."""
    assert cell_key_columns(frame) == list(CELL_COLUMNS)
    assert frame[list(CELL_COLUMNS)].drop_duplicates().shape[0] == 2


def test_cell_key_of_the_real_sweeps_matches_the_header() -> None:
    """The same rule on the CSVs on disk: params first, then ``sweep`` (skipped when absent)."""
    expected = {"lead": ["lead_tier", "lead_min", "commit_block_min", "billing_block_min"],
                "sigma": ["sigma"],
                "epsilon": ["error_knob", "error_value", "forecast_scale", "bell_shift_frac",
                            "p_cancel", "price_ratio"]}
    seen = 0
    for sweep, columns in expected.items():
        path = REPO / "outputs" / "sim" / sweep / f"{sweep}.csv"
        if not path.exists():
            continue
        seen += 1
        assert cell_key_columns(pd.read_csv(path, nrows=5)) == columns
    if not seen:
        pytest.skip("needs a sweep CSV under outputs/sim/ (run output, not distributed; create it with scripts/sim/run_sweep.py)")


def test_reference_cell_is_the_smallest_lead_min(frame: pd.DataFrame) -> None:
    """The replica tier, and the rows it selects."""
    cell = reference_cell(frame)
    assert cell.columns == CELL_COLUMNS
    assert cell.values == REFERENCE_CELL
    assert "lead_min = 0.5" in cell.description
    rows = cell.rows(frame)
    assert set(rows["lead_tier"]) == {"d_rep"}
    assert len(rows) == len(SEEDS) * len(POLICIES) * 3


def test_reference_cell_of_a_sigma_sweep_is_the_largest_sigma(frame: pd.DataFrame) -> None:
    """A sweep whose cell key is ``sigma`` is reported at its most synchronous cell.

    ``run_sweep`` writes the knob of the sigma sweep as the *first* column and repeats it inside
    the shared block, so the cell key of that CSV is a single ``sigma``: the frame is rebuilt here
    with the four lead columns and the shared copy dropped, and the grid value put in front.
    """
    synchrony = np.where(frame["lead_min"] == REFERENCE_CELL[1], 0.2, 0.7)
    sigma_like = frame.drop(columns=[*CELL_COLUMNS, "sigma"])
    sigma_like.insert(0, "sigma", synchrony)
    assert cell_key_columns(sigma_like) == ["sigma"]
    cell = reference_cell(sigma_like)
    assert cell.columns == ("sigma",)
    assert cell.values == (0.7,)
    assert "largest sigma" in cell.rule
    assert set(cell.rows(sigma_like)["sigma"]) == {0.7}


# --------------------------------------------------------------------------- derive
def test_derive_matches_the_hand_computed_base_tier(derived: pd.DataFrame) -> None:
    """Whole-run and weekly rows of the reference cell, against numbers written out by hand."""
    whole = _pick(derived, "calendar", 0)
    assert whole["run_minutes"] == pytest.approx(20160.0)
    assert whole["base_usd"] == pytest.approx(201.6)
    assert whole["base_hours"] == pytest.approx(336.0)
    assert whole["offline_total"] == pytest.approx(400.0)
    assert whole["excess_usd"] == pytest.approx(500.0 - 400.0)
    assert whole["attr_cost"] == pytest.approx(500.0 - 201.6)
    assert whole["attr_ratio"] == pytest.approx((500.0 - 201.6) / (400.0 - 201.6))
    assert whole["attr_gpu_hours"] == pytest.approx(500.0 - 336.0)

    week1 = _pick(derived, "calendar", 1)
    assert week1["run_minutes"] == pytest.approx(10080.0)
    assert week1["base_usd"] == pytest.approx(100.8)
    assert week1["base_hours"] == pytest.approx(168.0)
    assert week1["offline_total"] == pytest.approx(180.0)  # the oracle's *week 1*, not its run
    assert week1["excess_usd"] == pytest.approx(230.0 - 180.0)
    assert week1["attr_ratio"] == pytest.approx((230.0 - 100.8) / (180.0 - 100.8))
    assert week1["attr_gpu_hours"] == pytest.approx(250.0 - 168.0)


def test_derive_uses_the_weekly_oracle_not_the_whole_run_offline_cost(derived: pd.DataFrame
                                                                     ) -> None:
    """``offline_cost`` is a whole-run column and would be the wrong weekly denominator."""
    week2 = _pick(derived, "reactive", 2, seed=1)
    assert week2["offline_cost"] == pytest.approx(400.0)
    assert week2["offline_total"] == pytest.approx(220.0)
    assert week2["excess_usd"] == pytest.approx(380.0 - 220.0)


def test_attr_ratio_is_nan_when_the_optimum_sits_on_the_base_tier(derived: pd.DataFrame) -> None:
    """A denominator at the floor leaves the attributable ratio undefined, not infinite."""
    on_floor = derived[derived["lead_tier"] == FLOOR_CELL[0]]
    assert on_floor["offline_total"].eq(BASE_WHOLE_USD).any()
    whole = _pick(derived, "calendar", 0, cell=FLOOR_CELL)
    assert whole["offline_total"] == pytest.approx(BASE_WHOLE_USD)
    assert whole["attr_cost"] == pytest.approx(100.0)
    assert np.isnan(whole["attr_ratio"])
    assert whole["excess_usd"] == pytest.approx(100.0)  # excess survives the floor
    assert on_floor["attr_ratio"].isna().all()


def test_derive_is_pure(frame: pd.DataFrame) -> None:
    """The input frame keeps its columns and its values."""
    before = frame.copy()
    out = derive(frame, MIN_REPLICAS, C_RUN)
    pd.testing.assert_frame_equal(frame, before)
    assert "attr_ratio" in out.columns and "attr_ratio" not in frame.columns


def test_derive_without_a_floor_leaves_the_raw_costs(frame: pd.DataFrame) -> None:
    """``--min-replicas 0``: the attributable cost is the cost, and the ratio is ``cost_ratio``."""
    out = derive(frame, 0, C_RUN)
    row = _pick(out, "calendar", 0)
    assert row["base_usd"] == 0.0 and row["base_hours"] == 0.0
    assert row["attr_cost"] == pytest.approx(row["total_cost"])
    assert row["attr_ratio"] == pytest.approx(row["cost_ratio"])
    assert row["attr_gpu_hours"] == pytest.approx(row["gpu_hours"])


# --------------------------------------------------------------------------- verification
def _verify(frame: pd.DataFrame) -> Dict[str, object]:
    """The verification row of a frame, as the report builds it."""
    source = SweepFile("r30", "lead", Path("r30/lead/lead.csv"))
    return verification_row(source, derive(frame, MIN_REPLICAS, C_RUN))


def test_verification_passes_on_a_clean_frame(frame: pd.DataFrame) -> None:
    """Shape, coverage and both invariants on a frame that respects them."""
    row = _verify(frame)
    assert row["n_rows"] == len(frame)
    assert row["n_cells"] == 2
    assert row["seeds"] == "0,1"
    assert row["policies"] == "calendar,offline,reactive"
    assert row["nan_total_cost"] == 0 and row["nan_gpu_hours"] == 0 and row["nan_slo_in"] == 0
    assert row["dominance_violations"] == 0
    assert row["floor_violations"] == 0
    assert row["x_mean_min"] == pytest.approx(1.5)
    assert "calendar 4..4" in str(row["x_peak_range_by_policy"])
    assert row["sigma2_measured_range"] == "0.68..0.68"
    assert violation_count(pd.DataFrame([row])) == 0


def test_verification_flags_a_planted_dominance_violation(frame: pd.DataFrame) -> None:
    """A whole-run row cheaper than its own optimum is impossible and has to be counted."""
    planted = frame.copy()
    mask = ((planted["policy"] == "calendar") & (planted["week"] == 0)
            & (planted["seed"] == 0) & (planted["lead_tier"] == "d_rep"))
    planted.loc[mask, "total_cost"] = 399.0  # the oracle of that cell paid 400
    row = _verify(planted)
    assert row["dominance_violations"] == 1
    assert row["floor_violations"] == 0
    assert violation_count(pd.DataFrame([row])) == 1


def test_verification_flags_a_planted_floor_violation(frame: pd.DataFrame) -> None:
    """A trajectory below the always-on base tier is impossible and has to be counted."""
    planted = frame.copy()
    mask = (planted["policy"] == "reactive") & (planted["week"] == 1)
    planted.loc[mask, "gpu_hours"] = BASE_WEEK_HOURS - 1.0
    row = _verify(planted)
    assert row["floor_violations"] == int(mask.sum()) == 4
    assert row["dominance_violations"] == 0
    assert violation_count(pd.DataFrame([row])) == 4


# --------------------------------------------------------------------------- tables
def test_main_table_rows_and_means(derived: pd.DataFrame) -> None:
    """One row per policy, ordered by the excess over the optimum, with the sd over the seeds."""
    rows = reference_cell(derived).rows(derived)
    table = main_table(rows[rows["week"] == 0])
    assert list(table["policy"]) == ["offline", "calendar", "reactive"]
    assert len(table) == len(POLICIES)
    calendar = table[table["policy"] == "calendar"].iloc[0]
    assert calendar["total_cost_mean"] == pytest.approx((500.0 + 520.0) / 2.0)
    assert calendar["excess_usd_mean"] == pytest.approx((100.0 + 120.0) / 2.0)
    assert calendar["excess_usd_sd"] == pytest.approx(20.0 / np.sqrt(2.0))
    assert calendar["attr_ratio_mean"] == pytest.approx(
        ((500.0 - 201.6) / 198.4 + (520.0 - 201.6) / 198.4) / 2.0)
    assert calendar["attr_gpu_hours_mean"] == pytest.approx(500.0 - 336.0)
    assert calendar["slo_in_mean"] == pytest.approx(0.96)
    assert calendar["x_peak_mean"] == pytest.approx(4.0)
    offline = table[table["policy"] == "offline"].iloc[0]
    assert offline["excess_usd_mean"] == pytest.approx(0.0)
    assert offline["attr_ratio_mean"] == pytest.approx(1.0)


def test_weekly_table_has_a_regime_column_and_weekly_means(derived: pd.DataFrame) -> None:
    """Policies x weeks 1..2, week 1 in regime R1 and week 2 in R2."""
    rows = reference_cell(derived).rows(derived)
    table = weekly_table(rows)
    assert len(table) == len(POLICIES) * 2
    assert set(table["week"]) == {1, 2}
    assert dict(zip(table["week"], table["regime"]))[1] == "R1"
    assert dict(zip(table["week"], table["regime"]))[2] == "R2"
    week1 = table[(table["policy"] == "calendar") & (table["week"] == 1)].iloc[0]
    assert week1["excess_usd"] == pytest.approx(((230.0 - 180.0) + (240.0 - 180.0)) / 2.0)
    assert week1["attr_ratio"] == pytest.approx(
        ((230.0 - 100.8) / 79.2 + (240.0 - 100.8) / 79.2) / 2.0)
    assert week1["attr_gpu_hours"] == pytest.approx(250.0 - 168.0)
    assert "week" in table.columns and 0 not in set(table["week"])


def test_ratio_table_has_one_block_per_metric(derived: pd.DataFrame) -> None:
    """Three metric blocks, one column per price ratio, policies as rows."""
    rows = reference_cell(derived).rows(derived)
    whole = rows[rows["week"] == 0]
    table = ratio_table({6: whole, 30: whole}, ratios=(6, 10, 30))
    assert list(table.columns) == ["metric", "policy", "r6", "r30"]  # r10 never landed
    assert list(dict.fromkeys(table["metric"])) == ["excess_usd", "attr_ratio", "slo_in"]
    assert len(table) == 3 * len(POLICIES)
    excess = table[(table["metric"] == "excess_usd") & (table["policy"] == "reactive")].iloc[0]
    assert excess["r6"] == pytest.approx((300.0 + 320.0) / 2.0)
    assert excess["r30"] == pytest.approx(excess["r6"])


def test_cell_table_keeps_every_cell_key_column(derived: pd.DataFrame) -> None:
    """The per-cell tables (adoption, margin, trust, epsilon) report a row per cell and policy."""
    table = cell_table(derived[derived["week"] == 0])
    assert list(table.columns)[:5] == [*CELL_COLUMNS, "policy"]
    assert len(table) == 2 * len(POLICIES)
    assert set(table["lead_tier"]) == {"d_rep", "d_inst"}


# --------------------------------------------------------------------------- the CLI
def _write_sweep(root: Path, group: str, sweep: str, frame: pd.DataFrame) -> Path:
    """Write a frame where ``run_sweep.py --out <root>/<group>`` would have put it."""
    path = root / group / sweep / f"{sweep}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _run_report(root: Path, out: Path, extra: Optional[List[str]] = None) -> int:
    """Run the report CLI on ``root``."""
    cli = _load_script("batch_c_report")
    argv = ["--root", str(root), "--out", str(out), "--min-replicas", str(MIN_REPLICAS),
            "--c-run", str(C_RUN), "--ratios", "30", *(extra or [])]
    return int(cli.main(argv))


def test_report_writes_every_table_with_a_provenance_header(tmp_path: Path,
                                                            frame: pd.DataFrame) -> None:
    """The CLI writes CSV and Markdown, and the Markdown says where the numbers came from."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    source = _write_sweep(root, "r30", "lead", frame)
    assert _run_report(root, out) == 0
    for name in ("verification", "main_r30", "weekly_r30", "r_sweep", "tuning"):
        assert (out / f"{name}.csv").stat().st_size > 0, name
        assert (out / f"{name}.md").stat().st_size > 0, name
    text = (out / "main_r30.md").read_text(encoding="utf-8")
    assert text.startswith("# Batch C main table, r = 30")
    assert str(source) in text
    assert "`M = 1` replica(s)" in text and "`c_run = 0.01` USD per replica-minute" in text
    assert "minutes = 10080 * weeks" in text
    assert "Generated: " in text
    assert "Numbers are means over seeds, with the standard deviation in parentheses" in text
    assert "lead_min = 0.5" in text
    assert "| policy | total_cost | excess_usd | attr_ratio |" in text
    weekly = (out / "weekly_r30.md").read_text(encoding="utf-8")
    assert "| policy | w1* | w2 |" in weekly  # the R1 week is starred in the header
    assert "R1 weeks are starred" in weekly
    # the cells the batch had not written are skipped, not invented
    for absent in ("main_r6", "epsilon", "adoption_r30", "sensitivity_32b"):
        assert not (out / f"{absent}.csv").exists(), absent


def test_report_falls_back_to_the_sigma_sweep(tmp_path: Path, frame: pd.DataFrame) -> None:
    """Without a lead sweep the main table uses the sigma sweep and says so in the header."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    _write_sweep(root, "r30", "sigma", frame)
    assert _run_report(root, out) == 0
    text = (out / "main_r30.md").read_text(encoding="utf-8")
    assert "falls back to the `sigma` sweep" in text


def test_missing_inputs_are_skipped_with_exit_code_zero(tmp_path: Path) -> None:
    """A root the batch has not filled in yet is a warning, not a crash."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    root.mkdir(parents=True)
    assert _run_report(root, out) == 0
    assert (out / "verification.md").exists()
    assert pd.read_csv(out / "verification.csv").empty if (
        out / "verification.csv").stat().st_size > 1 else True
    assert not (out / "main_r30.csv").exists()


def test_report_exits_one_on_a_planted_violation(tmp_path: Path, frame: pd.DataFrame) -> None:
    """A run that beats its own optimum fails the report rather than being tabulated quietly."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    planted = frame.copy()
    mask = (planted["policy"] == "calendar") & (planted["week"] == 0)
    planted.loc[mask, "total_cost"] = 1.0
    _write_sweep(root, "r30", "lead", planted)
    assert _run_report(root, out) == 1
    table = pd.read_csv(out / "verification.csv")
    assert int(table["dominance_violations"].sum()) == int(mask.sum())


def test_report_reads_the_secondary_sweeps_and_the_tuning(tmp_path: Path,
                                                          frame: pd.DataFrame) -> None:
    """adoption / margin / trust / pool / epsilon and the 32B sensitivity land when the CSVs exist."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    _write_sweep(root, "r30", "lead", frame)
    for sweep in ("adoption", "margin", "trust", "pool"):
        _write_sweep(root, "r30", sweep, frame)
    _write_sweep(root, "eps", "epsilon", frame)
    _write_sweep(root, "r30_32b", "lead", frame)
    assert _run_report(root, out) == 0
    for name in ("adoption_r30", "margin_r30", "trust_r30", "pool_r30", "epsilon",
                 "sensitivity_32b"):
        assert (out / f"{name}.csv").stat().st_size > 0, name
        assert (out / f"{name}.md").stat().st_size > 0, name
    verification = pd.read_csv(out / "verification.csv")
    assert sorted(verification["file"]) == ["eps/epsilon", "r30/adoption", "r30/lead",
                                            "r30/margin", "r30/pool", "r30/trust",
                                            "r30_32b/lead"]


def test_the_pool_table_reports_every_cell_with_the_total_gpu_hours(tmp_path: Path,
                                                                    frame: pd.DataFrame) -> None:
    """The offset-pool sweep gets its own table, with total GPU-hours beside the cost columns."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    _write_sweep(root, "r30", "lead", frame)
    _write_sweep(root, "r30", "pool", frame)
    assert _run_report(root, out) == 0
    table = pd.read_csv(out / "pool_r30.csv")
    assert list(table.columns) == [*CELL_COLUMNS, "policy", "excess_usd", "attr_ratio", "slo_in",
                                   "gpu_hours"]
    assert len(table) == 2 * len(POLICIES)  # both cells of the fixture, every policy
    row = table[(table["lead_tier"] == REFERENCE_CELL[0])
                & (table["policy"] == "calendar")].iloc[0]
    assert row["gpu_hours"] == pytest.approx(GPU_HOURS["calendar"][0])
    assert row["excess_usd"] == pytest.approx((100.0 + 120.0) / 2.0)
    assert (out / "pool_r30.md").read_text(encoding="utf-8").startswith(
        "# Batch C offset-pool sweep at r = 30")


def test_main_r30_appends_the_slo_matched_baselines(tmp_path: Path,
                                                    frame: pd.DataFrame) -> None:
    """``r30_slo95/lead/lead.csv`` adds two labelled rows to the r = 30 main table."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    _write_sweep(root, "r30", "lead", frame)
    _write_sweep(root, "r30_slo95", "lead", _matched_frame(frame))
    assert _run_report(root, out) == 0
    table = pd.read_csv(out / "main_r30.csv")
    labels = [f"{p}{SLO_MATCHED_SUFFIX}" for p in MATCHED_POLICIES]
    assert list(table["policy"])[-2:] == labels
    assert len(table) == len(POLICIES) + len(MATCHED_POLICIES)
    row = table[table["policy"] == labels[0]].iloc[0]
    assert row["slo_in_mean"] == pytest.approx(MATCHED_SLO_IN["reactive"])
    assert row["gpu_hours_mean"] == pytest.approx(MATCHED_HOURS["reactive"])
    # no offline rows in that file: the whole-run optimum comes from the ``offline_cost`` column
    assert row["excess_usd_mean"] == pytest.approx(MATCHED_TOTAL["reactive"] - 400.0)
    assert row["attr_ratio_mean"] == pytest.approx(
        (MATCHED_TOTAL["reactive"] - BASE_WHOLE_USD) / (400.0 - BASE_WHOLE_USD))
    text = (out / "main_r30.md").read_text(encoding="utf-8")
    assert "r30_slo95/lead/lead.csv" in text and "slo_in >= 0.95" in text
    # the price-ratio table keeps comparing the freely tuned policies only
    assert not any(str(p).endswith(SLO_MATCHED_SUFFIX)
                   for p in pd.read_csv(out / "r_sweep.csv")["policy"])


def test_the_tuning_table_separates_the_slo_matched_selection(tmp_path: Path) -> None:
    """``tuning_r30`` and ``tuning_r30_slo95`` are two rows per policy, told apart by the floor."""
    root, out = tmp_path / "batchC", tmp_path / "report"
    for folder, floor in (("tuning_r30", "null"), ("tuning_r30_slo95", "0.95")):
        path = root / folder
        path.mkdir(parents=True)
        (path / "best.json").write_text(
            '{"slice": {"price_ratio": 30, "slo_floor": ' + floor + '}, "policies": '
            '{"reactive": {"config": {"hpa_up_util": 0.2}, "grid_size": 54, '
            '"tuning": {"cost_ratio_mean": 2.5}, "heldout": {"cost_ratio": 2.1, "slo_in": 0.9}}}}',
            encoding="utf-8")
    assert _run_report(root, out) == 0
    table = pd.read_csv(out / "tuning.csv")
    assert list(table["slice"]) == ["r30", "r30_slo95"]
    assert np.isnan(table.iloc[0]["slo_floor"])
    assert table.iloc[1]["slo_floor"] == pytest.approx(0.95)
    text = (out / "tuning.md").read_text(encoding="utf-8")
    assert "r30_slo95" in text and "0.950" in text


def test_tuning_table_counts_the_distinct_grid_scores(tmp_path: Path) -> None:
    """``n_distinct_scores`` exposes a grid that scored every configuration the same."""
    from pilot.batchc import read_tuning, tuning_table

    folder = tmp_path / "tuning_r30"
    folder.mkdir(parents=True)
    (folder / "best.json").write_text(
        '{"policies": {"reactive": {"config": {"hpa_up_util": 0.2}, "default": '
        '{"hpa_up_util": 0.7}, "rank_of_default": 29, "grid_size": 54, '
        '"tuning": {"cost_ratio_mean": 2.5}, "default_tuning": {"cost_ratio_mean": 3.0}, '
        '"heldout": {"cost_ratio": 2.1, "slo_in": 0.9}, '
        '"default_heldout": {"cost_ratio": 4.0, "slo_in": 0.7}}}}', encoding="utf-8")
    pd.DataFrame({"cost_ratio_tune_mean": [2.5, 2.5, 2.5 + 1e-12, 3.0]}).to_csv(
        folder / "reactive_grid.csv", index=False)
    run = read_tuning(tmp_path, 30)
    assert run is not None
    table = tuning_table([run])
    assert len(table) == 1
    row = table.iloc[0]
    assert row["price_ratio"] == 30
    assert row["chosen_config"] == "hpa_up_util=0.2"
    assert row["default_config"] == "hpa_up_util=0.7"
    assert row["tuning_ratio_chosen"] == pytest.approx(2.5)
    assert row["tuning_ratio_default"] == pytest.approx(3.0)
    assert row["heldout_ratio_chosen"] == pytest.approx(2.1)
    assert row["heldout_slo_in_default"] == pytest.approx(0.7)
    assert row["rank_of_default"] == 29 and row["grid_size"] == 54
    assert row["n_distinct_scores"] == 2  # 2.5 and 2.5 + 1e-12 are the same score


# --------------------------------------------------------------------------- the figures
def _epsilon_frame() -> pd.DataFrame:
    """A minimal forecast-scale grid with the timetable policy and its online correction."""
    return pd.DataFrame(
        [{"error_knob": "forecast_scale", "error_value": scale, "forecast_scale": scale,
          "price_ratio": 30.0, "sweep": "epsilon", "seed": seed, "weeks": N_WEEKS, "week": 0,
          "policy": policy, "cost_ratio": 1.0 + scale / 10.0 + 0.01 * seed,
          "total_cost": 500.0 + scale, "gpu_hours": 500.0, "eta": 100.0, "sigma": 0.7,
          "eta_norm": 0.6 + abs(scale - 1.0) / 10.0, "offline_cost": 400.0, "p_in": 0.001}
         for scale in (0.5, 1.0, 2.0) for seed in SEEDS
         for policy in ("calendar", "calendar_adapt", "lcp")])


def _f6_series(ax) -> Dict[str, List[float]]:
    """F6's series by legend label: ``errorbar`` keeps them in the containers, not in the lines."""
    return {str(container.get_label()): [float(v) for v in container.lines[0].get_ydata()]
            for container in ax.containers}


def test_f6_keeps_the_total_ratio_and_styles_the_online_correction(tmp_path: Path) -> None:
    """F6 reports ``cost_ratio`` and its own theory line whether or not there is a base tier.

    Lemma B bounds the total cost against the total optimum, so the floor belongs in both; the
    ``calendar_adapt`` control has to reach the legend with a style of its own.
    """
    figures = _load_script("make_figures")
    cfg = figures.apply_style()
    assert "calendar_adapt" in figures.F6_POLICIES
    data = _epsilon_frame()
    out = tmp_path / "figures"
    plain = _captured(figures, figures.figure_f6, data, out, cfg)
    floored = _captured(figures, figures.figure_f6, data, out, cfg,
                        FloorSpec(MIN_REPLICAS, C_RUN))
    name = "F6_cost_ratio_vs_forecast_scale"
    assert plain[name].get_ylabel() == "Cost ratio to the offline optimum"
    assert floored[name].get_ylabel() == plain[name].get_ylabel()
    assert _f6_series(floored[name]) == _f6_series(plain[name])  # the floor changes nothing here
    drawn = _f6_series(plain[name])
    assert "Timetable + online correction" in drawn  # LABELS entry of calendar_adapt
    assert "FtP-cal" in drawn
    theory = [line for line in plain[name].get_lines()
              if str(line.get_label()).startswith("Lemma B")]
    assert len(theory) == 1
    assert ([float(v) for v in theory[0].get_ydata()]
            == [float(v) for v in [line for line in floored[name].get_lines()
                                   if str(line.get_label()).startswith("Lemma B")
                                   ][0].get_ydata()])
    colours = {str(c.get_label()): c.lines[0].get_color() for c in plain[name].containers}
    assert colours["Timetable + online correction"] != colours["FtP-cal"]
    assert colours["Timetable + online correction"] in cfg.palette
    assert "Timetable + online correction" in plain[name].get_legend_handles_labels()[1]


def _figure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The synthetic frame with a synchrony axis, so F5 has more than one cell to compare."""
    out = frame.copy()
    on_reference = out["lead_tier"] == REFERENCE_CELL[0]
    out["sigma"] = np.where(on_reference, 0.2, 0.7)
    out["sigma2_measured"] = np.where(on_reference, 0.24, 0.70)
    return out


def _captured(module: ModuleType, call, *args, **kwargs) -> Dict[str, object]:
    """Call a figure function and keep the axes it drew, instead of writing the files."""
    drawn: Dict[str, object] = {}
    original = module.save

    def spy(fig, out_dir, name, exts=("pdf", "png")) -> None:
        drawn[name] = fig.get_axes()[0]
        original(fig, out_dir, name, exts)

    module.save = spy
    try:
        call(*args, **kwargs)
    finally:
        module.save = original
        plt.close("all")
    return drawn


def test_f4b_and_the_floor_aware_f5_run_on_the_synthetic_frame(tmp_path: Path,
                                                              frame: pd.DataFrame) -> None:
    """Both figures draw from the synthetic frame and write their PDF and PNG."""
    figures = _load_script("make_figures")
    cfg = figures.apply_style()
    floor = FloorSpec(min_replicas=MIN_REPLICAS, c_run=C_RUN)
    data = _figure_frame(frame)
    out = tmp_path / "figures"
    with_controls = pd.concat([data, data[data["policy"] == "reactive"].assign(policy="fixed"),
                               data[data["policy"] == "reactive"].assign(policy="static")],
                              ignore_index=True)
    axes = _captured(figures, figures.figure_f4b, with_controls, out, cfg, floor)
    assert "F4b_weekly_excess" in axes
    assert axes["F4b_weekly_excess"].get_xlabel() == "Teaching week"
    assert "USD per week" in axes["F4b_weekly_excess"].get_ylabel()
    drawn = {line.get_label() for line in axes["F4b_weekly_excess"].get_lines()}
    # the zero line, the flat line and the control that never scales are all left out
    assert drawn.isdisjoint({"Offline optimum", "Static peak", "Base tier only"})
    assert "Timetable FtP" in drawn
    assert axes["F4b_weekly_excess"].patches, "the R1 weeks should be shaded"
    assert [text.get_text() for text in axes["F4b_weekly_excess"].get_legend().get_texts()
            ].count("R1") == 1
    axes = _captured(figures, figures.figure_f5, data, out, cfg, floor)
    assert axes["F5_gain_vs_sigma"].get_ylabel() == "Cost above the base tier saved vs reactive"
    for name in ("F4b_weekly_excess", "F5_gain_vs_sigma"):
        for ext in ("pdf", "png"):
            assert (out / f"{name}.{ext}").stat().st_size > 0, f"{name}.{ext}"


def test_the_figures_keep_their_pre_floor_labels_without_a_floor(tmp_path: Path,
                                                                 frame: pd.DataFrame) -> None:
    """``--min-replicas 0`` draws F4 and F5 exactly as before the base tier existed."""
    figures = _load_script("make_figures")
    cfg = figures.apply_style()
    data = _figure_frame(frame)
    out = tmp_path / "figures"
    plain = _captured(figures, figures.figure_f4, data, out, cfg)["F4_cost_violation_pareto"]
    assert plain.get_ylabel() == "GPU-hours"
    assert _captured(figures, figures.figure_f5, data, out,
                     cfg)["F5_gain_vs_sigma"].get_ylabel() == "Cost saved vs reactive"
    # the whole-run GPU-hours of the most synchronous cell, policy by policy, as before the floor
    whole = data[(data["week"] == 0) & (data["sigma"] == data["sigma"].max())]
    expected = whole.groupby("policy")["gpu_hours"].mean()
    drawn = {line.get_label(): float(line.get_ydata()[0]) for line in plain.get_lines()}
    for policy, hours in expected.items():
        assert drawn[figures.LABELS[policy]] == pytest.approx(hours)


def test_f4_keeps_the_total_gpu_hours_per_week_above_a_floor(tmp_path: Path,
                                                             frame: pd.DataFrame) -> None:
    """With a base tier F4 divides by the teaching weeks but keeps the *total* resource use.

    The reader of an institution-wide deployment has to see how large the always-on floor is next
    to what the scheduler decides, so the attributable component stays in the tables.
    """
    figures = _load_script("make_figures")
    cfg = figures.apply_style()
    data = _figure_frame(frame)
    axes = _captured(figures, figures.figure_f4, data, tmp_path / "figures", cfg,
                     FloorSpec(MIN_REPLICAS, C_RUN))
    ax = axes["F4_cost_violation_pareto"]
    assert ax.get_ylabel() == "GPU-hours per week"
    whole = data[(data["week"] == 0) & (data["sigma"] == data["sigma"].max())]
    expected = whole.groupby("policy")["gpu_hours"].mean() / N_WEEKS
    drawn = {line.get_label(): float(line.get_ydata()[0]) for line in ax.get_lines()}
    for policy, hours in expected.items():
        assert drawn[figures.LABELS[policy]] == pytest.approx(hours)
        assert hours > BASE_WHOLE_HOURS / N_WEEKS  # the total, not the attributable component


def test_f4_draws_the_slo_matched_pair_as_hollow_markers(tmp_path: Path,
                                                         frame: pd.DataFrame) -> None:
    """``r30_slo95/lead/lead.csv`` adds two hollow points of the same colour and marker."""
    figures = _load_script("make_figures")
    cfg = figures.apply_style()
    data = _figure_frame(frame)
    matched = _matched_frame(frame)
    out = tmp_path / "figures"
    axes = _captured(figures, figures.figure_f4, data, out, cfg,
                     FloorSpec(MIN_REPLICAS, C_RUN), matched)
    ax = axes["F4_cost_violation_pareto"]
    hollow = {line.get_label(): line for line in ax.get_lines()
              if line.get_markerfacecolor() == "none"}
    assert set(hollow) == {f"Reactive (HPA){figures.F4_MATCHED_SUFFIX}",
                           f"Token velocity{figures.F4_MATCHED_SUFFIX}"}
    solid = {line.get_label(): line for line in ax.get_lines()
             if line.get_markerfacecolor() != "none"}
    same_style = hollow[f"Reactive (HPA){figures.F4_MATCHED_SUFFIX}"]
    assert same_style.get_color() == solid["Reactive (HPA)"].get_color()
    assert same_style.get_marker() == solid["Reactive (HPA)"].get_marker()
    assert float(same_style.get_xdata()[0]) == pytest.approx(MATCHED_SLO_IN["reactive"])
    assert (float(same_style.get_ydata()[0])
            == pytest.approx(MATCHED_HOURS["reactive"] / N_WEEKS))
    for ext in ("pdf", "png"):
        assert (out / f"F4_cost_violation_pareto.{ext}").stat().st_size > 0


def test_f4_reads_the_slo_matched_sweep_beside_the_plotted_group(tmp_path: Path,
                                                                 frame: pd.DataFrame) -> None:
    """``_slo_matched`` looks in the sibling directory the batch writes, and tolerates its absence."""
    figures = _load_script("make_figures")
    assert figures._slo_matched(tmp_path / "r30") is None
    path = tmp_path / figures.F4_MATCHED_GROUP / "lead" / "lead.csv"
    path.parent.mkdir(parents=True)
    _matched_frame(frame).to_csv(path, index=False)
    found = figures._slo_matched(tmp_path / "r30")
    assert found is not None
    assert set(found["policy"]) == {"reactive", "velocity"}


def test_the_floor_flags_need_a_price(tmp_path: Path) -> None:
    """``--min-replicas`` without ``--c-run`` is refused rather than priced at zero."""
    figures = _load_script("make_figures")
    parser = figures.build_parser()
    assert figures.floor_from(parser.parse_args([]), parser).on is False
    args = parser.parse_args(["--min-replicas", "1", "--c-run", "0.0333"])
    assert figures.floor_from(args, parser) == FloorSpec(1, 0.0333)
    with pytest.raises(SystemExit):
        figures.floor_from(parser.parse_args(["--min-replicas", "1"]), parser)
