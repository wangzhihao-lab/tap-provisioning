"""End to end: one course, two teaching weeks, every policy, plus the sweep and CSV round trip.

The last three tests cover the two-axis F6 pipeline: the epsilon sweep's price-ratio facets, the
``--p-in`` / ``--price-ratios`` options of ``scripts/sim/run_sweep.py`` and the figure functions
that consume the CSV.  The scripts are entry points rather than a package, so they are loaded by
path (``_load_script``).
"""

import importlib.util
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from pilot.sim import CostConfig, RunConfig, ServiceConfig, SweepConfig, WorkloadConfig
from pilot.sim import engine
from pilot.sim.metrics import to_frame
from pilot.sim.sweeps import EXACT, SWEEPS, run_sweep

REPO = Path(__file__).resolve().parents[1]
BUDGET_S = 10.0
SCRIPTS = REPO / "scripts" / "sim"


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/sim/<name>.py`` as a module; they are CLIs, not an installed package."""
    spec = importlib.util.spec_from_file_location(f"_sim_script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def one_course() -> RunConfig:
    """A single section of a single college over two teaching weeks."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")
    workload = WorkloadConfig(college="college_04", weeks=(1, 2), adoption=1.0, max_sections=1,
                              sigma=0.7)
    return RunConfig(workload=workload, service=ServiceConfig(lead_min=8.0), seed=5,
                     root=str(REPO))


def test_all_policies_on_one_course_within_budget(one_course: RunConfig) -> None:
    """Every policy runs, produces finite metrics and the whole comparison fits the budget."""
    start = time.perf_counter()
    stream = engine.build_stream(one_course)
    runs = engine.run_all(one_course, stream=stream)
    elapsed = time.perf_counter() - start
    assert set(runs) == set(engine.DEFAULT_POLICIES)
    assert stream.sessions["section_id"].nunique() == 1
    assert stream.n_requests > 0
    for name, run in runs.items():
        m = run.metrics
        assert np.isfinite(m.total_cost) and m.total_cost >= 0.0
        assert np.isfinite(m.gpu_hours) and m.gpu_hours >= 0.0
        assert m.n_in + m.n_off == stream.n_requests
        assert 0.0 <= m.slo_in <= 1.0 and 0.0 <= m.slo_off <= 1.0
        assert m.timeouts >= 0 and m.rejected >= 0
        assert m.cost_ratio >= 1.0 - 1e-9, name
    assert elapsed < BUDGET_S, f"one-course comparison took {elapsed:.1f} s"


def test_metrics_round_trip_through_a_frame(one_course: RunConfig) -> None:
    """``to_frame`` keeps one row per policy and all the metric columns."""
    runs = engine.run_all(one_course, policies=("static", "reactive", "calendar", "offline"))
    frame = to_frame([r.metrics for r in runs.values()])
    assert len(frame) == 4
    for column in ("gpu_hours", "starts", "reservation_blocks", "cost_ratio", "ttft_p95_in_s",
                   "ttft_p95_off_s", "slo_in", "slo_off", "timeouts", "rejected"):
        assert column in frame.columns


def test_sweep_writes_one_row_per_cell(tmp_path, one_course: RunConfig) -> None:
    """A reduced sigma sweep yields cells x seeds x policies whole-run rows, plus a row per week."""
    policies = ("static", "calendar", "offline")
    n_weeks = one_course.workload.n_weeks
    result = run_sweep("sigma", one_course, SweepConfig(), seeds=(0, 1), policies=policies,
                       quick=True)
    whole = result.frame[result.frame["week"] == 0]
    assert len(whole) == 2 * 2 * len(policies)
    assert len(result.frame) == 2 * 2 * len(policies) * (1 + n_weeks)
    assert set(result.frame.loc[result.frame["week"] > 0, "week"]) == set(one_course.workload.weeks)
    assert set(result.frame["policy"]) == set(policies)
    assert set(result.frame["sigma"]) == {0.0, 1.0}  # the grid is the mixing weight w
    assert whole["eta_norm"].between(0.0, 10.0).all()
    # every row carries the synchrony its own stream realised, which is what F5 is drawn against
    for column in ("sigma2_measured", "sigma2_pooled"):
        assert whole[column].between(0.0, 1.0).all()
    by_w = whole.groupby("sigma")["sigma2_measured"].mean()
    assert by_w.loc[0.0] < by_w.loc[1.0]
    assert by_w.loc[1.0] == pytest.approx(0.70, abs=0.05)
    assert by_w.loc[0.0] == pytest.approx(0.24, abs=0.05)
    path = tmp_path / "sigma" / "sigma.csv"
    result.write(path)
    assert path.exists() and path.stat().st_size > 0


def test_every_sweep_builder_produces_cells(one_course: RunConfig) -> None:
    """Every grid is wired and yields at least two cells in quick mode."""
    for name, builder in SWEEPS.items():
        cells = list(builder(one_course, SweepConfig(), True))
        assert len(cells) >= 2, name
        assert all(isinstance(params, dict) for params, _ in cells)


@pytest.fixture(scope="module")
def six_weeks() -> RunConfig:
    """The same single section over six teaching weeks: the longest horizon the CLI allows."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")
    workload = WorkloadConfig(college="college_04", weeks=tuple(range(1, 7)), adoption=1.0,
                              max_sections=1, sigma=0.7)
    return RunConfig(workload=workload, service=ServiceConfig(lead_min=8.0), seed=5,
                     root=str(REPO))


def test_weekly_rows_partition_the_run(six_weeks: RunConfig) -> None:
    """Per-week costs add up to the whole-run cost and carry their own in-class attainment."""
    result = run_sweep("sigma", six_weeks, SweepConfig(), seeds=(0,),
                       policies=("calendar", "offline"), quick=True)
    frame = result.frame
    assert set(frame.loc[frame["week"] > 0, "week"]) == set(range(1, 7))
    for (policy, sigma), part in frame.groupby(["policy", "sigma"]):
        whole = part[part["week"] == 0].iloc[0]
        weekly = part[part["week"] > 0]
        assert len(weekly) == 6
        assert weekly["total_cost"].sum() == pytest.approx(whole["total_cost"], rel=1e-9)
        assert weekly["n_in"].sum() + weekly["n_off"].sum() == whole["n_in"] + whole["n_off"]
        # a week without any in-class request has no attainment to report (NaN), which a single
        # cancelled session of a one-section slice can produce
        attained = weekly["slo_in"].between(0.0, 1.0) | (weekly["n_in"] == 0)
        assert attained.all(), (policy, sigma)


def test_six_week_run_keeps_every_policy_finite(six_weeks: RunConfig) -> None:
    """A six-week horizon runs end to end for all nine policies plus the two optima."""
    stream = engine.build_stream(six_weeks)
    assert stream.horizon_min == 6 * 7 * 1440
    runs = engine.run_all(six_weeks, stream=stream)
    assert "stale_calendar" in runs
    for name, run in runs.items():
        assert np.isfinite(run.metrics.total_cost), name
        assert run.metrics.cost_ratio >= 1.0 - 1e-9, name


def test_margin_sweep_only_moves_the_calendar_policies(one_course: RunConfig) -> None:
    """The ``margin`` grid changes policies 6, 7 and 9 and leaves the rest alone."""
    policies = ("lcp", "calendar", "bdinterp", "offline")
    frame = run_sweep("margin", one_course, SweepConfig(), seeds=(0,), policies=policies,
                      quick=True).frame
    whole = frame[frame["week"] == 0]
    assert set(whole["plan_margin"]) == {1.0, 1.5}
    piv = whole.pivot_table(index="policy", columns="plan_margin", values="gpu_hours")
    assert piv.loc["lcp", 1.0] == pytest.approx(piv.loc["lcp", 1.5])
    assert piv.loc["offline", 1.0] == pytest.approx(piv.loc["offline", 1.5])
    assert piv.loc["calendar", 1.5] > piv.loc["calendar", 1.0]


def test_epsilon_sweep_varies_one_knob_at_a_time(one_course: RunConfig) -> None:
    """Each cell of the calendar-error grid moves one knob and leaves the other two exact."""
    # price axis off: this test is about the error knobs, and without it the grid has one cell per
    # knob value rather than one per (knob value, price ratio).
    frame = run_sweep("epsilon", one_course, SweepConfig(price_ratio=()), seeds=(0,),
                      policies=("calendar",), quick=True).frame
    whole = frame[frame["week"] == 0]
    assert "price_ratio" not in whole.columns
    assert set(whole["error_knob"]) == {"forecast_scale", "bell_shift_frac", "p_cancel"}
    for knob, part in whole.groupby("error_knob"):
        others = [c for c in ("forecast_scale", "bell_shift_frac", "p_cancel") if c != knob]
        for other in others:
            assert set(part[other]) == {EXACT[other]}, (knob, other)
        assert part[knob].nunique() >= 2
    scale = whole[whole["error_knob"] == "forecast_scale"].set_index("error_value")
    assert scale.loc[0.5, "eta_norm"] > scale.loc[1.0, "eta_norm"]
    assert scale.loc[2.0, "eta_norm"] > scale.loc[1.0, "eta_norm"]


def test_epsilon_sweep_prices_every_facet_by_its_ratio(one_course: RunConfig) -> None:
    """The price axis solves ``p_in = ratio * c_run / mu`` per cell and records what it realised."""
    frame = run_sweep("epsilon", one_course, SweepConfig(price_ratio=(6.0, 18.0)), seeds=(0,),
                      policies=("calendar",), quick=True).frame
    whole = frame[frame["week"] == 0]
    assert set(whole["price_ratio"]) == {6.0, 18.0}
    assert (whole["p_in"] > 0.0).all()
    off_by = (whole["price_ratio_real"] / whole["price_ratio"] - 1.0).abs()
    assert off_by.max() < 0.01, off_by.max()
    # the ratio is dimensionless, so p_in itself has to move with it (and p_off follow p_in)
    p_in = whole.groupby("price_ratio")["p_in"].mean()
    assert p_in.loc[18.0] == pytest.approx(3.0 * p_in.loc[6.0], rel=1e-9)
    assert whole["p_off"].to_numpy() == pytest.approx(whole["p_in"].to_numpy() / 6.0, rel=1e-9)
    # the streams are shared across the facets: the same cell has the same measured error
    paired = whole.pivot_table(index=["error_knob", "error_value"], columns="price_ratio",
                               values="eta_norm")
    assert paired[6.0].to_numpy() == pytest.approx(paired[18.0].to_numpy())


def test_cli_price_options_override_the_cost_config() -> None:
    """``--p-in`` moves ``CostConfig`` (``p_off`` follows at ``p_in / 6``) and the ratios parse."""
    cli = _load_script("run_sweep")
    cfg = cli.base_config(cli.build_parser().parse_args(["--p-in", "0.003"]))
    assert cfg.cost.p_in == pytest.approx(0.003)
    assert cfg.cost.p_off == pytest.approx(0.0005)
    both = cli.base_config(cli.build_parser().parse_args(["--p-in", "0.003", "--p-off", "0.002"]))
    assert both.cost.p_off == pytest.approx(0.002)
    assert cli.base_config(cli.build_parser().parse_args([])).cost == CostConfig()
    assert cli.price_ratios(None) == SweepConfig().price_ratio
    assert cli.price_ratios(["6", "18"]) == (6.0, 18.0)
    assert cli.price_ratios(["none"]) == ()


def test_f6_figures_draw_from_a_tiny_frame(tmp_path) -> None:
    """F6's facets and F6b's scatter both run on a synthetic two-facet error grid."""
    figures = _load_script("make_figures")
    rows = [{"week": 0, "error_knob": "forecast_scale", "error_value": scale,
             "forecast_scale": scale, "price_ratio": ratio, "price_ratio_real": ratio,
             "policy": policy, "seed": seed, "cost_ratio": 1.0 + scale / 10.0 + 0.01 * seed,
             "eta": 100.0, "eta_norm": 0.6 + abs(scale - 1.0) / 10.0, "offline_cost": 50.0,
             "p_in": 0.01, "sigma": 0.7}
            for ratio in (6.0, 18.0) for scale in (0.5, 1.0, 20.0) for seed in (0, 1)
            for policy in ("calendar", "bdinterp@1.0", "lcp", "informed")]
    out = tmp_path / "figures"
    cfg = figures.apply_style()
    figures.figure_f6(pd.DataFrame(rows), out, cfg)
    figures.figure_f6b(pd.DataFrame(rows), out, cfg)
    for name in ("F6_cost_ratio_vs_forecast_scale", "F6b_cost_ratio_vs_calendar_error"):
        assert (out / f"{name}.pdf").stat().st_size > 0
        assert (out / f"{name}.png").stat().st_size > 0
