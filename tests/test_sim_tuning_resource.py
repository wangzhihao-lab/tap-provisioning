"""SLO-floor tuning by resource cost (``tune_baselines.py --slo-select resource``).

The floor-matched baseline is the cheapest configuration that meets the in-class floor, and its
cost is the capacity bill: mean ``run_cost + switch_cost`` over the tuning seeds.  The default rule
(first cell by the penalty-inclusive cost ratio) must stay what it was.  Pinned here:

* the floor is compared unrounded (0.9499999 is below 0.95);
* ties on resource cost keep the ranked order;
* no cell at the floor -> the unconstrained best, flagged ``floor_met = false``;
* a real run stores the tuning seeds' GPU-hours, starts, run and switch cost and attainment per
  cell, and ``best.json`` records the rule and what the cost rule would have picked.
"""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pilot.sim import WorkloadConfig

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "sim"
KNOBS = ("hpa_up_util", "hpa_down_util", "hpa_cooldown_min", "hpa_down_hold_min")


def _load(name: str, alias: str) -> ModuleType:
    """Import ``scripts/sim/<name>.py`` under a private name (see test_sim_tuning.py)."""
    key = f"_resource_test_{alias}"
    spec = importlib.util.spec_from_file_location(key, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tune() -> ModuleType:
    """A private copy of the tuner (its GRIDS get patched below)."""
    return _load("tune_baselines", "tune")


def _grid() -> pd.DataFrame:
    """Five ranked cells: (cost ratio, attainment, resource cost) chosen to separate the rules."""
    cells = [(1.00, 0.94, 10.0), (1.10, 0.96, 12.0), (1.20, 0.97, 9.0), (1.30, 0.95, 9.0),
             (1.40, 0.9499999, 5.0)]
    rows = []
    for idx, (ratio, slo, res) in enumerate(cells):
        rows.append({"policy": "reactive", "config_id": idx, "rank": idx + 1, "is_default": idx == 1,
                     "hpa_up_util": 0.1 * (idx + 1), "hpa_down_util": 0.1, "hpa_cooldown_min": 0,
                     "hpa_down_hold_min": 5, "cost_ratio_tune_mean": ratio,
                     "cost_ratio_tune_max": ratio, "slo_in_tune_mean": slo,
                     "resource_cost_tune_mean": res, "gpu_hours_tune_mean": res,
                     "starts_tune_mean": 1.0, "run_cost_tune_mean": res - 1.0,
                     "switch_cost_tune_mean": 1.0, "cost_ratio_heldout": ratio,
                     "slo_in_heldout": slo, "slo_first_in_heldout": slo, "viol_in_heldout": 1 - slo,
                     "gpu_hours_heldout": res, "starts_heldout": 1.0})
    return pd.DataFrame(rows)


def test_rules_on_a_synthetic_grid(tune: ModuleType) -> None:
    """Cost rule: first ranked cell at the floor.  Resource rule: cheapest capacity at the floor."""
    grid = _grid()
    by_cost = tune.chosen(grid, "reactive", KNOBS, 0.95)
    assert by_cost["config"]["hpa_up_util"] == pytest.approx(0.2)  # cell 1
    assert "selection" not in by_cost
    by_res = tune.chosen(grid, "reactive", KNOBS, 0.95, "resource")
    assert by_res["config"]["hpa_up_util"] == pytest.approx(0.3)  # cell 2: ties with 3, ranks first
    sel = by_res["selection"]
    assert sel["floor_met"] and sel["eligible"] == 3  # cells 1-3; 0.9499999 does not round up
    assert sel["resource_cost_mean"] == 9.0 and not sel["same_as_cost_rule"]
    assert sel["cost_rule_config"]["hpa_up_util"] == pytest.approx(0.2)
    nobody = tune.chosen(grid, "reactive", KNOBS, 0.99, "resource")
    assert nobody["config"]["hpa_up_util"] == pytest.approx(0.1)  # unconstrained best
    assert not nobody["selection"]["floor_met"] and nobody["selection"]["eligible"] == 0
    with pytest.raises(ValueError):
        tune.chosen(grid, "reactive", KNOBS, None, "resource")


def test_changed_policies_compares_the_chosen_configs() -> None:
    """The rerun decision of the rerun round: which policies' selection moved."""
    select = sys.modules["tuning_select"]
    old = {"policies": {"reactive": {"config": {"hpa_up_util": 0.4}},
                        "velocity": {"config": {"velocity_window_min": 5}}}}
    new = {"policies": {"reactive": {"config": {"hpa_up_util": 0.7}},
                        "velocity": {"config": {"velocity_window_min": 5}}}}
    assert select.changed_policies(old, new) == ["reactive"]
    assert select.changed_policies(old, old) == []


@pytest.fixture(scope="module")
def tiny_runs(tune: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """One tiny grid scored once per rule on one section and one teaching week."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")
    reactive = [{"hpa_up_util": 0.7, "hpa_down_util": 0.3, "hpa_cooldown_min": 3,
                 "hpa_down_hold_min": 5},
                {"hpa_up_util": 0.4, "hpa_down_util": 0.3, "hpa_cooldown_min": 0,
                 "hpa_down_hold_min": 45}]
    tune.GRIDS = {"reactive": lambda: reactive}
    spec = tune.TuningSpec(college="college_04", weeks=1, adoption=1.0, max_sections=1,
                           tuning_seeds=(0, 1), heldout_seed=2, root=str(REPO), slo_floor=0.0)
    out_cost, out_res = tmp_path_factory.mktemp("cost"), tmp_path_factory.mktemp("resource")
    cost = tune.run(spec, ["reactive"], out_cost)
    res = tune.run(replace(spec, slo_select="resource"), ["reactive"], out_res)
    return cost, res, out_cost, out_res


def test_a_real_run_stores_the_resource_columns_and_the_rule(tiny_runs: tuple) -> None:
    """Per-seed resource columns in the grid; the resource pick is its argmin; provenance kept."""
    cost, res, out_cost, out_res = tiny_runs
    grid = pd.read_csv(out_res / "reactive_grid.csv")
    for seed in (0, 1):
        for key in ("gpu_hours", "starts", "run_cost", "switch_cost", "slo_in"):
            assert f"{key}_seed{seed}" in grid.columns
    both = grid["run_cost_seed0"] + grid["switch_cost_seed0"] + grid["run_cost_seed1"] \
        + grid["switch_cost_seed1"]
    assert (grid["resource_cost_tune_mean"] - both / 2).abs().max() < 1e-9
    best = grid.sort_values(["resource_cost_tune_mean", "rank"]).iloc[0]
    section = res["policies"]["reactive"]
    assert section["config"]["hpa_up_util"] == pytest.approx(best["hpa_up_util"])
    assert section["selection"]["rule"] == "resource" and section["selection"]["floor_met"]
    assert res["slo_select"] == "resource" and res["slice"]["slo_select"] == "resource"
    on_disk = json.loads((out_res / "best.json").read_text(encoding="utf-8"))
    assert on_disk == res
    # the default rule's output carries no new keys, and picks the first ranked cell
    assert "slo_select" not in cost and "slo_select" not in cost["slice"]
    assert "selection" not in cost["policies"]["reactive"]
    first = pd.read_csv(out_cost / "reactive_grid.csv").iloc[0]
    assert cost["policies"]["reactive"]["config"]["hpa_up_util"] == pytest.approx(first["hpa_up_util"])
