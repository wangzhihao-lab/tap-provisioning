"""Baseline tuning (scripts/sim/tune_baselines.py) and the tuned-config round trip.

Two things have to hold for the tuned baselines to be trustworthy.  The grid search has to run end
to end and write a well-formed ``best.json`` - exercised here on a two-configuration grid over a
single section, which keeps the instance tiny while using the real engine - and that file has to
come back into a ``PolicyConfig`` with the right values *and the right types*, because it travels
through JSON, where an ``int`` knob becomes a float.  The reader lives in ``tuned_config`` and is
reached from ``run_sweep.py``, so the round trip is checked across that seam as well.

The scripts are CLIs rather than an installed package, so they are loaded by path.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pilot.sim import PolicyConfig, WorkloadConfig

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "sim"
SMALL_COLLEGE = "college_04"


def _load_script(name: str, alias: str = "") -> ModuleType:
    """Import ``scripts/sim/<name>.py`` as a module under a private name.

    The module goes into ``sys.modules`` before it is executed: the scripts use
    ``from __future__ import annotations``, so their dataclass annotations are strings and
    ``dataclasses`` resolves them through ``sys.modules[cls.__module__]``.  ``alias`` gives a second
    independent copy of the same file, which is how a test gets the unpatched grids back.
    """
    key = f"_tune_script_{alias or name}"
    spec = importlib.util.spec_from_file_location(key, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tune() -> ModuleType:
    """The tuner module."""
    return _load_script("tune_baselines")


@pytest.fixture(scope="module")
def tuned_config() -> ModuleType:
    """The JSON contract module the tuner writes and ``run_sweep`` reads."""
    return _load_script("tuned_config")


@pytest.fixture(scope="module")
def tiny_result(tune: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """Run a two-configuration grid for each policy on one section over one teaching week."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")
    velocity = [{"velocity_window_min": 2, "velocity_clip": 4.0, "velocity_hysteresis": True,
                 "velocity_persist_min": 2, "velocity_dwell_min": 3},
                {"velocity_window_min": 20, "velocity_clip": 4.0, "velocity_hysteresis": True,
                 "velocity_persist_min": 3, "velocity_dwell_min": 10}]
    reactive = [{"hpa_up_util": 0.7, "hpa_down_util": 0.3, "hpa_cooldown_min": 3,
                 "hpa_down_hold_min": 5},
                {"hpa_up_util": 0.4, "hpa_down_util": 0.3, "hpa_cooldown_min": 0,
                 "hpa_down_hold_min": 45}]
    tune.GRIDS = {"velocity": lambda: velocity, "reactive": lambda: reactive}
    spec = tune.TuningSpec(college=SMALL_COLLEGE, weeks=1, adoption=1.0, max_sections=1,
                           tuning_seeds=(0, 1), heldout_seed=2, root=str(REPO))
    out = tmp_path_factory.mktemp("tuning")
    payload = tune.run(spec, ["velocity", "reactive"], out)
    return payload, out, spec


def test_grid_runs_end_to_end_and_writes_both_csvs(tiny_result: tuple) -> None:
    """Each policy gets a CSV with one row per configuration and a rank column."""
    _, out, _ = tiny_result
    for policy in ("velocity", "reactive"):
        frame = pd.read_csv(out / f"{policy}_grid.csv")
        assert len(frame) == 2
        assert sorted(frame["rank"]) == [1, 2]
        assert frame["cost_ratio_tune_mean"].is_monotonic_increasing  # sorted by the objective
        assert bool(frame["is_default"].any())  # the default is in the grid, so before/after works
        assert (frame["cost_ratio_seed0"] >= 1.0).all()  # nothing beats the offline optimum
        assert (frame["cost_ratio_heldout"] >= 1.0).all()


def test_selection_uses_only_the_tuning_seeds(tiny_result: tuple) -> None:
    """The chosen row is the one with the lowest tuning mean, whatever the held-out seed says."""
    payload, out, _ = tiny_result
    for policy, section in payload["policies"].items():
        frame = pd.read_csv(out / f"{policy}_grid.csv")
        best = frame.sort_values(["cost_ratio_tune_mean", "config_id"]).iloc[0]
        for knob, value in section["config"].items():
            assert best[knob] == value
        assert section["tuning"]["cost_ratio_mean"] == pytest.approx(best["cost_ratio_tune_mean"])
        assert section["heldout"]["cost_ratio"] == pytest.approx(best["cost_ratio_heldout"])


def test_best_json_round_trips_into_policy_config(tiny_result: tuple,
                                                  tuned_config: ModuleType) -> None:
    """``best.json`` -> flat overrides -> ``PolicyConfig``, with the declared types preserved."""
    payload, out, _ = tiny_result
    path = out / "best.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    values = tuned_config.read_tuned(path)
    chosen = {k: v for section in payload["policies"].values()
              for k, v in section["config"].items()}
    assert values == chosen
    cfg = tuned_config.apply_policy_values(PolicyConfig(), values)
    base = PolicyConfig()
    for knob, value in chosen.items():
        assert getattr(cfg, knob) == value
        assert type(getattr(cfg, knob)) is type(getattr(base, knob))
    untouched = set(PolicyConfig().__dataclass_fields__) - set(chosen)
    assert all(getattr(cfg, k) == getattr(base, k) for k in untouched)


def test_run_sweep_applies_the_tuned_file_and_leaves_defaults_alone(tiny_result: tuple) -> None:
    """``--tuned`` reaches the base ``RunConfig``; without the flag nothing moves."""
    payload, out, _ = tiny_result
    sweep = _load_script("run_sweep")
    parser = sweep.build_parser()
    plain = sweep.base_config(parser.parse_args(["--quick"]))
    assert plain.policy == PolicyConfig()
    tuned = sweep.base_config(parser.parse_args(["--quick", "--tuned", str(out / "best.json")]))
    for section in payload["policies"].values():
        for knob, value in section["config"].items():
            assert getattr(tuned.policy, knob) == value
    inline = sweep.base_config(parser.parse_args(
        ["--quick", "--policy-config", '{"velocity_window_min": 7, "plan_margin": 1.25}']))
    assert inline.policy.velocity_window_min == 7
    assert inline.policy.plan_margin == pytest.approx(1.25)
    assert type(inline.policy.velocity_window_min) is int


def test_tuned_config_rejects_bad_input(tuned_config: ModuleType, tmp_path: Path) -> None:
    """Unknown fields, non-integral ints and disagreeing sections raise rather than pass through."""
    with pytest.raises(KeyError):
        tuned_config.policy_values({"not_a_field": 1})
    with pytest.raises(TypeError):
        tuned_config.policy_values({"velocity_window_min": 2.5})
    with pytest.raises(TypeError):
        tuned_config.policy_values({"hpa_mode": None})
    assert tuned_config.policy_values({"forecast_warm_start": None}) == {
        "forecast_warm_start": None}
    clash = tmp_path / "clash.json"
    clash.write_text(json.dumps({"policies": {"a": {"config": {"headroom": 1.1}},
                                              "b": {"config": {"headroom": 1.2}}}}))
    with pytest.raises(ValueError, match="disagree"):
        tuned_config.read_tuned(clash)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"policies": {}}))
    with pytest.raises(ValueError):
        tuned_config.read_tuned(empty)


def test_grids_stay_inside_their_declared_caps(tune: ModuleType) -> None:
    """Both production grids hold at most 60 configurations and contain the current default."""
    module = _load_script("tune_baselines", alias="grids")  # fresh: the fixture patches GRIDS
    for policy, grid in module.GRIDS.items():
        cells = grid()
        assert 0 < len(cells) <= 60
        assert module.defaults_for(policy) in cells
        assert len({tuple(sorted(c.items())) for c in cells}) == len(cells)  # no duplicates
    assert max(module.HPA_DOWN_HOLD) < module.HPA_HOLD_CAP_MIN


def test_ties_go_to_the_default_cell(tune: ModuleType) -> None:
    """A tie is no evidence for moving off the default, so the default wins it."""
    rows = [{"policy": "reactive", "config_id": i, "is_default": i == 2,
             "cost_ratio_tune_mean": 2.0 if i < 3 else 3.0} for i in range(5)]
    frame = tune.ranked(pd.DataFrame(rows))
    assert list(frame["config_id"]) == [2, 0, 1, 3, 4]
    assert list(frame["rank"]) == [1, 2, 3, 4, 5]
