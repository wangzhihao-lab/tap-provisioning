"""The validation-round helpers: run-name grammar, cell labelling, group assembly, comparisons."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.validation import (
    CELL_KEY,
    CELLS,
    RunFile,
    RunSpec,
    assemble,
    cell_means,
    compare,
    expected_runs,
    frontier_row,
    groups_of,
    load_run,
    parse_run,
    same_rows,
    wide,
)
from pilot.validation.tables import SLO_SUFFIX


@pytest.mark.parametrize("name, spec", [
    ("r30_d_rep", RunSpec(30, "d_rep", "d_rep", "main")),
    ("r100_d_inst", RunSpec(100, "d_inst", "d_rep", "main")),
    ("r30_d_inst_t8", RunSpec(30, "d_inst", "d_inst", "main")),
    ("r30_d_inst_t8_slo95", RunSpec(30, "d_inst", "d_inst", "slo95")),
    ("r30_d_rep_learn", RunSpec(30, "d_rep", "d_rep", "learn")),
    ("r30_d_inst_t8_heldout_sem2", RunSpec(30, "d_inst", "d_inst", "heldout_sem2")),
])
def test_parse_run(name: str, spec: RunSpec) -> None:
    assert parse_run(name) == spec


@pytest.mark.parametrize("name", ["tuning_r30", "r30", "r30_d_rep_extra", "offset"])
def test_parse_run_rejects_other_names(name: str) -> None:
    assert parse_run(name) is None


def test_group_key_merges_variants_and_separates_tunings() -> None:
    assert parse_run("r30_d_inst_t8").group == parse_run("r30_d_inst_t8_slo95").group
    assert parse_run("r30_d_inst").group != parse_run("r30_d_inst_t8").group
    assert parse_run("r30_d_rep").group != parse_run("r30_d_rep_heldout_sem2").group


def test_expected_runs_central_case_has_two_ratios_and_heldout() -> None:
    names = expected_runs("c2_32b_shared4k")
    assert "r100_d_rep" in names and "r30_d_inst_t8_heldout_sem2" in names
    assert "r100_d_rep" not in expected_runs("c1_7b_shared4k")


def _rows(policy: str, cell_key: tuple, seeds=(0, 1), cost=100.0, gpu=1008.0, slo=1.0,
          weeks=6, offline_cost=100.0) -> list:
    out = []
    for seed in seeds:
        for week in range(0, weeks + 1):
            row = dict(zip(CELL_KEY, cell_key))
            row.update({"sweep": "offset", "seed": seed, "week": week, "weeks": weeks, "policy": policy,
                        "total_cost": cost if week == 0 else cost / weeks,
                        "gpu_hours": gpu if week == 0 else gpu / weeks, "slo_in": slo,
                        "starts": 6.0 if week == 0 else 1.0, "x_peak": 2.0, "ttft_p95_in_s": 0.5,
                        "offline_cost": offline_cost, "mu": 400.0, "lead_min": 8.0,
                        "price_ratio_real": 30.0})
            out.append(row)
    return out


def _write(tmp_path: Path, case: str, run: str, rows: list) -> RunFile:
    path = tmp_path / "validation" / case / run / "offset" / "offset.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return RunFile(case, run, parse_run(run), path)


def test_load_run_labels_cells_and_divides_by_weeks(tmp_path: Path) -> None:
    keys = list(CELLS)
    rows = _rows("offline", keys[0], cost=120.0) + _rows("calendar", keys[0], cost=150.0, gpu=1200.0) \
        + _rows("calendar", keys[2], cost=130.0) + _rows("offline", keys[2], cost=100.0)
    run = _write(tmp_path, "c2_32b_shared4k", "r30_d_inst_t8", rows)
    frame = load_run(run)
    assert set(frame["cell"]) == {"A", "C"}
    whole = frame[(frame["week"] == 0) & (frame["policy"] == "calendar") & (frame["cell"] == "A")]
    assert np.allclose(whole["gpu_h_wk"], 200.0)
    assert np.allclose(whole["usd_wk"], 25.0)
    assert np.allclose(whole["excess_wk"], 5.0)  # (150 - 120) / 6
    weekly = frame[(frame["week"] == 1) & (frame["policy"] == "calendar") & (frame["cell"] == "A")]
    assert np.allclose(weekly["gpu_h_wk"], 200.0)  # weekly rows already cover one week


def test_assemble_adds_learn_and_relabels_slo_matched(tmp_path: Path) -> None:
    key = list(CELLS)[2]
    case = "c3_72b_shared4k"
    base = _write(tmp_path, case, "r30_d_rep", _rows("offline", key) + _rows("calendar", key, cost=110.0)
                  + _rows("velocity", key, cost=130.0, slo=0.9))
    learn = _write(tmp_path, case, "r30_d_rep_learn", _rows("offline", key) + _rows("calendar", key, cost=110.0)
                   + _rows("calendar_learn", key, cost=112.0))
    slo = _write(tmp_path, case, "r30_d_rep_slo95", _rows("offline", key) + _rows("velocity", key, cost=160.0, slo=0.99))
    groups = groups_of([base, learn, slo])
    assert len(groups) == 1
    frames = {r.run: load_run(r) for r in (base, learn, slo)}
    merged = assemble(groups[0], frames)
    policies = set(merged["policy"])
    assert policies == {"offline", "calendar", "velocity", "calendar_learn", f"velocity{SLO_SUFFIX}"}
    assert (merged[merged["policy"] == "offline"]["run"] == "r30_d_rep").all()
    cells = cell_means(merged)
    table = wide(cells, "usd_wk").set_index("policy")
    assert table.loc["calendar_learn", "C"] == pytest.approx(112.0 / 6)
    assert table.loc[f"velocity{SLO_SUFFIX}", "C"] == pytest.approx(160.0 / 6)
    assert same_rows(frames["r30_d_rep"], frames["r30_d_rep_learn"], "calendar") is True
    assert same_rows(frames["r30_d_rep"], frames["r30_d_rep_slo95"], "calendar") is None


def test_compare_and_frontier(tmp_path: Path) -> None:
    key = list(CELLS)[2]
    rows = (_rows("offline", key, gpu=1008.0) + _rows("calendar_learn", key, cost=110.0, gpu=1020.0, slo=0.99)
            + _rows("velocity", key, cost=130.0, gpu=1080.0, slo=0.96)
            + _rows("reactive", key, cost=140.0, gpu=1200.0, slo=0.999))
    run = _write(tmp_path, "c3_72b_shared4k", "r30_d_inst_t8", rows)
    cells = cell_means(load_run(run))
    comp = compare(cells, "C", "velocity").set_index("policy")
    assert comp.loc["calendar_learn", "d_gpu_h_wk"] == pytest.approx(-10.0)
    assert comp.loc["calendar_learn", "dominates"]
    assert not comp.loc["reactive", "dominates"]
    front = frontier_row(cells, "C", 0.95)
    assert front["timetable_best"] == "calendar_learn" and front["feedback_best"] == "velocity"
    assert front["saving_pct"] == pytest.approx(100.0 * (1 - 170.0 / 180.0))
    assert frontier_row(cells, "C", 0.995)["timetable_best"] is None
