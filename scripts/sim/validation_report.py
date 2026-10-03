"""Results tables of the validation round: five capacity cases x two tiers x six timing cells.

    uv run python scripts/sim/validation_report.py                 # outputs/validation_remote -> docs
    uv run python scripts/sim/validation_report.py --root outputs/validation --out /tmp/v

Reads every ``<case>/<run>/offset/offset.csv`` under ``--root`` (the boxes' mirrors or a flat
layout), groups the runs by stream and tuning (``pilot.validation``) and writes, under ``--out``:

* ``verification`` - which expected runs are present, their shape, and the round's invariants
  (offline optimum identical across the runs of a group; the calendar rows of a ``_learn`` run
  identical to its base run; no policy cheaper than the optimum; no run below the base tier);
* ``tuning`` - the baseline configurations selected per case and tier, and whether the SLO floor
  of the matched selection was reachable at all;
* ``groups/<case>_<group>`` - per group, policies x timing cells for GPU-hours per week, in-class
  SLO, USD per week and USD over the optimum per week;
* ``cells_long`` - the same numbers as one long CSV (means and sds over seeds);
* ``headline`` - in the two realistic cells (independent dispersion, planner at the bell / with
  the learned distribution), every policy against the re-tuned velocity baseline, plus the
  cheapest timetable vs cheapest feedback policy at common SLO floors;
* ``weekly_learn`` - the learning curve of the learned-timing planner by teaching week;
* ``heldout`` - the central cases on the other semester's timetable next to the in-sample run.

Exit status 1 when an invariant fails or an expected run is missing, so the note cannot be built
on a partial round without noticing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))

from pilot.batchc.render import Provenance, markdown_table, now_stamp, write_table  # noqa: E402
from pilot.validation import (  # noqa: E402
    BASE_USD_PER_WEEK,
    CASE_ORDER,
    CASES,
    CELL_ORDER,
    CELLS,
    METRIC_TITLES,
    METRICS,
    REALISTIC_CELLS,
    WEEK_HOURS,
    Group,
    RunFile,
    assemble,
    cell_means,
    compare,
    discover,
    discover_tuning,
    expected_runs,
    frontier_row,
    groups_of,
    load_run,
    same_rows,
    steady_state,
    tuning_rows,
    weekly_table,
    whole_run,
    wide,
)

logger = logging.getLogger("validation_report")

WEEKLY_METRICS = ("slo_in", "gpu_hours")
LEARN_POLICIES = ("offline", "calendar", "calendar_learn", "calendar_learn_prior", "calendar_adapt",
                  "stale_calendar", "forecast_ilp", "guard", "velocity", "reactive")
SLO_FLOORS = (0.95, 0.99)
DECIMALS: Dict[str, int] = {"gpu_h_wk": 1, "usd_wk": 0, "excess_wk": 1, "slo_in": 3, "x_peak": 1,
                            "starts_wk": 0, "ttft_p95_in_s": 2, "d_gpu_h_wk": 1, "d_gpu_pct": 1,
                            "d_slo": 3, "d_usd_wk": 0, "saving_pct": 1, "gpu_hours": 1,
                            "timetable_gpu_h_wk": 1, "feedback_gpu_h_wk": 1, "timetable_slo": 3,
                            "feedback_slo": 3, "slo_floor": 2, "max_slo_tune": 3}


def fmt(column: str, value: object) -> str:
    """Fixed decimals per metric; text passes through; NaN is ``n/a``."""
    if isinstance(value, str):
        return value
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    base = column[:-5] if column.endswith("_mean") else column
    return f"{number:.{DECIMALS.get(base, 3)}f}"


def to_markdown(frame: pd.DataFrame) -> str:
    """A frame as a Markdown table with the report's number formats."""
    if frame is None or frame.empty:
        return "_(no rows)_"
    header = [str(c) for c in frame.columns]
    rows = [[fmt(str(c), row[c]) for c in frame.columns] for _, row in frame.iterrows()]
    return markdown_table(header, rows)


def base_formula() -> str:
    """The per-week conventions, for the provenance block."""
    return (f"Per-week metrics: a run covers `weeks` teaching weeks and every whole-run number is "
            f"divided by it; one always-on replica is {WEEK_HOURS:.0f} GPU-hours and "
            f"{BASE_USD_PER_WEEK:.1f} USD per week (c_run = 0.0333 USD per replica-minute). "
            f"`excess_wk = (total_cost - offline_total) / weeks` with the optimum of the same "
            f"cell and seed.")


class Report:
    """Everything the CLI writes, sharing the loaded frames."""

    def __init__(self, root: Path, out: Path) -> None:
        self.root, self.out = root, out
        self.stamp = now_stamp()
        self.runs: List[RunFile] = discover(root)
        self.frames: Dict[str, Dict[str, pd.DataFrame]] = {}
        for run in self.runs:
            self.frames.setdefault(run.case, {})[run.run] = load_run(run)
        self.groups: List[Group] = [self._without_redundant_learn(g) for g in groups_of(self.runs)]
        self.assembled: Dict[Group, pd.DataFrame] = {
            g: assemble(g, self.frames.get(g.case, {})) for g in self.groups}
        self.cells: Dict[Group, pd.DataFrame] = {g: cell_means(f) for g, f in self.assembled.items()}
        self.problems: List[str] = []

    def _without_redundant_learn(self, group: Group) -> Group:
        """Drop a learn-only run whose policies the base run already carries (a re-run)."""
        base = [r for r in group.runs if r.spec.variant in ("main", "heldout_sem2")]
        if not base:
            return group
        have = set(self.frames[group.case][base[0].run]["policy"])
        kept = []
        for r in group.runs:
            if r.spec.is_learn_only:
                extra = set(self.frames[group.case][r.run]["policy"]) - have - {"offline"}
                if not extra:
                    logger.info("%s/%s: superseded by %s", group.case, r.run, base[0].run)
                    continue
            kept.append(r)
        return Group(group.case, group.ratio, group.tier, group.tuned_at, group.slots, tuple(kept))

    # -- writing helpers -------------------------------------------------------------------
    def provenance(self, title: str, sources: Sequence[Path], notes: Sequence[str] = ()) -> Provenance:
        rel = []
        for p in sources:
            try:
                rel.append(str(Path(p).resolve().relative_to(PILOT)))
            except ValueError:
                rel.append(str(p))
        return Provenance(title, tuple(rel), base_formula(), self.stamp, tuple(n for n in notes if n))

    def write(self, name: str, table: pd.DataFrame, title: str, sources: Sequence[Path],
              notes: Sequence[str] = (), markdown: Optional[str] = None) -> None:
        write_table(self.out, name, table, self.provenance(title, sources, notes),
                    markdown if markdown is not None else to_markdown(table))

    # -- verification ------------------------------------------------------------------------
    def write_verification(self) -> None:
        rows: List[Dict[str, object]] = []
        have = {(r.case, r.run): r for r in self.runs}
        for case in CASE_ORDER:
            for name in expected_runs(case):
                run = have.get((case, name))
                if run is None:
                    rows.append({"case": case, "run": name, "found": False})
                    self.problems.append(f"missing run {case}/{name}")
                    continue
                frame = self.frames[case][name]
                w = whole_run(frame)
                offline = w[w["policy"] == "offline"].set_index(["cell", "seed"])["total_cost"]
                joined = w.join(offline.rename("opt"), on=["cell", "seed"])
                cheaper = int((joined["total_cost"] < joined["opt"] - 1e-6).sum())
                below_base = int((w["gpu_h_wk"] < WEEK_HOURS - 1e-6).sum())
                if cheaper:
                    self.problems.append(f"{case}/{name}: {cheaper} rows cheaper than the optimum")
                if below_base:
                    self.problems.append(f"{case}/{name}: {below_base} rows below the base tier")
                rows.append({"case": case, "run": name, "found": True,
                             "tree": run.path.parents[3].name, "n_rows": len(frame),
                             "n_whole": len(w), "cells": "".join(sorted(w["cell"].unique())),
                             "n_seeds": int(w["seed"].nunique()),
                             "n_policies": int(w["policy"].nunique()),
                             "mu": float(w["mu"].mean()), "lead_min": float(w["lead_min"].iloc[0]),
                             "price_ratio_real": float(w["price_ratio_real"].mean()),
                             "cheaper_than_opt": cheaper, "below_base": below_base})
        table = pd.DataFrame(rows)
        checks = self._group_checks()
        notes = ["`cheaper_than_opt` and `below_base` must be 0.",
                 "Group checks (offline optimum identical across the runs of a group; calendar rows "
                 "of the `_learn` run identical to the base run):", *[f"- {c}" for c in checks]]
        self.write("verification", table, "Validation round: coverage and invariants",
                   [r.path for r in self.runs], notes)

    def _group_checks(self) -> List[str]:
        out: List[str] = []
        for g in self.groups:
            frames = self.frames[g.case]
            base = [r for r in g.runs if r.spec.variant in ("main", "heldout_sem2")]
            for r in g.runs:
                if not base or r.run == base[0].run:
                    continue
                same = same_rows(frames[base[0].run], frames[r.run], "offline")
                text = "n/a" if same is None else ("ok" if same else "DIFFER")
                out.append(f"{g.case} {base[0].run} vs {r.run}: offline {text}")
                if same is False:
                    self.problems.append(f"{g.case}: offline differs between {base[0].run} and {r.run}")
                if r.spec.is_learn_only:
                    for pol in ("calendar", "calendar_adapt"):
                        same = same_rows(frames[base[0].run], frames[r.run], pol)
                        text = "n/a" if same is None else ("ok" if same else "DIFFER")
                        out[-1] += f"; {pol} {text}"
                        if same is False:
                            self.problems.append(f"{g.case}: {pol} differs between {base[0].run} and {r.run}")
                    # calendar_learn's forecast draws were process-salted before 2026-09-21: report, do not flag
                    out[-1] += f"; calendar_learn max rel. diff {self._max_rel_diff(frames[base[0].run], frames[r.run], 'calendar_learn')}"
        return out

    @staticmethod
    def _max_rel_diff(a: pd.DataFrame, b: pd.DataFrame, policy: str) -> str:
        wa, wb = whole_run(a), whole_run(b)
        x = wa[wa["policy"] == policy].set_index(["cell", "seed"])["total_cost"]
        y = wb[wb["policy"] == policy].set_index(["cell", "seed"])["total_cost"]
        if x.empty or y.empty:
            return "n/a"
        joined = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
        return f"{float((joined['b'] / joined['a'] - 1.0).abs().max()):.2e}" if not joined.empty else "n/a"

    # -- tuning ------------------------------------------------------------------------------
    def write_tuning(self) -> None:
        paths = discover_tuning(self.root)
        table = tuning_rows(paths)
        notes = ["`floor_reachable` is whether any grid point reached `slo_floor` on the tuning "
                 "seeds; when it is `no`, the SLO-matched selection fell back to the cheapest point "
                 "and equals the free selection."]
        self.write("tuning", table, "Validation round: baseline tuning", paths, notes)

    # -- per group tables --------------------------------------------------------------------
    def write_groups(self) -> pd.DataFrame:
        long: List[pd.DataFrame] = []
        cell_note = "Cells: " + "; ".join(f"**{c.letter}** {c.short} = {c.description}"
                                          for c in CELLS.values()) + "."
        for g in self.groups:
            cells = self.cells[g]
            if cells.empty:
                continue
            part = cells.copy()
            for k, v in zip(("case", "ratio", "tier", "tuned_at", "slots"), g.key):
                part.insert(0, k, v)
            long.append(part)
            parts: List[str] = [cell_note, ""]
            for metric in METRICS + ("x_peak",):
                parts += [f"## {METRIC_TITLES[metric]}", "", to_markdown(wide(cells, metric)), ""]
            self.write(f"groups/{g.case}_{g.name}", cells,
                       f"{CASES[g.case].label}: {g.label}", [r.path for r in g.runs],
                       markdown="\n".join(parts))
        table = pd.concat(long, ignore_index=True) if long else pd.DataFrame()
        self.write("cells_long", table, "Validation round: every policy x cell x group",
                   [r.path for r in self.runs], markdown="_(see the CSV)_")
        return table

    # -- headline ----------------------------------------------------------------------------
    def write_headline(self) -> None:
        rows: List[pd.DataFrame] = []
        fronts: List[Dict[str, object]] = []
        for g in self.groups:
            if g.tier != g.tuned_at or g.slots != "main":
                continue  # the properly tuned groups only
            cells = self.cells[g]
            for cell in REALISTIC_CELLS:
                comp = compare(cells, cell, "velocity")
                if comp.empty:
                    continue
                for k, v in zip(("case", "ratio", "tier"), (g.case, g.ratio, g.tier)):
                    comp.insert(0, k, v)
                rows.append(comp)
                for floor in SLO_FLOORS:
                    fronts.append({"case": g.case, "ratio": g.ratio, "tier": g.tier,
                                   **frontier_row(cells, cell, floor)})
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        front = pd.DataFrame(fronts)
        parts = ["## Every policy against the re-tuned velocity baseline", "",
                 "`d_*` columns are the policy minus velocity; `dominates` = no more GPU-hours and "
                 "no less SLO than velocity.", ""]
        for (case, ratio, tier, cell), part in table.groupby(["case", "ratio", "tier", "cell"], sort=False):
            parts += [f"### {CASES[case].label}, r = {ratio}, {tier}, cell {cell} "
                      f"({CELLS_BY_LETTER[cell].short})", "",
                      to_markdown(part.drop(columns=["case", "ratio", "tier", "cell", "reference"])), ""]
        parts += ["## Cheapest timetable policy vs cheapest feedback policy at a common SLO floor",
                  "", "Timetable = calendar / calendar_adapt / calendar_learn; feedback = velocity / "
                  "reactive, free or SLO-matched tuning. `n/a` = no policy of that family reaches "
                  "the floor in that cell.", "", to_markdown(front), ""]
        self.write("headline", table, "Validation round: headline comparison in the realistic cells",
                   [r.path for r in self.runs], markdown="\n".join(parts))
        self.write("frontier", front, "Validation round: timetable vs feedback at common SLO floors",
                   [r.path for r in self.runs])

    # -- weekly learning curve -------------------------------------------------------------
    def write_weekly(self) -> None:
        parts: List[str] = ["Cell C (independent dispersion, planner with the distribution); "
                            "`calendar_learn` learns the timing from strictly earlier weeks, so week "
                            "1 is at the bell and weeks 2-6 are its held-out test.", ""]
        tables: List[pd.DataFrame] = []
        for g in self.groups:
            if g.tier != g.tuned_at or g.slots != "main":
                continue
            frame = self.assembled[g]
            if "calendar_learn" not in set(frame["policy"]):
                continue
            for metric in WEEKLY_METRICS:
                t = weekly_table(frame, "C", LEARN_POLICIES, metric)
                if t.empty:
                    continue
                parts += [f"## {CASES[g.case].label}, {g.label}: {metric}", "", to_markdown(t), ""]
                t = t.copy()
                t.insert(0, "metric", metric)
                for k, v in zip(("case", "ratio", "tier"), (g.case, g.ratio, g.tier)):
                    t.insert(0, k, v)
                tables.append(t)
        table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        self.write("weekly_learn", table, "Validation round: learned timing by teaching week",
                   [r.path for r in self.runs], markdown="\n".join(parts))

    # -- weeks 2-6 only ----------------------------------------------------------------------
    def write_steady(self) -> None:
        parts: List[str] = ["Means over teaching weeks 2-6 (per-week rows), so the learned-timing "
                            "planner is read on its held-out weeks and every other policy on the same "
                            "weeks. Cells B and C only.", ""]
        tables: List[pd.DataFrame] = []
        for g in self.groups:
            if g.tier != g.tuned_at or g.slots != "main":
                continue
            table = steady_state(self.assembled[g])
            if table.empty:
                continue
            table = table[table["cell"].isin(REALISTIC_CELLS)]
            for cell in REALISTIC_CELLS:
                part = table[table["cell"] == cell].drop(columns=["cell", "weeks"])
                if part.empty:
                    continue
                parts += [f"## {CASES[g.case].label}, {g.label}, cell {cell} "
                          f"({CELLS_BY_LETTER[cell].short}), weeks 2-6", "",
                          to_markdown(part[["policy", "gpu_h_wk_mean", "slo_in_mean", "usd_wk_mean"]]), ""]
            for k, v in zip(("case", "ratio", "tier"), (g.case, g.ratio, g.tier)):
                table.insert(0, k, v)
            tables.append(table)
        out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        self.write("steady_state", out, "Validation round: weeks 2-6 in the realistic cells",
                   [r.path for r in self.runs], markdown="\n".join(parts))

    # -- held-out semester -----------------------------------------------------------------
    def write_heldout(self) -> None:
        rows: List[pd.DataFrame] = []
        for g in self.groups:
            if g.slots != "heldout_sem2":
                continue
            twin = next((h for h in self.groups if h.key == (g.case, g.ratio, g.tier, g.tuned_at, "main")), None)
            for label, grp in (("in-sample 2026-2027-1", twin), ("held-out 2025-2026-2", g)):
                if grp is None:
                    continue
                cells = self.cells[grp]
                part = cells[cells["cell"].isin(REALISTIC_CELLS)][
                    ["policy", "cell", "gpu_h_wk_mean", "slo_in_mean", "usd_wk_mean", "excess_wk_mean"]].copy()
                for k, v in zip(("timetable", "tuned_at", "tier", "ratio", "case"),
                                (label, grp.tuned_at, grp.tier, grp.ratio, grp.case)):
                    part.insert(0, k, v)
                rows.append(part)
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        self.write("heldout", table, "Validation round: the other semester's timetable",
                   [r.path for r in self.runs if r.spec.variant == "heldout_sem2"],
                   ["The held-out run reuses the tuning of the in-sample run; the timetable, and "
                    "with it every session, is the other semester's."])

    def run(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "groups").mkdir(exist_ok=True)
        self.write_verification()
        self.write_tuning()
        self.write_groups()
        self.write_headline()
        self.write_weekly()
        self.write_steady()
        self.write_heldout()
        for problem in self.problems:
            logger.error(problem)
        logger.info("%d runs, %d groups -> %s (%d problem(s))", len(self.runs), len(self.groups),
                    self.out, len(self.problems))
        return 1 if self.problems else 0


CELLS_BY_LETTER = {c.letter: c for c in CELLS.values()}
assert tuple(CELLS_BY_LETTER) == CELL_ORDER


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description="Validation-round results tables")
    ap.add_argument("--root", type=Path, default=PILOT / "outputs" / "validation_remote")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "sim" / "validation" / "tables")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    return Report(Path(args.root), Path(args.out)).run()


if __name__ == "__main__":
    raise SystemExit(main())
