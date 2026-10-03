"""Finding and loading the validation round: cases, runs, timing cells, weekly metrics.

The validation round (``scripts/sim/cluster/jobs.py``) writes one ``offset`` sweep per
``<case>/<run>/offset/offset.csv``.  A run name follows the grammar
``r<ratio>_<tier>[_t8][_slo95 | _learn | _heldout_sem2]``: the price ratio, the provisioning tier,
``_t8`` when the feedback baselines were tuned at the instance tier (8 min) rather than at the
replica tier, and a variant - the SLO-matched baselines (``slo95``: floor tuning on the synchronous
slice; ``slo95disp``: on a slice with the measured dispersion), the learned-timing planner alone, or
the other semester's timetable.  ``parse_run`` turns a name into a ``RunSpec`` and ``discover`` finds
every run under a root, whether the boxes' mirrors (``box<k>/validation/<case>/<run>``) or a flat
``<case>/<run>`` layout.

Every metric the report shows is per teaching week, because a run covers six of them: GPU-hours
per week, USD per week, and the USD a policy paid over the offline optimum per week.  The optimum is
the ``offline`` row of the same cell and seed, so a run without offline rows falls back to the
``offline_cost`` column that ``run_sweep`` writes on every row.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEEK_HOURS: float = 168.0
DEFAULT_C_RUN: float = 0.0333  # USD per replica-minute of the round
BASE_USD_PER_WEEK: float = DEFAULT_C_RUN * 60.0 * WEEK_HOURS  # one always-on replica: 335.7
OFFLINE: str = "offline"
CELL_KEY: Tuple[str, ...] = ("trigger_offset_mean", "trigger_offset_sd", "trigger_offset_course_sd",
                             "trigger_offset_common_sd", "forecast_offset", "forecast_offset_sd")
RUN_RE = re.compile(r"^r(?P<ratio>\d+)_(?P<tier>d_rep|d_inst)(?P<t8>_t8)?"
                    r"(?:_(?P<variant>slo95disp|slo95|learnprior|learn|heldout_sem2_learnprior|heldout_sem2))?$")


@dataclass(frozen=True)
class Case:
    """One capacity case of batch B4: model, prompt regime, SLO and replica-tier lead."""

    name: str
    label: str
    slo_s: float
    lead_rep_min: float
    central: bool


CASES: Dict[str, Case] = {
    "c1_7b_shared4k": Case("c1_7b_shared4k", "7B, shared 4k prefix", 2.0, 0.6, False),
    "c2_32b_shared4k": Case("c2_32b_shared4k", "32B-AWQ, shared 4k prefix", 2.0, 1.0, True),
    "c3_72b_shared4k": Case("c3_72b_shared4k", "72B-AWQ, shared 4k prefix", 2.0, 1.5, True),
    "c4_32b_unique2k": Case("c4_32b_unique2k", "32B-AWQ, unique 2k prompts", 5.0, 1.0, False),
    "c5_72b_unique4k": Case("c5_72b_unique4k", "72B-AWQ, unique 4k prompts", 5.0, 1.5, False),
}
CASE_ORDER: Tuple[str, ...] = tuple(CASES)


@dataclass(frozen=True)
class Cell:
    """One timing cell of the offset sweep: what the bursts do and what the planner assumes."""

    letter: str
    short: str
    description: str


CELLS: Dict[Tuple[float, ...], Cell] = {
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0): Cell(
        "A", "sync", "bursts at the bell, planner at the bell (no timing error)"),
    (-8.7, 5.3, 4.0, 0.0, 0.0, 0.0): Cell(
        "B", "indep/bell", "independent dispersion (session sd 5.3, section sd 4.0), "
        "planner at the bell"),
    (-8.7, 5.3, 4.0, 0.0, -8.7, 6.6): Cell(
        "C", "indep/dist", "independent dispersion, planner knows the offset distribution "
        "(-8.7, sd 6.6)"),
    (-8.7, 0.0, 0.0, 5.3, 0.0, 0.0): Cell(
        "D", "common/bell", "common shock per bell (sd 5.3), planner at the bell"),
    (-8.7, 0.0, 0.0, 5.3, -8.7, 5.3): Cell(
        "E", "common/dist", "common shock per bell, planner knows the distribution (-8.7, sd 5.3)"),
    (-8.7, 3.7, 0.0, 3.7, -8.7, 5.3): Cell(
        "F", "mixed/dist", "half independent, half common (3.7 each), planner knows the "
        "distribution"),
}
CELL_ORDER: Tuple[str, ...] = tuple(c.letter for c in CELLS.values())
REALISTIC_CELLS: Tuple[str, ...] = ("B", "C")  # the measured structure is independent dispersion


@dataclass(frozen=True)
class RunSpec:
    """The parsed run name."""

    ratio: int
    tier: str  # d_rep | d_inst
    tuned_at: str  # d_rep | d_inst: the tier the feedback baselines were tuned at
    variant: str  # main | slo95 | slo95disp | learn | learnprior | heldout_sem2[_learnprior]

    @property
    def is_learn_only(self) -> bool:
        """A run that carries only learned-timing planners (plus offline), to be merged into its group."""
        return "learn" in self.variant

    @property
    def group(self) -> Tuple[int, str, str, str]:
        """Runs that share a group describe the same arrival stream and tuning."""
        slots = "heldout_sem2" if self.variant.startswith("heldout") else "main"
        return (self.ratio, self.tier, self.tuned_at, slots)

    @property
    def group_label(self) -> str:
        """``r30 d_inst, baselines tuned at d_inst, timetable 2026-2027-1``."""
        slots = ("held-out timetable 2025-2026-2" if self.variant.startswith("heldout")
                 else "timetable 2026-2027-1")
        return f"r = {self.ratio}, {self.tier}, baselines tuned at {self.tuned_at}, {slots}"


def parse_run(name: str) -> Optional[RunSpec]:
    """``r30_d_inst_t8_slo95`` -> ``RunSpec(30, 'd_inst', 'd_inst', 'slo95')``; None if not a run."""
    m = RUN_RE.match(name)
    if not m:
        return None
    tier = m.group("tier")
    tuned_at = "d_inst" if m.group("t8") else "d_rep"
    return RunSpec(int(m.group("ratio")), tier, tuned_at, m.group("variant") or "main")


@dataclass(frozen=True)
class RunFile:
    """One sweep CSV of the round."""

    case: str
    run: str
    spec: RunSpec
    path: Path


def discover(root: Path) -> List[RunFile]:
    """Every ``<case>/<run>/offset/offset.csv`` under ``root`` (any depth), one per (case, run).

    The same (case, run) can exist in several result trees: the boxes' mirrors, and a re-run tree
    whose name sorts later (``validation2`` after ``validation``).  The run from the tree whose
    name sorts last wins, so a re-run supersedes the original; the run it replaced is logged.
    """
    root = Path(root)
    found: Dict[Tuple[str, str], RunFile] = {}
    for path in sorted(root.glob("**/offset/offset.csv"), key=lambda p: (p.parents[3].name, str(p))):
        run_dir = path.parent.parent
        case = run_dir.parent.name
        spec = parse_run(run_dir.name)
        if case not in CASES or spec is None:
            continue
        key = (case, run_dir.name)
        if key in found:
            logger.info("run %s/%s: %s supersedes %s", case, run_dir.name, path, found[key].path)
        found[key] = RunFile(case, run_dir.name, spec, path)
    if not found:
        logger.warning("no validation runs under %s", root)
    return [found[k] for k in sorted(found, key=lambda k: (CASE_ORDER.index(k[0]), k[1]))]


def expected_runs(case: str) -> List[str]:
    """The runs the results note is built from: both tiers with baselines tuned at their own tier.

    Per price ratio the full policy list at the replica tier (re-run tree) and at the instance tier
    with the 8-minute tuning; the SLO-matched feedback baselines at both; and, for the central
    cases, the other semester's timetable at both.  The round-1 runs of the instance tier with the
    replica-tier tuning (``r*_d_inst``, ``r30_d_inst_slo95``, ``r30_d_inst_heldout_sem2``) and the
    ``_learn`` runs are sensitivity material: reported when present, never required.
    """
    central = CASES[case].central
    ratios = (30, 100) if central else (30,)
    names: List[str] = []
    for r in ratios:
        names += [f"r{r}_d_rep", f"r{r}_d_inst_t8"]
    names += ["r30_d_rep_slo95", "r30_d_inst_t8_slo95", "r30_d_inst_t8_slo95disp"]
    if central:
        names += ["r30_d_rep_heldout_sem2", "r30_d_inst_t8_heldout_sem2", "r30_d_rep_slo95disp"]
    return sorted(names)


def cell_of(row: pd.Series) -> str:
    """The letter of the timing cell a row belongs to, ``?`` for an unknown tuple."""
    key = tuple(round(float(row[c]), 3) for c in CELL_KEY)
    cell = CELLS.get(key)
    return cell.letter if cell is not None else "?"


def _offline_total(frame: pd.DataFrame) -> pd.Series:
    """``total_cost`` of the offline row of each row's (cell, seed, week); ``offline_cost`` as fallback."""
    keys = ["cell", "seed", "week"]
    oracle = frame[frame["policy"] == OFFLINE]
    if oracle.empty:
        logger.warning("no offline rows: weekly excess uses the offline_cost column (week 0 only)")
        week = pd.to_numeric(frame["week"], errors="coerce")
        return pd.to_numeric(frame["offline_cost"], errors="coerce").where(week == 0)
    lookup = oracle.groupby(keys, as_index=False)["total_cost"].mean().rename(
        columns={"total_cost": "offline_total"})
    merged = frame[keys].merge(lookup, on=keys, how="left", sort=False)
    merged.index = frame.index
    return merged["offline_total"]


def load_run(run: RunFile) -> pd.DataFrame:
    """Read one sweep CSV and add ``case``, ``run``, ``cell`` and the per-week metrics.

    Whole-run rows (``week = 0``) cover ``weeks`` teaching weeks, weekly rows one; the per-week
    columns divide by that span so both kinds of row are comparable.
    """
    frame = pd.read_csv(run.path)
    frame["case"] = run.case
    frame["run"] = run.run
    frame["cell"] = frame.apply(cell_of, axis=1)
    unknown = int((frame["cell"] == "?").sum())
    if unknown:
        logger.warning("%s/%s: %d rows in an unknown timing cell", run.case, run.run, unknown)
    weeks = pd.to_numeric(frame["weeks"], errors="coerce").fillna(1.0)
    week = pd.to_numeric(frame["week"], errors="coerce").fillna(0.0)
    span = weeks.where(week == 0, 1.0)
    frame["span_weeks"] = span
    frame["offline_total"] = _offline_total(frame)
    frame["gpu_h_wk"] = frame["gpu_hours"] / span
    frame["usd_wk"] = frame["total_cost"] / span
    frame["excess_wk"] = (frame["total_cost"] - frame["offline_total"]) / span
    frame["starts_wk"] = frame["starts"] / span
    frame["gpu_h_above_base_wk"] = frame["gpu_h_wk"] - WEEK_HOURS
    return frame


def whole_run(frame: pd.DataFrame) -> pd.DataFrame:
    """The whole-run rows."""
    return frame[pd.to_numeric(frame["week"], errors="coerce") == 0]


def weekly(frame: pd.DataFrame) -> pd.DataFrame:
    """The per-teaching-week rows."""
    return frame[pd.to_numeric(frame["week"], errors="coerce") > 0]


def load_all(runs: Sequence[RunFile]) -> pd.DataFrame:
    """Every run in one long frame."""
    frames = [load_run(r) for r in runs]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def same_rows(a: pd.DataFrame, b: pd.DataFrame, policy: str, atol: float = 1e-9) -> Optional[bool]:
    """Whether two runs agree on ``policy``'s whole-run ``total_cost`` per (cell, seed).

    None when either run lacks the policy.  Used for the two invariants of the round: the
    offline optimum is the same in every run of a group (same stream), and the calendar rows of a
    ``_learn`` run equal those of its base run (the learned-timing code left the stream untouched).
    """
    cols = ["cell", "seed", "total_cost"]
    wa, wb = whole_run(a), whole_run(b)
    x = wa[wa["policy"] == policy][cols].sort_values(["cell", "seed"])
    y = wb[wb["policy"] == policy][cols].sort_values(["cell", "seed"])
    if x.empty or y.empty:
        return None
    if len(x) != len(y):
        return False
    return bool(np.allclose(x["total_cost"].to_numpy(), y["total_cost"].to_numpy(), atol=atol))


__all__ = ["BASE_USD_PER_WEEK", "CASES", "CASE_ORDER", "CELLS", "CELL_KEY", "CELL_ORDER", "OFFLINE",
           "REALISTIC_CELLS", "WEEK_HOURS", "Case", "Cell", "RunFile", "RunSpec", "cell_of",
           "discover", "expected_runs", "load_all", "load_run", "parse_run", "same_rows",
           "weekly", "whole_run"]
