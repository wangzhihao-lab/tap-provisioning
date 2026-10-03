"""Canonical numbers, part 5: the laptop sensitivity runs with the resource-rule floor rules, TAP.

The runs behind sections 14 and 18-23 of ``paper_numbers.md`` and the supplementary tables
``ext_ablation``, ``ext_sensitivity_cells``, ``ext_reservation``, ``ext_adoption`` and ``ext_heldout``;
``canonical_sensitivity_tables.py`` formats them.  Every sensitivity run this laptop made in the
sensitivity round (``validation_local/sensitivity_sweeps``) or for the priority ablation
(``validation_local/priority_ablation``) is read together with two runs of its stream group:

* the *original* gives the offline optimum and the calendar planners;
* its *s3 re-run* (``runs/s3_consistency/<run>``, ``runs_s3_jobs.py``) gives the
  SLO-floor reactive and velocity rules at the configurations section 3 selected by resource cost;
* the *TAP job* of the group (``runs/tap/<job>``, JOBS_README.md) gives ``tap``; the R1 jobs,
  which run cells A-F, hold the lead times 1 / 1.5 / 8 min and the priority-on arm.

A run is read only if both checks of the rerun round passed against this very original
(``invariant.json``: status ``verified``), the TAP job ran with the original's learned-timing prior
(the one argument TAP reads that the offline rows cannot reveal) and the offline rows of the three
are equal on the same (timing cell, seed, week) keys in every column of the original, the TAP job
restricted to the original's cells; anything else raises.  Rows are whole-run means over seeds
10-14 with the rounding of the sections: GPU-hours and USD per week to 1 decimal, attainment to 3
(and to 4 in the floor tables), percentages from unrounded means.  A rule reaches the floor when
its unrounded mean in-class attainment is at least 0.95.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from canonical_extra import ABLATION_CASES, RUNS, WEEKS

LOCAL = "outputs/validation_local"
SENSITIVITY_SWEEPS = f"{LOCAL}/sensitivity_sweeps"
ABLATION = f"{LOCAL}/priority_ablation"
S3 = f"{RUNS}/s3_consistency"
TAP_JOBS = f"{RUNS}/tap"
OFFSET_KEYS: Tuple[str, ...] = ("trigger_offset_mean", "trigger_offset_sd",
                                "trigger_offset_course_sd", "trigger_offset_common_sd",
                                "forecast_offset", "forecast_offset_sd")
KEYS: Tuple[str, ...] = (*OFFSET_KEYS, "seed", "week")
NOT_COMPARED: Tuple[str, ...] = ("sweep", "policy")
SEEDS: Tuple[int, ...] = (10, 11, 12, 13, 14)
FROM_ORIGINAL: Tuple[str, ...] = ("offline", "calendar", "calendar_learn_prior", "calendar_adapt")
FEEDBACK: Tuple[str, ...] = ("reactive", "velocity")  # from the s3 re-run
SHORT: Dict[str, str] = {"c2_32b_shared4k": "c2", "c3_72b_shared4k": "c3"}
REPLICA_LEAD: Dict[str, float] = {"c2_32b_shared4k": 1.0, "c3_72b_shared4k": 1.5}  # R1 replica jobs
INSTANCE_LEAD = 8.0  # the R1 instance jobs
ABLATION_MODES: Tuple[Tuple[str, str], ...] = (
    ("on", "graded penalty, in-class priority on (main setting)"),
    ("off", "graded penalty, in-class priority off"),
    ("ungraded", "p_off = p_in, in-class priority on"))
CELL_KEYS: Tuple[str, ...] = ("trigger_offset_sd", "trigger_offset_course_sd",
                              "trigger_offset_common_sd", "forecast_offset_sd")
# sensitivity-round timing cells, keyed by (session sd, course sd, common sd, forecast sd) as recorded
SENSITIVITY_CELLS: Dict[Tuple[float, ...], str] = {
    (4.59, 3.46, 3.32, 6.6): "G: common share 0.25",
    (3.75, 2.83, 4.70, 6.6): "H: common share 0.5",
    (2.65, 2.00, 5.75, 6.6): "I: common share 0.75",
    (0.0, 0.0, 6.64, 6.6): "J: common share 1.0",
    (5.3, 4.0, 0.0, 0.0): "K: mean offset known, no spread",
    (12.0, 4.0, 0.0, 12.6): "L: wide independent spread (session sd 12)",
}
Suffixes = Tuple[Tuple[str, Dict[str, str]], ...]
RESERVATION_RUNS: Suffixes = (
    ("res", {"tier": "reservation, 8 min lead, 60 min commit and billing blocks"}),)
ADOPTION_RUNS: Suffixes = tuple((f"adopt{a}", {"section_adoption": a})
                                for a in ("0.25", "0.5", "0.75"))
HELDOUT_RUNS: Suffixes = (
    ("heldout_sym", {"timetable": "2025-26 sem. 2", "prior_from": "2026-27 sem. 1 (symmetric)"}),
    ("heldout_fwd", {"timetable": "2025-26 sem. 2", "prior_from": "2025-26 sem. 1 (forward)"}))


@dataclass(frozen=True)
class Run:
    """A laptop sensitivity run; its folders, relative to the repository, follow from the names."""

    case: str
    name: str  # the folder name the original and its s3 re-run share
    root: str  # sensitivity_sweeps or priority_ablation
    tap_job: str

    @property
    def original(self) -> str:
        return f"{self.root}/{self.name}"

    @property
    def s3(self) -> str:
        return f"{S3}/{self.name}"

    @property
    def tap(self) -> str:
        return f"{TAP_JOBS}/{self.tap_job}"


@dataclass(frozen=True)
class Stats:
    """Whole-run means over the seeds of one policy in one timing cell, unrounded."""

    gpu: float  # GPU-hours per week
    usd: float  # total cost per week, penalties included
    slo_in: float
    slo_off: float
    slo_min: float  # the lowest seed's in-class attainment
    excess: float  # USD per week over the offline optimum of the same seed
    timeouts: float  # per million requests
    seeds: int
    source: str  # the run folder the rows came from


@dataclass(frozen=True)
class Block:
    """One timing cell of one run: the columns that name it and its per-policy statistics."""

    case_id: str  # e.g. "c2_32b_shared4k"
    key: Dict[str, str]
    stats: Dict[str, Stats]

    @property
    def case(self) -> str:
        """The case as the sections name it, e.g. "c2 32B shared-4k"."""
        return ABLATION_CASES[self.case_id]


def tap_job(case: str, suffix: str) -> str:
    """The TAP job whose stream group holds the sensitivity-round run ``<case>_<suffix>``."""
    short = SHORT[case]
    if suffix.startswith("lead_d"):
        lead = float(suffix[len("lead_d"):])
        if lead == REPLICA_LEAD[case]:
            return f"r1_{short}_rep"
        if lead == INSTANCE_LEAD:
            return f"r1_{short}_inst"
        return f"r3_{short}_d{suffix[len('lead_d'):]}"
    if suffix.startswith("adopt"):
        return f"r6_{short}_{suffix}"
    if suffix.startswith("heldout_"):
        return f"r2_{short}_inst_{suffix[len('heldout_'):]}"
    return {"cells_GL": f"r4_{short}_cellsGL", "res": f"r5_{short}_res"}[suffix]


def sweep_run(case: str, suffix: str) -> Run:
    """A sensitivity-round run with its s3 re-run and TAP job."""
    return Run(case, f"{case}_{suffix}", SENSITIVITY_SWEEPS, tap_job(case, suffix))


def ablation_run(case: str, mode: str) -> Run:
    """An arm of the priority ablation with its s3 re-run and TAP job (R1 for priority on)."""
    short = SHORT[case]
    return Run(case, f"{case}_{mode}", ABLATION,
               f"r1_{short}_inst" if mode == "on" else f"r8_{short}_{mode}")


def _read(repo: Path, folder: str) -> pd.DataFrame:
    """The offset sweep CSV of a run folder, floats parsed exactly."""
    return pd.read_csv(repo / folder / "offset" / "offset.csv", float_precision="round_trip")


def _tap_prior(repo: Path, job: str) -> Optional[str]:
    """The ``--learn-prior-from`` a TAP job ran with (the START line of its log)."""
    from runs_groups import flag_value  # scripts/sim

    log = repo / RUNS / "logs" / f"{job}.log"
    starts = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith("START ")]
    if len(starts) != 1 or not (repo / RUNS / "logs" / f"{job}.DONE").exists():
        raise ValueError(f"{log}: {len(starts)} START lines, or the job has no DONE marker")
    command = starts[0].split(" ", 3)[3].split(" && ")[0]  # START <date> <time> <run> && <check>
    return flag_value(shlex.split(command), "--learn-prior-from")


def verified(repo: Path, run: Run) -> None:
    """Raise unless the s3 and TAP checks passed against this original and TAP had its prior."""
    from runs_groups import flag_value, laptop_originals  # scripts/sim

    csv = f"{run.original}/offset/offset.csv"  # the checks name files from the repository root
    s3 = json.loads((repo / run.s3 / "invariant.json").read_text(encoding="utf-8"))
    if s3.get("status") != "verified" or s3.get("orig") != csv or not s3["offline"]["equal"]:
        raise ValueError(f"{run.s3}/invariant.json: the s3 check did not pass against {csv}")
    tap = json.loads((repo / run.tap / "invariant.json").read_text(encoding="utf-8"))
    passed = [o for o in tap.get("originals", [])
              if o["path"] == csv and o["platform"] == "laptop" and o["equal"]]
    if tap.get("status") != "verified" or len(passed) != 1:
        raise ValueError(f"{run.tap}/invariant.json: the TAP check did not pass against {csv}")
    argv = {o.csv: o.argv for o in laptop_originals()}[csv]
    if _tap_prior(repo, run.tap_job) != flag_value(argv, "--learn-prior-from"):
        raise ValueError(f"{run.tap}: TAP ran with another timing prior than {run.original}")


def _offline(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    """The offline rows of a run; a repeated (timing cell, seed, week) key raises."""
    rows = frame[frame["policy"] == "offline"]
    if rows.empty or rows.duplicated(list(KEYS)).any():
        raise ValueError(f"{where}: no offline rows, or two share a (timing cell, seed, week) key")
    return rows


def check_offline(run: Run, original: pd.DataFrame, others: Dict[str, pd.DataFrame]) -> None:
    """Raise unless the offline rows of every run of ``others`` (folder -> rows) are the
    original's: the same keys, every column of the original present and equal."""
    base = _offline(original, run.original)
    cols = [c for c in base.columns if c not in KEYS and c not in NOT_COMPARED]
    for where, frame in others.items():
        rows = _offline(frame, where)
        missing = [c for c in cols if c not in rows.columns]
        merged = base.merge(rows, on=list(KEYS), how="outer", suffixes=("", "__other"),
                            indicator=True)
        if missing or not (merged["_merge"] == "both").all():
            raise ValueError(f"{where}: offline keys or columns differ from {run.original} "
                             f"({missing})")
        bad = [c for c in cols if not ((merged[c] == merged[f"{c}__other"])
                                       | (merged[c].isna() & merged[f"{c}__other"].isna())).all()]
        if bad:
            raise ValueError(f"{where}: offline rows differ from {run.original} in {bad}")


def load(repo: Path, run: Run) -> pd.DataFrame:
    """Whole-run rows (week 0, seeds 10-14) of the policies the sections report, each with the
    folder it came from; raises unless the three runs pass the checks of the module docstring."""
    verified(repo, run)
    original, s3, tap = _read(repo, run.original), _read(repo, run.s3), _read(repo, run.tap)
    tap = tap.merge(original[list(OFFSET_KEYS)].drop_duplicates(), on=list(OFFSET_KEYS))
    check_offline(run, original, {run.s3: s3, run.tap: tap})
    rules = [p for p in FEEDBACK if p in set(original["policy"])]
    missing = [p for p in rules if p not in set(s3["policy"])]
    missing += [] if (tap["policy"] == "tap").any() else ["tap"]
    if missing:
        raise ValueError(f"{run.name}: no laptop rows of {missing}")
    frame = pd.concat([original[original["policy"].isin(FROM_ORIGINAL)].assign(source=run.original),
                       s3[s3["policy"].isin(rules)].assign(source=run.s3),
                       tap[tap["policy"] == "tap"].assign(source=run.tap)], ignore_index=True)
    whole = frame[(frame["week"] == 0) & frame["seed"].isin(SEEDS)]
    for key, part in whole.groupby([*OFFSET_KEYS, "policy"]):
        if tuple(sorted(int(s) for s in part["seed"])) != SEEDS:
            raise ValueError(f"{run.name}: {key} does not hold one row per seed of {SEEDS}")
    return whole


def cell_stats(cell: pd.DataFrame) -> Dict[str, Stats]:
    """Per policy of one timing cell of ``load``'s rows, in the order the rows came."""
    offline = cell[cell["policy"] == "offline"].set_index("seed")["total_cost"]
    out: Dict[str, Stats] = {}
    for policy, part in cell.groupby("policy", sort=False):
        part = part.set_index("seed")
        out[str(policy)] = Stats(
            gpu=float(part["gpu_hours"].mean() / WEEKS),
            usd=float(part["total_cost"].mean() / WEEKS),
            slo_in=float(part["slo_in"].mean()), slo_off=float(part["slo_off"].mean()),
            slo_min=float(part["slo_in"].min()),
            excess=float(((part["total_cost"] - offline.reindex(part.index)) / WEEKS).mean()),
            timeouts=float((part["timeouts"] / (part["n_in"] + part["n_off"])).mean() * 1e6),
            seeds=int(len(part)), source=str(part["source"].iloc[0]))
    return out


def _single(repo: Path, run: Run, key: Dict[str, str]) -> Block:
    """The block of a run that holds one timing cell."""
    whole = load(repo, run)
    if len(whole[list(OFFSET_KEYS)].drop_duplicates()) != 1:
        raise ValueError(f"{run.name}: more than one timing cell")
    return Block(run.case, dict(key), cell_stats(whole))


def _lead(name: str) -> float:
    return float(name.split("_lead_d")[1])


def ablation_blocks(repo: Path) -> List[Block]:
    """Section 14: one block per case and arm of the priority ablation."""
    return [_single(repo, ablation_run(case, mode), {"setting": setting})
            for case in SHORT for mode, setting in ABLATION_MODES]


def lead_blocks(repo: Path) -> List[Block]:
    """Section 18: one block per case and lead time (every ``sensitivity_sweeps/<case>_lead_d*``)."""
    blocks: List[Block] = []
    for case in SHORT:
        found = (repo / SENSITIVITY_SWEEPS).glob(f"{case}_lead_d*/offset/offset.csv")
        for name in sorted((p.parents[1].name for p in found), key=_lead):
            blocks.append(_single(repo, sweep_run(case, name[len(case) + 1:]),
                                  {"lead_min": f"{_lead(name):g}"}))
    return blocks


def cells_blocks(repo: Path) -> List[Block]:
    """Section 19: one block per case and timing cell G-L."""
    blocks: List[Block] = []
    for case in SHORT:
        whole = load(repo, sweep_run(case, "cells_GL"))
        for key, part in whole.groupby(list(CELL_KEYS)):
            name = SENSITIVITY_CELLS.get(tuple(round(float(k), 2) for k in key), f"cell {key}")
            blocks.append(Block(case, {"cell": name}, cell_stats(part)))
    return blocks


def sweep_blocks(repo: Path, suffixes: Suffixes) -> List[Block]:
    """Sections 20-22: one block per case and run suffix (reservation, adoption, held-out)."""
    return [_single(repo, sweep_run(case, suffix), extra)
            for case in SHORT for suffix, extra in suffixes]


PAIRED_TIERS: Dict[Tuple[str, str], str] = {  # (case, lead time) of section 18 -> tier of 27a
    ("c2_32b_shared4k", "1"): "rep", ("c3_72b_shared4k", "1.5"): "rep",
    ("c2_32b_shared4k", "8"): "inst", ("c3_72b_shared4k", "8"): "inst"}


def check_paired(repo: Path, blocks: Sequence[Block]) -> int:
    """Raise unless every row of section 18 at the two measured tiers is the row of the same name
    of section 27a (``paired_cell_c.csv``: GPU-hours per week and attainment, unrounded); returns
    the number of rows compared."""
    from canonical_extra import FLOOR_DISPERSED, PAIRED_CSV

    paired = pd.read_csv(repo / PAIRED_CSV, float_precision="round_trip").query("week == 0")
    compared = 0
    for b in blocks:
        tier = PAIRED_TIERS.get((b.case_id, b.key["lead_min"]))
        if tier is None:
            continue
        for policy, s in b.stats.items():
            variant = FLOOR_DISPERSED if policy in FEEDBACK else "-"
            hit = paired[(paired["case"] == b.case_id) & (paired["tier"] == tier)
                         & (paired["policy"] == policy) & (paired["variant"] == variant)]
            if (len(hit) != 1 or abs(float(hit["gpu_h_per_week"].iloc[0]) - s.gpu) > 1e-9
                    or abs(float(hit["slo_in"].iloc[0]) - s.slo_in) > 1e-12):
                raise ValueError(f"{b.case_id}, d = {b.key['lead_min']}: {policy} is not the "
                                 f"{PAIRED_CSV} row ({tier}, {variant})")
            compared += 1
    return compared


__all__ = ["ABLATION_MODES", "ADOPTION_RUNS", "Block", "FEEDBACK", "HELDOUT_RUNS",
           "RESERVATION_RUNS", "Run", "Stats", "ablation_blocks", "cells_blocks", "check_offline",
           "check_paired", "lead_blocks", "load", "sweep_blocks", "verified"]
