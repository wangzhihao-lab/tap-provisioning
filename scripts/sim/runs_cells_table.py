"""The six timing cells A-F on the laptop's streams: every policy variant of one stream group.

    uv run python scripts/sim/runs_cells_table.py

Collects, per case x tier (c2 and c3, both tiers), the rows (seeds 10-14, six weeks, cells A-F) of
the four laptop runs of one stream group: the 13-policy run and the learned-prior run of the
cells-laptop queue (``runs/cells_laptop``, JOBS.txt: the cloud's six-cell commands run here),
the R1 TAP job (tap, offline) and section 3's six-week rerun of the resource-rule floor rules
(``tuning_resource/<case>_<tier>/eval``).  A run is read only if its job finished (DONE marker,
exit 0) and its check passed against the R1 job of the same case and tier (``invariant.json``:
status verified), and only if the offline rows of the four runs are equal on the same (timing
cell, seed, week) keys in every column (``canonical_sensitivity.check_offline``).  A policy two
runs hold must have the same rows in both, every column; it is read from the first run that holds
it (R1 job, 13-policy run, learned-prior run, rerun) and its note names the other.  Variants are
named as in ``paired_cell_c.csv``: a reactive / velocity rule of a cells-laptop run by the
``--tuned`` of its job's START line, the rules of section 3's rerun "floor (dispersed slice)",
every other policy "-".  Cell C must be the rows of ``paired_cell_c.csv`` of the same name in
every column but source and note, unrounded, whole run and each week.  Anything else raises.

Writes ``results/runs/cells_laptop.csv`` (long: the columns of ``paired_cell_c.csv``
plus ``cell``; week 0 = the six weeks) and ``cells_laptop.md`` (whole run, GPU-hours per week /
in-class attainment to 1 and 3 decimals, with the runs and the checks).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
REPO = PILOT
sys.path.insert(0, str(HERE))
from canonical_cells_laptop import CELLS_CSV, TUNED_VARIANTS, job_argv, sweep_frame  # noqa: E402
from canonical_extra import (RUNS, FLOOR_DISPERSED, PAIRED_CSV, TUNING_RESOURCE,  # noqa: E402
                             WEEKS)
from canonical_sensitivity import (KEYS as ROW_KEYS, NOT_COMPARED, SEEDS, TAP_JOBS, Run,  # noqa: E402
                                   check_offline)
from runs_groups import flag_value  # noqa: E402
from runs_paired_table import FEEDBACK, GROUPS, decision  # noqa: E402

logger = logging.getLogger("runs_cells_table")

CELLS_JOBS = f"{RUNS}/cells_laptop"  # repository-relative, as every folder here
CELLS: Tuple[str, ...] = tuple("ABCDEF")
RUN_NAMES: Dict[str, str] = {"inst": "r30_d_inst_t8", "rep": "r30_d_rep"}  # the cloud's run names
KINDS: Tuple[Tuple[str, str], ...] = (("", "cells-laptop run, 13 policies, --tuned {tuned}"),
                                      ("_learnprior", "cells-laptop run, learned-timing prior"))
KEYS: Tuple[str, ...] = ("case", "tier", "cell", "policy", "variant", "week")
VALUES: Tuple[str, ...] = ("seeds", "gpu_h_per_week", "starts_per_week", "usd_per_week", "slo_in",
                           "slo_in_min")
Source = Tuple[str, Dict[str, str], str]  # (run folder, {policy: variant}, note)


def _pilot(folder: str) -> str:
    """A repository-relative path as the runs and checks name it (relative to the repository root)."""
    return folder


def _finished(logs: Path, job: str) -> None:
    """Raise unless ``job`` has a DONE marker that records exit 0."""
    marker = logs / f"{job}.DONE"
    if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")).get("exit") != 0:
        raise ValueError(f"{marker}: missing, or the job did not exit 0")


def _pair_csv(case: str, tier: str) -> str:
    return f"{TAP_JOBS}/r1_{case[:2]}_{tier}/offset/offset.csv"


def _verified(repo: Path, folder: str, pair: str) -> None:
    """Raise unless the cells-laptop check of ``folder`` passed against the R1 job's ``pair``."""
    report = json.loads((repo / folder / "invariant.json").read_text(encoding="utf-8"))
    if (report.get("status") != "verified" or report.get("failures")
            or report.get("new") != _pilot(f"{folder}/offset/offset.csv")
            or report.get("pair") != _pilot(pair) or not report["offline_vs_pair"]["equal"]):
        raise ValueError(f"{folder}/invariant.json: the check did not pass against {pair}")


def cells_runs(repo: Path, case: str, tier: str) -> List[Source]:
    """The two cells-laptop runs of a case x tier with every policy but the offline optimum, a
    reactive / velocity rule named by the ``--tuned`` of its job's START line; raises unless the
    job finished, ran with ``--out`` this folder and passed its check against the R1 job."""
    logs = repo / CELLS_JOBS / "logs"
    out: List[Source] = []
    for suffix, note in KINDS:
        job = f"{case}_{RUN_NAMES[tier]}{suffix}"
        folder = f"{CELLS_JOBS}/{job}"
        argv = job_argv(repo, job, logs)
        _finished(logs, job)
        _verified(repo, folder, _pair_csv(case, tier))
        tuned = flag_value(argv, "--tuned")
        if tuned is None or flag_value(argv, "--out") != _pilot(folder):
            raise ValueError(f"{job}: no --tuned, or --out is not {_pilot(folder)}")
        variant = TUNED_VARIANTS[Path(tuned).parent.name]
        policies = pd.read_csv(repo / folder / "offset" / "offset.csv", usecols=["policy"])["policy"]
        out.append((folder, {p: variant if p in FEEDBACK else "-" for p in policies.unique()
                             if p != "offline"}, note.format(tuned=Path(tuned).parent.name)))
    return out


def group_sources(repo: Path, case: str, tier: str) -> List[Source]:
    """Every run of a case x tier in reading order: R1 job, cells-laptop runs, section 3's rerun."""
    tap_job = f"r1_{case[:2]}_{tier}"
    _finished(repo / RUNS / "logs", tap_job)
    decided = decision(case, tier)
    if decided is None or not decided["rerun"]:
        raise ValueError(f"{case} {tier}: section 3 has not rerun the floor rules")
    rerun = f"{TUNING_RESOURCE}/{case}_{tier}/eval"
    _finished(repo / RUNS / "logs", f"s3_eval_{case[:2]}_{tier}")
    report = json.loads((repo / rerun / "invariant.json").read_text(encoding="utf-8"))
    if report.get("status") != "verified" or not any(
            o["path"] == _pilot(_pair_csv(case, tier)) and o["platform"] == "laptop" and o["equal"]
            for o in report.get("originals", [])):
        raise ValueError(f"{rerun}/invariant.json: the check did not pass against {tap_job}")
    out: List[Source] = [(f"{TAP_JOBS}/{tap_job}", {"tap": "-"}, "R1 job"),
                         (f"{TAP_JOBS}/{tap_job}", {"offline": "-"},
                          "R1 job; every run of the group has these offline rows")]
    out += cells_runs(repo, case, tier)
    changed = [str(p) for p in decided["changed"]]  # type: ignore[attr-defined]
    out += [(rerun, {p: FLOOR_DISPERSED}, "section-3 resource rule (rerun; configuration "
             + ("changed)" if p in changed else "kept)")) for p in FEEDBACK]
    return out


def _rows(frame: pd.DataFrame, policy: str, where: str) -> pd.DataFrame:
    """One policy's rows in file order; raises unless every (cell, seed, week) is there once."""
    rows = frame[frame["policy"] == policy].drop(columns=["sweep"])
    keys = rows[["cell", "seed", "week"]].drop_duplicates()
    if (len(keys) != len(rows) or len(rows) != len(CELLS) * len(SEEDS) * (WEEKS + 1)
            or set(rows["cell"]) != set(CELLS) or set(rows["seed"]) != set(SEEDS)
            or set(rows["week"]) != set(range(WEEKS + 1)) or (rows["weeks"] != WEEKS).any()):
        raise ValueError(f"{where}: {policy} does not hold cells A-F x seeds 10-14 x weeks 0-6 once")
    return rows


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """The two row sets are equal in every column (NaN equal to NaN), in (cell, seed, week) order."""
    a, b = (x.sort_values(["cell", "seed", "week"]).reset_index(drop=True) for x in (a, b))
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    return bool(((a == b) | (a.isna() & b.isna())).all().all())


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Means over the seeds per cell and week, as ``runs_paired_table.collect`` computes them."""
    per = rows["weeks"].where(rows["week"] == 0, 1).astype(float)  # whole-run rows -> per week
    rows = rows.assign(gpu_h_per_week=rows["gpu_hours"] / per, starts_per_week=rows["starts"] / per,
                       usd_per_week=(rows["run_cost"] + rows["switch_cost"]) / per)
    return (rows.groupby(["cell", "week"], as_index=False)
            .agg(seeds=("seed", "nunique"), gpu_h_per_week=("gpu_h_per_week", "mean"),
                 starts_per_week=("starts_per_week", "mean"), usd_per_week=("usd_per_week", "mean"),
                 slo_in=("slo_in", "mean"), slo_in_min=("slo_in", "min")))


def collect(repo: Path, case: str, tier: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """The long table of one case x tier and what was checked on the way."""
    sources = group_sources(repo, case, tier)
    frames = {f: sweep_frame(repo, f"{f}/offset/offset.csv") for f in dict.fromkeys(s[0] for s in sources)}
    anchor, *others = frames
    check_offline(Run(case, anchor.rsplit("/", 1)[1], TAP_JOBS, anchor.rsplit("/", 1)[1]),
                  frames[anchor], {f: frames[f] for f in others})
    picked: Dict[Tuple[str, str], Tuple[pd.DataFrame, str, str, List[str]]] = {}
    for folder, wanted, note in sources:
        for policy, variant in wanted.items():
            rows = _rows(frames[folder], policy, folder)
            if (policy, variant) not in picked:
                picked[(policy, variant)] = (rows, folder, note, [])
            elif _same(picked[(policy, variant)][0], rows):
                picked[(policy, variant)][3].append(folder.rsplit("/", 1)[1])
            else:
                raise ValueError(f"{case} {tier}: {policy} ({variant}) differs between "
                                 f"{picked[(policy, variant)][1]} and {folder}")
    parts = []
    for (policy, variant), (rows, folder, note, also) in sorted(picked.items()):
        extra = f"; the same rows in {', '.join(also)}" if also else ""
        parts.append(aggregate(rows).assign(case=case, tier=tier, policy=policy, variant=variant,
                                            source=_pilot(f"{folder}/offset/offset.csv"),
                                            note=note + extra))
    offline = frames[anchor][frames[anchor]["policy"] == "offline"]
    checks = {"offline_rows": len(offline), "offline_runs": len(others),
              "offline_columns": len([c for c in offline.columns
                                      if c not in ROW_KEYS and c not in NOT_COMPARED])}
    return pd.concat(parts, ignore_index=True), checks


def check_paired(repo: Path, table: pd.DataFrame) -> Dict[str, int]:
    """Raise unless every cell-C row is the row of ``paired_cell_c.csv`` of the same case, tier,
    policy, variant and week in every column but source and note; returns the rows compared."""
    paired = pd.read_csv(repo / PAIRED_CSV, float_precision="round_trip")
    on = [k for k in KEYS if k != "cell"]
    merged = table[table["cell"] == "C"].merge(paired, on=on, how="left", suffixes=("", "__paired"),
                                               indicator=True)
    lacking = merged[merged["_merge"] != "both"]
    if not lacking.empty:
        raise ValueError(f"cell C rows without a {PAIRED_CSV} row: "
                         f"{lacking[on].drop_duplicates().to_dict('records')[:5]}")
    for col in VALUES:
        bad = merged[merged[col] != merged[f"{col}__paired"]]
        if not bad.empty:
            raise ValueError(f"cell C {col} differs from {PAIRED_CSV}: "
                             f"{bad[on + [col, col + '__paired']].to_dict('records')[:5]}")
    return {"whole_run": int((merged["week"] == 0).sum()), "weekly": int((merged["week"] > 0).sum())}


def summary(table: pd.DataFrame, checks: Dict[Tuple[str, str], Dict[str, int]],
            paired: Dict[str, int]) -> str:
    """The markdown summary: per case x tier the runs, the checks and the whole-run table."""
    lines = ["# Timing cells A-F on the laptop's streams: every policy variant of one stream group",
             "", (__doc__ or "").split("\n\n")[2].strip(), "",
             f"Cell C equals `{PAIRED_CSV}`: {paired['whole_run']} whole-run and {paired['weekly']} "
             "weekly rows compared, every column but source and note, unrounded.", ""]
    for (case, tier), part in table.groupby(["case", "tier"], sort=False):
        c = checks[(str(case), str(tier))]
        lines += [f"## {case}, {'instance tier (d = 8)' if tier == 'inst' else 'replica tier'}", "",
                  f"Offline rows of {c['offline_runs']} runs equal to the R1 job's: {c['offline_rows']} "
                  f"rows x {c['offline_columns']} columns.", "",
                  "| policy | variant | " + " | ".join(CELLS) + " | source | note |",
                  "|---|---|" + "---|" * len(CELLS) + "---|---|"]
        whole = part[part["week"] == 0]
        for (policy, variant), rows in whole.groupby(["policy", "variant"], sort=False):
            by = rows.set_index("cell")
            vals = [f"{by.loc[x, 'gpu_h_per_week']:.1f} / {by.loc[x, 'slo_in']:.3f}" for x in CELLS]
            lines.append(f"| {policy} | {variant} | " + " | ".join(vals)
                         + f" | `{rows['source'].iloc[0]}` | {rows['note'].iloc[0]} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Write the CSV and the summary; raise on any failed check rather than write a partial table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parts, checks = [], {}
    for case, tier, _lead in GROUPS:
        part, checks[(case, tier)] = collect(REPO, case, tier)
        parts.append(part)
        logger.info("%s %s: %d rows, offline rows of %d runs equal to the R1 job's", case, tier,
                    len(part), checks[(case, tier)]["offline_runs"])
    table = pd.concat(parts, ignore_index=True)
    group = {(case, tier): i for i, (case, tier, _lead) in enumerate(GROUPS)}
    table = (table.assign(_group=[group[k] for k in zip(table["case"], table["tier"])])
             .sort_values(["_group", "cell", "policy", "variant", "week"], kind="stable")
             [[*KEYS, "source", "note", *VALUES]].reset_index(drop=True))
    paired = check_paired(REPO, table)
    out = REPO / CELLS_CSV
    table.to_csv(out, index=False)
    back = pd.read_csv(out, float_precision="round_trip")
    if not back.equals(table):
        raise ValueError(f"{out} does not read back as written")
    out.with_suffix(".md").write_text(summary(table, checks, paired), encoding="utf-8")
    logger.info("wrote %s (%d rows); cell C: %d whole-run and %d weekly rows equal to %s", out,
                len(table), paired["whole_run"], paired["weekly"], PAIRED_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
