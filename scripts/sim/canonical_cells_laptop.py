"""Canonical numbers, part 6: the six timing cells on the laptop's streams (sections 7 and 32).

TAP and the offline optimum come from the R1 jobs (``runs/tap/r1_<case>_<tier>``: cells A-F at
r = 30), TAP's phase share, re-solves and final size ratio from each job's ``diagnostics.csv``; the
same readers serve the R7 jobs at r = 100 (``canonical_r100_laptop``, section 33).  The baselines
come from every laptop run of the job's stream group: in cells A-F section 3's six-week rerun of
the resource-rule floor rules (``tuning_resource/<case>_<tier>/eval``) and the two runs of the
cells-laptop queue (``runs/cells_laptop``, ``runs_cells_table.cells_runs``), and in cell
C also the runs section 27a draws on (``runs_paired_table.sources``) and the main-caliber
block of the billing round (``billing/<case>_<tier>/ref_main``).  A run enters only if its offline
rows equal the TAP job's on the same (timing cell, seed, week) keys in every column
(``canonical_sensitivity.check_offline``); cells are the letters of ``SweepConfig.trigger_offset``
(``run_sweep.cell_letter``).  A row that two runs produce seed for seed in every column is listed
once with both sources; two different rows under one name raise.  Cloud runs of the same arguments
draw other streams (JOBS_README.md, Platform) and never enter.  Section 7 prints
``cells_laptop.csv`` (``runs_cells_table.py``), which must hold the rows of section 32 of the
same cell and name.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from canonical_extra import (RUNS, LAPTOP_POLICIES, LAPTOP_TIERS, PAIRED_CSV, SLO_FLOOR,
                             VARIANT_WORDS, WEEKS, _variant_words)
from canonical_laptop import FEEDBACK, PLANNERS
from canonical_sensitivity import (LOCAL, OFFSET_KEYS, SEEDS, TAP_JOBS, Run, Stats, cell_stats,
                                   check_offline)

logger = logging.getLogger("canonical_cells_laptop")

NEAR = 0.01  # a mean attainment this close to the floor is also given to 4 decimals
TUNED_VARIANTS: Dict[str, str] = {  # --tuned folder -> variant, as in runs_paired_table
    "tuning8_r30": "cost (tier's own)", "tuning_r30": "cost (replica-tier tuning)",
    "tuning_r30_slo95": "floor (synchronous slice)"}
KEPT_NOTE = ", rows equal the laptop original's"  # true of cell C only, the original's one cell
CELLS_CSV = "results/runs/cells_laptop.csv"  # section 7, from runs_cells_table.py
Source = Tuple[str, Dict[str, str], str]  # (run folder, {policy: variant}, note)


@dataclass(frozen=True, eq=False)
class TapJob:
    """A finished TAP job: its rows with their cell letter, and TAP's rows of diagnostics.csv."""

    name: str  # e.g. "r1_c2_inst"
    case: str  # e.g. "c2_32b_shared4k"
    tier: str  # "inst" | "rep"
    frame: pd.DataFrame
    diag: pd.DataFrame

    @property
    def folder(self) -> str:
        return f"{TAP_JOBS}/{self.name}"


@dataclass(frozen=True, eq=False)
class Baseline:
    """One laptop baseline row of one cell: its rows (every seed and week), every run giving it."""

    policy: str
    variant: str
    rows: pd.DataFrame
    stats: Stats
    sources: Tuple[str, ...]
    notes: Tuple[str, ...]


@dataclass(frozen=True, eq=False)
class CellTables:
    """Section 32: TAP per cell, every paired laptop baseline row, the rows no laptop run holds."""

    tap: pd.DataFrame
    baselines: pd.DataFrame
    missing: pd.DataFrame
    cell_c_checked: int  # rows compared with section 27a (paired_cell_c.csv)
    cells_checked: int  # rows compared with section 7 (cells_laptop.csv)


def sweep_frame(repo: Path, csv: str) -> pd.DataFrame:
    """Every row of a sweep CSV (floats parsed exactly) with its timing-cell letter."""
    from run_sweep import cell_letter  # scripts/sim

    frame = pd.read_csv(repo / csv, float_precision="round_trip")
    letters = [cell_letter(r) for r in frame[list(OFFSET_KEYS)].to_dict("records")]
    if "" in letters:
        raise ValueError(f"{csv}: rows outside the timing cells of SweepConfig.trigger_offset")
    return frame.assign(cell=letters)


def _whole(frame: pd.DataFrame, where: str) -> pd.DataFrame:
    """Whole-run rows of seeds 10-14; raises unless each (cell, policy) has one row per seed."""
    whole = frame[(frame["week"] == 0) & frame["seed"].isin(SEEDS)]
    if (whole["weeks"] != WEEKS).any():
        raise ValueError(f"{where}: a run of other than {WEEKS} weeks")
    for key, part in whole.groupby(["cell", "policy"]):
        if tuple(sorted(int(s) for s in part["seed"])) != SEEDS:
            raise ValueError(f"{where}: {key} does not hold one row per seed of {SEEDS}")
    return whole


def tap_jobs(repo: Path, prefix: str) -> List[TapJob]:
    """The finished TAP jobs ``<prefix>_<case>_<tier>`` ("r1" or "r7"), in case and tier order."""
    from runs_groups import CASES  # scripts/sim

    jobs: List[TapJob] = []
    for folder in sorted((repo / TAP_JOBS).glob(f"{prefix}_*")):
        parts = folder.name.split("_")
        if len(parts) != 3 or parts[1] not in CASES or parts[2] not in LAPTOP_TIERS:
            continue
        if not (repo / RUNS / "logs" / f"{folder.name}.DONE").exists():
            logger.info("%s is not finished, skipped", folder.name)
            continue
        diag = pd.read_csv(folder / "diagnostics.csv", float_precision="round_trip")
        jobs.append(TapJob(folder.name, CASES[parts[1]].name, parts[2],
                           sweep_frame(repo, f"{TAP_JOBS}/{folder.name}/offset/offset.csv"),
                           diag[diag["policy"] == "tap"]))
    return jobs


def job_stats(job: TapJob) -> Dict[str, Dict[str, Stats]]:
    """Per cell, the whole-run statistics of the job's policies (offline and tap)."""
    whole = _whole(job.frame, job.folder).assign(source=job.folder)
    return {str(cell): cell_stats(part) for cell, part in whole.groupby("cell")}


def _attainment(s: Stats) -> Dict[str, str]:
    """Mean and lowest-seed in-class attainment (3 decimals, 4 near the floor) and the verdict."""
    near = abs(s.slo_in - SLO_FLOOR) <= NEAR
    return {"slo_in": f"{s.slo_in:.3f}",
            "slo_in_4dp_if_within_0.01_of_0.95": f"{s.slo_in:.4f}" if near else "",
            "lowest_seed": f"{s.slo_min:.3f}",
            "lowest_seed_4dp_if_within_0.01_of_0.95": f"{s.slo_min:.4f}" if near else "",
            "reaches_0.95": "yes" if s.slo_in >= SLO_FLOOR else "no"}


def platform_of(folder: str) -> str:
    if not folder.startswith(f"{LOCAL}/"):
        raise ValueError(f"{folder}: not a laptop run")
    return "laptop"


def tap_row(job: TapJob, cell: str, stats: Dict[str, Stats],
             labels: Mapping[str, str]) -> Dict[str, object]:
    """TAP's numbers and diagnostics and the offline optimum in one cell of a TAP job."""
    tap, off = stats["tap"], stats["offline"]
    diag = job.diag[job.diag["cell"] == cell]
    if tuple(sorted(int(s) for s in diag["seed"])) != SEEDS:
        raise ValueError(f"{job.folder}/diagnostics.csv: cell {cell} lacks a TAP row per seed")
    return {"case": labels.get(job.case, job.case), "tier": LAPTOP_TIERS[job.tier],
            "lead_min": f"{float(job.frame['lead_min'].iloc[0]):g}", "cell": cell,
            "tap_gpu_h_wk": f"{tap.gpu:.1f}", **_attainment(tap),
            "phase_share_min": f"{float(diag['phase_share'].min()):.6f}",
            "n_replans_min / max": f"{int(diag['n_replans'].min())} / "
                                   f"{int(diag['n_replans'].max())}",
            "final_ratio_min / max": f"{float(diag['final_ratio'].min()):.3f} / "
                                     f"{float(diag['final_ratio'].max()):.3f}",
            "eps": f"{float(diag['trust'].iloc[0]):g}",
            "offline_gpu_h_wk": f"{off.gpu:.1f}", "offline_slo_in": f"{off.slo_in:.3f}",
            "seeds": tap.seeds, "platform": platform_of(job.folder),
            "source": f"{job.folder}/offset/offset.csv; {job.folder}/diagnostics.csv"}


def job_argv(repo: Path, job: str, logs: Optional[Path] = None) -> List[str]:
    """The run command of a finished job (the START line of its log), split; the log is read from
    ``logs``, by default the rerun round's (``runs/logs``)."""
    logs = repo / RUNS / "logs" if logs is None else logs
    starts = [ln for ln in (logs / f"{job}.log").read_text(encoding="utf-8").splitlines()
              if ln.startswith("START ")]
    if len(starts) != 1 or not (logs / f"{job}.DONE").exists():
        raise ValueError(f"{job}: {len(starts)} START lines, or no DONE marker")
    return shlex.split(starts[0].split(" ", 3)[3].split(" && ")[0])  # START <date> <time> <run>


def _sources(repo: Path, job: TapJob) -> List[Source]:
    """Every laptop baseline run of a TAP job's stream group (none for a group without one)."""
    import runs_paired_table as pt  # scripts/sim

    lead = {(case, tier): lead for case, tier, lead in pt.GROUPS}.get((job.case, job.tier))
    if lead is None:
        return []
    out: List[Source] = []
    for path, wanted, note in pt.sources(job.case, job.tier, lead):
        keep = {p: v for p, v in wanted.items() if p not in ("tap", "offline")}
        if keep:
            out.append((str(path.parents[1].relative_to(repo)), keep, note))
    ref = f"{RUNS}/billing/{job.case}_{job.tier}/ref_main"
    if (repo / ref / "offset" / "offset.csv").exists():
        from runs_groups import flag_value  # scripts/sim

        tuned = flag_value(job_argv(repo, f"s4_ref_main_{job.name.split('_')[1]}_{job.tier}"),
                           "--tuned")
        if tuned is None:
            raise ValueError(f"{ref}: its job ran without --tuned")
        variant = TUNED_VARIANTS[Path(tuned).parent.name]
        policies = pd.read_csv(repo / ref / "offset" / "offset.csv", usecols=["policy"])["policy"]
        out.append((ref, {p: variant if p in FEEDBACK else "-" for p in policies.unique()
                          if p != "offline"}, "billing round, main caliber (s4_ref_main)"))
    from runs_cells_table import cells_runs  # scripts/sim; it imports this module
    for folder, wanted, note in cells_runs(repo, job.case, job.tier):
        out.append((folder, {p: v for p, v in wanted.items() if p not in ("tap", "offline")}, note))
    return out


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """The two row sets are equal in every column (NaN equal to NaN)."""
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    return bool(((a == b) | (a.isna() & b.isna())).all().all())


def _baselines(repo: Path, job: TapJob) -> Dict[Tuple[str, str], Baseline]:
    """Every laptop baseline row of a TAP job's streams, keyed by (cell, name)."""
    frames: Dict[str, pd.DataFrame] = {}
    out: Dict[Tuple[str, str], Baseline] = {}
    for folder, wanted, note in _sources(repo, job):
        platform_of(folder)
        if folder not in frames:
            frame = sweep_frame(repo, f"{folder}/offset/offset.csv")
            base = job.frame[job.frame["cell"].isin(set(frame["cell"]))]
            check_offline(Run(job.case, job.name, TAP_JOBS, job.name), base, {folder: frame})
            frames[folder] = frame
        frame = frames[folder]
        whole = _whole(frame, folder).assign(source=folder)
        for policy, variant in wanted.items():
            name = _variant_words(policy, variant, job.tier)
            for cell in sorted(set(frame["cell"])):
                rows = frame[(frame["cell"] == cell) & (frame["policy"] == policy)]
                if rows.empty:
                    raise ValueError(f"{folder}: no {policy} rows in cell {cell}")
                rows = rows.drop(columns=["sweep"]).sort_values(["seed", "week"])
                rows = rows.reset_index(drop=True)
                here = note if cell == "C" else note.replace(KEPT_NOTE, "")
                prev = out.get((cell, name))
                if prev is None:
                    part = whole[(whole["cell"] == cell) & (whole["policy"] == policy)]
                    out[(cell, name)] = Baseline(policy, variant, rows, cell_stats(part)[policy],
                                                 (folder,), (here,))
                elif _same(prev.rows, rows):
                    out[(cell, name)] = replace(prev, sources=prev.sources + (folder,),
                                                notes=prev.notes + (here,))
                else:
                    raise ValueError(f"{job.name}, cell {cell}: {name} differs between "
                                     f"{prev.sources} and {folder}")
    return out


def to_end(frame: pd.DataFrame, cols: Sequence[str] = ("platform", "source")) -> pd.DataFrame:
    """``frame`` with ``cols`` moved to the end."""
    return frame[[c for c in frame.columns if c not in cols] + [c for c in cols if c in frame]]


def _rank(policy: str, variant: str) -> Tuple[int, int]:
    """Section 27a's order: policy, then tuning variant."""
    policies, variants = list(LAPTOP_POLICIES), list(VARIANT_WORDS)
    return (policies.index(policy) if policy in policies else len(policies),
            variants.index(variant) if variant in variants else -1)


def _order(b: Baseline) -> Tuple[int, int]:
    return _rank(b.policy, b.variant)


def _group(policy: str) -> str:
    """The groups of section 27d."""
    return "planner" if policy in PLANNERS else "feedback rule" if policy in FEEDBACK else "other"


def _baseline_row(where: Dict[str, object], name: str, b: Baseline,
                  tap: Stats) -> Dict[str, object]:
    """A row of section 32b: the baseline against TAP in the same cell."""
    s = b.stats
    return {**where, "group": _group(b.policy), "policy": name, "gpu_h_wk": f"{s.gpu:.1f}",
            **_attainment(s), "tap_gpu_h_wk": f"{tap.gpu:.1f}", "tap_slo_in": f"{tap.slo_in:.3f}",
            "extra_over_tap_pct": f"{100.0 * (s.gpu / tap.gpu - 1.0):+.1f}",
            "extra_over_tap_pct_int": f"{100.0 * (s.gpu / tap.gpu - 1.0):+.0f}",
            "seeds": s.seeds,
            "platform": "; ".join(dict.fromkeys(platform_of(f) for f in b.sources)),
            "note": "; ".join(dict.fromkeys(b.notes)),
            "source": "; ".join(f"{f}/offset/offset.csv" for f in b.sources)}


def _missing(repo: Path, job: TapJob, cells: Sequence[str],
             found: Dict[Tuple[str, str], Baseline],
             where: Dict[str, object]) -> List[Dict[str, object]]:
    """The cells where a policy variant of a TAP job's streams has no laptop row, one row per set
    of cells (every baseline, for streams without a laptop baseline run)."""
    if not found:
        report = json.loads((repo / job.folder / "invariant.json").read_text(encoding="utf-8"))
        mirrors = [o["platform"] for o in report.get("originals", [])]
        return [{**where, "cells_without_a_laptop_row": ", ".join(cells), "cells_with_one": "",
                 "policies": "every baseline", "laptop_row": "no laptop row",
                 "why": f"no laptop run of these streams besides the TAP job ({job.name}/"
                        f"invariant.json: {mirrors.count('laptop')} laptop and "
                        f"{mirrors.count('cloud')} cloud mirrors)"}]
    groups: Dict[Tuple[str, ...], List[str]] = {}
    for _, name in sorted(found, key=lambda k: _order(found[k])):
        lacking = tuple(c for c in cells if (c, name) not in found)
        if lacking and name not in groups.get(lacking, []):
            groups.setdefault(lacking, []).append(name)
    return [{**where, "cells_without_a_laptop_row": ", ".join(lacking),
             "cells_with_one": ", ".join(c for c in cells if c not in lacking),
             "policies": "; ".join(names), "laptop_row": "no laptop row",
             "why": "their laptop runs hold only the cells with one"}
            for lacking, names in groups.items()]


def _check_cell_c(repo: Path, job: TapJob, stats: Dict[str, Stats],
                  found: Dict[Tuple[str, str], Baseline]) -> int:
    """Raise unless cell C holds section 27a's rows of the case and tier (paired_cell_c.csv: the
    same names, GPU-hours per week and attainment, unrounded); returns the rows compared."""
    paired = pd.read_csv(repo / PAIRED_CSV, float_precision="round_trip")
    part = paired[(paired["case"] == job.case) & (paired["tier"] == job.tier)
                  & (paired["week"] == 0)]
    if part.empty:
        return 0
    mine = {name: b.stats for (cell, name), b in found.items() if cell == "C"}
    mine.update({p: stats[p] for p in ("tap", "offline")})
    names = [_variant_words(p, v, job.tier) for p, v in zip(part["policy"], part["variant"])]
    if sorted(mine) != sorted(names):
        raise ValueError(f"{job.name}, cell C: rows differ from {PAIRED_CSV}: "
                         f"{sorted(set(mine) ^ set(names))}")
    for name, gpu, slo in zip(names, part["gpu_h_per_week"], part["slo_in"]):
        if abs(float(gpu) - mine[name].gpu) > 1e-9 or abs(float(slo) - mine[name].slo_in) > 1e-12:
            raise ValueError(f"{job.name}, cell C: {name} is not the {PAIRED_CSV} row")
    return len(names)


def cells_csv(repo: Path) -> pd.DataFrame:
    """``cells_laptop.csv`` (floats parsed exactly) with each variant named as in section 27a."""
    table = pd.read_csv(repo / CELLS_CSV, float_precision="round_trip")
    return table.assign(name=[_variant_words(p, v, t) for p, v, t
                              in zip(table["policy"], table["variant"], table["tier"])])


def _check_cells_csv(repo: Path, job: TapJob, stats: Dict[str, Dict[str, Stats]],
                     found: Dict[Tuple[str, str], Baseline]) -> int:
    """Raise unless the whole-run rows of ``cells_laptop.csv`` (section 7) of the case and tier are
    the rows of the same cell and name here, TAP's and the offline optimum's included (GPU-hours per
    week and attainment unrounded, and as both sections print them), and every row here outside
    cell C is one of them; returns the rows compared."""
    table = cells_csv(repo)
    part = table[(table["case"] == job.case) & (table["tier"] == job.tier) & (table["week"] == 0)]
    if part.empty:
        return 0
    mine = {key: b.stats for key, b in found.items()}
    mine.update({(cell, p): by[p] for cell, by in stats.items() for p in ("tap", "offline")})
    theirs = {(c, n): (float(g), float(s)) for c, n, g, s
              in zip(part["cell"], part["name"], part["gpu_h_per_week"], part["slo_in"])}
    lacking = sorted(set(theirs) - set(mine)) + sorted(k for k in mine if k[0] != "C"
                                                       and k not in theirs)
    if lacking:
        raise ValueError(f"{job.name}: rows in only one of cells_laptop.csv and section 32: {lacking}")
    for (cell, name), (gpu, slo) in theirs.items():
        s = mine[(cell, name)]
        if (abs(gpu - s.gpu) > 1e-9 or abs(slo - s.slo_in) > 1e-12
                or f"{gpu:.1f} / {slo:.3f}" != f"{s.gpu:.1f} / {s.slo_in:.3f}"):
            raise ValueError(f"{job.name}, cell {cell}: {name} is not the cells_laptop.csv row")
    return len(theirs)


def cells_tables(repo: Path, labels: Mapping[str, str]) -> List[Tuple[str, pd.DataFrame]]:
    """Section 7a: per case and tier, every policy variant of ``cells_laptop.csv`` in cells A-F,
    whole run: GPU-hours per week (1 decimal) / in-class attainment (3 decimals), 27a's order."""
    whole = cells_csv(repo).query("week == 0")
    out: List[Tuple[str, pd.DataFrame]] = []
    for (case, tier), part in whole.groupby(["case", "tier"], sort=False):
        rows: List[Dict[str, object]] = []
        for name, block in sorted(part.groupby("name"), key=lambda item: _rank(
                str(item[1]["policy"].iloc[0]), str(item[1]["variant"].iloc[0]))):
            by = block.set_index("cell")
            if len(set(block["source"])) != 1:
                raise ValueError(f"{case} {tier}: {name} comes from more than one run")
            rows.append({"policy": name, **{c: f"{by.loc[c, 'gpu_h_per_week']:.1f} / "
                                               f"{by.loc[c, 'slo_in']:.3f}" for c in sorted(by.index)},
                         "run": block["source"].iloc[0], "source": CELLS_CSV})
        out.append((f"{labels.get(str(case), str(case))}, {LAPTOP_TIERS[str(tier)]}",
                    pd.DataFrame(rows)))
    return out


def cells_weekly_table(repo: Path, labels: Mapping[str, str]) -> pd.DataFrame:
    """Section 7b: in-class attainment by teaching week (3 decimals), cell C, instance tier, every
    policy variant of ``cells_laptop.csv``, 27a's order."""
    part = cells_csv(repo).query("cell == 'C' and tier == 'inst' and week > 0")
    rows: List[Dict[str, object]] = []
    for case, block in part.groupby("case", sort=False):
        for name, weeks in sorted(block.groupby("name"), key=lambda item: _rank(
                str(item[1]["policy"].iloc[0]), str(item[1]["variant"].iloc[0]))):
            by = weeks.set_index("week")["slo_in"]
            rows.append({"case": labels.get(str(case), str(case)), "tier": LAPTOP_TIERS["inst"],
                         "policy": name,
                         **{f"w{w}": f"{float(by.loc[w]):.3f}" for w in sorted(by.index)},
                         "source": CELLS_CSV})
    return pd.DataFrame(rows)


def six_cell_tables(repo: Path, labels: Mapping[str, str]) -> CellTables:
    """Section 32: TAP per timing cell of every R1 job and every laptop baseline row that pairs."""
    tap_rows: List[Dict[str, object]] = []
    base_rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    checked = cells = 0
    for job in tap_jobs(repo, "r1"):
        stats, found = job_stats(job), _baselines(repo, job)
        where: Dict[str, object] = {"case": labels.get(job.case, job.case),
                                    "tier": LAPTOP_TIERS[job.tier]}
        for cell, by in stats.items():
            here = sorted(((n, b) for (c, n), b in found.items() if c == cell),
                          key=lambda item: _order(item[1]))
            tap_rows.append({**tap_row(job, cell, by, labels), "laptop_baseline_rows": len(here)})
            base_rows += [_baseline_row({**where, "cell": cell}, n, b, by["tap"]) for n, b in here]
        missing += _missing(repo, job, list(stats), found, where)
        checked += _check_cell_c(repo, job, stats["C"], found)
        cells += _check_cells_csv(repo, job, stats, found)
    return CellTables(to_end(pd.DataFrame(tap_rows)), pd.DataFrame(base_rows),
                      pd.DataFrame(missing), checked, cells)


__all__ = ["CELLS_CSV", "CellTables", "TapJob", "cells_csv", "cells_tables", "cells_weekly_table",
           "job_argv", "job_stats", "platform_of", "six_cell_tables", "sweep_frame", "tap_jobs",
           "tap_row", "to_end"]
