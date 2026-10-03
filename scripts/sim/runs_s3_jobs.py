"""Job list of the s3-consistency batch: the laptop sensitivity runs with the resource-rule baselines.

    uv run python scripts/sim/runs_s3_jobs.py

the rerun round re-selected the SLO-floor baselines (reactive, velocity) by resource cost
(``runs/tuning_resource/<case>_{rep,inst}/best.json``).  The laptop sensitivity runs of
2026-09-21 - ``sensitivity_sweeps`` (lead-time curve, cells G-L, reservation tier, adoption, held-out
semester) and ``priority_ablation`` (in-class priority on / off, ungraded penalties) - ran those
rules at the old floor configurations.  Each such run gets one job: its argument list
(``runs_groups.laptop_originals``, checked here against its CSV, its log and its chain log)
with ``--policies offline`` plus the feedback rules it ran, ``--tuned`` swapped for the
resource-rule ``best.json`` of the same case and tier, and ``--out s3_consistency/<its name>``,
followed by ``runs_s3_check.py``.  The calendar planners and TAP read no tuned knob, so they
are not re-run.

Every job's arguments are parsed as run_sweep.py parses them before anything is written: the stream
key and the timing cells must be the original's, and the run config may differ from the original's
only by the new ``best.json``.  Writes ``outputs/validation_local/runs/s3_consistency/JOBS.txt``
(run by ``runs_runner.py --jobs <it> --logs <its folder>/logs``).
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runs_groups import (CASES, LOCAL, PILOT, Original, cells_of,  # noqa: E402
                              check_laptop_reconstruction, flag_value, laptop_originals,
                              run_config, stream_key)
from runs_runner import parse  # noqa: E402
from tuned_config import read_tuned  # noqa: E402
from tuning_select import changed_policies  # noqa: E402

OUT = f"{LOCAL}/runs/s3_consistency"
LOGS = f"{OUT}/logs"
TUNING = f"{LOCAL}/runs/tuning_resource"
FLOOR_TIER = {"tuning_r30_slo95disp": "rep", "tuning8_r30_slo95disp": "inst"}  # old floor tunings
RULES = ("reactive", "velocity")
# wall of a re-run = the original's wall x 1.3: the queue's offline+reactive+velocity cell-C jobs
# under six workers (s4_eval_floor c2_inst / c3_inst / c3_rep: 911 / 1162 / 1091 s) against the
# five-policy originals of the same stream (sensitivity lead_d8 / lead_d8 / lead_d1.5: 760 / 798 / 838 s)
SCALE = 1.3
# peak RSS in GB, from the queue's TAP runs of the same streams (r3/r5/r8: 1.86-2.33, r2 held-out:
# 2.17-2.52, r6 adoption 0.25 / 0.5 / 0.75: 0.93-0.98 / 1.35-1.44 / 1.77-1.82)
MEM_GB = {"adopt0.25": 1.0, "adopt0.5": 1.5, "adopt0.75": 1.9, "heldout": 2.5, "other": 2.3}
APPLIED = re.compile(r"policy overrides applied: (\{.*\})")
SWEPT = re.compile(r"sweep offset: (\d+) rows \(\d+ whole-run\) in ([\d.]+) s")
CHAIN = re.compile(r"^\[([\d-]+ [\d:]+)\] (START|EXIT) (?:(\d+) )?(\S+) (\S+)$")


@dataclass(frozen=True)
class S3Job:
    """One re-run and the original it repeats."""

    job_id: str
    orig: Original
    old_tuned: str
    new_tuned: str
    policies: Tuple[str, ...]
    changed: Tuple[str, ...]
    argv: Tuple[str, ...]
    wall_min: float
    mem: float

    @property
    def out(self) -> str:
        """Output folder of the re-run."""
        return str(flag_value(self.argv, "--out"))

    def line(self) -> str:
        """``id after=- mem=.. est=.. :: run && check``, as runs_runner.py reads it."""
        check = ["--new", f"{self.out}/offset/offset.csv", "--orig", self.orig.csv, "--policies",
                 *self.policies, "--old-tuned", self.old_tuned, "--new-tuned", self.new_tuned,
                 "--report", f"{self.out}/invariant.json", "--log", f"{LOGS}/{self.job_id}.log"]
        command = (shlex.join(["uv", "run", "python", "scripts/sim/run_sweep.py", *self.argv]) + " && "
                   + shlex.join(["uv", "run", "python", "scripts/sim/runs_s3_check.py", *check]))
        return f"{self.job_id} after=- mem={self.mem:g} est={self.wall_min * SCALE:.0f} :: {command}"


def with_values(argv: Sequence[str], flag: str, values: Sequence[str]) -> List[str]:
    """``argv`` with the values of its single ``flag`` replaced in place."""
    at = [i for i, tok in enumerate(argv) if tok == flag]
    if len(at) != 1:
        raise ValueError(f"{flag} occurs {len(at)} times in {list(argv)}")
    stop = at[0] + 1
    while stop < len(argv) and not argv[stop].startswith("--"):
        stop += 1
    return [*argv[:at[0] + 1], *values, *argv[stop:]]


def values_of(argv: Sequence[str], flag: str) -> List[str]:
    """Every value after the single ``flag``."""
    at = argv.index(flag) + 1
    stop = at
    while stop < len(argv) and not argv[stop].startswith("--"):
        stop += 1
    return list(argv[at:stop])


def chain_wall(orig: Original, case: str, label: str) -> float:
    """Minutes between the run's START and its EXIT 0 in the chain log."""
    start = end = None
    for line in (PILOT / orig.out).parent.joinpath("logs", "chain.log").read_text().splitlines():
        m = CHAIN.match(line.strip())
        if m is None or m[4] != case or m[5] != label:
            continue
        when = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S")
        if m[2] == "START":
            start = when
        elif m[3] == "0":
            end = when
    if start is None or end is None:
        raise RuntimeError(f"{orig.out}: no START / EXIT 0 pair for '{case} {label}' in chain.log")
    return (end - start).total_seconds() / 60.0


def evidence(orig: Original, old_tuned: str) -> None:
    """The rebuilt arguments against the run's CSV and log; raises on any disagreement."""
    bad = check_laptop_reconstruction(orig)
    if bad:
        raise RuntimeError(f"{orig.out}: CSV columns {bad} disagree with the rebuilt arguments")
    log = (PILOT / orig.out).parent.joinpath("logs", f"{Path(orig.out).name}.log").read_text()
    applied, swept = APPLIED.findall(log), SWEPT.findall(log)
    if len(applied) != 1 or ast.literal_eval(applied[0]) != read_tuned(PILOT / old_tuned):
        raise RuntimeError(f"{orig.out}: its log did not apply {old_tuned}")
    frame = pd.read_csv(PILOT / orig.csv, usecols=["policy", "seed", "week"])
    args, cfg = run_config(orig.argv)
    checks = {"rows": (len(frame), int(swept[0][0]) if swept else -1),
              "policies": (sorted(set(frame["policy"])), sorted(args.policies)),
              "seeds": (sorted(set(frame["seed"])), sorted(args.seeds)),
              "weeks": (sorted(set(frame["week"])), list(range(cfg.workload.n_weeks + 1))),
              "cells": (cells_of(orig.argv, None), cells_of(["--sweep", "offset"], PILOT / orig.csv))}
    wrong = {k: v for k, v in checks.items() if v[0] != v[1]}
    if wrong:
        raise RuntimeError(f"{orig.out}: CSV / log disagree with the rebuilt arguments: {wrong}")


def dry_parse(job: S3Job) -> None:
    """The new arguments as run_sweep.py parses them: same stream, cells, config but the knobs."""
    _, old = run_config(job.orig.argv)
    args, new = run_config(job.argv)
    want = replace(old.policy, **read_tuned(PILOT / job.new_tuned))
    if (stream_key(job.argv) != stream_key(job.orig.argv) or cells_of(job.argv) != cells_of(job.orig.argv)
            or replace(new, policy=old.policy) != old or new.policy != want
            or tuple(args.policies) != job.policies):
        raise RuntimeError(f"{job.job_id}: the new arguments change more than the tuned knobs")


def build(orig: Original) -> S3Job:
    """The re-run of one original that ran a feedback rule at an old floor configuration."""
    name = Path(orig.out).name
    key, case = next((k, c) for k, c in CASES.items() if name.startswith(f"{c.name}_"))
    suffix = name[len(case.name) + 1:]
    old_tuned = str(flag_value(orig.argv, "--tuned"))
    tier = FLOOR_TIER[Path(old_tuned).parent.name]
    new_tuned = f"{TUNING}/{case.name}_{tier}/best.json"
    note = json.loads((PILOT / TUNING / f"{case.name}_{tier}" / "compare.json").read_text())
    old_cfg = json.loads((PILOT / old_tuned).read_text())
    changed = changed_policies(old_cfg, json.loads((PILOT / new_tuned).read_text()))
    if note["old"] != old_tuned or note["new"] != new_tuned or note["changed"] != changed:
        raise RuntimeError(f"{name}: {TUNING}/{case.name}_{tier}/compare.json disagrees")
    evidence(orig, old_tuned)
    rules = tuple(p for p in RULES if p in values_of(orig.argv, "--policies"))
    policies = ("offline", *rules)
    argv = with_values(with_values(with_values(orig.argv, "--policies", policies), "--tuned",
                                   [new_tuned]), "--out", [f"{OUT}/{name}"])
    ablation = "priority_ablation" in orig.out
    label = ({"on": "priority-on", "off": "priority-off"}.get(suffix, suffix) if ablation
             else suffix)
    group = "heldout" if suffix.startswith("heldout") else suffix if suffix in MEM_GB else "other"
    job = S3Job(f"s3c_{key}_{'prio_' if ablation else ''}{suffix}", orig, old_tuned, new_tuned,
                policies, tuple(changed), tuple(argv), chain_wall(orig, case.name, label),
                MEM_GB[group])
    dry_parse(job)
    return job


def main() -> int:
    """Build, verify and write the job list; print what was left out and why."""
    jobs: List[S3Job] = []
    skipped: Dict[str, str] = {}
    for orig in laptop_originals():
        tuned = flag_value(orig.argv, "--tuned")
        if not (PILOT / orig.csv).exists():
            skipped[orig.out] = "no CSV on disk"
        elif tuned is None or Path(tuned).parent.name not in FLOOR_TIER:
            skipped[orig.out] = f"not run at an old floor configuration ({tuned})"
        elif not set(RULES) & set(values_of(orig.argv, "--policies")):
            skipped[orig.out] = "runs no feedback rule"
        else:
            jobs.append(build(orig))
    jobs.sort(key=lambda j: (-j.wall_min, j.job_id))  # longest first: the shortest tail
    target = PILOT / OUT / "JOBS.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    head = ["# s3-consistency jobs: <id> after=<deps> mem=<GB> est=<min> :: <command>   (run from the repository root)",
            f"# written by scripts/sim/runs_s3_jobs.py; runner: runs_runner.py --jobs {OUT}/JOBS.txt "
            f"--logs {LOGS}"]
    target.write_text("\n".join(head + [j.line() for j in jobs]) + "\n", encoding="utf-8")
    parsed = parse(target)
    for job, line in zip(jobs, parsed):
        run_part = shlex.split(line.command.split(" && ")[0])[4:]
        if line.job_id != job.job_id or tuple(run_part) != job.argv:
            raise RuntimeError(f"{job.job_id}: the written line does not parse back to its arguments")
    for job in jobs:
        print(f"{job.job_id:26s} {job.old_tuned.split('/')[-2]:22s} -> {job.new_tuned.split('/')[-2]:22s} "
              f"changed {','.join(job.changed):17s} policies {' '.join(job.policies):26s} "
              f"orig {job.wall_min:5.1f} min est {job.wall_min * SCALE:4.0f} mem {job.mem:g}")
    for out, why in skipped.items():
        print(f"left out {out}: {why}")
    total = sum(j.wall_min * SCALE for j in jobs)
    print(f"wrote {target} ({len(jobs)} jobs, est {total:.0f} job-min, {total / 6 / 60:.1f} h on six "
          f"workers, longest {max(j.wall_min for j in jobs) * SCALE:.0f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
