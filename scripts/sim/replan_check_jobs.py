"""Job list of the replan-boundary check (the re-solve boundary check, fixed integer re-solve).

    uv run python scripts/sim/replan_check_jobs.py [--first ID ...] [--scale GROUP=F ...]

One job per archived run (``replan_check_sources``): the archived ``run_sweep.py`` command with
``--policies`` cut to the affected policies plus ``offline`` (original order and spelling) and
``--out`` moved to ``replan_check/<group>/<run>``, between a code stamp and the row-by-row compare
(``replan_check_compare.py``).  Two more jobs repeat the TAP prefix audit (c2, c3) with the commands
their logs recorded; the old JSONs were saved beforehand in ``replan_check/audit_before/``.

Estimates (``est``, minutes): the archived wall time times the share of its work that is re-run,
the work of one (cell, seed) counted as 2 for ``offline`` (stream, solve and replay) and 1 for
every other policy, so a run whose policy list is unchanged keeps its archived time; ``--scale
GROUP=F`` multiplies a group's estimates (calibrated on a trial job).  Jobs are listed longest
first, the ``--first`` ids ahead of them (a canary of the oldest originals).  ``mem=2.5`` GB each.
Writes ``replan_check/JOBS.txt`` and ``replan_check/runs.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
for _p in (PILOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from runs_groups import flag_value  # noqa: E402
from replan_check_compare import code_hash  # noqa: E402
from replan_check_sources import (AFFECTED, CHECK, LOCAL, SCRIPT, Command, check_run,  # noqa: E402
                                  commands, inventory, parsed)

COMPARE = "uv run python scripts/sim/replan_check_compare.py"
MEM_GB = 2.5
AUDITS = ("c2", "c3")
FIRST = ("f6_32b__r30_d_inst",)


@dataclass(frozen=True)
class Job:
    """One line of the job list and what the summary needs to read its result."""

    job_id: str
    kind: str  # "run" | "audit"
    group: str
    name: str
    orig: str  # archived CSV (run) or JSON (audit)
    new: str  # the re-run's CSV or JSON
    report: str
    policies: Tuple[str, ...]
    week0_runs: Dict[str, int]
    source: str
    orig_wall_min: float
    est_min: float
    command: str


def replace_values(argv: Sequence[str], flag: str, values: Sequence[str]) -> List[str]:
    """``argv`` with the values of its single ``flag`` replaced in place."""
    idx = [i for i, tok in enumerate(argv) if tok == flag]
    if len(idx) != 1:
        raise ValueError(f"{flag} appears {len(idx)} times")
    j = idx[0] + 1
    while j < len(argv) and not argv[j].startswith("--"):
        j += 1
    return [*argv[:idx[0] + 1], *values, *argv[j:]]


def policies_of(argv: Sequence[str]) -> List[str]:
    """The values of ``--policies``."""
    i = list(argv).index("--policies") + 1
    out = []
    while i < len(argv) and not argv[i].startswith("--"):
        out.append(argv[i])
        i += 1
    return out


def strip(argv: Sequence[str]) -> List[str]:
    """``argv`` without the values of ``--policies`` and ``--out`` (the flags stay in place)."""
    return replace_values(replace_values(argv, "--policies", []), "--out", [])


def work(policies: Sequence[str]) -> float:
    """Work of one (cell, seed): 2 for offline (stream, solve, replay), 1 per other policy."""
    return 2.0 + sum(1.0 for p in policies if p != "offline")


def group_and_name(csv: str) -> Tuple[str, str]:
    """``(group, run)``: runs's subdirectory or validation_local's, and the rest of the path."""
    parts = Path(csv).parents[1].relative_to(LOCAL).parts
    return (parts[1], "/".join(parts[2:])) if parts[0] == "runs" else (parts[0], "/".join(parts[1:]))


def run_job(csv: str, counts: Dict[str, int], cmd: Command, scale: Dict[str, float]) -> Job:
    """The re-run job of one archived run."""
    group, name = group_and_name(csv)
    old = policies_of(cmd.argv)
    kept = [p for p in old if p == "offline" or p in AFFECTED]
    affected = [p for p in kept if p != "offline"]
    if sorted(affected) != sorted(counts) or "offline" not in kept:
        raise ValueError(f"{csv}: --policies {old} vs week-0 policies {counts}")
    new_out = f"{CHECK}/{group}/{name}"
    argv = replace_values(replace_values(cmd.argv, "--policies", kept), "--out", [new_out])
    if (strip(argv) != strip(cmd.argv)
            or {**parsed(argv), "policies": None, "out": None} != {**parsed(cmd.argv), "policies": None, "out": None}):
        raise ValueError(f"{csv}: the re-run command differs from the archived one beyond --policies/--out")
    new_csv = f"{new_out}/{Path(csv).parent.name}/{Path(csv).name}"
    report = f"{new_out}/report.json"
    command = (f"{COMPARE} stamp --out {new_out} && uv run python {SCRIPT} {shlex.join(argv)} && "
               f"{COMPARE} compare --new {new_csv} --orig {csv} --policies {' '.join(affected)} "
               f"--report {report}")
    wall = (cmd.wall_s or 0.0) / 60.0
    est = wall * work(kept) / work(old) * scale.get(group, 1.0)
    return Job(f"{group}__{name.replace('/', '__')}", "run", group, name, csv, new_csv, report,
               tuple(affected), counts, cmd.source, round(wall, 1), max(2.0, round(est)), command)


def audit_job(case: str) -> Job:
    """The TAP prefix audit of one case, as its log recorded it, then compared with the saved JSON."""
    log = PILOT / LOCAL / "runs" / "logs" / f"audit_tap_prefix_{case}.log"
    text = log.read_text(encoding="utf-8")
    start = re.search(r"^START \S+ \S+ (.*)$", text, re.M)
    wall = re.search(r"^EXIT \S+ \S+ 0 wall_s=(\d+)", text, re.M)
    if start is None or wall is None:
        raise ValueError(f"{log}: no START / EXIT 0 line")
    command = start.group(1)
    out = flag_value(shlex.split(command), "--out")
    saved = PILOT / CHECK / "audit_before" / Path(out).name  # type: ignore[arg-type]
    if out != f"{LOCAL}/runs/audit/tap_prefix_{case}.json" or not saved.exists():
        raise ValueError(f"{log}: --out {out} or no saved copy {saved}")
    base = f"{CHECK}/audit/tap_prefix_{case}"
    report = f"{base}/report.json"
    full = (f"{COMPARE} stamp --out {base} && {command} && {COMPARE} audit --new {out} "
            f"--orig {saved.relative_to(PILOT)} --report {report}")
    minutes = int(wall.group(1)) / 60.0
    return Job(f"audit__tap_prefix_{case}", "audit", "audit", f"tap_prefix_{case}",
               str(saved.relative_to(PILOT)), str(out), report, ("tap",), {}, f"START line of {log.relative_to(PILOT)}",
               round(minutes, 1), max(2.0, round(minutes)), full)


def build(scale: Dict[str, float], first: Sequence[str]) -> List[Job]:
    """All jobs, checked, in queue order."""
    cmds = commands()
    jobs: List[Job] = []
    for csv, counts in inventory():
        cmd = cmds[str(Path(csv).parents[1])]
        problems = check_run(csv, cmd)
        if problems:
            raise ValueError(f"{csv}: {problems}")
        jobs.append(run_job(csv, counts, cmd, scale))
    jobs += [audit_job(case) for case in AUDITS]
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids) or not set(first) <= set(ids):
        raise ValueError("duplicate job ids or unknown --first ids")
    lead = [next(j for j in jobs if j.job_id == i) for i in first]
    rest = sorted((j for j in jobs if j.job_id not in first), key=lambda j: (-j.est_min, j.job_id))
    return lead + rest


def header(jobs: Sequence[Job], stamp: Dict[str, object]) -> List[str]:
    """Comment block of JOBS.txt."""
    runs = [j for j in jobs if j.kind == "run"]
    totals: Dict[str, int] = {}
    for j in runs:
        for p, n in j.week0_runs.items():
            totals[p] = totals.get(p, 0) + n
    saved = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
             for p in sorted((PILOT / CHECK / "audit_before").glob("*.json"))}
    return [
        "# replan-check jobs: <id> after=<deps> mem=<GB> est=<min> :: <command>   (run from the repository root)",
        f"# written {datetime.now():%Y-%m-%d %H:%M} by scripts/sim/replan_check_jobs.py; simulator tree "
        f"sha256 {str(stamp['tree_sha256'])[:16]} ({stamp['files']} files)",
        "# Replan check: every archived run with",
        "# tap / calendar_adapt / forecast_ilp re-run on the fixed code; each command is the archived one",
        "# with --policies cut to <affected> + offline and --out under replan_check/ (the sources and the",
        "# per-run assertions are in scripts/sim/replan_check_sources.py); compare exits 3 if offline rows",
        "# do not reproduce (the runner halts), 0 for identical / differs.",
        f"# {len(runs)} runs: " + ", ".join(f"{p} {n}" for p, n in sorted(totals.items()))
        + f" whole-run policy runs; plus the TAP prefix audits c2, c3 (saved copies {saved}).",
        "# est = archived wall x re-run share of the work (offline 2, other policies 1 per cell x seed).",
        "# Run: nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- caffeinate -i nice -n 10 uv run python \\",
        f"#   scripts/sim/runs_runner.py --jobs {CHECK}/JOBS.txt --logs {CHECK}/logs --workers 6 \\",
        f"#   --mem-budget-gb 14 > {CHECK}/logs/runner.out 2>&1 < /dev/null &",
        "# Summary (any time): uv run python scripts/sim/replan_check_summary.py",
        "",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", nargs="*", default=list(FIRST), help="job ids to put at the head")
    ap.add_argument("--scale", nargs="*", default=[], metavar="GROUP=F",
                    help="multiply a group's estimates (e.g. f6_32b=0.8)")
    args = ap.parse_args(argv)
    scale = {k: float(v) for k, v in (item.split("=", 1) for item in args.scale)}
    jobs = build(scale, args.first)
    stamp = code_hash()
    lines = header(jobs, stamp)
    for job in jobs:
        lines.append(f"# {job.group}/{job.name}: {job.orig} ({job.source}); week-0 runs "
                     f"{job.week0_runs or '-'}; archived wall {job.orig_wall_min:g} min")
        lines.append(f"{job.job_id} after=- mem={MEM_GB:g} est={job.est_min:g} :: {job.command}")
    target = PILOT / CHECK
    target.mkdir(parents=True, exist_ok=True)
    (target / "JOBS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"written": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "code": stamp,
                "jobs": [asdict(j) for j in jobs]}
    (target / "runs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    est = sum(j.est_min for j in jobs)
    print(f"{len(jobs)} jobs ({sum(j.kind == 'run' for j in jobs)} runs), est total {est / 60:.1f} h "
          f"of job time -> {CHECK}/JOBS.txt, runs.json")
    by_group: Dict[str, List[Job]] = {}
    for job in jobs:
        by_group.setdefault(job.group, []).append(job)
    for group, members in by_group.items():
        print(f"  {group:18s} {len(members):3d} jobs  est {sum(j.est_min for j in members) / 60:5.1f} h  "
              f"longest {max(j.est_min for j in members):.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
