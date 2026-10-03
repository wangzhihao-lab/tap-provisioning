"""Queue runner for ``outputs/validation_local/runs/JOBS.txt``.

    nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- caffeinate -i nice -n 10 \\
        uv run python scripts/sim/runs_runner.py --workers 6 > .../logs/runner.out 2>&1 &

Jobs are started in file order as far as three limits allow: at most ``--workers`` at a time, the
sum of their ``mem=`` estimates within ``--mem-budget-gb`` (a job always starts when nothing else
runs), and every job named in ``after=`` finished.  Each job runs ``bash -c <command>`` from
the repository root under ``/usr/bin/time -l`` (its peak RSS lands in the log); its log starts with a
``START`` and ends with an ``EXIT`` line, and a zero exit writes ``<id>.DONE``.  A restart skips
every job with a DONE marker, so the runner resumes where it stopped.  A failed job blocks the jobs
that wait for it; exit code 3 (a stream-identity check failed, ``runs_check.py``) stops the
queue: running jobs finish, nothing new starts, ``logs/HALT`` says why.  A job with a
``logs/<id>.SKIP`` marker is left out (``touch`` one to drop a queued job without a restart).
``logs/status.txt`` is rewritten every minute; ``logs/runner.pgid`` holds the process group to stop
everything with ``kill -TERM -<pgid>``.  JOBS.txt is re-read when it changes, so a job appended to
it joins the queue without a restart (a running job keeps the command it was started with).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, TextIO

PILOT = Path(__file__).resolve().parents[2]
ROOT = PILOT / "outputs" / "validation_local" / "runs"
HALT_CODE = 3
LINE = re.compile(r"^(?P<id>\S+)\s+after=(?P<after>\S+)\s+mem=(?P<mem>[\d.]+)\s+est=(?P<est>[\d.]+)"
                  r"\s+::\s+(?P<cmd>.+)$")
RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)


@dataclass(frozen=True)
class Job:
    """One line of JOBS.txt."""

    job_id: str
    after: tuple
    mem: float
    est: float
    command: str


def parse(path: Path) -> List[Job]:
    """Jobs of a JOBS.txt, in file order; comment and blank lines are skipped."""
    jobs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = LINE.match(raw.strip())
        if m is None:
            raise ValueError(f"malformed job line: {raw[:120]}")
        after = () if m["after"] == "-" else tuple(m["after"].split(","))
        jobs.append(Job(m["id"], after, float(m["mem"]), float(m["est"]), m["cmd"]))
    ids = [j.job_id for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate job ids in the job list")
    return jobs


def stamp() -> str:
    """Local wall-clock time for the logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Running:
    """A started job."""

    job: Job
    proc: subprocess.Popen
    log: TextIO
    started: float


class Runner:
    """The queue (module docstring)."""

    def __init__(self, jobs: List[Job], logs: Path, workers: int, mem_budget: float,
                 source: Optional[Path] = None, only: Optional[Set[str]] = None) -> None:
        self.jobs, self.logs = jobs, logs
        self.source, self.only = source, only
        self.mtime = source.stat().st_mtime if source is not None else 0.0
        self.workers, self.mem_budget = workers, mem_budget
        self.running: Dict[str, Running] = {}
        self.failed: Set[str] = set()
        self.blocked: Set[str] = set()
        self.halted: Optional[str] = None
        self.t0 = time.time()
        logs.mkdir(parents=True, exist_ok=True)

    def done(self, job_id: str) -> bool:
        """Finished with exit 0, in this session or an earlier one."""
        return (self.logs / f"{job_id}.DONE").exists()

    def note(self, message: str) -> None:
        """Append one line to ``runner.log``."""
        with open(self.logs / "runner.log", "a", encoding="utf-8") as fh:
            fh.write(f"[{stamp()}] {message}\n")

    def launch(self, job: Job) -> None:
        """Start one job under ``/usr/bin/time -l``."""
        log = open(self.logs / f"{job.job_id}.log", "a", encoding="utf-8")
        log.write(f"START {stamp()} {job.command}\n")
        log.flush()
        proc = subprocess.Popen(["/usr/bin/time", "-l", "/bin/bash", "-c", job.command], cwd=PILOT,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.running[job.job_id] = Running(job, proc, log, time.time())
        self.note(f"START {job.job_id} pid={proc.pid} mem={job.mem:g} est={job.est:.0f}min")

    def reap(self) -> None:
        """Collect finished jobs: EXIT lines, DONE / FAILED markers, the halt flag."""
        for job_id, run in list(self.running.items()):
            code = run.proc.poll()
            if code is None:
                continue
            wall = time.time() - run.started
            run.log.write(f"EXIT {stamp()} {code} wall_s={wall:.0f}\n")
            run.log.close()
            text = (self.logs / f"{job_id}.log").read_text(encoding="utf-8", errors="replace")
            rss = RSS.findall(text)
            peak_gb = int(rss[-1]) / 2 ** 30 if rss else float("nan")
            self.note(f"EXIT {code} {job_id} wall={wall / 60:.1f}min peak_rss={peak_gb:.2f}GB")
            marker = {"exit": code, "wall_s": round(wall), "peak_rss_gb": round(peak_gb, 2),
                      "end": stamp(), "est_min": run.job.est}
            if code == 0:
                (self.logs / f"{job_id}.DONE").write_text(json.dumps(marker) + "\n")
            else:
                self.failed.add(job_id)
                (self.logs / f"{job_id}.FAILED").write_text(json.dumps(marker) + "\n")
                if code == HALT_CODE and self.halted is None:
                    self.halted = f"{job_id} failed its stream-identity check (exit {code})"
                    (self.logs / "HALT").write_text(self.halted + "\n")
                    self.note(f"HALT {self.halted}")
            del self.running[job_id]

    def skipped(self, job_id: str) -> bool:
        """Dropped by hand with a SKIP marker."""
        return (self.logs / f"{job_id}.SKIP").exists()

    def pending(self) -> List[Job]:
        """Jobs that are neither done, running, failed, blocked nor skipped."""
        return [j for j in self.jobs if not self.done(j.job_id) and j.job_id not in self.running
                and j.job_id not in self.failed and j.job_id not in self.blocked
                and not self.skipped(j.job_id)]

    def schedule(self) -> None:
        """Start what the limits allow, in file order."""
        for job in self.pending():
            if any(d in self.failed or d in self.blocked or self.skipped(d) for d in job.after):
                self.blocked.add(job.job_id)
                self.note(f"BLOCKED {job.job_id} (a dependency failed)")
                continue
            if self.halted or len(self.running) >= self.workers:
                continue
            if not all(self.done(d) for d in job.after):
                continue
            in_use = sum(r.job.mem for r in self.running.values())
            if self.running and in_use + job.mem > self.mem_budget:
                continue
            self.launch(job)

    def status(self) -> None:
        """Rewrite ``status.txt``: counts, running jobs, failures, a rough ETA."""
        done = [j for j in self.jobs if self.done(j.job_id)]
        pend = self.pending()
        left = sum(j.est for j in pend) + sum(max(0.0, r.job.est - (time.time() - r.started) / 60)
                                             for r in self.running.values())
        scale = [json.loads((self.logs / f"{j.job_id}.DONE").read_text()) for j in done]
        ratio = (sum(m["wall_s"] / 60 for m in scale) / max(1e-9, sum(m["est_min"] for m in scale))
                 if scale else 1.0)
        longest = max([j.est for j in pend] + [max(0.0, r.job.est - (time.time() - r.started) / 60)
                                               for r in self.running.values()] + [0.0])
        eta_h = max(left / max(1, self.workers), longest) * ratio / 60
        lines = [f"{stamp()}  done {len(done)}/{len(self.jobs)}  running {len(self.running)}  "
                 f"pending {len(pend)}  failed {len(self.failed)}  blocked {len(self.blocked)}",
                 f"rough ETA {eta_h:.1f} h (remaining est {left:.0f} min x observed/est {ratio:.2f} "
                 f"/ {self.workers} workers)" + (f"  HALTED: {self.halted}" if self.halted else "")]
        for job_id, run in self.running.items():
            lines.append(f"  running {job_id}  {(time.time() - run.started) / 60:.0f} min "
                         f"(est {run.job.est:.0f})")
        lines += [f"  FAILED {j}" for j in sorted(self.failed)]
        lines += [f"  blocked {j}" for j in sorted(self.blocked)]
        (self.logs / "status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def refresh(self) -> None:
        """Re-read the job list if its file changed (new jobs join, pending ones take new lines)."""
        if self.source is None or self.source.stat().st_mtime == self.mtime:
            return
        try:
            jobs = parse(self.source)
        except ValueError as exc:
            self.note(f"job list not reloaded: {exc}")
            return
        self.mtime = self.source.stat().st_mtime
        self.jobs = [j for j in jobs if self.only is None or j.job_id in self.only]
        self.note(f"job list reloaded: {len(self.jobs)} jobs")

    def loop(self, poll_s: float = 10.0) -> int:
        """Run until nothing is running and nothing more can start."""
        (self.logs / "runner.pgid").write_text(f"{os.getpgid(0)}\n")
        self.note(f"runner up: {len(self.jobs)} jobs, {len(self.pending())} to run, "
                  f"workers {self.workers}, memory budget {self.mem_budget:g} GB, pgid {os.getpgid(0)}")
        last = 0.0
        while True:
            self.reap()
            self.refresh()
            self.schedule()
            if time.time() - last > 60 or not self.running:
                self.status()
                last = time.time()
            startable = [j for j in self.pending() if all(self.done(d) for d in j.after)]
            if not self.running and (self.halted or not startable):
                break
            time.sleep(poll_s)
        self.status()
        self.note(f"runner exit: {len(self.failed)} failed, {len(self.blocked)} blocked"
                  + (f", halted ({self.halted})" if self.halted else ""))
        return 1 if (self.failed or self.halted) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, default=ROOT / "JOBS.txt")
    ap.add_argument("--logs", type=Path, default=ROOT / "logs")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mem-budget-gb", type=float, default=22.0)
    ap.add_argument("--only", nargs="*", default=None, help="run just these job ids")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs that would run")
    args = ap.parse_args(argv)
    jobs = parse(args.jobs)
    if args.only:
        unknown = set(args.only) - {j.job_id for j in jobs}
        if unknown:
            raise SystemExit(f"unknown job ids: {sorted(unknown)}")
        jobs = [j for j in jobs if j.job_id in set(args.only)]
    runner = Runner(jobs, args.logs, max(1, args.workers), args.mem_budget_gb, args.jobs,
                    set(args.only) if args.only else None)
    if args.dry_run:
        for job in runner.pending():
            print(f"{job.job_id:28s} after={','.join(job.after) or '-':40s} mem={job.mem:g} est={job.est:.0f}")
        return 0
    if (args.logs / "HALT").exists():
        raise SystemExit(f"{args.logs / 'HALT'} exists: {(args.logs / 'HALT').read_text().strip()}; "
                         "remove it after the cause is fixed")
    return runner.loop()


if __name__ == "__main__":
    raise SystemExit(main())
