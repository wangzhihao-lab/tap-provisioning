#!/usr/bin/env python3
"""Batch-B stage driver: runs on the rented GPU box under nohup and owns the budget.

Five resumable stages, each with its own wall-clock timeout, its own log file and
a marker under ``<root>/q2_results`` (``stage_<name>.done`` on success,
``stage_<name>.failed`` on failure, skipped when ``.done`` is already there)::

    check_patterns -> coldstart -> concurrency -> prefix -> summarize

Everything money-related lives here.  A hard budget watchdog is armed before the
first stage, ``heartbeat.txt`` is rewritten every 30 s so the laptop can tell a
long stage from a hang, and on every exit path (success, stage failure, timeout,
unexpected exception) the driver writes ``$OUT/STATUS``, waits up to 30 min for
the laptop's ``PULLED`` marker and then halts the box.  Results are never
deleted.

    python stages.py run-all --root <dir on the box> --cap-hours 14.5
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hardware import detect_tier, record_environment
from layout import STAGES, RemoteLayout, StageSpec, iso_now
from measurements import STAGE_IMPLS, StepContext, select_models
from serving import write_header

logger = logging.getLogger("q2.remote.stages")


@dataclass(frozen=True)
class SequenceResult:
    ran: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed is None


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def run_stage_sequence(
    stages: Sequence[StageSpec],
    runner: Callable[[StageSpec], None],
    layout: RemoteLayout,
    on_stage: Callable[[str], None] | None = None,
) -> SequenceResult:
    """Run the stages in order, skipping the ones already marked ``.done``.

    Stops at the first failure: a later stage is worthless if an earlier one did
    not produce its inputs, and idle GPU time is billed.
    """
    ran: list[str] = []
    skipped: list[str] = []
    for spec in stages:
        if layout.done(spec.name).exists():
            logger.info("stage %s already done; skipping", spec.name)
            skipped.append(spec.name)
            continue
        if on_stage is not None:
            on_stage(spec.name)
        layout.failed(spec.name).unlink(missing_ok=True)
        try:
            runner(spec)
        except Exception as exc:  # any stage failure must be recorded, never raised on
            logger.exception("stage %s failed", spec.name)
            _write(layout.failed(spec.name), f"{iso_now()}\n{type(exc).__name__}: {exc}")
            return SequenceResult(tuple(ran), tuple(skipped), spec.name,
                                  f"{type(exc).__name__}: {exc}")
        _write(layout.done(spec.name), iso_now())
        ran.append(spec.name)
    return SequenceResult(tuple(ran), tuple(skipped))


class Heartbeat:
    """Rewrites ``heartbeat.txt`` (ISO time, epoch, stage) so the laptop can spot a hang."""

    def __init__(self, path: Path, interval_s: float = 30.0) -> None:
        self._path = path
        self._interval_s = interval_s
        self._stage = "starting"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, stage: str = "starting") -> None:
        self._stage = stage
        self._write()
        self._thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)
        self._thread.start()

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        self._write()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._write()

    def _write(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                f"{iso_now()}\t{int(time.time())}\t{self._stage}\n", encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("heartbeat write failed: %s", exc)


def start_watchdog(
    layout: RemoteLayout, cap_hours: float, dry_run: bool, on_cap: str = "keep"
) -> subprocess.Popen[bytes] | None:
    """Arm the budget cap.  ``on_cap`` is the operator's instruction: ``halt`` powers the box off
    at the cap, ``keep`` only writes a note (whether a box is halted is decided by the operator
    for every run, never by a default: on some providers a halted instance cannot be restarted).
    """
    cap_s = max(60, int(cap_hours * 3600))
    if dry_run or on_cap != "halt":
        note = shlex.quote(str(layout.no_shutdown_note))
        action = f"echo 'budget cap reached; would have run: sync; shutdown -h now' >> {note}"
    else:
        action = "sync; shutdown -h now"
    try:
        proc = subprocess.Popen(
            ["nohup", "sh", "-c", f"sleep {cap_s}; {action}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        logger.error("could not arm the budget watchdog: %s", exc)
        return None
    _write(layout.watchdog_pid, str(proc.pid))
    logger.warning("budget watchdog armed: halt in %.2f h (pid %d)", cap_s / 3600, proc.pid)
    return proc


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path) -> None:
    """Refuse to start a second driver: two vLLM servers on one GPU is paid-for OOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                pid = int(path.read_text(encoding="utf-8").split()[0])
            except (OSError, IndexError, ValueError):
                pid = 0
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise SystemExit(f"run_all already running as pid {pid}; nothing to do") from None
            logger.warning("clearing stale lock %s (pid %s)", path, pid or "unknown")
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")
        return
    raise SystemExit(f"could not acquire {path}")


def tail(path: Path, n_lines: int, max_bytes: int = 256 * 1024) -> list[str]:
    if not path.exists():
        return [f"(no log at {path})"]
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - max_bytes))
            blob = fh.read()
    except OSError as exc:
        return [f"(cannot read {path}: {exc})"]
    return blob.decode("utf-8", "replace").splitlines()[-n_lines:]


def write_status(layout: RemoteLayout, result: SequenceResult, tail_lines: int = 50) -> str:
    """Write ``$OUT/STATUS``; its first line is what the laptop polls for."""
    if result.ok:
        body = [
            "DONE",
            f"finished_at={iso_now()}",
            f"stages_ran={','.join(result.ran) or '-'}",
            f"stages_skipped={','.join(result.skipped) or '-'}",
        ]
    else:
        stage = result.failed or "unknown"
        body = [
            "FAILED",
            f"stage={stage}",
            f"finished_at={iso_now()}",
            f"error={result.error}",
            f"stages_ran={','.join(result.ran) or '-'}",
            f"--- last {tail_lines} lines of logs/stage_{stage}.log ---",
            *tail(layout.stage_log(stage), tail_lines),
        ]
    _write(layout.status, "\n".join(body))
    return body[0]


def wait_for_pulled(layout: RemoteLayout, timeout_s: float, poll_s: float = 5.0) -> bool:
    logger.warning("waiting up to %.0f min for the laptop's marker %s",
                   timeout_s / 60, layout.pulled)
    limit = time.monotonic() + timeout_s
    while time.monotonic() < limit:
        if layout.pulled.exists():
            logger.warning("PULLED seen; halting now")
            return True
        time.sleep(poll_s)
    logger.warning("no PULLED marker after %.0f min", timeout_s / 60)
    return False


def halt(layout: RemoteLayout, dry_run: bool, action: str = "keep") -> None:
    """``action`` is the operator's instruction for this outcome: ``halt`` or ``keep``."""
    subprocess.run(["sync"], check=False, timeout=180)
    if dry_run or action != "halt":
        with layout.no_shutdown_note.open("a", encoding="utf-8") as fh:
            fh.write(f"{iso_now()} would have run: sync; shutdown -h now (instruction: {action})\n")
        logger.warning("NOT halting (%s); noted in %s",
                       "--dry-run" if dry_run else f"instruction {action}", layout.no_shutdown_note)
        return
    logger.warning("running: shutdown -h now")
    subprocess.run(["shutdown", "-h", "now"], check=False, timeout=60)


def _build_context(layout: RemoteLayout, args: argparse.Namespace) -> StepContext:
    """Pick the interpreter, probe the GPU, and let the tier set the serve arguments."""
    if args.python:
        python = Path(args.python)
    elif layout.venv_python.exists():
        python = layout.venv_python
    else:
        python = Path(sys.executable)
    tier = detect_tier(args.max_model_len, args.gpu_mem_util)
    models = select_models(tier)
    record_environment(layout, tier, [model.slug for model in models])
    band = (float(args.ramp_band[0]), float(args.ramp_band[1]))
    if args.dry_run:
        return StepContext(
            layout=layout, python=python, tier=tier, models=models, dry_run=True,
            port=args.port, ramp_band=band, health_timeout_s=60.0, cooldown_s=1.0,
            concurrency_duration_s=5.0, burst_k=4, repeats=1,
        )
    return StepContext(layout=layout, python=python, tier=tier, models=models,
                       port=args.port, ramp_band=band)


def _run_stage(ctx: StepContext, spec: StageSpec) -> None:
    logger.info("stage %s starting (timeout %.0f min)", spec.name, spec.timeout_s / 60)
    write_header(ctx.layout.stage_log(spec.name),
                 f"stage {spec.name} (timeout {spec.timeout_s / 60:.0f} min) | {ctx.header()}")
    STAGE_IMPLS[spec.name](ctx, time.monotonic() + spec.timeout_s)
    logger.info("stage %s finished", spec.name)


def _add_file_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("run-all",))
    p.add_argument("--root", required=True, help="working directory on the box")
    p.add_argument("--out-name", default="b1")
    p.add_argument("--cap-hours", type=float, default=14.5, help="hard budget cap")
    p.add_argument("--wait-pulled-minutes", type=float, default=30.0)
    p.add_argument("--python", default=None, help="interpreter for the measurement scripts")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="override the value the detected GPU tier would choose")
    p.add_argument("--gpu-mem-util", type=float, default=None,
                   help="override --gpu-memory-utilization from the GPU tier")
    p.add_argument("--ramp-band", type=float, nargs=2, metavar=("LO", "HI"), default=[60.0, 120.0])
    p.add_argument("--dry-run", action="store_true", help="fake server, no GPU, no shutdown")
    p.add_argument("--on-done", choices=("halt", "keep"), required=True,
                   help="operator's instruction after a verified pull: halt the box or keep it")
    p.add_argument("--on-fail", choices=("halt", "keep"), required=True,
                   help="operator's instruction after a failed stage: halt the box or keep it")
    p.add_argument("--on-cap", choices=("halt", "keep"), required=True,
                   help="operator's instruction when --cap-hours is reached: halt or keep")
    p.add_argument("--only", default=None,
                   help="comma-separated stage names to run instead of the whole list")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(message)s")
    layout = RemoteLayout(root=Path(args.root).expanduser(), out_name=args.out_name)
    for directory in (layout.results, layout.out, layout.logs):
        directory.mkdir(parents=True, exist_ok=True)
    _add_file_log(layout.logs / "stages.log")
    acquire_lock(layout.run_all_pid)
    watchdog = start_watchdog(layout, args.cap_hours, args.dry_run, args.on_cap)
    heartbeat = Heartbeat(layout.heartbeat)
    heartbeat.start()
    try:
        # Inside the try: a GPU probe that blows up must still reach the shutdown
        # path below, or the box idles until the budget watchdog fires hours later.
        ctx = _build_context(layout, args)
        selected = tuple(STAGES)
        if args.only:
            wanted = [name.strip() for name in args.only.split(",") if name.strip()]
            unknown = [name for name in wanted if name not in {s.name for s in STAGES}]
            if unknown:
                raise ValueError(f"unknown stage(s) in --only: {unknown}")
            selected = tuple(spec for spec in STAGES if spec.name in wanted)
        result = run_stage_sequence(selected, lambda spec: _run_stage(ctx, spec), layout,
                                    heartbeat.set_stage)
    except BaseException as exc:  # the box must not survive an unexpected exit either
        logger.exception("driver aborted")
        result = SequenceResult(failed="driver", error=f"{type(exc).__name__}: {exc}")
    heartbeat.set_stage("finalizing")
    logger.info("STATUS=%s", write_status(layout, result))
    subprocess.run(["sync"], check=False, timeout=180)
    if result.failed is not None:
        heartbeat.set_stage("failed")
        logger.error("stage %s failed: %s; instruction for this outcome: %s (cap %.1f h: %s)",
                     result.failed, result.error, args.on_fail, args.cap_hours, args.on_cap)
        if args.on_fail == "halt":
            # never before the laptop has pulled: it writes PULLED after its verified pull and
            # its own grace period; until then the box stays up (the watchdog still caps it)
            heartbeat.set_stage("failed_waiting_for_pull")
            wait_for_pulled(layout, args.cap_hours * 3600.0)
        heartbeat.stop()
        if args.dry_run and watchdog is not None:
            watchdog.terminate()
        halt(layout, args.dry_run, args.on_fail)
        return 1
    heartbeat.set_stage("waiting_for_pull")
    wait_s = min(120.0, args.wait_pulled_minutes * 60) if args.dry_run \
        else args.wait_pulled_minutes * 60
    pulled = wait_for_pulled(layout, wait_s)
    heartbeat.stop()
    if args.dry_run and watchdog is not None:
        watchdog.terminate()  # a real run keeps its watchdog: last line of defence
    if not pulled and args.on_done == "halt" and not args.dry_run:
        # Data before money: a halt here would leave the last rounds only on the data disk
        # (B5, 2026-09-22: the laptop orchestrator died with its session, no PULLED arrived, the
        # box halted on its own and the run's last two rounds had to be recovered from a clone).
        # The budget watchdog armed at the start is the only thing that may still halt the box.
        heartbeat.set_stage("done_waiting_for_pull")
        logger.error("run finished but the laptop never wrote %s; NOT halting (data first). "
                     "Pull the results and halt from the console, or wait for the %.1f h cap.",
                     layout.pulled, args.cap_hours)
        return 0
    halt(layout, args.dry_run, args.on_done)
    layout.run_all_pid.unlink(missing_ok=True)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
