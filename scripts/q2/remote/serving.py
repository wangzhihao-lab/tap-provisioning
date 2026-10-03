#!/usr/bin/env python3
"""Subprocess and server plumbing for the box: run it, bound it, kill it properly.

Two things the stages need and nothing else does:

* ``run_step`` runs one subprocess with its output appended to a stage log and a
  hard bound from the stage deadline.  A timeout kills the whole process group,
  because ``vllm`` and the load generators spawn children.
* ``ManagedServer`` starts a server, polls ``/health`` until it answers 200, and
  stops it with SIGTERM followed by SIGKILL after a grace period.

Deliberately independent of the stage definitions, so the call sites stay
readable and this file has no import cycle with ``measurements.py``.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout import iso_now

try:  # installed by bootstrap.sh on the box; the laptop test env does not need it
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

logger = logging.getLogger("q2.remote.serving")


class StageError(RuntimeError):
    """A stage could not complete; the driver records it and stops."""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def write_header(path: Path, text: str) -> None:
    """Append a header line to a log, so every log says which hardware it is from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n##### {iso_now()} | {text}\n")


def kill_process_group(proc: subprocess.Popen[bytes], grace_s: float = 30.0) -> None:
    """SIGTERM the whole process group, SIGKILL whatever is left after ``grace_s``."""
    if proc.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            logger.warning("pid %d still alive %.0fs after %s", proc.pid, wait_s, sig.name)
    logger.error("pid %d refused to die", proc.pid)


def run_step(
    cmd: Sequence[str], log_path: Path, deadline: float,
    cwd: Path | None = None, header: str = "",
) -> None:
    """Run one subprocess, bounded by ``deadline`` (monotonic clock)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    remaining = deadline - time.monotonic()
    printable = " ".join(shlex.quote(part) for part in cmd)
    if remaining <= 1.0:
        raise StageError(f"out of time before running {printable}")
    logger.info("$ %s", printable)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {iso_now()} (<= {remaining:.0f}s) {header}\n$ {printable}\n")
        fh.flush()
        proc = subprocess.Popen(list(cmd), stdout=fh, stderr=subprocess.STDOUT,
                                cwd=None if cwd is None else str(cwd), start_new_session=True)
        try:
            code = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            kill_process_group(proc, grace_s=15.0)
            raise StageError(f"timed out after {remaining:.0f}s: {printable}") from None
    if code != 0:
        raise StageError(f"exit {code} from {printable} (see {log_path})")


@dataclass(frozen=True)
class ServerSpec:
    cmd: tuple[str, ...]
    base_url: str
    log_path: Path
    label: str
    header: str = ""


class ManagedServer:
    """Start a server, wait for ``/health`` 200, stop it with SIGTERM then SIGKILL."""

    def __init__(self, spec: ServerSpec, grace_s: float = 30.0) -> None:
        self.spec = spec
        self._grace_s = grace_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._fh: Any = None

    def __enter__(self) -> ManagedServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.spec.log_path.open("a", encoding="utf-8")
        self._fh.write(f"\n=== {iso_now()} {self.spec.header}\n$ {' '.join(self.spec.cmd)}\n")
        self._fh.flush()
        logger.info("starting %s: %s", self.spec.label, " ".join(self.spec.cmd))
        # FlashInfer's JIT sampler needs a CUDA >= 12.9 toolkit for SM 12.x GPUs (Blackwell); the
        # provider image shipped an older nvcc, so vLLM's engine died at sampler init on 2026-09-18.
        # The attention backend is FLASH_ATTN anyway; force the torch-native sampler.
        env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        self._proc = subprocess.Popen(list(self.spec.cmd), stdout=self._fh, env=env,
                                      stderr=subprocess.STDOUT, start_new_session=True)

    def wait_health(self, timeout_s: float, deadline: float) -> float:
        """Poll ``/health`` until it answers 200; raise if it never does."""
        if requests is None:
            raise StageError("requests is not installed; cannot poll /health")
        url = f"{self.spec.base_url.rstrip('/')}/health"
        started = time.monotonic()
        limit = min(started + timeout_s, deadline)
        session = requests.Session()
        while time.monotonic() < limit:
            if self._proc is not None and self._proc.poll() is not None:
                raise StageError(
                    f"{self.spec.label} exited {self._proc.returncode} before /health "
                    f"(see {self.spec.log_path})"
                )
            try:
                if session.get(url, timeout=3).status_code == 200:
                    waited = time.monotonic() - started
                    logger.info("%s healthy after %.1fs", self.spec.label, waited)
                    return waited
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise StageError(f"{self.spec.label}: /health not 200 within {timeout_s:g}s")

    def stop(self) -> None:
        if self._proc is not None:
            logger.info("stopping %s", self.spec.label)
            kill_process_group(self._proc, self._grace_s)
            self._proc = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
