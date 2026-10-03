#!/usr/bin/env python3
"""The one engine of the ``single`` mode, and the server plumbing both modes share.

Plan/07 section 11: co-resident capped vLLM processes on one card share the memory bandwidth
and give no capacity scaling, so the pool runs ONE vLLM engine (fixed arguments below) and the
replicas are virtual admission units (``pool_replicas.py``).  ``start_engine`` starts it through
``remote/serving.py``'s ``ManagedServer``, waits for ``/health`` in a worker thread, sends one
warm-up completion and returns the record that goes into ``pool_meta.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import requests

REMOTE_DIR = Path(__file__).resolve().parent / "remote"
if str(REMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DIR))

from serving import ManagedServer, ServerSpec, StageError  # noqa: E402

if TYPE_CHECKING:
    from pool_replicas import ReplicaSpec

logger = logging.getLogger("q2.pool.engine")

ENGINE_GPU_FRAC = 0.90
ENGINE_MAX_NUM_SEQS = 256
ENGINE_MAX_MODEL_LEN = 8192


def engine_command(spec: "ReplicaSpec", port: int) -> tuple[str, ...]:
    """The one big engine of the ``single`` mode: fixed arguments."""
    if spec.dry_run:
        return (spec.python, spec.fake_server, "--port", str(port), "--model", spec.model,
                "--capacity", str(ENGINE_MAX_NUM_SEQS), "--time-scale",
                f"{spec.fake_time_scale:g}", "--log-level", "WARNING")
    return (spec.vllm_bin, "serve", spec.model, "--host", "localhost", "--port", str(port),
            "--dtype", spec.dtype, "--max-model-len", str(ENGINE_MAX_MODEL_LEN),
            "--gpu-memory-utilization", f"{ENGINE_GPU_FRAC:g}",
            "--max-num-seqs", str(ENGINE_MAX_NUM_SEQS), "--enable-prefix-caching")


class ReplicaServer(ManagedServer):
    """``ManagedServer`` with its process visible, for liveness checks and the pid record."""

    @property
    def proc(self) -> Optional[subprocess.Popen[bytes]]:
        return self._proc


def warm_up(base_url: str, model: str, timeout_s: float) -> Optional[str]:
    """One tiny completion; the error text, or None when the server answered 200."""
    payload = {"model": model, "messages": [{"role": "user", "content": "预热"}],
               "max_tokens": 1, "temperature": 0.0}
    try:
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                             timeout=(10, timeout_s))
    except requests.RequestException as exc:
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    if resp.status_code != 200:
        return f"HTTP {resp.status_code}: {resp.text[:200]}"
    return None


async def start_engine(spec: "ReplicaSpec", out_dir: Path, port: int
                       ) -> tuple[ReplicaServer, dict[str, Any]]:
    """Start the single engine, wait for ``/health``, warm it up; raise ``StageError`` if not."""
    server_spec = ServerSpec(engine_command(spec, port), f"http://localhost:{port}",
                             out_dir / "engine.log", "engine",
                             header=f"single engine dry_run={spec.dry_run}")
    server = ReplicaServer(server_spec, spec.stop_grace_s)
    started = time.time()
    server.start()
    record: dict[str, Any] = {"mode": "single", "url": server_spec.base_url, "port": port,
                              "args": list(server_spec.cmd),
                              "pid": None if server.proc is None else server.proc.pid,
                              "started_wallclock": started, "health_after_s": None,
                              "ready_wallclock": None, "warmup_s": None,
                              "note": "single mode ignores --gpu-frac and --max-num-seqs for the "
                                      "engine (fixed args above); --max-num-seqs only sets the "
                                      "default per-replica K"}
    try:
        waited = await asyncio.to_thread(server.wait_health, spec.health_timeout_s,
                                         time.monotonic() + spec.health_timeout_s)
        healthy = time.time()
        error = await asyncio.to_thread(warm_up, server_spec.base_url, spec.model,
                                        spec.warmup_timeout_s)
        if error is not None:
            raise StageError(f"engine warm-up failed: {error}")
    except BaseException:
        await asyncio.to_thread(server.stop)
        raise
    record.update({"health_after_s": round(waited, 3), "ready_wallclock": time.time(),
                   "warmup_s": round(time.time() - healthy, 3)})
    (out_dir / "engine_server.json").write_text(json.dumps(record, indent=2) + "\n",
                                                encoding="utf-8")
    logger.info("engine ready at %s after %.1fs", server_spec.base_url, waited)
    return server, record


__all__ = ["ENGINE_GPU_FRAC", "ENGINE_MAX_MODEL_LEN", "ENGINE_MAX_NUM_SEQS", "ReplicaServer",
           "engine_command", "start_engine", "warm_up"]
