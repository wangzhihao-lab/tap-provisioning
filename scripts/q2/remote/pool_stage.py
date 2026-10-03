#!/usr/bin/env python3
"""Stage ``pool`` (batch B' step two): the policy-driven replica pool.

Reads ``<scripts>/pool_inputs/pool_runs.json`` (``pool_runs_dry.json`` under ``--dry-run`` when
it exists) and runs, in order: ``calib_multi`` (a fixed pool of N replicas behind the router,
the closed-loop concurrency curve through the router, ``summarize.py``) and then one round per
entry of ``runs``: the pool is started with ``T0 = now + head_s``, the router's ``/health`` is
awaited, ``replay_client.py`` is started with ``--t0-delay`` so that its first request is
scheduled at ``T0``, and the stage waits for the client and then for the pool, which ends itself
after ``T0 + duration + drain``.  Every round writes ``<label>.done`` in its folder and a folder
with that marker is skipped, so the stage resumes where a previous attempt stopped.

``pool_runs.json`` keys the stage reads: ``trace``, ``max_offset_s``, ``duration_min``, ``head_s``
(default 240 s and, outside a dry run, never less: a plan round starts up to five vLLM replicas
at once), ``abandon_s`` and ``give_up_s`` (both default 60 s: the router drops a request after
``abandon_s`` and the client stops retrying it after ``give_up_s``), optional
``replica_concurrency`` (the router's K; default the replicas' ``--max-num-seqs``),
``max_replicas`` (the pool's x_max cap, default 5) and ``gpu_frac`` (process mode only, default
0.19), ``engine`` ("process" or "single", default "process") with ``cold_start_s`` (the emulated
cold-start sequence of the single mode), and ``minute_s`` (dry run only).

Outputs: ``<out>/pool/calib_multi/`` and ``<out>/pool/runs/<label>/``.  The stage never kills a
pool with SIGKILL first: the pool's SIGTERM handler is what stops the replicas, and after a pool
that did not exit cleanly the replica records are read back and any replica process still alive
is killed, so a vLLM replica cannot outlive its round and hold the GPU for the next one.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import requests

from layout import iso_now
from measurements import StepContext, _popen_step, _replay_client_cmd, _step
from serving import StageError, free_port, kill_process_group

logger = logging.getLogger("q2.remote.pool")

POOL_DIR = "pool"
POOL_MODEL_SLUG = "qwen2.5-7b"
POOL_GPU_FRAC = 0.19
POOL_MAX_NUM_SEQS = 10
POOL_MAX_MODEL_LEN = 8192
POOL_MAX_REPLICAS = 5
POOL_ENGINE = "process"  # or "single": one engine, virtual replicas
POOL_REPLICA_PORT_BASE = 8101
POOL_COOLDOWN_S = 30.0
POOL_HEAD_S = 240.0  # pool start -> T0; five concurrent vLLM launches need more than 180 s
POOL_ABANDON_S = 60.0  # the simulator's abandon_in_s; the client's --give-up-s matches it
POOL_EXIT_GRACE_S = 240.0  # SIGTERM -> the pool drains and stops up to 5 vLLM replicas
CALIB_DURATION_MIN = 30  # the fixed pool is stopped by the stage long before this
DRY_MINUTE_S = 6.0
ROUTER_POLL_S = 1.0


def load_pool_runs(ctx: StepContext) -> dict[str, Any]:
    folder = ctx.layout.scripts / "pool_inputs"
    path = folder / "pool_runs.json"
    if ctx.dry_run and (folder / "pool_runs_dry.json").exists():
        path = folder / "pool_runs_dry.json"
    if not path.exists():
        raise StageError(f"{path} missing: run make_bell_trace.py (laptop) first")
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["_path"] = str(path)
    return spec


def _minute_s(ctx: StepContext, spec: dict[str, Any]) -> float:
    return float(spec.get("minute_s", DRY_MINUTE_S)) if ctx.dry_run else 60.0


def _pool_cmd(ctx: StepContext, spec: dict[str, Any], out: Path, port: int, policy: str,
              t0: float, duration_min: int, lead_gate_min: int,
              context: Optional[str] = None, plan: Optional[str] = None,
              fixed: Optional[int] = None) -> list[str]:
    model = next((m for m in ctx.models if m.slug == POOL_MODEL_SLUG), ctx.models[0])
    engine = _engine(spec)
    cmd = [str(ctx.python), ctx.script("replica_pool.py"), "--out", str(out),
           "--model", ctx.model_path(model), "--vllm-bin", ctx.vllm_bin,
           "--python", str(ctx.python), "--port", str(port),
           "--replica-port-base", str(POOL_REPLICA_PORT_BASE),
           "--max-replicas", str(int(spec.get("max_replicas", POOL_MAX_REPLICAS))),
           "--max-num-seqs", str(POOL_MAX_NUM_SEQS), "--max-model-len", str(POOL_MAX_MODEL_LEN),
           "--abandon-s", f"{_abandon_s(spec):g}", "--engine", engine,
           "--policy", policy, "--t0-wallclock", f"{t0:.3f}",
           "--duration-min", str(duration_min), "--lead-gate-min", str(lead_gate_min),
           "--minute-s", f"{_minute_s(ctx, spec):g}",
           "--health-timeout-s", f"{ctx.health_timeout_s:g}"]
    if engine == "single":
        cmd += ["--cold-start-s", _cold_start_list(spec)]
    else:  # the single engine has fixed arguments; gpu_frac is a process-mode knob
        cmd += ["--gpu-frac", f"{float(spec.get('gpu_frac', POOL_GPU_FRAC)):g}"]
    if spec.get("replica_concurrency") is not None:
        cmd += ["--replica-concurrency", str(int(spec["replica_concurrency"]))]
    if context:
        cmd += ["--context", ctx.script(context)]
    if plan:
        cmd += ["--plan", ctx.script(plan)]
    if fixed is not None:
        cmd += ["--fixed-replicas", str(fixed)]
    if ctx.dry_run:
        cmd += ["--dry-run", "--drain-grace-s", "5"]
    return cmd


def _engine(spec: dict[str, Any]) -> str:
    engine = str(spec.get("engine", POOL_ENGINE))
    if engine not in ("process", "single"):
        raise StageError(f"pool_runs.json: engine must be process or single, not {engine!r}")
    return engine


def _cold_start_list(spec: dict[str, Any]) -> str:
    """``cold_start_s`` as the comma list ``replica_pool.py --cold-start-s`` takes."""
    raw = spec.get("cold_start_s", [30.0])
    values = [float(v) for v in (raw if isinstance(raw, (list, tuple)) else [raw])]
    if not values:
        raise StageError("pool_runs.json: cold_start_s is empty")
    return ",".join(f"{v:g}" for v in values)


def _abandon_s(spec: dict[str, Any]) -> float:
    return float(spec.get("abandon_s", POOL_ABANDON_S))


def _give_up_s(spec: dict[str, Any]) -> float:
    return float(spec.get("give_up_s", POOL_ABANDON_S))


def _head_s(ctx: StepContext, spec: dict[str, Any]) -> float:
    head = float(spec.get("head_s", POOL_HEAD_S))
    if not ctx.dry_run and head < POOL_HEAD_S:
        logger.warning("head_s %g in %s raised to %g (concurrent vLLM launches at pool start)",
                       head, spec["_path"], POOL_HEAD_S)
        head = POOL_HEAD_S
    return head


def _router_port(ctx: StepContext) -> int:
    return free_port() if ctx.dry_run else ctx.port


def _wait_router(proc: subprocess.Popen[bytes], base_url: str, deadline: float,
                 timeout_s: float = 600.0) -> None:  # single mode: the engine starts before the router
    limit = min(time.monotonic() + timeout_s, deadline)
    session = requests.Session()
    while time.monotonic() < limit:
        if proc.poll() is not None:
            raise StageError(f"replica_pool.py exited {proc.returncode} before its router answered")
        try:
            if session.get(f"{base_url}/health", timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(ROUTER_POLL_S)
    raise StageError(f"router at {base_url} not healthy within {timeout_s:g}s")


def _wait_admitted(proc: subprocess.Popen[bytes], base_url: str, n: int, deadline: float,
                   timeout_s: float) -> float:
    started = time.monotonic()
    limit = min(started + timeout_s, deadline)
    session = requests.Session()
    while time.monotonic() < limit:
        if proc.poll() is not None:
            raise StageError(f"replica_pool.py exited {proc.returncode} while replicas started")
        try:
            resp = session.get(f"{base_url}/health", timeout=3)
            if resp.status_code == 200 and int(resp.json().get("admitted", 0)) >= n:
                waited = time.monotonic() - started
                logger.info("%d replicas admitted after %.0fs", n, waited)
                return waited
        except (requests.RequestException, ValueError):
            pass
        time.sleep(ROUTER_POLL_S)
    raise StageError(f"fewer than {n} replicas admitted within {timeout_s:g}s")


def _kill_leftover_replicas(run_dir: Path) -> list[int]:
    """After a pool that did not stop cleanly: kill replica processes it recorded but never stopped."""
    killed: list[int] = []
    for path in sorted(run_dir.glob("replica_*_server.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = rec.get("pid")
        if not pid or rec.get("stop_wallclock") is not None:
            continue
        try:
            cmdline = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                                     text=True, check=False, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        if "vllm" not in cmdline and "fake_server" not in cmdline:
            continue
        try:
            os.killpg(int(pid), signal.SIGKILL)
            killed.append(int(pid))
        except (ProcessLookupError, PermissionError):
            continue
    if killed:
        logger.error("killed leftover replica processes %s under %s", killed, run_dir)
    return killed


def _finish_pool(proc: subprocess.Popen[bytes], run_dir: Path, deadline: float,
                 what: str, terminate: bool) -> None:
    """Wait for the pool to end (or SIGTERM it), then make sure nothing of it survives."""
    clean = False
    try:
        if terminate:
            kill_process_group(proc, grace_s=POOL_EXIT_GRACE_S)
        else:
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    logger.error("%s ran past the stage deadline; SIGTERM", what)
                    kill_process_group(proc, grace_s=POOL_EXIT_GRACE_S)
                    raise StageError(f"{what} ran past the stage deadline")
                time.sleep(1.0)
        clean = proc.returncode == 0
        if not clean:
            raise StageError(f"{what} exited {proc.returncode} (see its pool.log)")
    finally:
        if not clean:
            _kill_leftover_replicas(run_dir)


def _calib_multi(ctx: StepContext, spec: dict[str, Any], out: Path, deadline: float) -> None:
    calib = spec.get("calib_multi") or {}
    if not calib:
        logger.info("no calib_multi block; skipping")
        return
    done = out / "calib_multi.done"
    if done.exists():
        logger.info("calib_multi already done; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    n = int(calib.get("replicas", POOL_MAX_REPLICAS))
    port = _router_port(ctx)
    base_url = f"http://localhost:{port}"
    pool_dir = out / "pool"
    cmd = _pool_cmd(ctx, spec, pool_dir, port, "fixed", time.time(), CALIB_DURATION_MIN, 0,
                    fixed=n)
    proc = _popen_step(ctx, "pool", cmd)
    started = time.time()
    try:
        _wait_router(proc, base_url, deadline)
        admitted_after = _wait_admitted(proc, base_url, n, deadline, ctx.health_timeout_s * 2)
        model = next((m for m in ctx.models if m.slug == POOL_MODEL_SLUG), ctx.models[0])
        curve = [str(ctx.python), ctx.script("concurrency_curve.py"), "--out", str(out),
                 "--model", ctx.model_path(model), "--base-url", base_url,
                 "--levels", str(calib.get("levels", "10,20,30,40,50")),
                 "--duration", f"{float(calib.get('duration_s', 30.0)):g}"]
        _step(ctx, "pool", curve, deadline)
        lo, hi = ctx.ramp_band
        _step(ctx, "pool", (str(ctx.python), ctx.script("summarize.py"), "--results", str(out),
                            "--out", str(out), "--ramp-band", f"{lo:g}", f"{hi:g}",
                            "--no-figure"), deadline)
    finally:
        _finish_pool(proc, pool_dir, deadline, "calib_multi pool", terminate=True)
    (out / "calib_multi_meta.json").write_text(json.dumps({
        "schema": "q2.pool.calib_multi.v1", "dry_run": ctx.dry_run, "replicas": n,
        "levels": calib.get("levels"), "duration_s": calib.get("duration_s"),
        "engine": _engine(spec), "cold_start_s": _cold_start_list(spec).split(","),
        "replica_concurrency": spec.get("replica_concurrency"),
        "router": base_url, "pool_started_wallclock": started,
        "replicas_admitted_after_s": round(admitted_after, 1), "finished": iso_now(),
        "note": "concurrency_curve.py ran against the router, not its own fake server, so its "
                "records say dry_run=false even in a dry run; this file is the tag",
    }, indent=2) + "\n", encoding="utf-8")
    done.write_text(iso_now() + "\n", encoding="utf-8")
    time.sleep(ctx.cooldown_s if ctx.dry_run else POOL_COOLDOWN_S)


def _one_run(ctx: StepContext, spec: dict[str, Any], run: dict[str, Any], out: Path,
             deadline: float) -> None:
    label = str(run["label"])
    run_dir = out / "runs" / label
    done = run_dir / f"{label}.done"
    if done.exists():
        logger.info("round %s already done; skipping", label)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    minute_s = _minute_s(ctx, spec)
    duration_min = int(run.get("duration_min", spec["duration_min"]))
    head_s = _head_s(ctx, spec)
    max_offset = min(float(spec["max_offset_s"]), duration_min * minute_s)
    port = _router_port(ctx)
    base_url = f"http://localhost:{port}"
    t0 = time.time() + head_s
    cmd = _pool_cmd(ctx, spec, run_dir, port, str(run["policy"]), t0, duration_min,
                    int(run.get("lead_gate_min", 0)), context=run.get("context"),
                    plan=run.get("plan"), fixed=run.get("fixed_replicas"))
    pool = _popen_step(ctx, "pool", cmd)
    client: Optional[subprocess.Popen[bytes]] = None
    try:
        _wait_router(pool, base_url, deadline)
        delay = t0 - time.time()
        if delay < 1.0:
            logger.warning("round %s: the router took the whole head; t0 is %.1fs away", label, delay)
        model = next((m for m in ctx.models if m.slug == POOL_MODEL_SLUG), ctx.models[0])
        client_cmd = _replay_client_cmd(ctx, model, base_url, run_dir, label, 1, max_offset,
                                        max(0.0, delay), warmup=0,
                                        trace=ctx.script(str(spec["trace"])),
                                        give_up_s=_give_up_s(spec))
        client = _popen_step(ctx, "pool", client_cmd)
        while client.poll() is None:
            if time.monotonic() > deadline:
                raise StageError(f"replay client {label} ran past the stage deadline")
            if pool.poll() is not None:
                raise StageError(f"pool {label} exited {pool.returncode} while the client ran")
            time.sleep(1.0)
        if client.returncode != 0:
            raise StageError(f"replay client {label} exited {client.returncode}")
        _finish_pool(pool, run_dir, deadline, f"pool {label}", terminate=False)
    except BaseException:
        if client is not None and client.poll() is None:
            client.kill()
        try:
            _finish_pool(pool, run_dir, deadline, f"pool {label}", terminate=True)
        except StageError as exc:  # the first error is the one worth reporting
            logger.error("while cleaning up round %s: %s", label, exc)
        raise
    (run_dir / "round.json").write_text(json.dumps({
        "schema": "q2.pool.round.v1", "label": label, "dry_run": ctx.dry_run, "run": run,
        "t0_wallclock": t0, "head_s": head_s, "minute_s": minute_s, "duration_min": duration_min,
        "max_offset_s": max_offset, "abandon_s": _abandon_s(spec), "give_up_s": _give_up_s(spec),
        "replica_concurrency": spec.get("replica_concurrency"), "engine": _engine(spec),
        "cold_start_s": _cold_start_list(spec).split(","), "router": base_url,
        "finished": iso_now(),
    }, indent=2) + "\n", encoding="utf-8")
    done.write_text(iso_now() + "\n", encoding="utf-8")
    time.sleep(ctx.cooldown_s if ctx.dry_run else POOL_COOLDOWN_S)


def run_pool_stage(ctx: StepContext, deadline: float) -> None:
    spec = load_pool_runs(ctx)
    out = ctx.layout.out / POOL_DIR
    out.mkdir(parents=True, exist_ok=True)
    logger.info("pool stage from %s: engine=%s, calib_multi + %d rounds", spec["_path"],
                _engine(spec), len(spec["runs"]))
    _calib_multi(ctx, spec, out / "calib_multi", deadline)
    for run in spec["runs"]:
        _one_run(ctx, spec, run, out, deadline)


__all__ = ["load_pool_runs", "run_pool_stage"]


CALIB_SINGLE_DIR = "calib_single"
CALIB_SINGLE_LEVELS = "1,2,5,8,10,12,15,20"
CALIB_SINGLE_DURATION_S = 30.0
CALIB_SINGLE_PROMPT_MEDIAN_TOKENS = 1020.0  # the bell stream's Azure-length prompts
CALIB_SINGLE_PROMPT_SIGMA = 0.78
CALIB_SINGLE_MAX_TOKENS = 210  # the stream's mean output length


def run_calib_single_stage(ctx: StepContext, deadline: float) -> None:
    """One capped replica calibrated in isolation on this card.

    The replica is the pool's unit (``--max-num-seqs`` K, ``gpu_frac`` of the card, from
    ``pool_inputs/pool_runs.json`` when it exists so that the calibration and the run agree);
    ``concurrency_curve.py`` sweeps ``CALIB_SINGLE_LEVELS`` through it with the bell stream's
    prompt regime and ``summarize.py`` writes ``<out>/calib_single/T3_calibration.csv``, which
    ``make_bell_trace.py`` reads on the laptop before the ``pool`` stage is launched.
    """
    spec: dict[str, Any] = {}
    runs_file = Path(ctx.script("pool_inputs/pool_runs.json"))
    if runs_file.exists():
        spec = json.loads(runs_file.read_text(encoding="utf-8"))
    out = ctx.layout.out / CALIB_SINGLE_DIR
    out.mkdir(parents=True, exist_ok=True)
    model = next((m for m in ctx.models if m.slug == POOL_MODEL_SLUG), ctx.models[0])
    port = free_port() if ctx.dry_run else POOL_REPLICA_PORT_BASE
    gpu_frac = float(spec.get("gpu_frac", POOL_GPU_FRAC))
    max_num_seqs = int(spec.get("replica_concurrency", POOL_MAX_NUM_SEQS))
    if ctx.dry_run:
        cmd: tuple[str, ...] = (str(ctx.python), ctx.script("fake_server.py"), "--port", str(port),
                                "--model", ctx.model_path(model), "--capacity", str(max_num_seqs),
                                "--time-scale", "0.05", "--log-level", "WARNING")
    else:
        cmd = (ctx.vllm_bin, "serve", ctx.model_path(model), "--host", "localhost",
               "--port", str(port), "--dtype", model.dtype,
               "--max-model-len", str(POOL_MAX_MODEL_LEN),
               "--gpu-memory-utilization", f"{gpu_frac:g}", "--max-num-seqs", str(max_num_seqs),
               "--enable-prefix-caching")
    from serving import ManagedServer, ServerSpec  # noqa: PLC0415  (measurements imports us lazily)
    server = ManagedServer(ServerSpec(cmd, f"http://localhost:{port}",
                                      ctx.layout.logs / "server_calib_single.log",
                                      f"capped replica {model.slug}",
                                      header=f"stage=calib_single {ctx.header()}"),
                           ctx.shutdown_grace_s)
    launched = time.time()
    with server:
        health_s = server.wait_health(ctx.health_timeout_s, deadline)
        (out / "server.json").write_text(json.dumps({
            "schema": "q2.calib_single.v1", "model": ctx.model_path(model), "gpu_frac": gpu_frac,
            "max_num_seqs": max_num_seqs, "max_model_len": POOL_MAX_MODEL_LEN,
            "launched_wallclock": launched, "health_after_s": health_s, "dry_run": ctx.dry_run,
            "gpu_tier": ctx.tier.as_dict()}, indent=2) + "\n", encoding="utf-8")
        curve = [str(ctx.python), ctx.script("concurrency_curve.py"), "--out", str(out),
                 "--model", ctx.model_path(model),
                 "--duration", f"{5.0 if ctx.dry_run else CALIB_SINGLE_DURATION_S:g}",
                 "--prompt-median-tokens", f"{CALIB_SINGLE_PROMPT_MEDIAN_TOKENS:g}",
                 "--prompt-sigma", f"{CALIB_SINGLE_PROMPT_SIGMA:g}",
                 "--max-tokens", str(CALIB_SINGLE_MAX_TOKENS), "--cooldown", "5",
                 "--base-url", server.spec.base_url]
        curve += ["--levels", "1,5"] if ctx.dry_run else ["--levels", CALIB_SINGLE_LEVELS]
        _step(ctx, "calib_single", curve, deadline)
    _step(ctx, "calib_single", (str(ctx.python), ctx.script("summarize.py"), "--results", str(out),
                                "--out", str(out), "--ramp-band", "60", "120", "--no-figure"),
          deadline)
