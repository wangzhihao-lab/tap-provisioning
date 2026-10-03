#!/usr/bin/env python3
"""Q2 tier (i): replica start lead time for a vLLM OpenAI-compatible server.

Launches ``vllm serve <model>`` and timestamps the stages of a cold replica
start: process start, weights loaded, CUDA-graph capture done, ``/health`` OK,
first completion returned (32 output tokens) and the steady-state TTFT of a
second request.  R runs per model, server killed between runs, page cache
dropped between runs when running as root.  One JSON object per run is appended
to ``<out>/coldstart/coldstart.jsonl``; the timestamped raw server log of each
run is kept under ``<out>/coldstart/logs/``.

    python coldstart_vllm.py --out results/b1 --runs 3      # real; --dry-run needs no GPU
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger("q2.coldstart")

DEFAULT_MODELS = (
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct-AWQ",
)
MILESTONES = ("weights_loaded", "cuda_graph_done", "engine_ready", "server_listening")


@dataclass(frozen=True)
class VLLMLogPatterns:
    """Assumption: vLLM >= 0.6 / V1 engine wording. Re-check on the box before batch B."""

    weights_loaded: tuple[str, ...] = (
        r"Model loading took", r"Loading model weights took",
        r"Loading safetensors checkpoint shards:\s*100%",
    )
    cuda_graph_done: tuple[str, ...] = (
        r"Graph capturing finished", r"Capturing CUDA graph shapes:\s*100%",
        r"Capturing CUDA graphs.*100%",
    )
    engine_ready: tuple[str, ...] = (
        r"init engine \(profile, create kv cache, warmup model\) took", r"Init engine .* took",
    )
    server_listening: tuple[str, ...] = (r"Application startup complete", r"Uvicorn running on")


@dataclass(frozen=True)
class ColdStartConfig:
    models: tuple[str, ...] = DEFAULT_MODELS
    runs: int = 3
    host: str = "localhost"
    port: int = 8000
    dtype: str = "auto"
    max_model_len: int = 4096
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.90
    extra_args: tuple[str, ...] = ()
    vllm_bin: str = "vllm"
    ready_timeout_s: float = 900.0
    request_timeout_s: float = 300.0
    probe_max_tokens: int = 32
    shutdown_grace_s: float = 30.0
    cooldown_s: float = 10.0
    drop_caches: bool = True
    out_dir: Path = Path("results")
    dry_run: bool = False
    seed: int = 20260917


# Dry-run only: (weights_s, graph_s, startup_tail_s, first_completion_s, steady_ttft_s)
_DRY_PROFILES: dict[str, tuple[float, float, float, float, float]] = {
    "7b": (11.5, 8.5, 2.8, 1.35, 0.075),
    "13b": (18.0, 10.5, 3.2, 1.75, 0.098),
    "14b": (19.0, 11.0, 3.3, 1.80, 0.100),
    "32b": (28.0, 14.5, 4.0, 2.40, 0.130),
    "70b": (44.0, 19.0, 5.0, 3.10, 0.180),
}


class LogWatcher:
    """Timestamps the first log line matching each milestone pattern group."""

    def __init__(self, patterns: VLLMLogPatterns, sink: Path) -> None:
        self._groups = {n: [re.compile(p) for p in getattr(patterns, n)] for n in MILESTONES}
        self.marks: dict[str, float] = {}
        sink.parent.mkdir(parents=True, exist_ok=True)
        self._fh = sink.open("w", encoding="utf-8")

    def feed(self, line: str, offset_s: float) -> None:
        self._fh.write(f"{offset_s:9.3f}\t{line}\n")
        for name, regexes in self._groups.items():
            if name not in self.marks and any(rx.search(line) for rx in regexes):
                self.marks[name] = offset_s
                logger.debug("milestone %s at t+%.3fs: %s", name, offset_s, line[:110])

    def close(self) -> None:
        self._fh.close()


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _profile_for(model: str) -> tuple[float, float, float, float, float]:
    low = model.lower()
    return _DRY_PROFILES[next((k for k in _DRY_PROFILES if k in low), "7b")]


def build_cmd(cfg: ColdStartConfig, model: str) -> list[str]:
    cmd = [cfg.vllm_bin, "serve", model, "--host", cfg.host, "--port", str(cfg.port)]
    cmd += ["--dtype", cfg.dtype, "--max-model-len", str(cfg.max_model_len)]
    cmd += ["--gpu-memory-utilization", f"{cfg.gpu_memory_utilization:g}"]
    if cfg.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    return cmd + list(cfg.extra_args)


def drop_page_cache() -> bool:
    """Drop the Linux page cache; True only if it actually happened."""
    if os.name != "posix" or not Path("/proc/sys/vm/drop_caches").exists():
        logger.info("page cache not dropped: no /proc/sys/vm/drop_caches (non-Linux host)")
        return False
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        logger.info("page cache not dropped: not root; weights may be served from RAM")
        return False
    try:
        subprocess.run(["sync"], check=True, timeout=60)
        Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("page cache drop failed: %s", exc)
        return False
    return True


def _pump_logs(proc: subprocess.Popen[bytes], watcher: LogWatcher, t0: float) -> None:
    """Split the merged stdout/stderr on both \\n and \\r so progress bars are seen."""
    if proc.stdout is None:
        return
    fd, buf = proc.stdout.fileno(), b""
    while True:
        try:
            chunk = os.read(fd, 8192)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        while True:
            cuts = [i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0]
            if not cuts:
                break
            raw, buf = buf[: min(cuts)], buf[min(cuts) + 1 :]
            text = raw.decode("utf-8", "replace").strip()
            if text:
                watcher.feed(text, time.monotonic() - t0)


def wait_health(base_url: str, deadline: float) -> float | None:
    session = requests.Session()
    while time.monotonic() < deadline:
        try:
            if session.get(f"{base_url}/health", timeout=3).status_code == 200:
                return time.monotonic()
        except requests.RequestException:
            pass
        time.sleep(0.2)
    return None


def probe_completion(
    base_url: str, model: str, max_tokens: int, timeout_s: float
) -> dict[str, Any]:
    """Send one streaming completion; return TTFT, end-to-end latency and chunk count."""
    prompt = "Explain what a hash table is to a first-year undergraduate."
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    start, ttft, n_chunks = time.monotonic(), None, 0
    url = f"{base_url}/v1/completions"
    with requests.post(url, json=payload, stream=True, timeout=(5, timeout_s)) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            body = raw[5:].strip()
            if body == b"[DONE]":
                break
            try:
                choice = json.loads(body).get("choices", [{}])[0]
            except (json.JSONDecodeError, IndexError):
                continue
            text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
            if text:
                n_chunks += 1
                ttft = time.monotonic() - start if ttft is None else ttft
    now = time.monotonic()
    return {"ttft_s": ttft, "e2e_s": now - start, "chunks": n_chunks, "done_at": now}


def _simulate_run(cfg: ColdStartConfig, model: str, run_idx: int, watcher: LogWatcher) -> dict:
    """Feed synthetic vLLM log lines through the real parser on a virtual clock."""
    rng = np.random.default_rng(cfg.seed + run_idx * 977 + zlib.crc32(model.encode()) % 10_000)

    def jit(x: float) -> float:
        return float(x * rng.normal(1.0, 0.06))

    w_s, g_s, tail_s, first_s, steady_s = _profile_for(model)
    t_w = jit(w_s) * (1.22 if run_idx == 0 else 1.0)
    t_g = t_w + jit(g_s)
    t_eng = t_g + jit(1.4)
    t_listen = t_eng + jit(tail_s)
    for offset, line in (
        (0.30, "INFO 09-17 10:00:00 [api_server.py:1601] vLLM API server version 0.10.1"),
        (0.35, f"INFO 09-17 10:00:00 [api_server.py:1602] args: model={model}"),
        (1.10, "INFO 09-17 10:00:01 [gpu_model_runner.py:2120] Starting to load model ..."),
        (t_w * 0.5, "Loading safetensors checkpoint shards:  50% Completed | 4/8"),
        (t_w * 0.99, "Loading safetensors checkpoint shards: 100% Completed | 8/8"),
        (t_w, f"INFO [gpu_model_runner.py:2178] Model loading took 14.99 GiB, {t_w:.3f} seconds"),
        (t_g * 0.99, "Capturing CUDA graphs (decode, FULL): 100%|##########| 67/67"),
        (t_g, f"INFO [gpu_model_runner.py:2999] Graph capturing finished in {t_g - t_w:.0f} secs"),
        (t_eng, "INFO [core.py:214] init engine (profile, create kv cache, warmup model) took "
                f"{t_eng:.2f} seconds"),
        (t_listen - 0.05, "INFO:     Started server process [12345]"),
        (t_listen, "INFO:     Application startup complete."),
        (t_listen + 0.01, "INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C)"),
    ):
        watcher.feed(line, float(offset))
    t_health, first, steady = t_listen + jit(0.35), jit(first_s), jit(steady_s)
    return {
        "health_ok_s": t_health, "first_completion_ttft_s": first * 0.55,
        "first_completion_e2e_s": first, "first_completion_done_s": t_health + first,
        "steady_ttft_s": steady, "steady_e2e_s": steady + 0.012 * cfg.probe_max_tokens,
        "output_chunks": cfg.probe_max_tokens, "exit_code": 0,
        "cmd": " ".join(build_cmd(cfg, model)) + "   # not executed (--dry-run)",
    }


def _run_server(cfg: ColdStartConfig, model: str, watcher: LogWatcher) -> dict[str, Any]:
    cmd = build_cmd(cfg, model)
    base_url = f"http://{cfg.host}:{cfg.port}"
    logger.info("launching: %s", " ".join(cmd))
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )
    pump = threading.Thread(target=_pump_logs, args=(proc, watcher, t0), daemon=True)
    pump.start()
    out: dict[str, Any] = {"cmd": " ".join(cmd)}
    try:
        health_at = wait_health(base_url, t0 + cfg.ready_timeout_s)
        if health_at is None:
            out["error"] = f"/health not OK within {cfg.ready_timeout_s:g}s"
            return out
        out["health_ok_s"] = health_at - t0
        n_tok, t_req = cfg.probe_max_tokens, cfg.request_timeout_s
        first = probe_completion(base_url, model, n_tok, t_req)
        out["first_completion_ttft_s"] = first["ttft_s"]
        out["first_completion_e2e_s"] = first["e2e_s"]
        out["first_completion_done_s"] = first["done_at"] - t0
        second = probe_completion(base_url, model, n_tok, t_req)
        out["steady_ttft_s"], out["steady_e2e_s"] = second["ttft_s"], second["e2e_s"]
        out["output_chunks"] = second["chunks"]
    except requests.RequestException as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _shutdown(proc, cfg.shutdown_grace_s)
        pump.join(timeout=5)
        out["exit_code"] = proc.returncode
    return out


def _shutdown(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=grace_s)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        logger.error("server pid %d refused to die", proc.pid)


def run_once(cfg: ColdStartConfig, model: str, run_idx: int) -> dict[str, Any]:
    log_path = cfg.out_dir / "coldstart" / "logs" / f"{_slug(model)}_run{run_idx}.log"
    watcher = LogWatcher(VLLMLogPatterns(), log_path)
    dropped = drop_page_cache() if cfg.drop_caches else False
    rec: dict[str, Any] = {
        "schema": "q2.coldstart.v1", "model": model, "run": run_idx, "dry_run": cfg.dry_run,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "page_cache_dropped": dropped, "dtype": cfg.dtype, "max_model_len": cfg.max_model_len,
        "enable_prefix_caching": cfg.enable_prefix_caching, "raw_log": str(log_path),
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
    }
    try:
        if cfg.dry_run:
            rec.update(_simulate_run(cfg, model, run_idx, watcher))
        else:
            rec.update(_run_server(cfg, model, watcher))
    finally:
        watcher.close()
    for name in MILESTONES:
        rec[f"{name}_s"] = watcher.marks.get(name)
    if rec["weights_loaded_s"] is not None and rec["cuda_graph_done_s"] is not None:
        rec["graph_capture_s"] = rec["cuda_graph_done_s"] - rec["weights_loaded_s"]
    rec["missing_milestones"] = [
        n for n in ("weights_loaded", "cuda_graph_done", "server_listening")
        if n not in watcher.marks
    ]
    return rec


def _summarise(records: list[dict[str, Any]]) -> None:
    keys = ("weights_loaded_s", "cuda_graph_done_s", "health_ok_s", "first_completion_done_s")
    for model in dict.fromkeys(r["model"] for r in records):
        rows = [r for r in records if r["model"] == model]
        parts = []
        for key in keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            parts.append(f"{key[:-2]}={np.median(vals):.1f}s" if vals else f"{key[:-2]}=n/a")
        logger.info("%s (n=%d): %s", model, len(rows), "  ".join(parts))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = ColdStartConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", default=",".join(d.models), help="comma-separated HF model ids")
    p.add_argument("--runs", type=int, default=d.runs)
    p.add_argument("--host", default=d.host)
    p.add_argument("--port", type=int, default=d.port)
    p.add_argument("--dtype", default=d.dtype)
    p.add_argument("--max-model-len", type=int, default=d.max_model_len)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=d.gpu_memory_utilization)
    p.add_argument("--vllm-bin", default=d.vllm_bin)
    p.add_argument("--extra-arg", action="append", default=[], help="extra `vllm serve` argument")
    p.add_argument("--ready-timeout", type=float, default=d.ready_timeout_s)
    p.add_argument("--probe-max-tokens", type=int, default=d.probe_max_tokens)
    p.add_argument("--cooldown", type=float, default=d.cooldown_s)
    p.add_argument("--no-drop-caches", action="store_true")
    p.add_argument("--out", type=Path, default=d.out_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    cfg = ColdStartConfig(
        models=tuple(m.strip() for m in a.models.split(",") if m.strip()),
        runs=a.runs, host=a.host, port=a.port, dtype=a.dtype,
        max_model_len=a.max_model_len, enable_prefix_caching=not a.no_prefix_caching,
        gpu_memory_utilization=a.gpu_memory_utilization, extra_args=tuple(a.extra_arg),
        vllm_bin=a.vllm_bin, ready_timeout_s=a.ready_timeout,
        probe_max_tokens=a.probe_max_tokens, cooldown_s=a.cooldown,
        drop_caches=not a.no_drop_caches, out_dir=a.out, dry_run=a.dry_run, seed=a.seed,
    )
    out_path = cfg.out_dir / "coldstart" / "coldstart.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with out_path.open("a", encoding="utf-8") as fh:
        for model in cfg.models:
            for run_idx in range(cfg.runs):
                logger.info("cold start %s run %d/%d", model, run_idx + 1, cfg.runs)
                rec = run_once(cfg, model, run_idx)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(rec)
                if rec["missing_milestones"]:
                    logger.warning("unmatched log milestones: %s", rec["missing_milestones"])
                if not cfg.dry_run and cfg.cooldown_s > 0:
                    time.sleep(cfg.cooldown_s)
    logger.info("wrote %d runs to %s", len(records), out_path)
    _summarise(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
