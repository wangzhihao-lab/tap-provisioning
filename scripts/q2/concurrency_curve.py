#!/usr/bin/env python3
"""Q2 / T3: closed-loop concurrency-latency-throughput curve of one serving replica.

N concurrent simulated students hammer an OpenAI-compatible endpoint for a fixed
window per level, N in {1,5,10,20,50,100,200}.  Each student sends a tutoring
style prompt whose length is drawn log-normally around 300 tokens, asks for 256
output tokens and streams the answer so TTFT is measured at the first content
chunk.  Per-request records go to ``<out>/concurrency/requests_N<level>.jsonl``,
per-level aggregates to ``<out>/concurrency/concurrency_summary.jsonl``.

    python concurrency_curve.py --base-url http://localhost:8000 --model Qwen/...
    python concurrency_curve.py --dry-run --out results/dryrun   # local fake server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger("q2.concurrency")

DEFAULT_LEVELS = (1, 5, 10, 20, 50, 100, 200)
DRY_LEVELS = (1, 5, 10, 20)
WORDS_PER_TOKEN = 0.75

_TOPICS = (
    "binary search trees", "hash collisions", "big-O of merge sort", "SQL joins",
    "normalisation to 3NF", "pandas groupby", "matplotlib subplots", "linked list reversal",
    "graph traversal", "dynamic programming", "B+ tree indexes", "transaction isolation",
)
_FILLER = (
    "The lecturer worked through the example on the slide but I lost the thread halfway.",
    "Here is the code I wrote in the lab session and the error message it printed.",
    "Please explain the idea first and only then show the corrected implementation.",
    "I am a second-year undergraduate and I have not seen this notation before.",
    "Our assignment is due on Friday and this is the part I cannot finish.",
)


@dataclass(frozen=True)
class LoadConfig:
    base_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    levels: tuple[int, ...] = DEFAULT_LEVELS
    duration_s: float = 90.0
    max_tokens: int = 256
    prompt_median_tokens: float = 300.0
    prompt_sigma: float = 0.5
    temperature: float = 0.0
    api_key: str = ""
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 300.0
    cooldown_s: float = 10.0
    out_dir: Path = Path("results")
    dry_run: bool = False
    seed: int = 20260917
    shared_prefix_tokens: int = 0  # course material every request carries (prefix-cache friendly)
    unique_prefix_tokens: int = 0  # per-request distinct context, e.g. a student's own notes (no cache hits)


def model_slug(model: str) -> str:
    """Filename-safe model tag.

    Without it, a second model written into the same ``--out`` silently overwrites
    the first model's per-request records (the summary JSONL is appended, the
    per-request file is not), and RUNBOOK section 4 runs three models into one
    ``$OUT``.
    """
    tail = model.rstrip("/").split("/")[-1]
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in tail).strip("_")
    return safe or "model"


_VOCAB = (
    "matrix vector gradient theorem lemma proof invariant recursion iteration pointer buffer "
    "index schema tuple relation transaction lock cache thread process kernel scheduler queue "
    "latency throughput bandwidth protocol packet socket router encoding decoding entropy "
    "variance estimator likelihood posterior prior sample population regression classifier "
    "cluster embedding attention token layer neuron activation loss optimizer epoch batch"
).split()
_SHARED_PREFIX_CACHE: dict[int, str] = {}


def _random_text(rng: np.random.Generator, n_tokens: int) -> str:
    """About ``n_tokens`` tokens of random technical words (a distinct context per call)."""
    n_words = max(1, int(n_tokens * WORDS_PER_TOKEN))
    return " ".join(rng.choice(_VOCAB, size=n_words))


def shared_prefix(n_tokens: int) -> str:
    """The same ``n_tokens``-token block for every request (built once from a fixed seed)."""
    if n_tokens not in _SHARED_PREFIX_CACHE:
        text = _random_text(np.random.default_rng(12345), n_tokens)
        _SHARED_PREFIX_CACHE[n_tokens] = "Course material for today:\n" + text + "\n\n"
    return _SHARED_PREFIX_CACHE[n_tokens]


def make_prompt(rng: np.random.Generator, cfg: LoadConfig) -> tuple[str, int]:
    """Return (prompt, estimated prompt tokens) with a log-normal question length.

    With ``shared_prefix_tokens`` every prompt starts with the same course-material block, the
    prefix-cache friendly shape of a RAG assistant over one syllabus; with
    ``unique_prefix_tokens`` each request carries its own random context (a student's own notes
    or code), which no cache can reuse.  Both are counted in the token estimate.
    """
    mu = float(np.log(cfg.prompt_median_tokens))
    n_tokens = int(np.clip(rng.lognormal(mu, cfg.prompt_sigma), 40, 4096))
    n_words = max(12, int(n_tokens * WORDS_PER_TOKEN))
    head = f"I have a question about {rng.choice(_TOPICS)} from today's class. "
    words = head.split()
    while len(words) < n_words:
        words.extend(str(rng.choice(_FILLER)).split())
    question = " ".join(words[:n_words])
    parts = []
    if cfg.shared_prefix_tokens > 0:
        parts.append(shared_prefix(cfg.shared_prefix_tokens))
    if cfg.unique_prefix_tokens > 0:
        parts.append("My notes:\n" + _random_text(rng, cfg.unique_prefix_tokens) + "\n\n")
    parts.append(question)
    total = n_tokens + cfg.shared_prefix_tokens + cfg.unique_prefix_tokens
    return "".join(parts), total


def _post_streaming(session: requests.Session, cfg: LoadConfig, prompt: str) -> dict[str, Any]:
    """One streaming chat completion; returns TTFT, end-to-end latency, output tokens."""
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": "You are a teaching assistant. Be concise."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
    }
    start = time.monotonic()
    ttft: float | None = None
    n_out = 0
    usage: dict[str, Any] | None = None
    try:
        with session.post(
            url, json=payload, headers=headers, stream=True,
            timeout=(cfg.connect_timeout_s, cfg.read_timeout_s),
        ) as resp:
            if resp.status_code != 200:
                return {"ok": False, "error": f"http {resp.status_code}",
                        "e2e_s": time.monotonic() - start}
            for raw in resp.iter_lines():
                if not raw or not raw.startswith(b"data:"):
                    continue
                body = raw[5:].strip()
                if body == b"[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or usage
                choices = obj.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or choices[0].get("text") or ""
                if text:
                    n_out += 1
                    if ttft is None:
                        ttft = time.monotonic() - start
    except requests.RequestException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}", "e2e_s": time.monotonic() - start}
    if usage and usage.get("completion_tokens"):
        n_out = int(usage["completion_tokens"])
    return {"ok": ttft is not None, "ttft_s": ttft, "e2e_s": time.monotonic() - start,
            "output_tokens": n_out, "error": None if ttft is not None else "no content"}


class _Window:
    """Start/deadline of the measurement window, published when the barrier trips."""

    start: float = 0.0
    deadline: float = 0.0


def _worker(
    worker_id: int, cfg: LoadConfig, window: _Window,
    sink: list[dict[str, Any]], lock: threading.Lock, barrier: threading.Barrier,
) -> None:
    rng = np.random.default_rng(cfg.seed + worker_id)
    session = requests.Session()
    try:
        barrier.wait(timeout=60)
    except threading.BrokenBarrierError:
        return
    while time.monotonic() < window.deadline:
        prompt, n_in = make_prompt(rng, cfg)
        started = time.monotonic()
        result = _post_streaming(session, cfg, prompt)
        record = {"worker": worker_id, "t_start_s": started - window.start,
                  "prompt_tokens_est": n_in, **result}
        with lock:
            sink.append(record)
    session.close()


def run_level(cfg: LoadConfig, level: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one concurrency level and return (per-request records, aggregate)."""
    sink: list[dict[str, Any]] = []
    lock = threading.Lock()
    window = _Window()

    def _trip() -> None:
        window.start = time.monotonic()
        window.deadline = window.start + cfg.duration_s

    barrier = threading.Barrier(level + 1, action=_trip)
    threads = [
        threading.Thread(
            target=_worker, args=(i, cfg, window, sink, lock, barrier),
            name=f"student-{i}", daemon=True,
        )
        for i in range(level)
    ]
    for th in threads:
        th.start()
    barrier.wait(timeout=60)
    for th in threads:
        th.join(timeout=cfg.duration_s + cfg.read_timeout_s + 30)
    window_s = time.monotonic() - window.start
    return sink, aggregate(cfg, level, sink, window_s)


def aggregate(
    cfg: LoadConfig, level: int, records: list[dict[str, Any]], window_s: float
) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    ttfts = np.array([r["ttft_s"] for r in ok], dtype=float)
    e2es = np.array([r["e2e_s"] for r in ok], dtype=float)
    out_tokens = int(sum(r.get("output_tokens", 0) for r in ok))
    n_all = len(records)
    agg: dict[str, Any] = {
        "schema": "q2.concurrency.v1",
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": cfg.model, "level": level, "dry_run": cfg.dry_run,
        "duration_s": cfg.duration_s, "window_s": round(window_s, 3),
        "max_tokens": cfg.max_tokens, "prompt_median_tokens": cfg.prompt_median_tokens,
        "shared_prefix_tokens": cfg.shared_prefix_tokens,
        "unique_prefix_tokens": cfg.unique_prefix_tokens,
        "requests_total": n_all, "requests_ok": len(ok),
        "error_rate": (n_all - len(ok)) / n_all if n_all else float("nan"),
        "throughput_req_s": len(ok) / window_s if window_s > 0 else float("nan"),
        "output_tokens_total": out_tokens,
        "throughput_tokens_s": out_tokens / window_s if window_s > 0 else float("nan"),
    }
    for name, arr in (("ttft", ttfts), ("e2e", e2es)):
        for tag, q in (("p50", 50), ("p95", 95), ("p99", 99)):
            agg[f"{name}_{tag}_s"] = float(np.percentile(arr, q)) if arr.size else None
        agg[f"{name}_mean_s"] = float(arr.mean()) if arr.size else None
    agg["prompt_tokens_mean"] = (
        float(np.mean([r["prompt_tokens_est"] for r in records])) if records else None
    )
    return agg


def _start_dry_run_server(cfg: LoadConfig, time_scale: float) -> tuple[Any, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_server import FakeServerConfig, start_fake_server

    server, _thread, base_url = start_fake_server(
        FakeServerConfig(model=cfg.model, time_scale=time_scale, seed=cfg.seed, capacity=16)
    )
    return server, base_url


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = LoadConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=d.base_url)
    p.add_argument("--model", default=d.model)
    p.add_argument("--levels", default=None, help="comma-separated, default 1,5,10,20,50,100,200")
    p.add_argument("--duration", type=float, default=None, help="seconds per level (default 90)")
    p.add_argument("--max-tokens", type=int, default=d.max_tokens)
    p.add_argument("--prompt-median-tokens", type=float, default=d.prompt_median_tokens)
    p.add_argument("--prompt-sigma", type=float, default=d.prompt_sigma)
    p.add_argument("--shared-prefix-tokens", type=int, default=d.shared_prefix_tokens,
                   help="identical course-material block on every prompt (prefix-cache friendly)")
    p.add_argument("--unique-prefix-tokens", type=int, default=d.unique_prefix_tokens,
                   help="distinct random context per request (no prefix-cache reuse)")
    p.add_argument("--api-key", default=d.api_key)
    p.add_argument("--read-timeout", type=float, default=d.read_timeout_s)
    p.add_argument("--cooldown", type=float, default=d.cooldown_s)
    p.add_argument("--out", type=Path, default=d.out_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fake-time-scale", type=float, default=0.08, help="dry-run speed-up factor")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _build_cfg(a: argparse.Namespace) -> LoadConfig:
    if a.levels:
        levels = tuple(int(x) for x in a.levels.split(",") if x.strip())
    else:
        levels = DRY_LEVELS if a.dry_run else DEFAULT_LEVELS
    duration = a.duration if a.duration is not None else (5.0 if a.dry_run else 90.0)
    return LoadConfig(
        base_url=a.base_url, model=a.model, levels=levels, duration_s=duration,
        max_tokens=a.max_tokens, prompt_median_tokens=a.prompt_median_tokens,
        prompt_sigma=a.prompt_sigma, api_key=a.api_key, read_timeout_s=a.read_timeout,
        cooldown_s=0.0 if a.dry_run else a.cooldown, out_dir=a.out,
        dry_run=a.dry_run, seed=a.seed,
        shared_prefix_tokens=a.shared_prefix_tokens, unique_prefix_tokens=a.unique_prefix_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    logging.basicConfig(
        level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = _build_cfg(a)
    server = None
    if cfg.dry_run:
        server, base_url = _start_dry_run_server(cfg, a.fake_time_scale)
        cfg = replace(cfg, base_url=base_url)
        logger.info("dry run against fake server %s (time_scale=%g)", base_url, a.fake_time_scale)
    out_dir = cfg.out_dir / "concurrency"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "concurrency_summary.jsonl"
    aggs: list[dict[str, Any]] = []
    try:
        with summary_path.open("a", encoding="utf-8") as sfh:
            for level in cfg.levels:
                logger.info("level N=%d for %.0fs", level, cfg.duration_s)
                records, agg = run_level(cfg, level)
                req_path = out_dir / f"requests_{model_slug(cfg.model)}_N{level:03d}.jsonl"
                with req_path.open("w", encoding="utf-8") as rfh:
                    for rec in records:
                        rfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                agg["requests_file"] = str(req_path)
                sfh.write(json.dumps(agg, ensure_ascii=False) + "\n")
                sfh.flush()
                aggs.append(agg)
                if cfg.cooldown_s > 0:
                    time.sleep(cfg.cooldown_s)
    finally:
        if server is not None:
            server.shutdown()
    logger.info("wrote %d levels to %s", len(aggs), summary_path)
    for agg in aggs:
        logger.info(
            "N=%-4d ok=%-5d err=%.1f%%  TTFT p50/p95/p99 = %.3f/%.3f/%.3f s  "
            "%.2f req/s  %.0f tok/s",
            agg["level"], agg["requests_ok"], 100 * agg["error_rate"],
            agg["ttft_p50_s"] or float("nan"), agg["ttft_p95_s"] or float("nan"),
            agg["ttft_p99_s"] or float("nan"), agg["throughput_req_s"],
            agg["throughput_tokens_s"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
