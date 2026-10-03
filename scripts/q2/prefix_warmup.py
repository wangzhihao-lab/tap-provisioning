#!/usr/bin/env python3
"""Q2 / T3: does one warm-up request before a classroom burst buy anything?

With prefix caching enabled, K requests that share a long course-material system
prompt (2-4k synthetic tokens) are issued as a burst, in two arms:

  cold  - no warm-up; the shared prefix is cold when the burst starts
  warm  - one warm-up request with the same prefix is issued first

Each repeat uses a fresh nonce inside the course material so the ``cold`` arm is
genuinely cold even on a server that has served earlier repeats.  Per-request
records go to ``<out>/prefix/prefix_requests.jsonl`` and per-arm aggregates plus
the arm difference to ``<out>/prefix/prefix_summary.jsonl``.

    python prefix_warmup.py --base-url http://localhost:8000 --model Qwen/...
    python prefix_warmup.py --dry-run --out results/dryrun   # local fake server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger("q2.prefix")

WORDS_PER_TOKEN = 0.75
ARMS = ("cold", "warm")

_SYLLABUS = (
    "Week {i}: the lecture covers the definition, the standard construction and two worked",
    "examples; the lab asks students to implement the operation and to measure its running",
    "time on the provided dataset; the assessed exercise is due before the next session and",
    "must be submitted through the course platform with the completed report template.",
    "Common mistakes from previous cohorts are listed at the end of the handout, together",
    "with the marking rubric and the reference solution released after the deadline.",
)
_QUESTIONS = (
    "Why does my implementation return the wrong value for the empty input?",
    "Can you walk me through the third example on the handout?",
    "What is the difference between the two variants introduced in the lecture?",
    "My code compiles but the test on line 40 fails; what should I check first?",
    "How do I explain this result in the report without copying the reference solution?",
)


@dataclass(frozen=True)
class PrefixConfig:
    base_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    prefix_tokens: int = 3000
    burst_k: int = 20
    repeats: int = 3
    max_tokens: int = 32
    warmup_max_tokens: int = 8
    warmup_settle_s: float = 1.0
    temperature: float = 0.0
    api_key: str = ""
    concurrent: bool = False
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 300.0
    out_dir: Path = Path("results")
    dry_run: bool = False
    seed: int = 20260917


def make_course_material(rng: np.random.Generator, n_tokens: int, nonce: str) -> str:
    """Deterministic synthetic course material of roughly ``n_tokens`` tokens."""
    n_words = max(64, int(n_tokens * WORDS_PER_TOKEN))
    words: list[str] = f"COURSE PACK {nonce} - Data Structures and Databases, autumn term.".split()
    week = 1
    while len(words) < n_words:
        for line in _SYLLABUS:
            words.extend(line.format(i=week).split())
        words.append(str(rng.integers(1000, 9999)))
        week += 1
    return " ".join(words[:n_words])


def _post_streaming(
    cfg: PrefixConfig, system: str, question: str, max_tokens: int
) -> dict[str, Any]:
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
    }
    start = time.monotonic()
    ttft: float | None = None
    n_out = 0
    try:
        with requests.post(
            url, json=payload, headers=headers, stream=True,
            timeout=(cfg.connect_timeout_s, cfg.read_timeout_s),
        ) as resp:
            if resp.status_code != 200:
                return {"ok": False, "error": f"http {resp.status_code}", "ttft_s": None}
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
                text = (choice.get("delta") or {}).get("content") or choice.get("text") or ""
                if text:
                    n_out += 1
                    if ttft is None:
                        ttft = time.monotonic() - start
    except requests.RequestException as exc:
        return {"ok": False, "error": type(exc).__name__, "ttft_s": None}
    return {"ok": ttft is not None, "ttft_s": ttft, "e2e_s": time.monotonic() - start,
            "output_tokens": n_out, "error": None if ttft is not None else "no content"}


def run_arm(cfg: PrefixConfig, arm: str, repeat: int) -> list[dict[str, Any]]:
    """Run one arm of one repeat; returns the K burst records (warm-up excluded)."""
    rng = np.random.default_rng(cfg.seed + repeat * 101 + ARMS.index(arm))
    nonce = f"{uuid.uuid4().hex[:8]}-r{repeat}-{arm}"
    system = make_course_material(rng, cfg.prefix_tokens, nonce)
    if arm == "warm":
        warm = _post_streaming(cfg, system, "Summarise the course pack in one sentence.",
                               cfg.warmup_max_tokens)
        logger.debug("warm-up request ttft=%s", warm.get("ttft_s"))
        time.sleep(cfg.warmup_settle_s)
    questions = [
        f"[student {i:02d}] {_QUESTIONS[i % len(_QUESTIONS)]}" for i in range(cfg.burst_k)
    ]
    t0 = time.monotonic()
    if cfg.concurrent:
        with ThreadPoolExecutor(max_workers=cfg.burst_k) as pool:
            results = list(pool.map(
                lambda q: _post_streaming(cfg, system, q, cfg.max_tokens), questions
            ))
    else:
        results = [_post_streaming(cfg, system, q, cfg.max_tokens) for q in questions]
    return [
        {"schema": "q2.prefix.request.v1", "arm": arm, "repeat": repeat, "index": i,
         "model": cfg.model, "prefix_tokens": cfg.prefix_tokens, "nonce": nonce,
         "dry_run": cfg.dry_run, "burst_elapsed_s": round(time.monotonic() - t0, 4), **res}
        for i, res in enumerate(results)
    ]


def summarise_arm(cfg: PrefixConfig, arm: str, repeat: int, recs: list[dict]) -> dict[str, Any]:
    ttfts = np.array([r["ttft_s"] for r in recs if r.get("ok")], dtype=float)
    first = next((r["ttft_s"] for r in recs if r["index"] == 0 and r.get("ok")), None)
    return {
        "schema": "q2.prefix.arm.v1",
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": cfg.model, "arm": arm, "repeat": repeat, "dry_run": cfg.dry_run,
        "prefix_tokens": cfg.prefix_tokens, "burst_k": cfg.burst_k,
        "concurrent": cfg.concurrent, "n_ok": int(ttfts.size),
        "ttft_first_s": first,
        "ttft_mean_s": float(ttfts.mean()) if ttfts.size else None,
        "ttft_median_s": float(np.median(ttfts)) if ttfts.size else None,
        "ttft_p95_s": float(np.percentile(ttfts, 95)) if ttfts.size else None,
    }


def _delta(cold: dict[str, Any], warm: dict[str, Any]) -> dict[str, Any]:
    def diff(key: str) -> float | None:
        a, b = cold.get(key), warm.get(key)
        return None if a is None or b is None else float(a - b)

    return {
        "schema": "q2.prefix.delta.v1", "model": cold["model"], "repeat": cold["repeat"],
        "dry_run": cold["dry_run"], "prefix_tokens": cold["prefix_tokens"],
        "burst_k": cold["burst_k"],
        "ttft_first_gain_s": diff("ttft_first_s"),
        "ttft_mean_gain_s": diff("ttft_mean_s"),
        "ttft_p95_gain_s": diff("ttft_p95_s"),
        "cold_ttft_first_s": cold.get("ttft_first_s"),
        "warm_ttft_first_s": warm.get("ttft_first_s"),
        "cold_ttft_mean_s": cold.get("ttft_mean_s"),
        "warm_ttft_mean_s": warm.get("ttft_mean_s"),
    }


def _start_dry_run_server(cfg: PrefixConfig, time_scale: float) -> tuple[Any, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_server import FakeServerConfig, start_fake_server

    server, _thread, base_url = start_fake_server(
        FakeServerConfig(model=cfg.model, time_scale=time_scale, seed=cfg.seed, prefix_cache=True)
    )
    return server, base_url


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    d = PrefixConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=d.base_url)
    p.add_argument("--model", default=d.model)
    p.add_argument("--prefix-tokens", type=int, default=d.prefix_tokens, help="2000-4000")
    p.add_argument("--burst-k", type=int, default=d.burst_k)
    p.add_argument("--repeats", type=int, default=d.repeats)
    p.add_argument("--max-tokens", type=int, default=d.max_tokens)
    p.add_argument("--warmup-max-tokens", type=int, default=d.warmup_max_tokens)
    p.add_argument("--warmup-settle", type=float, default=d.warmup_settle_s)
    p.add_argument("--concurrent", action="store_true", help="issue the K burst requests at once")
    p.add_argument("--api-key", default=d.api_key)
    p.add_argument("--read-timeout", type=float, default=d.read_timeout_s)
    p.add_argument("--out", type=Path, default=d.out_dir)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fake-time-scale", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    cfg = PrefixConfig(
        base_url=a.base_url, model=a.model, prefix_tokens=a.prefix_tokens, burst_k=a.burst_k,
        repeats=a.repeats, max_tokens=a.max_tokens, warmup_max_tokens=a.warmup_max_tokens,
        warmup_settle_s=0.05 if a.dry_run else a.warmup_settle, concurrent=a.concurrent,
        api_key=a.api_key, read_timeout_s=a.read_timeout, out_dir=a.out,
        dry_run=a.dry_run, seed=a.seed,
    )
    server = None
    if cfg.dry_run:
        server, base_url = _start_dry_run_server(cfg, a.fake_time_scale)
        cfg = replace(cfg, base_url=base_url)
        logger.info("dry run against fake server %s", base_url)
    out_dir = cfg.out_dir / "prefix"
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path, sum_path = out_dir / "prefix_requests.jsonl", out_dir / "prefix_summary.jsonl"
    deltas: list[dict[str, Any]] = []
    try:
        with req_path.open("a", encoding="utf-8") as rfh, \
                sum_path.open("a", encoding="utf-8") as sfh:
            for repeat in range(cfg.repeats):
                arms: dict[str, dict[str, Any]] = {}
                for arm in ARMS:
                    logger.info("repeat %d/%d arm=%s K=%d", repeat + 1, cfg.repeats,
                                arm, cfg.burst_k)
                    recs = run_arm(cfg, arm, repeat)
                    for rec in recs:
                        rfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    arms[arm] = summarise_arm(cfg, arm, repeat, recs)
                    sfh.write(json.dumps(arms[arm], ensure_ascii=False) + "\n")
                delta = _delta(arms["cold"], arms["warm"])
                sfh.write(json.dumps(delta, ensure_ascii=False) + "\n")
                sfh.flush()
                rfh.flush()
                deltas.append(delta)
    finally:
        if server is not None:
            server.shutdown()
    logger.info("wrote %d repeats to %s", len(deltas), sum_path)
    for d in deltas:
        logger.info(
            "repeat %d: first-request TTFT cold %.3f s vs warm %.3f s (gain %.3f s); "
            "burst mean gain %.3f s",
            d["repeat"], d["cold_ttft_first_s"] or float("nan"),
            d["warm_ttft_first_s"] or float("nan"), d["ttft_first_gain_s"] or float("nan"),
            d["ttft_mean_gain_s"] or float("nan"),
        )
    gains = [d["ttft_first_gain_s"] for d in deltas if d["ttft_first_gain_s"] is not None]
    if gains:
        logger.info("median first-request TTFT gain from one warm-up: %.3f s",
                    float(np.median(gains)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
