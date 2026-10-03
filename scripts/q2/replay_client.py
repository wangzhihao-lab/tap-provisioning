#!/usr/bin/env python3
"""Replay a real prompted burst against a live server and record every request's TTFT (F10).

The trace (``make_replay_trace.py``) holds one row per student request of one real class burst:
``offset_s`` from the first request, ``prompt_chars`` and ``reply_chars``.  ``--scale k`` sends
every request ``k`` times (copies jittered by up to 2 s) so that the same burst shape is replayed
at ``k`` times the institution's per-class intensity.  Requests are dispatched at their offsets
from ``t0 = start + --t0-delay`` whatever the server is doing: a request that finds no server (the
"late replica" condition) retries every 0.5 s until it is answered or ``--give-up-s`` has passed,
and its TTFT is counted from the *scheduled* arrival - which is exactly what a student waiting for
a replica that is still starting experiences.  Output length is pinned to the reply the platform
really produced (``ceil(reply_chars / 1.45)`` tokens, ``ignore_eos``), so the load is the measured
one and not the model's mood.

Writes ``<out>/<label>.jsonl`` (one record per request) and ``<out>/<label>_summary.json``.
Standalone: only ``requests``; runs unchanged against ``fake_server.py`` with ``--dry-run``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("replay_client")

CHARS_PER_TOKEN = 1.45
FILLER = "请结合课堂内容说明这个概念的定义、适用场景和一个具体例子，并指出常见错误。"
SLO_S = 2.0
BIN_S = 10.0
_local = threading.local()


@dataclass(frozen=True)
class Request:
    idx: int
    offset_s: float
    student: int
    prompt_chars: int
    reply_chars: int

    @property
    def max_tokens(self) -> int:
        return int(min(320, max(16, math.ceil(self.reply_chars / CHARS_PER_TOKEN))))


def load_trace(path: Path, scale: int, max_offset: float, seed: int) -> list[Request]:
    rng = random.Random(seed)
    rows: list[Request] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            offset = float(row["offset_s"])
            if offset > max_offset:
                continue
            for k in range(scale):
                jitter = rng.uniform(0.0, 2.0) if k else 0.0
                rows.append(Request(0, offset + jitter, int(row["student"]),
                                    int(row["prompt_chars"]), int(row["reply_chars"])))
    rows.sort(key=lambda r: r.offset_s)
    return [Request(i, r.offset_s, r.student, r.prompt_chars, r.reply_chars)
            for i, r in enumerate(rows)]


def make_prompt(idx: int, n_chars: int) -> str:
    """A unique prefix per request defeats prefix caching; the body has the measured length."""
    body = (FILLER * (n_chars // len(FILLER) + 1))[: max(8, n_chars)]
    return f"[{idx}] 课程问题：{body}"


def _session() -> requests.Session:
    if getattr(_local, "session", None) is None:
        _local.session = requests.Session()
    return _local.session


def one_request(req: Request, url: str, model: str, t0: float, give_up_s: float,
                dry_run: bool) -> dict[str, Any]:
    payload = {"model": model, "messages": [{"role": "user",
                                             "content": make_prompt(req.idx, req.prompt_chars)}],
               "max_tokens": req.max_tokens, "temperature": 0.0, "stream": True,
               "ignore_eos": True}
    t_sched = t0 + req.offset_s
    t_send = time.perf_counter()
    attempts, n_chunks = 0, 0
    t_first: float | None = None
    t_done: float | None = None
    error: str | None = None
    while True:
        attempts += 1
        try:
            with _session().post(url, json=payload, stream=True, timeout=(10, 900)) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        n_chunks += 1
                        if t_first is None:
                            t_first = time.perf_counter()
                t_done = time.perf_counter()
                error = None
                break
        except Exception as exc:  # connection refused while the server starts, resets, timeouts
            error = f"{type(exc).__name__}: {str(exc)[:100]}"
            if time.perf_counter() - t_sched >= give_up_s:
                break
            time.sleep(0.5)
    return {"idx": req.idx, "student": req.student, "offset_s": round(req.offset_s, 3),
            "prompt_chars": req.prompt_chars, "max_tokens": req.max_tokens,
            "sched_lag_s": round(t_send - t_sched, 4), "attempts": attempts,
            "ttft_s": None if t_first is None else round(t_first - t_sched, 4),
            "ttft_from_send_s": None if t_first is None else round(t_first - t_send, 4),
            "done_s": None if t_done is None else round(t_done - t_sched, 4),
            "n_chunks": n_chunks, "error": error, "dry_run": dry_run}


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 4)


def summarize(records: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    ttft = [r["ttft_s"] for r in records if r["ttft_s"] is not None]
    n = len(records)
    bins: list[dict[str, Any]] = []
    horizon = max((r["offset_s"] for r in records), default=0.0)
    for b in range(int(horizon // BIN_S) + 1):
        lo, hi = b * BIN_S, (b + 1) * BIN_S
        part = [r for r in records if lo <= r["offset_s"] < hi]
        t = [r["ttft_s"] for r in part if r["ttft_s"] is not None]
        bins.append({"bin_start_s": lo, "n_arrivals": len(part), "n_answered": len(t),
                     "ttft_p50_s": _quantile(t, 0.5), "ttft_p95_s": _quantile(t, 0.95),
                     "share_within_slo": (round(sum(1 for v in t if v <= SLO_S) / len(part), 4)
                                          if part else None)})
    return {**meta, "n_requests": n, "n_answered": len(ttft),
            "n_gave_up": sum(1 for r in records if r["ttft_s"] is None),
            "retried_requests": sum(1 for r in records if r["attempts"] > 1),
            "ttft_p50_s": _quantile(ttft, 0.5), "ttft_p90_s": _quantile(ttft, 0.9),
            "ttft_p95_s": _quantile(ttft, 0.95), "ttft_max_s": _quantile(ttft, 1.0),
            "share_within_slo": round(sum(1 for v in ttft if v <= SLO_S) / n, 4) if n else None,
            "share_within_5s": round(sum(1 for v in ttft if v <= 5.0) / n, 4) if n else None,
            "sched_lag_p95_s": _quantile([r["sched_lag_s"] for r in records], 0.95),
            "bins": bins}


def warm_up(url: str, model: str, n: int) -> None:
    for i in range(n):
        try:
            payload = {"model": model, "messages": [{"role": "user", "content": f"预热 {i}"}],
                       "max_tokens": 32, "temperature": 0.0}
            _session().post(url, json=payload, timeout=(10, 300)).raise_for_status()
        except Exception as exc:
            logger.warning("warm-up %d failed: %s", i, exc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--max-offset", type=float, default=480.0)
    ap.add_argument("--t0-delay", type=float, default=0.0,
                    help="seconds between process start and the first scheduled request")
    ap.add_argument("--give-up-s", type=float, default=180.0)
    ap.add_argument("--warmup", type=int, default=0, help="sequential requests before t0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reqs = load_trace(args.trace, args.scale, args.max_offset, args.seed)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    start_wall = time.time()
    if args.warmup:
        warm_up(url, args.model, args.warmup)
    t0 = time.perf_counter() + args.t0_delay
    t0_wall = time.time() + max(0.0, t0 - time.perf_counter())
    logger.info("%s: %d requests (scale %d, offsets <= %.0f s), t0 in %.1f s",
                args.label, len(reqs), args.scale, args.max_offset, args.t0_delay)
    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def run(req: Request) -> None:
        rec = one_request(req, url, args.model, t0, args.give_up_s, args.dry_run)
        with lock:
            results.append(rec)

    with ThreadPoolExecutor(max_workers=max(64, min(4096, len(reqs)))) as pool:
        for req in reqs:
            wait = t0 + req.offset_s - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            pool.submit(run, req)
    results.sort(key=lambda r: r["idx"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"{args.label}.jsonl").open("w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {"label": args.label, "scale": args.scale, "max_offset_s": args.max_offset,
            "t0_delay_s": args.t0_delay, "give_up_s": args.give_up_s, "model": args.model,
            "trace": str(args.trace), "dry_run": args.dry_run, "chars_per_token": CHARS_PER_TOKEN,
            "slo_s": SLO_S, "process_start_wallclock": start_wall, "t0_wallclock": t0_wall,
            "t0_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0_wall)),
            "finished_wallclock": time.time()}
    summary = summarize(results, meta)
    (args.out / f"{args.label}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("%s: answered %d/%d, gave up %d, TTFT p50 %s p95 %s, within %.0f s: %s",
                args.label, summary["n_answered"], summary["n_requests"], summary["n_gave_up"],
                summary["ttft_p50_s"], summary["ttft_p95_s"], SLO_S, summary["share_within_slo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
