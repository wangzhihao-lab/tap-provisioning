"""Replica pool, two-class admission and TTFT.

The pool is a set of warm replicas driven by a per-minute trajectory ``x_t`` that the engine hands
over (the policy decided it ``d`` minutes earlier).  Each replica holds at most ``K`` concurrent
streams.  A request waits in its class queue until a stream is free; its TTFT is the queue wait
plus the first-token latency at the concurrency it is admitted into.  In-class requests have strict
priority over off-class ones (the graded-SLO executor).

Two deliberate simplifications:

* a balanced router is assumed, so the concurrency a request is admitted into is
  ``ceil(active / replicas)`` rather than the load of a specific replica;
* TTFT and the per-stream decode rate are frozen at the concurrency seen at admission, which makes
  every completion time known when the request starts and keeps the loop event-driven.
"""

import heapq
import logging
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ServiceConfig
from .workload import IN_CLASS, ArrivalStream

logger = logging.getLogger(__name__)

_AT_N = re.compile(r"^(?P<metric>[a-z0-9_]+)@N=(?P<n>\d+)$")


@dataclass(frozen=True)
class ServiceResult:
    """Per-request outcome of one replay."""

    ttft_s: np.ndarray  # NaN for requests that never started
    kind: np.ndarray
    minute: np.ndarray
    served: np.ndarray
    timed_out: np.ndarray
    rejected: np.ndarray

    @property
    def n_requests(self) -> int:
        """Number of requests replayed."""
        return int(self.ttft_s.size)


def _knee_concurrency(ns: np.ndarray, ys: np.ndarray, slo_s: float, fallback: int) -> int:
    """Largest concurrency whose TTFT p95 stays within ``slo_s``.

    TTFT grows superlinearly with concurrency, so a global linear fit misplaces the knee; instead
    interpolate linearly between the two measured levels that bracket the target.  If the grid never
    reaches the target, extrapolate along the last segment but never beyond 1.5x the top level.
    """
    if ns.size == 0:
        return fallback
    if ys[0] > slo_s:
        raise ValueError(f"TTFT p95 at concurrency 1 is {ys[0]:.2f} s, above the {slo_s:g} s SLO: "
                         "this prompt regime cannot meet the SLO on this model at any replica count")
    for i in range(1, ns.size):
        if ys[i] > slo_s:
            frac = (slo_s - ys[i - 1]) / max(1e-9, ys[i] - ys[i - 1])
            return max(1, int(round(ns[i - 1] + frac * (ns[i] - ns[i - 1]))))
    if ns.size >= 2 and ys[-1] > ys[-2]:
        slope = (ys[-1] - ys[-2]) / (ns[-1] - ns[-2])
        k = ns[-1] + (slo_s - ys[-1]) / slope
        return int(round(min(k, 1.5 * ns[-1])))
    return int(round(1.5 * ns[-1]))


def load_service_config(path: Path, model: Optional[str] = None, slo_in_s: float = 2.0,
                        base: Optional[ServiceConfig] = None, lead_stat: str = "median") -> ServiceConfig:
    """Build a ``ServiceConfig`` from a T3 calibration CSV (batch B format).

    Recognised rows are ``ttft_p95_s@N=<n>``, ``throughput_tokens_s@N=<n>`` and the cold-start
    milestones ``first_completion_done_s`` / ``server_listening_s``.  ``K`` is the largest
    concurrency whose TTFT p95 stays at or below ``slo_in_s``, extrapolated linearly when the grid
    does not reach the target.
    """
    base = base or ServiceConfig()
    df = pd.read_csv(path)
    if model is None:
        model = str(df["model"].iloc[0])
    df = df[df["model"] == model]
    if df.empty:
        raise ValueError(f"no rows for model {model!r} in {path}")
    ttft: List[Tuple[int, float]] = []
    thru: List[Tuple[int, float]] = []
    scalars = {}
    for row in df.itertuples(index=False):
        m = _AT_N.match(str(row.metric))
        if m is None:
            scalars[str(row.metric)] = float(row.median)
            continue
        if m.group("metric") == "ttft_p95_s":
            ttft.append((int(m.group("n")), float(row.median)))
        elif m.group("metric") == "throughput_tokens_s":
            thru.append((int(m.group("n")), float(row.median)))
    if not ttft:
        raise ValueError(f"no ttft_p95_s@N=... rows for model {model!r} in {path}")
    ttft.sort()
    ns = np.array([n for n, _ in ttft], dtype=float)
    ys = np.array([y for _, y in ttft], dtype=float)
    floor = float(ys[0])
    k = _knee_concurrency(ns, ys, slo_in_s, base.max_concurrency)
    decode = max((v for _, v in thru), default=base.decode_tokens_per_s)
    per_stream = next((v for n, v in sorted(thru) if n == 1), base.per_stream_tokens_per_s)
    lead_s = scalars.get("first_completion_done_s", scalars.get("server_listening_s", 0.0))
    if lead_stat == "max":  # first start on a fresh node: the slowest run (weights + kernel compile)
        mx = df[df["metric"] == "first_completion_done_s"]["max"]
        lead_s = float(mx.iloc[0]) if not mx.empty else lead_s
    logger.info("T3 calibration %s: K=%d decode=%.0f tok/s lead=%.1f s", model, k, decode, lead_s)
    return ServiceConfig(
        source=str(path), model=model, max_concurrency=k, decode_tokens_per_s=decode,
        per_stream_tokens_per_s=per_stream, ttft_floor_s=floor, ttft_at_k_s=slo_in_s,
        prefill_tokens_per_s=base.prefill_tokens_per_s,
        lead_min=lead_s / 60.0 if lead_s else base.lead_min,
        commit_block_min=base.commit_block_min, billing_block_min=base.billing_block_min,
        abandon_in_s=base.abandon_in_s,
        abandon_off_s=base.abandon_off_s, max_queue_in=base.max_queue_in,
        max_queue_off=base.max_queue_off)


def _event_minutes(replicas: np.ndarray) -> np.ndarray:
    """Minutes at which the warm replica count changes (minute 0 always included)."""
    if replicas.size == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(replicas, prepend=replicas[0] - 1))
    return change.astype(np.int64)


def _next_class(queues: Tuple[Deque[int], Deque[int]], t_arrive_s: np.ndarray, priority: bool) -> int:
    """The queue to serve next: in-class first, or the earlier arrival when the priority is off."""
    if not queues[1]:
        return 0
    if not queues[0]:
        return 1
    if priority:
        return 0
    return 0 if float(t_arrive_s[queues[0][0]]) <= float(t_arrive_s[queues[1][0]]) else 1


def replay(stream: ArrivalStream, replicas: np.ndarray, cfg: ServiceConfig) -> ServiceResult:
    """Push ``stream`` through a pool that holds ``replicas[t]`` warm replicas in minute ``t``."""
    n = stream.n_requests
    ttft = np.full(n, np.nan, dtype=float)
    served = np.zeros(n, dtype=bool)
    timed_out = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)
    minute = np.minimum((stream.t_arrive_s // 60.0).astype(np.int64), stream.horizon_min - 1)
    x = np.rint(np.asarray(replicas, dtype=float)).astype(np.int64)
    if x.size < stream.horizon_min:
        x = np.pad(x, (0, stream.horizon_min - x.size), mode="edge")

    queues: Tuple[Deque[int], Deque[int]] = (deque(), deque())
    caps = (cfg.max_queue_in, cfg.max_queue_off)
    abandon = (cfg.abandon_in_s, cfg.abandon_off_s)
    done: List[float] = []
    active = 0
    i = 0
    boundaries = _event_minutes(x)
    b = 0
    now = 0.0
    n_rep = int(x[0])

    while i < n or done or queues[0] or queues[1]:
        nxt = []
        if i < n:
            nxt.append(float(stream.t_arrive_s[i]))
        if done:
            nxt.append(done[0])
        if b < boundaries.size:
            nxt.append(float(boundaries[b]) * 60.0)
        if not nxt:
            break
        now = min(nxt)
        while done and done[0] <= now + 1e-9:
            heapq.heappop(done)
            active -= 1
        while b < boundaries.size and float(boundaries[b]) * 60.0 <= now + 1e-9:
            n_rep = int(x[boundaries[b]])
            b += 1
        while i < n and float(stream.t_arrive_s[i]) <= now + 1e-9:
            cls = 0 if int(stream.kind[i]) == IN_CLASS else 1
            if len(queues[cls]) >= caps[cls]:
                rejected[i] = True
            else:
                queues[cls].append(i)
            i += 1
        capacity = n_rep * cfg.max_concurrency
        for cls in (0, 1):  # drop the requests that have waited past their class tolerance
            q = queues[cls]
            while q and now - float(stream.t_arrive_s[q[0]]) > abandon[cls]:
                timed_out[q.popleft()] = True
        while active < capacity and (queues[0] or queues[1]):
            cls = _next_class(queues, stream.t_arrive_s, cfg.in_class_priority)
            j = queues[cls].popleft()
            wait = now - float(stream.t_arrive_s[j])
            conc = max(1, math.ceil((active + 1) / max(1, n_rep)))
            first = cfg.ttft_p95_s(conc)
            if cfg.prefill_tokens_per_s > 0.0:
                first += float(stream.prompt_tokens[j]) / cfg.prefill_tokens_per_s
            ttft[j] = wait + first
            served[j] = True
            rate = cfg.stream_tokens_per_s(conc)
            heapq.heappush(done, now + first + float(stream.output_tokens[j]) / rate)
            active += 1
    for cls in (0, 1):
        for j in queues[cls]:
            timed_out[j] = True
    logger.debug("replay: %d served, %d timed out, %d rejected", int(served.sum()),
                 int(timed_out.sum()), int(rejected.sum()))
    return ServiceResult(ttft, stream.kind.copy(), minute, served, timed_out, rejected)


__all__ = ["ServiceResult", "load_service_config", "replay"]
