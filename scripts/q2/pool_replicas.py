#!/usr/bin/env python3
"""Replica lifecycle for the pool.

    launching -> healthy (/health 200) -> ready (one max_tokens=1 warm-up answered)
              -> admitted (gate passed and ready) -> draining (no new requests; once in flight
                 is zero or the drain grace is over) -> stopping (SIGTERM sent) -> stopped
    side exits: cancelled (never served), failed (process gone or timed out before ready)

Two engines:

* ``process``: each replica is its own ``vllm serve`` process (``fake_server.py`` in a dry run),
  started through ``remote/serving.py``'s ``ManagedServer`` (its own session, output to
  ``replica_<id>.log``, SIGTERM then SIGKILL on stop); the blocking health poll and the stop run
  in worker threads so the router's event loop never waits on them.
* ``single`` (``virtual=True``): one vLLM engine serves every replica (``start_engine``); a
  replica is an admission unit only.  Its "cold start" is an emulated delay taken from the
  measured sequence ``cold_start_s`` (cycled by launch ordinal), after which it is ready, gated
  and admitted exactly like a process replica; drain, cancel and stop only change its state.

Every transition is one line of ``pool_events.jsonl`` and a rewrite of
``replica_<id>_server.json`` in both modes, with the same event names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import urlparse

REMOTE_DIR = Path(__file__).resolve().parent / "remote"
if str(REMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DIR))

from pool_engine import ReplicaServer, warm_up  # noqa: E402
from serving import ServerSpec, StageError, free_port  # noqa: E402

logger = logging.getLogger("q2.pool.replicas")

TIMES = ("launch", "health", "ready", "admit", "drain", "stop")
LIVE_STATES = ("launching", "healthy", "ready", "admitted", "draining", "stopping")
MAX_CONSECUTIVE_FAILURES = 2


@dataclass(frozen=True)
class ReplicaSpec:
    """How to start one replica: the command template and the poll budgets."""

    python: str
    model: str
    vllm_bin: str = "vllm"
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    gpu_frac: float = 0.19
    max_num_seqs: int = 10
    fake_capacity: int = 10  # dry run: fake_server.py --capacity (= the router's K)
    port_base: int = 8101
    max_live: int = 5  # processes on the GPU at once; a launch waits for a slot below this
    slot_wait_s: float = 150.0  # longest wait for a slot (a drain grace plus a stop grace)
    health_timeout_s: float = 600.0
    warmup_timeout_s: float = 300.0
    stop_grace_s: float = 30.0
    drain_grace_s: float = 60.0
    dry_run: bool = False
    dry_launch_delay_s: float = 2.0
    fake_server: str = ""
    fake_time_scale: float = 0.05

    def command(self, port: int) -> tuple[str, ...]:
        if self.dry_run:
            return (self.python, self.fake_server, "--port", str(port), "--model", self.model,
                    "--capacity", str(self.fake_capacity), "--time-scale",
                    f"{self.fake_time_scale:g}", "--log-level", "WARNING")
        return (self.vllm_bin, "serve", self.model, "--host", "localhost", "--port", str(port),
                "--dtype", self.dtype, "--max-model-len", str(self.max_model_len),
                "--gpu-memory-utilization", f"{self.gpu_frac:g}",
                "--max-num-seqs", str(self.max_num_seqs), "--enable-prefix-caching")


class Replica:
    """One replica and its record; the controller reads the public fields, nothing else."""

    def __init__(self, rid: int, port: int, decision_minute: int, gate_minute: int,
                 server: Optional[ReplicaServer], base_url: Optional[str] = None,
                 emulated_cold_start_s: Optional[float] = None) -> None:
        self.rid = rid
        self.port = port
        self.decision_minute = decision_minute
        self.gate_minute = gate_minute
        self.drain_minute: Optional[int] = None
        self.state = "launching"
        self.in_flight = 0
        self.server = server  # None for a virtual replica (one engine serves them all)
        self.base_url = server.spec.base_url if server is not None else str(base_url)
        self.emulated_cold_start_s = emulated_cold_start_s
        self.times: dict[str, Optional[float]] = {k: None for k in TIMES}
        self.health_after_s: Optional[float] = None
        self.error: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.pid: Optional[int] = None  # recorded at launch; the process handle goes at stop
        self.task: Optional[asyncio.Task[None]] = None
        self.stop_task: Optional[asyncio.Task[None]] = None

    def record(self, extra: dict[str, Any]) -> dict[str, Any]:
        health = self.times["health"]
        ready = self.times["ready"]
        admit = self.times["admit"]
        return {
            "schema": "q2.pool.replica.v1", "replica_id": self.rid, "port": self.port,
            "pid": self.pid, "state": self.state, "decision_minute": self.decision_minute,
            "gate_minute": self.gate_minute, "drain_minute": self.drain_minute,
            **{f"{k}_wallclock": v for k, v in self.times.items()},
            "health_after_s": self.health_after_s,
            "ready_after_s": None if ready is None or self.times["launch"] is None
            else round(ready - self.times["launch"], 3),
            "warmup_s": None if ready is None or health is None else round(ready - health, 3),
            "gate_wait_s": None if admit is None or ready is None
            else round(max(0.0, admit - ready), 3),
            "error": self.error, "exit_code": self.exit_code,
            "cmd": None if self.server is None else list(self.server.spec.cmd),
            "virtual": self.server is None, "emulated_cold_start_s": self.emulated_cold_start_s,
            "base_url": self.base_url,
            **extra,
        }

    @property
    def virtual(self) -> bool:
        return self.server is None


class ReplicaManager:
    """Launch, admit, drain and stop replicas; the controller decides, this class executes."""

    def __init__(self, spec: ReplicaSpec, out_dir: Path, t0: float, minute_s: float,
                 dry_run: bool, virtual: bool = False, engine_url: str = "",
                 cold_start_s: Sequence[float] = (30.0,)) -> None:
        self.spec = spec
        self.out_dir = out_dir
        self.t0 = float(t0)
        self.minute_s = float(minute_s)
        self.dry_run = dry_run
        self.virtual = virtual
        self.engine_url = engine_url.rstrip("/")
        self.cold_start_s = tuple(float(v) for v in cold_start_s) or (30.0,)
        if virtual and not self.engine_url:
            raise ValueError("virtual replicas need engine_url")
        self._replicas: list[Replica] = []
        self._next_id = 0
        self._ports_used: set[int] = set()
        self._events = (out_dir / "pool_events.jsonl").open("a", encoding="utf-8")
        self.consecutive_failures = 0
        self.fatal: Optional[str] = None
        self.on_admit: Optional[Callable[[], Any]] = None  # the router's dispatch hook
        # vLLM sizes its KV cache from the memory it sees free while it profiles, so two engines
        # starting on one card at the same time each read the other's allocations as their own
        # overhead and abort with "No available memory for the cache blocks" (B5, 2026-09-22):
        # a launch holds this lock from the process start until /health answers or it fails.
        self._launch_lock = asyncio.Lock()

    # ----------------------------------------------------------- bookkeeping
    def replicas(self) -> list[Replica]:
        return list(self._replicas)

    def live(self) -> list[Replica]:
        return [r for r in self._replicas if r.state in LIVE_STATES]

    def minute_of(self, ts: float) -> int:
        return int(math.floor((ts - self.t0) / self.minute_s))

    def event(self, replica: Optional[Replica], name: str, detail: Any = None,
              ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        rec = {"ts": round(ts, 3), "minute": self.minute_of(ts),
               "replica_id": None if replica is None else replica.rid,
               "port": None if replica is None else replica.port, "event": name,
               "decision_minute": None if replica is None else replica.decision_minute,
               "gate_minute": None if replica is None else replica.gate_minute,
               "in_flight": None if replica is None else replica.in_flight,
               "detail": detail, "dry_run": self.dry_run}
        self._events.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._events.flush()
        if replica is not None:
            self._write_record(replica)
        logger.info("event %s %s %s", name, "" if replica is None else f"replica {replica.rid}",
                    "" if detail is None else detail)

    def _write_record(self, replica: Replica) -> None:
        path = self.out_dir / f"replica_{replica.rid}_server.json"
        path.write_text(json.dumps(replica.record({"dry_run": self.dry_run}), indent=2) + "\n",
                        encoding="utf-8")

    def close(self) -> None:
        self._events.close()

    # ----------------------------------------------------------- transitions
    def launch(self, decision_minute: int, gate_minute: int) -> Replica:
        rid = self._next_id
        self._next_id += 1
        if self.virtual:
            delay = self.cold_start_s[rid % len(self.cold_start_s)]
            port = urlparse(self.engine_url).port or 0
            replica = Replica(rid, port, decision_minute, gate_minute, None,
                              base_url=self.engine_url, emulated_cold_start_s=delay)
            self._replicas.append(replica)
            replica.times["launch"] = time.time()  # the decision instant is the launch
            self.event(replica, "launch", {"virtual": True, "emulated_cold_start_s": delay,
                                           "engine": self.engine_url})
        else:
            port = self._pick_port(rid)
            spec = ServerSpec(self.spec.command(port), f"http://localhost:{port}",
                              self.out_dir / f"replica_{rid}.log", f"replica {rid}",
                              header=f"replica {rid} decision_minute={decision_minute} "
                                     f"gate_minute={gate_minute} dry_run={self.dry_run}")
            replica = Replica(rid, port, decision_minute, gate_minute,
                              ReplicaServer(spec, self.spec.stop_grace_s))
            self._replicas.append(replica)
        replica.task = asyncio.get_running_loop().create_task(self._lifecycle(replica),
                                                              name=f"replica-{rid}")
        return replica

    def _pick_port(self, rid: int) -> int:
        port = self.spec.port_base + rid
        if self.dry_run:  # a laptop may have anything on 81xx; and never reuse a port in one run
            port = free_port()
            while port in self._ports_used:
                port = free_port()
        self._ports_used.add(port)
        return port

    def gate_ts(self, replica: Replica) -> float:
        return self.t0 + replica.gate_minute * self.minute_s

    def admit_due(self, minute: int) -> list[Replica]:
        """Ready replicas whose gate minute is ``<= minute`` start serving now."""
        out = []
        for r in self._replicas:
            if r.state == "ready" and r.gate_minute <= minute:
                self._admit(r)
                out.append(r)
        return out

    def _admit(self, replica: Replica) -> None:
        now = time.time()
        replica.state = "admitted"
        replica.times["admit"] = now
        ready = replica.times["ready"] or now
        self.event(replica, "admit", {"late_by_s": round(max(0.0, now - self.gate_ts(replica)), 3),
                                      "gate_wait_s": round(max(0.0, now - ready), 3)})
        if self.on_admit is not None:
            self.on_admit()  # queued requests may go to the new replica at once

    def cancel(self, replica: Replica, reason: str) -> None:
        """A launch that will never serve: stop the process, do not count it as a failure."""
        if replica.state not in ("launching", "healthy", "ready"):
            return
        replica.state = "cancelled"
        self.event(replica, "cancel", reason)
        self._schedule_stop(replica)

    def drain(self, replica: Replica) -> None:
        if replica.state != "admitted":
            return
        replica.state = "draining"
        replica.times["drain"] = time.time()
        self.event(replica, "drain", {"in_flight": replica.in_flight})
        self._schedule_stop(replica, wait_in_flight=True)

    def _schedule_stop(self, replica: Replica, wait_in_flight: bool = False) -> None:
        if replica.stop_task is not None:
            return
        replica.stop_task = asyncio.get_running_loop().create_task(
            self._stop(replica, wait_in_flight), name=f"stop-{replica.rid}")

    async def _stop(self, replica: Replica, wait_in_flight: bool) -> None:
        if wait_in_flight:
            limit = time.monotonic() + self.spec.drain_grace_s
            while replica.in_flight > 0 and time.monotonic() < limit:
                await asyncio.sleep(0.5)
            if replica.in_flight > 0:
                self.event(replica, "drain_grace_expired", {"in_flight": replica.in_flight})
        if replica.state == "draining":
            replica.state = "stopping"  # from here on the router must not hand it queued work
        if replica.server is not None:
            await asyncio.to_thread(replica.server.stop)
            proc = replica.server.proc
            replica.exit_code = None if proc is None else proc.returncode
        if replica.state == "stopping":
            replica.state = "stopped"
        replica.times["stop"] = time.time()
        self.event(replica, "stop", {"in_flight": replica.in_flight})

    async def stop_all(self, drain_grace_s: Optional[float] = None,
                       still_busy: Optional[Callable[[], bool]] = None) -> None:
        """End of run: no launch survives, in flight (and ``still_busy``: the router's queue)
        drains for at most the grace, then every process is stopped."""
        grace = self.spec.drain_grace_s if drain_grace_s is None else drain_grace_s
        for r in self._replicas:
            if r.state == "admitted":
                r.state = "draining"
                r.times["drain"] = time.time()
                self.event(r, "drain", {"in_flight": r.in_flight, "end_of_run": True})
            elif r.state in ("launching", "healthy", "ready"):
                r.state = "cancelled"
                self.event(r, "cancel", "end of run")
        limit = time.monotonic() + grace
        while (any(r.in_flight > 0 for r in self._replicas)
               or (still_busy is not None and still_busy())) and time.monotonic() < limit:
            await asyncio.sleep(0.5)
        for r in self._replicas:
            if r.in_flight > 0:
                self.event(r, "drain_grace_expired", {"in_flight": r.in_flight})
        for r in self._replicas:
            if r.state in ("draining", "cancelled", "failed"):
                self._schedule_stop(r)
        tasks = [r.stop_task for r in self._replicas if r.stop_task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ----------------------------------------------------------- lifecycle task
    async def _lifecycle(self, replica: Replica) -> None:
        try:
            if replica.virtual:
                await self._bring_up_virtual(replica)
            else:
                await self._bring_up(replica)
        except Exception as exc:  # a launch that dies must never take the router down
            logger.exception("replica %d lifecycle crashed", replica.rid)
            self._fail(replica, f"{type(exc).__name__}: {exc}")

    async def _bring_up_virtual(self, replica: Replica) -> None:
        """An admission unit over the shared engine: emulated cold start, then ready and gated."""
        delay = float(replica.emulated_cold_start_s or 0.0)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:  # in small steps, so a cancel is noticed promptly
            await asyncio.sleep(min(0.25, deadline - time.monotonic()))
            if replica.state == "cancelled":
                return
        if replica.state == "cancelled":
            return
        now = time.time()
        replica.health_after_s = round(delay, 3)
        replica.state = "healthy"
        replica.times["health"] = now
        self.event(replica, "healthy", {"health_after_s": replica.health_after_s, "virtual": True})
        replica.state = "ready"
        replica.times["ready"] = now
        self.consecutive_failures = 0
        self.event(replica, "ready", {"gate_minute": replica.gate_minute, "virtual": True})
        if self.gate_ts(replica) <= time.time():
            self._admit(replica)

    async def _bring_up(self, replica: Replica) -> None:
        if self.dry_run and self.spec.dry_launch_delay_s > 0:
            await asyncio.sleep(self.spec.dry_launch_delay_s)  # a pretend cold start
        waited = await self._wait_for_slot(replica)
        if replica.state == "cancelled":
            return
        if waited >= 0.5:
            self.event(replica, "slot_wait", {"waited_s": round(waited, 1)})
        lock_wait = time.monotonic()
        async with self._launch_lock:  # one engine profiles the card at a time
            if replica.state == "cancelled":
                return
            queued_for = time.monotonic() - lock_wait
            if queued_for >= 0.5:
                self.event(replica, "launch_wait", {"waited_s": round(queued_for, 1)})
            replica.times["launch"] = time.time()
            replica.server.start()
            replica.pid = None if replica.server.proc is None else replica.server.proc.pid
            self.event(replica, "launch", {"pid": replica.pid})
            try:
                waited = await asyncio.to_thread(
                    replica.server.wait_health, self.spec.health_timeout_s,
                    time.monotonic() + self.spec.health_timeout_s)
            except StageError as exc:
                if replica.state != "cancelled":
                    self._fail(replica, str(exc))
                return
        if replica.state == "cancelled":
            return
        replica.health_after_s = round(waited, 3)
        replica.state = "healthy"
        replica.times["health"] = time.time()
        self.event(replica, "healthy", {"health_after_s": replica.health_after_s})
        error = await asyncio.to_thread(self._warm_up, replica)
        if replica.state == "cancelled":
            return
        if error is not None:
            self._fail(replica, f"warm-up failed: {error}")
            return
        replica.state = "ready"
        replica.times["ready"] = time.time()
        self.consecutive_failures = 0
        self.event(replica, "ready", {"gate_minute": replica.gate_minute})
        if self.gate_ts(replica) <= time.time():
            self._admit(replica)  # gate already open (or missed): ready is admitted, no wait
        await self._watch(replica)

    def _processes_alive(self) -> int:
        return sum(1 for r in self._replicas
                   if r.times["launch"] is not None and r.times["stop"] is None)

    async def _wait_for_slot(self, replica: Replica) -> float:
        """A replica being drained still holds its GPU share: do not start the sixth process."""
        started = time.monotonic()
        while (self._processes_alive() >= self.spec.max_live and replica.state != "cancelled"
               and time.monotonic() - started < self.spec.slot_wait_s):
            await asyncio.sleep(1.0)
        waited = time.monotonic() - started
        if self._processes_alive() >= self.spec.max_live and replica.state != "cancelled":
            logger.warning("replica %d: no free slot after %.0fs; launching anyway",
                           replica.rid, waited)
        return waited

    def _warm_up(self, replica: Replica) -> Optional[str]:
        """One tiny completion, so the first student does not pay for the engine's first step."""
        return warm_up(replica.base_url, self.spec.model, self.spec.warmup_timeout_s)

    async def _watch(self, replica: Replica) -> None:
        """Notice a process that dies while it is supposed to be serving."""
        while replica.state in ("ready", "admitted", "draining"):
            await asyncio.sleep(2.0)
            proc = replica.server.proc
            if proc is None:
                return
            code = proc.poll()
            if code is not None and replica.state in ("ready", "admitted"):
                replica.exit_code = code
                self._fail(replica, f"process exited {code} while {replica.state}",
                           phase=replica.state)
                return

    def _fail(self, replica: Replica, why: str, phase: str = "launch") -> None:
        if replica.state in ("stopped", "cancelled", "failed"):
            return
        replica.state = "failed"
        replica.error = why
        self.consecutive_failures += 1
        self.event(replica, "failed", {"phase": phase, "error": why,
                                       "consecutive": self.consecutive_failures})
        self._schedule_stop(replica)
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES and self.fatal is None:
            self.fatal = (f"{self.consecutive_failures} consecutive replica failures; "
                          f"last: {why}")
            logger.error("fatal: %s", self.fatal)


__all__ = ["MAX_CONSECUTIVE_FAILURES", "Replica", "ReplicaManager", "ReplicaSpec"]
