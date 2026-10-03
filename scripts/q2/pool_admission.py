#!/usr/bin/env python3
"""The router's admission layer: ``K`` streams per replica, one FIFO queue, abandonment.

The counterpart of the simulator's service model (``pilot.sim.service.replay``): a replica runs
at most ``K`` streams, everything else waits in one first-come queue, and an in-class request
that has waited longer than ``abandon_in_s`` is dropped.  Without this layer the queue would form
inside vLLM (unbounded) and a request that waited three minutes would still be served, so under
overload the two worlds would diverge for a reason that has nothing to do with the policies.
Pure asyncio, no aiohttp: the tests drive it with fake replica objects.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence


class Abandoned(Exception):
    """A request waited longer than ``abandon_s`` since its first arrival (or is a stale retry)."""


class ServingReplica(Protocol):
    rid: int
    state: str
    in_flight: int


@dataclass
class _Waiter:
    first_arrival: float
    arrival: float
    fut: "asyncio.Future[Any]"


class AdmissionQueue:
    """FIFO admission in front of the replicas: ``K`` streams each, the rest wait, stale ones go.

    ``acquire`` reserves a slot (``in_flight`` is incremented here, so a woken waiter can never
    lose its slot to a later arrival) or parks the request; ``release`` frees the slot and hands
    it to the queue head.  Abandonment is measured from the request's *first* arrival, so a
    retry of an ``[idx]`` that has been waiting for longer than ``abandon_s`` is refused at once.
    At the end of the run (``end_of_run``) draining replicas still take queued work, so the tail
    is served the way the simulator's horizon tail serves it.
    """

    def __init__(self, replicas: Callable[[], Sequence[ServingReplica]], concurrency: int,
                 abandon_s: float, minute_of: Callable[[float], int],
                 clock: Callable[[], float] = time.time) -> None:
        self._replicas = replicas
        self.concurrency = int(concurrency)
        self.abandon_s = float(abandon_s)
        self._minute_of = minute_of
        self._clock = clock
        self._queue: deque[_Waiter] = deque()
        self.end_of_run = False
        self.n_abandoned = 0
        self.abandoned_by_minute: Counter[int] = Counter()

    def __len__(self) -> int:
        return sum(1 for w in self._queue if not w.fut.done())

    def free_replica(self) -> Optional[ServingReplica]:
        """The serving replica with the most free slots (ties: smallest id), or None."""
        states = ("admitted", "draining") if self.end_of_run else ("admitted",)
        cands = [r for r in self._replicas()
                 if r.state in states and r.in_flight < self.concurrency]
        return min(cands, key=lambda r: (r.in_flight, r.rid)) if cands else None

    def abandoned_in(self, minute: int) -> int:
        return int(self.abandoned_by_minute.get(minute, 0))

    def _abandon(self, waiter: _Waiter, now: float, why: str) -> None:
        self.n_abandoned += 1
        self.abandoned_by_minute[self._minute_of(now)] += 1
        if not waiter.fut.done():
            waiter.fut.set_exception(Abandoned(why))

    def _expire(self, waiter: _Waiter) -> None:
        if waiter.fut.done():
            return
        try:
            self._queue.remove(waiter)
        except ValueError:
            pass
        self._abandon(waiter, self._clock(), f"queued longer than {self.abandon_s:g}s")

    async def acquire(self, first_arrival: float, arrival: float) -> tuple[ServingReplica, float]:
        """Wait for a slot; returns ``(replica, queue_wait_s)`` or raises ``Abandoned``."""
        now = self._clock()
        if now - first_arrival > self.abandon_s:
            self.n_abandoned += 1
            self.abandoned_by_minute[self._minute_of(now)] += 1
            raise Abandoned(f"first arrival {now - first_arrival:.1f}s ago, over {self.abandon_s:g}s")
        if not self._queue:
            replica = self.free_replica()
            if replica is not None:
                replica.in_flight += 1
                return replica, 0.0
        loop = asyncio.get_running_loop()
        waiter = _Waiter(first_arrival, arrival, loop.create_future())
        self._queue.append(waiter)
        handle = loop.call_later(max(0.0, first_arrival + self.abandon_s - now), self._expire, waiter)
        try:
            replica = await waiter.fut
        finally:
            handle.cancel()
            if waiter in self._queue:  # cancelled while waiting: leave no ghost in the line
                self._queue.remove(waiter)
        return replica, self._clock() - arrival

    def release(self, replica: ServingReplica) -> int:
        """A forwarded request finished: free its slot and forward what the slot allows."""
        replica.in_flight -= 1
        return self.dispatch()

    def dispatch(self) -> int:
        """Forward queue heads while there are free slots; returns how many were forwarded."""
        n = 0
        while self._queue:
            waiter = self._queue[0]
            if waiter.fut.done():
                self._queue.popleft()
                continue
            now = self._clock()
            if now - waiter.first_arrival > self.abandon_s:
                self._queue.popleft()
                self._abandon(waiter, now, f"queued longer than {self.abandon_s:g}s")
                continue
            replica = self.free_replica()
            if replica is None:
                break
            self._queue.popleft()
            replica.in_flight += 1
            waiter.fut.set_result(replica)
            n += 1
        return n

    def abandon_all(self, why: str) -> int:
        """End of run: nothing left will be served."""
        n = 0
        now = self._clock()
        while self._queue:
            waiter = self._queue.popleft()
            if not waiter.fut.done():
                self._abandon(waiter, now, why)
                n += 1
        return n


__all__ = ["Abandoned", "AdmissionQueue", "ServingReplica"]
