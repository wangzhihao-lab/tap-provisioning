#!/usr/bin/env python3
"""Controller logic of the replica pool: arrival ledger, decision source, minute controller.

Pure state and arithmetic, no network and no subprocesses, so the tests can drive the controller
with fake replica objects.  The structure mirrors ``pilot.sim.engine.trajectory`` (the replica-pool
section 3): ``installed[]`` has ``duration + lead + 1`` slots, its first ``lead`` slots come from
the policy's ``prefix``, and the decision taken at minute ``t`` writes ``installed[t + lead]``
only.  What differs from the simulator is that ``x_now`` is the number of replicas actually
admitted, and that every write to ``installed`` is followed by a reconcile against the replicas
that exist (section 5.3 step 5): launch the shortfall with the tier's gate, or cancel launches
that no minute still needs and register drains at ``s`` for the rest.

A ``pool_minutes.csv`` row is complete only when its minute is over, so row ``t`` is emitted at
tick ``t + 1`` (``finish`` emits the last one) with the arrivals and tokens of minute ``t``
itself; the decision fields of the row are the ones taken at tick ``t``.

The router's admission layer (``K`` streams per replica, FIFO queue, abandonment) lives in
``pool_admission.py``; the ledger here keeps the first arrival of every ``[idx]`` it needs.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from simpolicies import (
    POLICY_FACTORY,
    CostConfig,
    PolicyConfig,
    PolicyContext,
    ServiceConfig,
    SimState,
    as_int_replicas,
)

logger = logging.getLogger("q2.pool.control")

CHARS_PER_TOKEN = 1.45  # replay_client.py's convention
PROMPT_OVERHEAD_TOKENS = 12  # chat template around one user message
IDX_RE = re.compile(r"^\s*\[(\d+)\]")
FEEDBACK_POLICIES = ("reactive", "velocity")
POLICY_KINDS = FEEDBACK_POLICIES + ("plan", "fixed")
PENDING_STATES = ("launching", "healthy", "ready")  # started, not serving yet
SERVING_STATES = ("admitted",)
MINUTE_COLUMNS = ("t", "ts", "arrivals", "tokens", "in_flight", "queued", "abandoned", "admitted",
                  "installed_t", "s", "target_s", "launched_ids", "drained_ids", "cancelled_ids")


# ------------------------------------------------------------------ arrivals
def prompt_text(body: dict[str, Any]) -> str:
    """The text of a chat/completions body, every message concatenated (as fake_server does)."""
    if "prompt" in body:
        prompt = body["prompt"]
        return prompt if isinstance(prompt, str) else "\n".join(map(str, prompt))
    return "\n".join(str(m.get("content", "")) for m in body.get("messages", ())
                     if isinstance(m, dict))


def parse_idx(text: str) -> Optional[int]:
    """The ``[idx]`` prefix replay_client.py puts on every prompt, or None."""
    match = IDX_RE.match(text)
    return int(match.group(1)) if match else None


def requested_max_tokens(body: dict[str, Any]) -> int:
    value = body.get("max_tokens") or body.get("max_completion_tokens") or 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def estimate_tokens(text: str, max_tokens: int) -> int:
    """Prompt tokens at 1.45 chars/token, the template overhead and the requested output."""
    return int(math.ceil(len(text) / CHARS_PER_TOKEN)) + PROMPT_OVERHEAD_TOKENS + int(max_tokens)


@dataclass(frozen=True)
class ArrivalRecord:
    idx: Optional[int]
    minute: int
    tokens: int
    duplicate: bool
    first_arrival: float  # this request's own arrival, or the first sighting of its ``[idx]``


class ArrivalLedger:
    """Per-minute arrivals and token estimates, one count per distinct ``[idx]``."""

    def __init__(self, n_minutes: int, t0: float, minute_s: float) -> None:
        self.arrivals = np.zeros(n_minutes, dtype=float)
        self.tokens = np.zeros(n_minutes, dtype=float)
        self.t0 = float(t0)
        self.minute_s = float(minute_s)
        self.n_requests = 0
        self.n_unique = 0
        self._first: dict[int, float] = {}

    def minute_of(self, ts: float) -> int:
        return int(math.floor((ts - self.t0) / self.minute_s))

    def record(self, body: dict[str, Any], ts: float) -> ArrivalRecord:
        """Count one request; a retry of an ``[idx]`` already seen is routed but not counted."""
        text = prompt_text(body)
        idx = parse_idx(text)
        tokens = estimate_tokens(text, requested_max_tokens(body))
        minute = self.minute_of(ts)
        self.n_requests += 1
        duplicate = idx is not None and idx in self._first
        first = ts
        if duplicate:
            first = self._first[idx]  # type: ignore[index]
        else:
            if idx is not None:
                self._first[idx] = ts
            self.n_unique += 1
            if 0 <= minute < self.arrivals.size:
                self.arrivals[minute] += 1.0
                self.tokens[minute] += float(tokens)
        return ArrivalRecord(idx, minute, tokens, duplicate, first)

    def first_arrival(self, idx: int) -> Optional[float]:
        return self._first.get(idx)


# ------------------------------------------------------------------ decision source
def _filtered(cls: type, values: dict[str, Any], what: str) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        logger.warning("%s: ignoring unknown keys %s", what, unknown)
    return cls(**{k: v for k, v in values.items() if k in known})


def load_context(path: Path) -> tuple[PolicyContext, dict[str, Any]]:
    """``context.json`` -> the vendored ``PolicyContext`` and the raw dict."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    horizon = int(raw["horizon_min"])
    lead = int(raw["lead_min"])
    cost = _filtered(CostConfig, dict(raw.get("cost", {})), "cost")
    policy = _filtered(PolicyConfig, dict(raw.get("policy", {})), "policy")
    service = _filtered(ServiceConfig, dict(raw.get("service", {})), "service")
    ctx = PolicyContext(horizon_min=horizon, mu=float(raw["mu"]), lead_min=lead, cost=cost,
                        policy=policy, service=service,
                        lam_hat_in=np.zeros(horizon + lead + 1, dtype=float),
                        lam_hat_off=np.zeros(horizon + lead + 1, dtype=float))
    return ctx, raw


@dataclass(frozen=True)
class PoolPlan:
    """A precomputed trajectory (``plan.json``); ``x[t]`` beyond the end holds the last value."""

    label: str
    lead_min: int
    x: tuple[int, ...]

    def at(self, s: int) -> int:
        return int(self.x[min(max(0, s), len(self.x) - 1)])

    @classmethod
    def load(cls, path: Path) -> "PoolPlan":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        x = tuple(int(round(float(v))) for v in raw["x"])
        if not x:
            raise ValueError(f"{path}: empty plan")
        return cls(str(raw.get("label", Path(path).stem)), int(raw.get("lead_min", 0)), x)


class PolicyRunner:
    """One decision source: a vendored feedback policy, a precomputed plan or a constant."""

    def __init__(self, kind: str, lead: int, ctx: Optional[PolicyContext] = None,
                 plan: Optional[PoolPlan] = None, fixed: Optional[int] = None) -> None:
        if kind not in POLICY_KINDS:
            raise ValueError(f"unknown policy kind {kind!r}; known: {POLICY_KINDS}")
        self.kind = kind
        self.lead = int(lead)
        self.plan = plan
        self.fixed = fixed
        self.policy = None
        if kind in FEEDBACK_POLICIES:
            if ctx is None:
                raise ValueError(f"policy {kind} needs a context.json")
            self.policy = POLICY_FACTORY[kind](ctx)
            self.policy.reset()
            self.policy.start_run()
        elif kind == "plan":
            if plan is None:
                raise ValueError("policy plan needs a plan.json")
        elif fixed is None:
            raise ValueError("policy fixed needs --fixed-replicas")

    @property
    def name(self) -> str:
        if self.kind == "plan" and self.plan is not None:
            return f"plan:{self.plan.label}"
        return self.kind if self.kind != "fixed" else f"fixed:{self.fixed}"

    def prefix(self) -> np.ndarray:
        """Targets for minutes ``0 .. lead-1`` (``Policy.prefix`` / the plan's head)."""
        if self.policy is not None:
            return np.asarray(self.policy.prefix(self.lead), dtype=float)
        if self.plan is not None:
            return np.asarray([self.plan.at(s) for s in range(self.lead)], dtype=float)
        return np.full(self.lead, float(self.fixed or 0), dtype=float)

    def decide(self, t: int, state: SimState) -> float:
        """The raw target for minute ``t + lead`` (rounded by the controller)."""
        if self.policy is not None:
            return float(self.policy.decide(t, state))
        if self.plan is not None:
            return float(self.plan.at(t + self.lead))
        return float(self.fixed or 0)


# ------------------------------------------------------------------ controller
class ReplicaLike(Protocol):
    rid: int
    state: str
    decision_minute: int
    gate_minute: int
    drain_minute: Optional[int]
    in_flight: int


class ReplicaManagerLike(Protocol):
    def replicas(self) -> Sequence[ReplicaLike]: ...
    def launch(self, decision_minute: int, gate_minute: int) -> ReplicaLike: ...
    def cancel(self, replica: ReplicaLike, reason: str) -> None: ...
    def drain(self, replica: ReplicaLike) -> None: ...
    def admit_due(self, minute: int) -> Sequence[ReplicaLike]: ...


@dataclass
class MinuteRow:
    t: int
    ts: float
    in_flight: int
    admitted: int
    installed_t: int
    s: int
    target_s: int
    launched_ids: tuple[int, ...]
    drained_ids: tuple[int, ...]
    cancelled_ids: tuple[int, ...]
    queued: int = 0  # router queue length at the tick
    arrivals: Optional[int] = None
    tokens: Optional[int] = None
    abandoned: Optional[int] = None  # requests abandoned during minute t

    def as_csv(self) -> str:
        ids = [";".join(str(i) for i in v) for v in
               (self.launched_ids, self.drained_ids, self.cancelled_ids)]
        blank = ""
        cells = [self.t, f"{self.ts:.3f}", blank if self.arrivals is None else self.arrivals,
                 blank if self.tokens is None else self.tokens, self.in_flight, self.queued,
                 blank if self.abandoned is None else self.abandoned, self.admitted,
                 self.installed_t, self.s, self.target_s, *ids]
        return ",".join(str(c) for c in cells)


def counts_for(replica: ReplicaLike, s: int) -> bool:
    """Whether ``replica`` will be serving during minute ``s`` if nothing else changes."""
    if replica.state in SERVING_STATES or (replica.state in PENDING_STATES
                                           and replica.gate_minute <= s):
        return replica.drain_minute is None or replica.drain_minute > s
    return False


class MinuteController:
    """``installed[]``, the per-tick decision and the reconcile step of the pool."""

    def __init__(self, runner: PolicyRunner, ledger: ArrivalLedger, manager: ReplicaManagerLike,
                 lead: int, gate_lead: int, duration: int, min_replicas: int,
                 cost: Optional[CostConfig] = None, max_replicas: int = 5,
                 queue_len: Callable[[], int] = lambda: 0,
                 abandoned_in: Callable[[int], int] = lambda _m: 0) -> None:
        if gate_lead > lead:
            raise ValueError(f"lead_gate_min {gate_lead} exceeds the policy lead {lead}")
        self.runner = runner
        self.ledger = ledger
        self.manager = manager
        self._queue_len = queue_len
        self._abandoned_in = abandoned_in
        self.lead = int(lead)
        self.gate_lead = int(gate_lead)
        self.duration = int(duration)
        self.min_replicas = int(min_replicas)
        self.cost = cost
        self.max_replicas = int(max_replicas)
        self.installed = np.zeros(self.duration + self.lead + 1, dtype=float)
        prefix = runner.prefix()
        if self.lead:
            padded = np.pad(prefix, (0, max(0, self.lead - prefix.size)), mode="edge")
            self.installed[:self.lead] = self._cap(padded[:self.lead])
        self.rows: list[MinuteRow] = []
        self._pending_row: Optional[MinuteRow] = None
        self.targets_raw: dict[int, float] = {}

    # ----------------------------------------------------------- helpers
    def _cap(self, values: np.ndarray) -> np.ndarray:
        capped = np.minimum(values, self.max_replicas)
        if np.any(capped != values):
            logger.warning("targets above --max-replicas %d capped: %s", self.max_replicas,
                           values.tolist())
        return capped

    def _will(self, s: int) -> int:
        return sum(1 for r in self.manager.replicas() if counts_for(r, s))

    def admitted_count(self) -> int:
        return sum(1 for r in self.manager.replicas() if r.state in SERVING_STATES)

    def in_flight(self) -> int:
        return sum(r.in_flight for r in self.manager.replicas())

    def _cancel_is_safe(self, replica: ReplicaLike, s: int) -> bool:
        """A pending launch may go only if no minute in ``[gate, s)`` still needs it."""
        for m in range(max(0, replica.gate_minute), s):
            without = sum(1 for r in self.manager.replicas()
                          if r is not replica and counts_for(r, m))
            if without < int(self.installed[m]):
                return False
        return True

    def reconcile(self, t: int, s: int) -> tuple[list[int], list[int], list[int]]:
        """Make the replicas that will exist at minute ``s`` match ``installed[s]``.

        Returns ``(launched, drained, cancelled)`` replica ids.  ``t`` is the current minute
        (-1 before the run: the prefix launches carry the gate ``s`` itself).
        """
        want = int(self.installed[s])
        launched: list[int] = []
        drained: list[int] = []
        cancelled: list[int] = []
        will = self._will(s)
        if want > will:
            gate = s if t < 0 else t + self.gate_lead
            for _ in range(want - will):
                launched.append(self.manager.launch(t, gate).rid)
            return launched, drained, cancelled
        excess = will - want
        if excess <= 0:
            return launched, drained, cancelled
        pending = [r for r in self.manager.replicas()
                   if counts_for(r, s) and r.state in PENDING_STATES and r.gate_minute > t]
        for r in sorted(pending, key=lambda r: (-r.gate_minute, -r.rid)):
            if excess == 0:
                break
            if self._cancel_is_safe(r, s):
                self.manager.cancel(r, f"installed[{s}]={want} below the {will} that existed")
                cancelled.append(r.rid)
                excess -= 1
        if excess > 0:
            counting = [r for r in self.manager.replicas() if counts_for(r, s)]
            for r in sorted(counting, key=lambda r: (r.in_flight, -r.rid))[:excess]:
                r.drain_minute = s
                drained.append(r.rid)
        return launched, drained, cancelled

    # ----------------------------------------------------------- the two entry points
    def prestart(self) -> tuple[list[int], list[int], list[int]]:
        """Launch the prefix (minutes ``0 .. lead-1``) at pool start, each with its own gate."""
        launched: list[int] = []
        drained: list[int] = []
        cancelled: list[int] = []
        for s in range(min(self.lead, self.installed.size)):
            a, b, c = self.reconcile(-1, s)
            launched += a
            drained += b
            cancelled += c
        return launched, drained, cancelled

    def due_stops(self, t: int) -> list[int]:
        """Replicas registered to leave at ``t``: drain the serving ones, cancel the rest."""
        out: list[int] = []
        for r in list(self.manager.replicas()):
            if r.drain_minute is None or r.drain_minute > t:
                continue
            if r.state in SERVING_STATES:
                self.manager.drain(r)
                out.append(r.rid)
            elif r.state in PENDING_STATES:
                self.manager.cancel(r, f"registered to leave at minute {r.drain_minute}")
                out.append(r.rid)
        return out

    def tick(self, t: int, now: float) -> tuple[MinuteRow, Optional[MinuteRow]]:
        """One controller tick at ``T0 + t * minute_s``.

        Returns ``(row_t, row_previous)``: the still-open row of ``t`` and the row of ``t - 1``,
        now complete (None at ``t = 0``).  A row carries the decisions of its tick: replicas
        launched then, replicas registered to drain at ``s`` and launches cancelled then.  The
        stops executed at ``t`` (registered at an earlier tick) are in the events log.
        """
        left_now = self.due_stops(t)
        if left_now:
            logger.info("minute %d: replicas %s leave as registered", t, left_now)
        self.manager.admit_due(t)
        state = SimState(lam_in=self.ledger.arrivals, lam_off=np.zeros_like(self.ledger.arrivals),
                         tokens=self.ledger.tokens, x_now=float(self.admitted_count()), t=t)
        raw = self.runner.decide(t, state)
        target = as_int_replicas(raw, self.cost, floor=self.min_replicas)
        target = float(self._cap(np.asarray([target]))[0])
        s = t + self.lead
        self.installed[s] = target
        self.targets_raw[t] = float(raw)
        launched: list[int] = []
        drained: list[int] = []
        cancelled: list[int] = []
        if s < self.duration:
            launched, drained, cancelled = self.reconcile(t, s)
        else:
            logger.info("minute %d: target %g for s=%d lies beyond the window; not acted on",
                        t, target, s)
        previous = self._close_previous_row(t)
        row = MinuteRow(t=t, ts=now, in_flight=self.in_flight(), admitted=self.admitted_count(),
                        installed_t=int(self.installed[t]), s=s, target_s=int(target),
                        launched_ids=tuple(launched), drained_ids=tuple(drained),
                        cancelled_ids=tuple(cancelled), queued=int(self._queue_len()))
        self._pending_row = row
        return row, previous

    def _close_previous_row(self, t: int) -> Optional[MinuteRow]:
        row = self._pending_row
        if row is None:
            return None
        row.arrivals = int(self.ledger.arrivals[row.t])
        row.tokens = int(self.ledger.tokens[row.t])
        row.abandoned = int(self._abandoned_in(row.t))
        self.rows.append(row)
        self._pending_row = None
        return row

    def finish(self) -> Optional[MinuteRow]:
        """Close the last open row once minute ``duration - 1`` is over."""
        return self._close_previous_row(self.duration)


__all__ = ["ArrivalLedger", "ArrivalRecord", "MINUTE_COLUMNS", "MinuteController", "MinuteRow",
           "PoolPlan", "PolicyRunner", "counts_for", "estimate_tokens", "load_context",
           "parse_idx", "prompt_text", "requested_max_tokens"]
