"""Replica pool (batch B' step two): controller logic, vendored policies, arrival ledger.

Pure logic only: the controller is driven with fake replica objects, no network, no
subprocesses.  The modules under test live in ``scripts/q2`` (standalone scripts), so that
directory goes on ``sys.path`` first, like ``test_q2_remote.py`` does for ``scripts/q2/remote``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

Q2_DIR = Path(__file__).resolve().parents[1] / "scripts" / "q2"
for folder in (Q2_DIR, Q2_DIR / "remote"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import pool_control  # noqa: E402
import simpolicies  # noqa: E402
import vendor_policies  # noqa: E402
from pool_admission import Abandoned, AdmissionQueue  # noqa: E402
from pool_replicas import ReplicaManager, ReplicaSpec  # noqa: E402
from pool_control import (  # noqa: E402
    ArrivalLedger,
    MinuteController,
    PolicyRunner,
    PoolPlan,
    estimate_tokens,
    parse_idx,
)


# ------------------------------------------------------------------ fakes
class FakeReplica:
    def __init__(self, rid: int, decision_minute: int, gate_minute: int) -> None:
        self.rid = rid
        self.decision_minute = decision_minute
        self.gate_minute = gate_minute
        self.drain_minute: Optional[int] = None
        self.state = "launching"
        self.in_flight = 0


class FakeManager:
    """Replicas as state records; ``ready(rid)`` stands in for the health/warm-up path."""

    def __init__(self) -> None:
        self.items: list[FakeReplica] = []
        self.log: list[tuple] = []

    def replicas(self) -> list[FakeReplica]:
        return list(self.items)

    def launch(self, decision_minute: int, gate_minute: int) -> FakeReplica:
        r = FakeReplica(len(self.items), decision_minute, gate_minute)
        self.items.append(r)
        self.log.append(("launch", r.rid, decision_minute, gate_minute))
        return r

    def cancel(self, replica: FakeReplica, reason: str) -> None:
        replica.state = "cancelled"
        self.log.append(("cancel", replica.rid))

    def drain(self, replica: FakeReplica) -> None:
        replica.state = "draining"
        self.log.append(("drain", replica.rid))

    def admit_due(self, minute: int) -> list[FakeReplica]:
        out = []
        for r in self.items:
            if r.state == "ready" and r.gate_minute <= minute:
                r.state = "admitted"
                out.append(r)
        return out

    def ready(self, *rids: int) -> None:
        for r in self.items:
            if r.rid in rids and r.state in ("launching", "healthy"):
                r.state = "ready"

    def admitted_ids(self) -> list[int]:
        return [r.rid for r in self.items if r.state == "admitted"]

    def fail(self, rid: int) -> None:
        self.items[rid].state = "failed"


class ScriptedRunner:
    """A decision source that answers with a fixed list of targets (raw, before rounding)."""

    def __init__(self, lead: int, prefix: list[float], targets: list[float]) -> None:
        self.lead = lead
        self._prefix = prefix
        self._targets = targets
        self.seen_x_now: list[float] = []
        self.seen_t: list[int] = []
        self.name = "scripted"

    def prefix(self) -> np.ndarray:
        return np.asarray(self._prefix, dtype=float)

    def decide(self, t: int, state: simpolicies.SimState) -> float:
        self.seen_x_now.append(state.x_now)
        self.seen_t.append(state.t)
        return self._targets[t]


def _controller(lead: int, gate: int, duration: int, prefix: list[float], targets: list[float],
                min_replicas: int = 1) -> tuple[MinuteController, FakeManager, ScriptedRunner]:
    manager = FakeManager()
    runner = ScriptedRunner(lead, prefix, targets)
    ledger = ArrivalLedger(duration + lead + 1, t0=1000.0, minute_s=60.0)
    ctl = MinuteController(runner, ledger, manager, lead, gate, duration, min_replicas,
                           cost=simpolicies.CostConfig(x_max=5), max_replicas=5)
    return ctl, manager, runner


# ------------------------------------------------------------------ (a) controller
def test_prefix_fills_installed_and_prestart_launches_with_gate_s() -> None:
    ctl, manager, _ = _controller(lead=8, gate=8, duration=12,
                                  prefix=[1, 1, 2, 2, 3, 3, 3, 2], targets=[2] * 12)
    assert ctl.installed[:8].tolist() == [1, 1, 2, 2, 3, 3, 3, 2]
    launched, drained, cancelled = ctl.prestart()
    # one replica per step up, each gated at the first minute that needs it
    assert [(r.rid, r.gate_minute) for r in manager.items] == [(0, 0), (1, 2), (2, 4)]
    assert launched == [0, 1, 2] and cancelled == []
    # the step down at minute 7 cannot cancel a launch minutes 4-6 still need: it is a drain at 7
    assert drained == [2] and manager.items[2].drain_minute == 7


def test_lead1_gate0_launch_is_immediate_and_serves_when_ready() -> None:
    ctl, manager, runner = _controller(lead=1, gate=0, duration=6, prefix=[1],
                                       targets=[1, 3, 3, 1, 1, 1])  # decision t -> minute t+1
    ctl.prestart()
    assert [(r.rid, r.gate_minute) for r in manager.items] == [(0, 0)]
    manager.ready(0)
    manager.admit_due(0)
    row0, prev = ctl.tick(0, 1000.0)
    assert prev is None and row0.admitted == 1 and row0.installed_t == 1
    assert ctl.installed[1] == 1 and row0.launched_ids == ()
    row1, prev = ctl.tick(1, 1060.0)  # target 3 for s = 2: two launches with gate = t = 1
    assert prev is not None and prev.t == 0
    assert ctl.installed[2] == 3 and row1.launched_ids == (1, 2)
    assert [r.gate_minute for r in manager.items[1:]] == [1, 1]
    # ready mid-minute: admitted as soon as the (already open) gate allows, i.e. at the next tick
    manager.ready(1, 2)
    manager.admit_due(1)
    assert manager.admitted_ids() == [0, 1, 2]
    row2, _ = ctl.tick(2, 1120.0)
    assert row2.admitted == 3 and row2.installed_t == 3 and row2.launched_ids == ()
    assert runner.seen_x_now == [1.0, 1.0, 3.0]  # x_now is the admitted count, not installed[t]
    ctl.tick(3, 1180.0)  # target 1 for s = 4: two admitted replicas registered to drain at 4
    row3 = ctl.rows[-1]
    assert row3.t == 2
    assert sorted(r.drain_minute for r in manager.items if r.drain_minute is not None) == [4, 4]
    assert ctl.installed[4] == 1
    row4, _ = ctl.tick(4, 1240.0)  # the registered drains execute at 4, before the decision
    assert [r.state for r in manager.items].count("draining") == 2
    assert row4.admitted == 1


def test_lead8_gate8_cancels_a_launch_nobody_needs_but_keeps_one_a_minute_needs() -> None:
    # targets: minute t decides installed[t + 8]
    targets = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ctl, manager, _ = _controller(lead=8, gate=8, duration=16, prefix=[1] * 8, targets=targets)
    ctl.prestart()
    manager.ready(0)
    manager.admit_due(0)
    r0, _ = ctl.tick(0, 1000.0)  # installed[8] = 3 -> two launches gated at 8
    assert r0.launched_ids == (1, 2) and [r.gate_minute for r in manager.items[1:]] == [8, 8]
    r1, _ = ctl.tick(1, 1060.0)  # installed[9] = 3: nothing to do
    assert r1.launched_ids == () and r1.drained_ids == () and r1.cancelled_ids == ()
    r2, _ = ctl.tick(2, 1120.0)  # installed[10] = 1: minutes 8-9 still need both -> drains at 10
    assert r2.cancelled_ids == () and sorted(r2.drained_ids) == [1, 2]
    assert all(manager.items[i].drain_minute == 10 for i in (1, 2))
    # a fresh step up then down within the lead: the launch of tick 3 is needed by nobody
    ctl.installed[11] = 1  # (no change) keep going
    targets[3] = 2  # installed[11] = 2 -> one launch gated at 11
    r3, _ = ctl.tick(3, 1180.0)
    assert r3.launched_ids == (3,) and manager.items[3].gate_minute == 11
    targets[4] = 1  # installed[12] = 1 -> replica 3 is needed for minute 11 only -> drain at 12
    r4, _ = ctl.tick(4, 1240.0)
    assert r4.cancelled_ids == () and r4.drained_ids == (3,)
    # whereas a launch whose gate minute itself no longer needs it is cancelled outright
    targets[5] = 2  # installed[13] = 2
    r5, _ = ctl.tick(5, 1300.0)
    assert r5.launched_ids == (4,) and manager.items[4].gate_minute == 13
    ctl.installed[13] = 1  # the minute that asked for it changes its mind (not reachable via
    targets[6] = 1  # decide() in the simulator's structure; exercised directly here)
    r6, _ = ctl.tick(6, 1360.0)
    assert r6.cancelled_ids == (4,) and manager.items[4].state == "cancelled"


def test_gate_semantics_admit_only_at_the_gate_minute() -> None:
    ctl, manager, _ = _controller(lead=8, gate=8, duration=12, prefix=[1] * 8, targets=[2] * 12)
    ctl.prestart()
    manager.ready(0)
    ctl.tick(0, 1000.0)  # launches replica 1 with gate 8
    manager.ready(1)  # ready early (a fast cold start)
    for t in range(1, 8):
        row, _ = ctl.tick(t, 1000.0 + 60 * t)
        assert row.admitted == 1, t  # gate 8 has not passed
    row, _ = ctl.tick(8, 1480.0)
    assert row.admitted == 2 and manager.items[1].state == "admitted"


def test_targets_beyond_the_window_are_recorded_but_not_acted_on() -> None:
    ctl, manager, _ = _controller(lead=8, gate=8, duration=10, prefix=[1] * 8, targets=[4] * 10)
    ctl.prestart()
    manager.ready(0)
    r0, _ = ctl.tick(0, 1000.0)  # s = 8 < 10: acted on
    assert len(r0.launched_ids) == 3
    r2, _ = ctl.tick(2, 1120.0)  # s = 10 == duration: recorded only
    assert ctl.installed[10] == 4 and r2.launched_ids == () and r2.target_s == 4
    assert len(manager.items) == 4


def test_a_failed_launch_is_relaunched_at_the_next_tick() -> None:
    ctl, manager, _ = _controller(lead=1, gate=0, duration=6, prefix=[1], targets=[2, 2, 2, 2, 2, 2])
    ctl.prestart()
    manager.ready(0)
    manager.admit_due(0)
    r0, _ = ctl.tick(0, 1000.0)
    assert r0.launched_ids == (1,)
    manager.fail(1)  # process died before /health
    r1, _ = ctl.tick(1, 1060.0)
    assert r1.launched_ids == (2,) and manager.items[2].gate_minute == 1


def test_rows_close_with_the_arrivals_of_their_own_minute() -> None:
    manager = FakeManager()
    runner = ScriptedRunner(1, [1], [1, 1, 1])
    ledger = ArrivalLedger(5, t0=1000.0, minute_s=60.0)
    queued = [4]
    abandoned = {0: 0, 1: 2, 2: 1}
    ctl = MinuteController(runner, ledger, manager, 1, 0, 3, 1, cost=simpolicies.CostConfig(x_max=5),
                           queue_len=lambda: queued[0], abandoned_in=lambda m: abandoned[m])
    ctl.prestart()
    manager.ready(0)
    ctl.ledger.arrivals[:3] = [5, 7, 9]
    ctl.ledger.tokens[:3] = [50, 70, 90]
    ctl.tick(0, 1000.0)
    _, prev = ctl.tick(1, 1060.0)
    assert prev is not None and (prev.t, prev.arrivals, prev.tokens) == (0, 5, 50)
    assert prev.queued == 4 and prev.abandoned == 0
    ctl.tick(2, 1120.0)
    last = ctl.finish()
    assert last is not None and (last.t, last.arrivals, last.tokens) == (2, 9, 90)
    assert last.abandoned == 1
    assert [r.t for r in ctl.rows] == [0, 1, 2]
    header = pool_control.MINUTE_COLUMNS
    cells = ctl.rows[1].as_csv().split(",")
    assert len(cells) == len(header) and cells[header.index("arrivals")] == "7"
    assert cells[header.index("queued")] == "4" and cells[header.index("abandoned")] == "2"


def test_decisions_only_see_minutes_before_t() -> None:
    """The vendored reactive policy through the controller: util uses lam_in[t-1], x_now = admitted."""
    ctx = simpolicies.PolicyContext(
        horizon_min=6, mu=30.0, lead_min=1, cost=simpolicies.CostConfig(x_max=5),
        policy=simpolicies.PolicyConfig(min_replicas=1, hpa_up_util=0.4, hpa_down_util=0.2,
                                        hpa_cooldown_min=0),
        service=simpolicies.ServiceConfig(), lam_hat_in=np.zeros(8), lam_hat_off=np.zeros(8))
    runner = PolicyRunner("reactive", 1, ctx=ctx)
    manager = FakeManager()
    ledger = ArrivalLedger(8, t0=0.0, minute_s=60.0)
    ctl = MinuteController(runner, ledger, manager, 1, 0, 6, 1, cost=ctx.cost)
    ctl.prestart()
    manager.ready(0)
    manager.admit_due(0)
    ledger.arrivals[0] = 60.0  # minute 0 is busy, but tick 0 cannot see it
    row0, _ = ctl.tick(0, 0.0)
    assert row0.target_s == 1
    row1, _ = ctl.tick(1, 60.0)  # sees minute 0: util 2.0 -> proportional scale-up to 5
    assert row1.target_s == 5 and len(row1.launched_ids) == 4


def test_plan_runner_uses_x_at_t_plus_lead_and_holds_the_last_value() -> None:
    plan = PoolPlan("p", 2, (1, 2, 3, 4))
    runner = PolicyRunner("plan", 2, plan=plan)
    assert runner.prefix().tolist() == [1.0, 2.0]
    state = simpolicies.SimState(np.zeros(4), np.zeros(4), np.zeros(4))
    assert [runner.decide(t, state) for t in range(4)] == [3.0, 4.0, 4.0, 4.0]


def test_targets_are_rounded_up_floored_and_capped() -> None:
    ctl, manager, _ = _controller(lead=1, gate=0, duration=3, prefix=[1], targets=[0.2, 2.1, 9.0],
                                  min_replicas=1)
    ctl.prestart()
    manager.ready(0)
    r0, _ = ctl.tick(0, 0.0)
    r1, _ = ctl.tick(1, 60.0)
    r2, _ = ctl.tick(2, 120.0)
    assert (r0.target_s, r1.target_s, r2.target_s) == (1, 3, 5)


# ------------------------------------------------------------------ (a2) admission queue
class Serving:
    def __init__(self, rid: int, state: str = "admitted") -> None:
        self.rid = rid
        self.state = state
        self.in_flight = 0


def _queue(reps: list[Serving], k: int, abandon_s: float, clock=None) -> AdmissionQueue:
    kwargs = {} if clock is None else {"clock": clock}
    return AdmissionQueue(lambda: reps, k, abandon_s, minute_of=lambda ts: int(ts // 60), **kwargs)


def test_queue_caps_streams_per_replica_and_releases_fifo() -> None:
    async def scenario() -> None:
        reps = [Serving(0)]
        q = _queue(reps, k=2, abandon_s=60.0, clock=lambda: 100.0)
        r1, w1 = await q.acquire(100.0, 100.0)
        r2, w2 = await q.acquire(100.0, 100.0)
        assert (r1, r2) == (reps[0], reps[0]) and reps[0].in_flight == 2 and (w1, w2) == (0.0, 0.0)
        third = asyncio.ensure_future(q.acquire(100.0, 100.0))
        fourth = asyncio.ensure_future(q.acquire(100.0, 100.0))
        await asyncio.sleep(0.01)
        assert not third.done() and len(q) == 2  # both wait: the replica is full
        q.release(reps[0])  # one stream finishes -> exactly one slot -> the queue head
        await asyncio.sleep(0.01)
        assert third.done() and not fourth.done()
        assert third.result()[0] is reps[0] and reps[0].in_flight == 2
        reps.append(Serving(1))  # a second replica is admitted -> dispatch fills it
        q.dispatch()
        await asyncio.sleep(0.01)
        assert fourth.done() and fourth.result()[0] is reps[1] and reps[1].in_flight == 1
        # most free slots first, ties by id: replica 1 (1 of 2 used) beats replica 0 (2 of 2)
        r5, _ = await q.acquire(100.0, 100.0)
        assert r5 is reps[1]
        q.release(reps[1])
        q.release(reps[1])
        q.release(reps[0])
        r6, _ = await q.acquire(100.0, 100.0)
        assert r6 is reps[1]  # 0 used on 1, 1 used on 0

    asyncio.run(scenario())


def test_queue_skips_draining_replicas_until_the_end_of_run() -> None:
    async def scenario() -> None:
        reps = [Serving(0, "draining"), Serving(1, "stopping")]
        q = _queue(reps, k=2, abandon_s=60.0)
        assert q.free_replica() is None
        q.end_of_run = True
        assert q.free_replica() is reps[0]  # draining takes the tail, stopping never does

    asyncio.run(scenario())


def test_stale_queued_request_is_abandoned_and_counted_in_its_minute() -> None:
    async def scenario() -> None:
        import time as _time

        reps = [Serving(0)]
        q = _queue(reps, k=1, abandon_s=0.05)  # real clock: the expiry timer has to fire
        first, _ = await q.acquire(_time.time(), _time.time())
        assert first is reps[0]
        with pytest.raises(Abandoned):
            await q.acquire(_time.time(), _time.time())  # queued behind the full replica
        assert q.n_abandoned == 1 and len(q) == 0 and reps[0].in_flight == 1
        assert sum(q.abandoned_by_minute.values()) == 1

    asyncio.run(scenario())


def test_stale_retry_is_refused_at_once_by_first_arrival() -> None:
    async def scenario() -> None:
        reps = [Serving(0)]
        clock = [1000.0]
        q = _queue(reps, k=5, abandon_s=60.0, clock=lambda: clock[0])
        r, _ = await q.acquire(first_arrival=950.0, arrival=1000.0)  # 50 s old: still fine
        assert r is reps[0]
        clock[0] = 1011.0
        with pytest.raises(Abandoned):  # a retry of an idx first seen 61 s ago
            await q.acquire(first_arrival=950.0, arrival=1011.0)
        assert q.n_abandoned == 1 and q.abandoned_in(int(1011.0 // 60)) == 1

    asyncio.run(scenario())


def test_ledger_gives_the_first_arrival_of_an_idx_and_no_idx_requests_their_own() -> None:
    ledger = ArrivalLedger(5, t0=1000.0, minute_s=60.0)
    first = ledger.record(_body(7, 10, 8), 1001.0)
    retry = ledger.record(_body(7, 10, 8), 1050.0)
    assert (first.first_arrival, retry.first_arrival, retry.duplicate) == (1001.0, 1001.0, True)
    assert ledger.first_arrival(7) == 1001.0 and ledger.first_arrival(8) is None
    probe = ledger.record({"messages": [{"role": "user", "content": "no prefix"}]}, 1070.0)
    again = ledger.record({"messages": [{"role": "user", "content": "no prefix"}]}, 1075.0)
    assert (probe.first_arrival, again.first_arrival) == (1070.0, 1075.0)  # each on its own
    assert not probe.duplicate and not again.duplicate


def test_queue_abandon_all_flushes_the_line() -> None:
    async def scenario() -> None:
        reps: list[Serving] = []  # nothing admitted yet
        q = _queue(reps, k=2, abandon_s=60.0, clock=lambda: 100.0)
        waits = [asyncio.ensure_future(q.acquire(100.0, 100.0)) for _ in range(3)]
        await asyncio.sleep(0.01)
        assert len(q) == 3
        assert q.abandon_all("pool ended") == 3
        await asyncio.sleep(0.01)
        assert all(w.done() and isinstance(w.exception(), Abandoned) for w in waits)
        assert len(q) == 0 and q.n_abandoned == 3

    asyncio.run(scenario())


# ------------------------------------------------------------------ (a3) virtual replicas
def _virtual_manager(tmp_path: Path, t0: float, cold: tuple[float, ...]) -> ReplicaManager:
    spec = ReplicaSpec(python="python", model="fake", dry_run=True, drain_grace_s=0.5)
    return ReplicaManager(spec, tmp_path, t0=t0, minute_s=60.0, dry_run=True, virtual=True,
                          engine_url="http://localhost:8101/", cold_start_s=cold)


def _events(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in
            (tmp_path / "pool_events.jsonl").read_text(encoding="utf-8").splitlines()]


def test_virtual_replica_emulates_the_cold_start_then_is_gated_and_admitted(tmp_path: Path) -> None:
    async def scenario() -> None:
        t0 = time.time() + 0.3  # gate minute 0 opens 0.3 s from now
        manager = _virtual_manager(tmp_path, t0, (0.1,))
        r = manager.launch(decision_minute=-1, gate_minute=0)
        assert r.virtual and r.base_url == "http://localhost:8101" and r.port == 8101
        assert r.state == "launching" and r.emulated_cold_start_s == 0.1
        await asyncio.sleep(0.2)
        assert r.state == "ready" and r.health_after_s == 0.1  # ready before the gate: waits
        assert r.times["launch"] is not None and r.times["ready"] is not None
        assert manager.admit_due(-1) == []
        await asyncio.sleep(0.2)
        assert manager.admit_due(0) == [r] and r.state == "admitted"
        # one whose gate minute is already behind us is admitted as soon as it is ready
        late = manager.launch(decision_minute=-2, gate_minute=-2)
        await asyncio.sleep(0.2)
        assert late.state == "admitted" and late.times["admit"] is not None
        manager.close()

    asyncio.run(scenario())
    names = [e["event"] for e in _events(tmp_path) if e["replica_id"] == 0]
    assert names == ["launch", "healthy", "ready", "admit"]
    rec = json.loads((tmp_path / "replica_0_server.json").read_text(encoding="utf-8"))
    assert rec["virtual"] is True and rec["emulated_cold_start_s"] == 0.1
    assert rec["health_after_s"] == 0.1 and rec["cmd"] is None and rec["pid"] is None
    assert all(rec[f"{k}_wallclock"] is not None for k in ("launch", "health", "ready", "admit"))


def test_virtual_cold_starts_cycle_through_the_measured_sequence(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = _virtual_manager(tmp_path, time.time(), (0.05, 0.02, 0.03))
        reps = [manager.launch(0, 0) for _ in range(5)]
        assert [r.emulated_cold_start_s for r in reps] == [0.05, 0.02, 0.03, 0.05, 0.02]
        assert len({r.base_url for r in reps}) == 1  # one engine behind every replica
        await asyncio.sleep(0.15)
        assert all(r.state == "admitted" for r in reps)
        manager.close()

    asyncio.run(scenario())


def test_virtual_drain_and_cancel_change_state_without_a_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = _virtual_manager(tmp_path, time.time(), (0.02,))
        served = manager.launch(0, 0)
        await asyncio.sleep(0.1)
        assert served.state == "admitted"
        served.in_flight = 1
        manager.drain(served)
        assert served.state == "draining"
        await asyncio.sleep(0.1)
        assert served.state == "draining"  # in flight is not zero: waits (up to the grace)
        served.in_flight = 0
        await asyncio.sleep(0.7)
        assert served.state == "stopped" and served.times["stop"] is not None
        pending = manager.launch(1, 9)  # still in its emulated cold start
        manager.cancel(pending, "not needed")
        await asyncio.sleep(0.1)
        assert pending.state == "cancelled" and pending.times["ready"] is None
        await manager.stop_all(0.1)
        manager.close()

    asyncio.run(scenario())
    by_replica: dict[int, list[str]] = {}
    for e in _events(tmp_path):
        if e["replica_id"] is not None:
            by_replica.setdefault(e["replica_id"], []).append(e["event"])
    assert by_replica[0] == ["launch", "healthy", "ready", "admit", "drain", "stop"]
    assert by_replica[1] == ["launch", "cancel", "stop"]


# ------------------------------------------------------------------ (a4) stage pass-through
def test_stage_passes_engine_and_cold_starts_through(tmp_path: Path) -> None:
    import pool_stage
    from layout import RemoteLayout
    from measurements import StepContext

    ctx = StepContext(layout=RemoteLayout(root=tmp_path), python=Path("/venv/bin/python"),
                      dry_run=True)
    single = {"engine": "single", "cold_start_s": [3, 2], "abandon_s": 20, "max_replicas": 4,
              "gpu_frac": 0.22, "replica_concurrency": 3, "_path": "x"}
    cmd = pool_stage._pool_cmd(ctx, single, tmp_path / "out", 8000, "fixed", 1.0, 30, 0, fixed=4)
    assert cmd[cmd.index("--engine") + 1] == "single"
    assert cmd[cmd.index("--cold-start-s") + 1] == "3,2"
    assert cmd[cmd.index("--max-replicas") + 1] == "4"
    assert cmd[cmd.index("--replica-concurrency") + 1] == "3"
    assert cmd[cmd.index("--abandon-s") + 1] == "20"
    assert "--gpu-frac" not in cmd  # the single engine has fixed arguments
    process = {"_path": "x"}
    cmd = pool_stage._pool_cmd(ctx, process, tmp_path / "out", 8000, "reactive", 1.0, 45, 8,
                               context="pool_inputs/c.json")
    assert cmd[cmd.index("--engine") + 1] == "process" and "--cold-start-s" not in cmd
    assert cmd[cmd.index("--gpu-frac") + 1] == "0.19"
    assert cmd[cmd.index("--max-replicas") + 1] == "5"
    assert cmd[cmd.index("--context") + 1].endswith("pool_inputs/c.json")
    assert "--dry-run" in cmd
    with pytest.raises(Exception, match="engine must be"):
        pool_stage._pool_cmd(ctx, {"engine": "dual", "_path": "x"}, tmp_path, 8000, "fixed",
                             1.0, 1, 0, fixed=1)


# ------------------------------------------------------------------ (b) vendored package
def test_vendored_package_exports_and_registry() -> None:
    assert set(simpolicies.POLICY_FACTORY) >= {"reactive", "velocity"}
    assert simpolicies.POLICY_FACTORY["reactive"] is simpolicies.ReactivePolicy
    assert simpolicies.POLICY_FACTORY["velocity"] is simpolicies.TokenVelocityPolicy
    for name in ("PolicyContext", "SimState", "as_int_replicas", "CostConfig", "PolicyConfig",
                 "ServiceConfig", "register_policy"):
        assert hasattr(simpolicies, name), name


def test_manifest_matches_the_sources_and_the_vendored_files() -> None:
    package = Q2_DIR / "simpolicies"
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == vendor_policies.SCHEMA
    for name, entry in manifest["sources"].items():
        src = vendor_policies.SOURCES[name]
        assert hashlib.sha256(src.read_bytes()).hexdigest() == entry["sha256"], (
            f"{name}: pilot/src changed; run vendor_policies.py")
    for name, entry in manifest["generated"].items():
        blob = (package / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"], name
    assert vendor_policies.stale_files() == []


def test_vendored_files_differ_from_the_sources_only_in_the_import_lines() -> None:
    package = Q2_DIR / "simpolicies"
    for name, src in vendor_policies.SOURCES.items():
        expected, _ = vendor_policies.rewrite_imports(src.read_text(encoding="utf-8"))
        assert (package / name).read_text(encoding="utf-8") == expected, name


def test_vendored_and_real_policy_agree_on_a_trajectory() -> None:
    pytest.importorskip("pilot.sim")
    from pilot.sim import policies as real  # noqa: F401  (registration)
    from pilot.sim.policies.velocity import TokenVelocityPolicy as RealVelocity

    lam = np.array([10, 40, 90, 150, 200, 210, 180, 120, 60, 20], dtype=float)
    tokens = lam * 300.0
    kwargs = dict(horizon_min=10, mu=60.0, lead_min=8,
                  policy=simpolicies.PolicyConfig(min_replicas=1, velocity_hysteresis=False),
                  service=simpolicies.ServiceConfig(), lam_hat_in=np.zeros(10),
                  lam_hat_off=np.zeros(10))
    ctx_v = simpolicies.PolicyContext(cost=simpolicies.CostConfig(x_max=5), **kwargs)
    ctx_r = ctx_v  # the frozen dataclasses are structurally identical copies
    vendored, real_policy = simpolicies.TokenVelocityPolicy(ctx_v), RealVelocity(ctx_r)
    for p in (vendored, real_policy):
        p.reset()
        p.start_run()
    outs = []
    for p in (vendored, real_policy):
        state = simpolicies.SimState(lam, np.zeros(10), tokens)
        seq = []
        for t in range(10):
            state.t = t
            state.x_now = 1.0
            seq.append(p.decide(t, state))
        outs.append(seq)
    assert outs[0] == outs[1]


# ------------------------------------------------------------------ (c) arrivals
def test_idx_prefix_and_token_estimate() -> None:
    assert parse_idx("[17] 课程问题：xx") == 17
    assert parse_idx("  [3] x") == 3
    assert parse_idx("no prefix [3]") is None
    # ceil(len / 1.45) + 12 + max_tokens
    assert estimate_tokens("a" * 145, 100) == 100 + 12 + 100
    assert estimate_tokens("a" * 146, 0) == 101 + 12


def _body(idx: int, n_chars: int, max_tokens: int) -> dict:
    return {"messages": [{"role": "user", "content": f"[{idx}] " + "x" * n_chars}],
            "max_tokens": max_tokens, "stream": True}


def test_ledger_dedups_retries_by_idx_and_bins_by_minute() -> None:
    ledger = ArrivalLedger(5, t0=1000.0, minute_s=60.0)
    first = ledger.record(_body(1, 100, 50), 1001.0)
    again = ledger.record(_body(1, 100, 50), 1005.0)  # the client's retry after a 503
    other = ledger.record(_body(2, 29, 16), 1061.0)  # 29 chars + 4 prefix chars = 33
    assert (first.minute, first.duplicate) == (0, False)
    assert again.duplicate is True and again.idx == 1
    assert (other.minute, other.duplicate) == (1, False)
    assert ledger.arrivals.tolist() == [1, 1, 0, 0, 0]
    assert ledger.tokens[0] == first.tokens and first.tokens == estimate_tokens("[1] " + "x" * 100, 50)
    assert ledger.tokens[1] == estimate_tokens("[2] " + "x" * 29, 16)
    assert (ledger.n_requests, ledger.n_unique) == (3, 2)
    # a request without the prefix (the concurrency probe) is always counted
    ledger.record({"messages": [{"role": "system", "content": "s"},
                                {"role": "user", "content": "q"}], "max_tokens": 8}, 1062.0)
    ledger.record({"messages": [{"role": "user", "content": "q"}], "max_tokens": 8}, 1063.0)
    assert ledger.arrivals[1] == 3
    # before T0 or past the array: routed, not binned
    before = ledger.record(_body(9, 10, 1), 900.0)
    assert before.minute == -2 and ledger.arrivals.sum() == 4
