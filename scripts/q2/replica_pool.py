#!/usr/bin/env python3
"""Policy-driven replica pool on one GPU box: router + replica manager + minute controller.

Batch B' step two.  One process serves ``POST /v1/chat/completions``
on ``--port`` and is the admission layer in front of the replicas: at most
``--replica-concurrency`` streams per admitted replica, the rest wait in one FIFO queue, and a
request whose first arrival is older than ``--abandon-s`` is answered 503 ``abandoned`` (the
simulator's ``abandon_in_s``).  ``--engine process``: every replica is its own ``vllm serve``
(``fake_server.py`` under ``--dry-run``) on ``--replica-port-base + id``.  ``--engine single``:
one engine on ``--replica-port-base`` serves every replica, and a replica is a virtual admission
unit whose cold start is an emulated delay from ``--cold-start-s`` (section 11: co-resident
processes on one card share the memory bandwidth and give no capacity scaling).  A minute
controller ticks at ``T0 + minute_s * t`` and runs the vendored
simulator policy (``simpolicies/``: the same ``reactive`` / ``velocity`` code as
``pilot.sim``), a precomputed plan or a constant, with the simulator's decision structure:
``decide(t)`` writes ``installed[t + lead]``, the first ``lead`` minutes come from the prefix.

Outputs under ``--out``: ``pool_minutes.csv`` (one row per minute), ``pool_requests.jsonl``,
``pool_events.jsonl``, ``replica_<id>_server.json`` and ``replica_<id>.log`` per replica,
``pool_meta.json`` (arguments, the vendored package's MANIFEST, T0, wall clocks) and
``pool.log``.  Exit 0 after the window and the drain, or after SIGTERM/SIGINT (every replica is
stopped first); exit 1 when two consecutive replica launches fail.

    replica_pool.py --out DIR --model PATH --vllm-bin PATH --python PATH --policy reactive \\
        --context context.json --t0-wallclock EPOCH --duration-min 45 --lead-gate-min 0
    replica_pool.py --dry-run --minute-s 6 ...     # fake replicas, 45 minutes in 4.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pool_admission import Abandoned, AdmissionQueue  # noqa: E402
from pool_control import (  # noqa: E402
    MINUTE_COLUMNS,
    ArrivalLedger,
    MinuteController,
    PolicyRunner,
    PoolPlan,
    load_context,
)
from pool_engine import ReplicaServer, start_engine  # noqa: E402
from pool_replicas import ReplicaManager, ReplicaSpec  # noqa: E402
from serving import StageError  # noqa: E402  (pool_replicas put remote/ on the path)

logger = logging.getLogger("q2.replica_pool")

UPSTREAM_TIMEOUT_S = 900.0
UPSTREAM_CONNECT_S = 10.0
CLIENT_MAX_BYTES = 4 * 1024 * 1024
MANIFEST_PATH = HERE / "simpolicies" / "MANIFEST.json"


class Router:
    """The aiohttp front: admission queue over the replicas, streamed pass-through."""

    def __init__(self, manager: ReplicaManager, ledger: ArrivalLedger, out_dir: Path,
                 concurrency: int, abandon_s: float, dry_run: bool) -> None:
        self.manager = manager
        self.ledger = ledger
        self.dry_run = dry_run
        self.queue = AdmissionQueue(manager.replicas, concurrency, abandon_s, ledger.minute_of)
        manager.on_admit = self.queue.dispatch
        self.current_minute = -1
        self.n_forwarded = 0
        self.n_upstream_errors = 0
        self._log = (out_dir / "pool_requests.jsonl").open("a", encoding="utf-8")
        self._session: Optional[aiohttp.ClientSession] = None
        self._runner: Optional[web.AppRunner] = None

    async def start(self, port: int) -> None:
        app = web.Application(client_max_size=CLIENT_MAX_BYTES)
        app.add_routes([web.post("/v1/chat/completions", self.chat),
                        web.get("/health", self.health), web.get("/stats", self.stats)])
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "localhost", port, backlog=1024).start()
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=UPSTREAM_TIMEOUT_S, sock_connect=UPSTREAM_CONNECT_S))
        logger.info("router listening on http://localhost:%d (K=%d per replica, abandon %gs)",
                    port, self.queue.concurrency, self.queue.abandon_s)

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
        if self._runner is not None:
            await self._runner.cleanup()
        self._log.close()

    def snapshot(self) -> dict[str, Any]:
        reps = self.manager.replicas()
        return {"minute": self.current_minute, "admitted": sum(r.state == "admitted" for r in reps),
                "in_flight": sum(r.in_flight for r in reps), "queued": len(self.queue),
                "requests": self.ledger.n_requests, "unique_arrivals": self.ledger.n_unique,
                "forwarded": self.n_forwarded, "abandoned": self.queue.n_abandoned,
                "upstream_errors": self.n_upstream_errors,
                "replica_concurrency": self.queue.concurrency, "abandon_s": self.queue.abandon_s,
                "replicas": [{"id": r.rid, "port": r.port, "state": r.state,
                              "in_flight": r.in_flight, "gate_minute": r.gate_minute,
                              "drain_minute": r.drain_minute} for r in reps],
                "dry_run": self.dry_run}

    async def health(self, _request: web.Request) -> web.Response:
        snap = self.snapshot()
        return web.json_response({"admitted": snap["admitted"], "minute": snap["minute"],
                                  "queued": snap["queued"], "dry_run": self.dry_run})

    async def stats(self, _request: web.Request) -> web.Response:
        return web.json_response(self.snapshot())

    def _record(self, **fields: Any) -> None:
        self._log.write(json.dumps(fields, ensure_ascii=False) + "\n")
        self._log.flush()

    async def chat(self, request: web.Request) -> web.StreamResponse:
        arrival = time.time()
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("body is not an object")
        except (ValueError, aiohttp.ClientError) as exc:
            return web.json_response({"error": {"message": f"bad json: {exc}"}}, status=400)
        rec = self.ledger.record(body, arrival)
        base = {"idx": rec.idx, "arrival_ts": round(arrival, 3), "minute": rec.minute,
                "duplicate": rec.duplicate, "first_arrival_ts": round(rec.first_arrival, 3),
                "tokens_est": rec.tokens}
        try:
            replica, waited = await self.queue.acquire(rec.first_arrival, arrival)
        except Abandoned as exc:
            self._record(**base, replica_id=None, in_flight_at_admit=None,
                         queue_wait_s=round(time.time() - arrival, 4), forwarded=False,
                         abandoned=True, proxy_ttft_s=None, status=503, client_gone=False,
                         done_ts=round(time.time(), 3))
            return web.json_response({"error": "abandoned", "detail": str(exc)}, status=503)
        self.n_forwarded += 1
        at_admit = replica.in_flight
        dispatched = time.time()
        first: Optional[float] = None
        status: Any = None
        client_gone = False
        response: Optional[web.StreamResponse] = None
        assert self._session is not None
        try:
            async with self._session.post(f"{replica.base_url}/v1/chat/completions",
                                          json=body) as upstream:
                status = upstream.status
                response = web.StreamResponse(status=upstream.status)
                ctype = upstream.headers.get("Content-Type")
                if ctype:
                    response.content_type = ctype.split(";")[0].strip()
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if first is None:
                        first = time.time()
                    if not await self._forward(response, chunk):
                        client_gone = True  # the student hung up: stop reading, free the replica
                        break
                if not client_gone:
                    client_gone = not await self._forward(response, None)
                return response
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.n_upstream_errors += 1
            status = f"error:{type(exc).__name__}"
            if response is None:
                return web.json_response({"error": {"message": f"upstream {status}",
                                                    "replica": replica.rid}}, status=502)
            return response  # headers already sent: the client sees a truncated stream
        finally:
            self.queue.release(replica)
            self._record(**base, replica_id=replica.rid, in_flight_at_admit=at_admit,
                         queue_wait_s=round(waited, 4), forwarded=True, abandoned=False,
                         proxy_ttft_s=None if first is None else round(first - arrival, 4),
                         service_ttft_s=None if first is None else round(first - dispatched, 4),
                         status=status, client_gone=client_gone, done_ts=round(time.time(), 3))

    @staticmethod
    async def _forward(response: web.StreamResponse, chunk: Optional[bytes]) -> bool:
        """Write one upstream chunk (None: EOF) downstream; False once the client has gone."""
        try:
            if chunk is None:
                await response.write_eof()
            else:
                await response.write(chunk)
        except ConnectionResetError:  # aiohttp.ClientConnectionResetError is a subclass
            return False
        return True


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True, help="model path (the served model name)")
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--python", default=sys.executable, help="interpreter for fake_server.py")
    p.add_argument("--port", type=int, default=8000, help="router port")
    p.add_argument("--replica-port-base", type=int, default=8101)
    p.add_argument("--max-replicas", type=int, default=5)
    p.add_argument("--gpu-frac", type=float, default=0.19)
    p.add_argument("--max-num-seqs", type=int, default=10)
    p.add_argument("--replica-concurrency", type=int, default=None,
                   help="K streams the router allows per replica; default --max-num-seqs")
    p.add_argument("--abandon-s", type=float, default=60.0,
                   help="a request whose first arrival is older than this is answered 503")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--engine", choices=("process", "single"), default="process",
                   help="process: one vLLM per replica; single: one engine, virtual replicas")
    p.add_argument("--cold-start-s", default="30",
                   help="single mode: comma list of emulated cold starts, cycled by launch ordinal")
    p.add_argument("--policy", choices=("reactive", "velocity", "plan", "fixed"), required=True)
    p.add_argument("--context", type=Path, default=None, help="context.json (section 5.6)")
    p.add_argument("--plan", type=Path, default=None, help="plan.json for --policy plan")
    p.add_argument("--fixed-replicas", type=int, default=None, help="for --policy fixed")
    p.add_argument("--t0-wallclock", type=float, default=None,
                   help="epoch seconds of minute 0; default: 5 s from now")
    p.add_argument("--duration-min", type=int, default=45)
    p.add_argument("--lead-gate-min", type=int, choices=(0, 8), default=0,
                   help="0: a replica serves as soon as it is ready; 8: not before "
                        "T0 + minute_s * (decision minute + 8)")
    p.add_argument("--minute-s", type=float, default=60.0, help="seconds per controller minute")
    p.add_argument("--dry-run", action="store_true", help="fake_server.py replicas, no GPU")
    p.add_argument("--dry-launch-delay-s", type=float, default=2.0,
                   help="dry run only: pretend cold start before the fake server is spawned")
    p.add_argument("--drain-grace-s", type=float, default=60.0)
    p.add_argument("--health-timeout-s", type=float, default=600.0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _parse_cold_starts(text: str) -> tuple[float, ...]:
    values = tuple(float(v) for v in str(text).split(",") if v.strip())
    if not values or any(v < 0 for v in values):
        raise SystemExit(f"--cold-start-s needs non-negative seconds, got {text!r}")
    return values


def _setup_logging(out: Path, level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    handler = logging.FileHandler(out / "pool.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise StageError(f"{MANIFEST_PATH} missing: run vendor_policies.py first")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_meta(out: Path, meta: dict[str, Any]) -> None:
    (out / "pool_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")


async def _sleep_until(ts: float, stop: asyncio.Event) -> bool:
    """Sleep until wall clock ``ts``; True if ``stop`` fired first."""
    while True:
        delay = ts - time.time()
        if delay <= 0:
            return stop.is_set()
        try:
            await asyncio.wait_for(stop.wait(), timeout=min(delay, 1.0))
            return True
        except asyncio.TimeoutError:
            continue


async def run_pool(args: argparse.Namespace) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out, args.log_level)
    started = time.time()
    t0 = float(args.t0_wallclock) if args.t0_wallclock is not None else started + 5.0
    minute_s = float(args.minute_s)
    manifest = _load_manifest()
    ctx, ctx_raw = (None, {})
    if args.context is not None:
        ctx, ctx_raw = load_context(args.context)
    plan = PoolPlan.load(args.plan) if args.plan is not None else None
    lead = int(ctx.lead_min) if ctx is not None else (plan.lead_min if plan is not None else 1)
    lead = max(1, lead)
    if ctx is not None and plan is not None and plan.lead_min != lead:
        logger.warning("plan %s says lead_min %d, context says %d; the context wins",
                       plan.label, plan.lead_min, lead)
    min_replicas = int(ctx.policy.min_replicas) if ctx is not None else 1
    if args.policy == "fixed":
        min_replicas = int(args.fixed_replicas or 0)
    runner = PolicyRunner(args.policy, lead, ctx=ctx, plan=plan, fixed=args.fixed_replicas)
    ledger = ArrivalLedger(args.duration_min + lead + 1, t0, minute_s)
    concurrency = args.max_num_seqs if args.replica_concurrency is None else args.replica_concurrency
    spec = ReplicaSpec(python=str(args.python), model=args.model, vllm_bin=args.vllm_bin,
                       dtype=args.dtype, max_model_len=args.max_model_len,
                       gpu_frac=args.gpu_frac, max_num_seqs=args.max_num_seqs,
                       fake_capacity=concurrency,
                       port_base=args.replica_port_base, max_live=args.max_replicas,
                       health_timeout_s=args.health_timeout_s,
                       drain_grace_s=args.drain_grace_s, dry_run=args.dry_run,
                       dry_launch_delay_s=args.dry_launch_delay_s,
                       fake_server=str(HERE / "fake_server.py"))
    cold_starts = _parse_cold_starts(args.cold_start_s)
    single = args.engine == "single"
    engine_url = f"http://localhost:{args.replica_port_base}"
    manager = ReplicaManager(spec, out, t0, minute_s, args.dry_run, virtual=single,
                             engine_url=engine_url if single else "", cold_start_s=cold_starts)
    router = Router(manager, ledger, out, concurrency, args.abandon_s, args.dry_run)
    controller = MinuteController(runner, ledger, manager, lead, args.lead_gate_min,
                                  args.duration_min, min_replicas,
                                  cost=None if ctx is None else ctx.cost,
                                  max_replicas=args.max_replicas,
                                  queue_len=lambda: len(router.queue),
                                  abandoned_in=router.queue.abandoned_in)
    meta: dict[str, Any] = {
        "schema": "q2.pool_meta.v1", "dry_run": args.dry_run,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "policy": runner.name, "lead_min": lead, "lead_gate_min": args.lead_gate_min,
        "min_replicas": min_replicas, "duration_min": args.duration_min, "minute_s": minute_s,
        "replica_concurrency": concurrency, "abandon_s": args.abandon_s,
        "engine": {"mode": args.engine, "url": engine_url if single else None,
                   "args": None, "started_wallclock": None, "health_after_s": None,
                   "ready_wallclock": None, "warmup_s": None,
                   "note": ("single: one vLLM engine with fixed args, replicas are virtual "
                            "admission units; --gpu-frac and --max-num-seqs do not reach the "
                            "engine (--max-num-seqs only sets the default K)" if single else
                            "process: one vLLM per replica with --gpu-frac/--max-num-seqs")},
        "cold_start_s": list(cold_starts) if single else None,
        "t0_wallclock": t0, "t0_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
        "installed_prefix": controller.installed[:lead].tolist(),
        "context": ctx_raw, "plan": None if plan is None else {"label": plan.label,
                                                               "lead_min": plan.lead_min,
                                                               "x": list(plan.x)},
        "simpolicies_manifest": manifest, "python": sys.version, "aiohttp": aiohttp.__version__,
        "started_wallclock": started, "finished_wallclock": None, "end_reason": None,
        "exit_code": None,
    }
    _write_meta(out, meta)
    minutes_path = out / "pool_minutes.csv"
    minutes_fh = minutes_path.open("a", encoding="utf-8")
    if minutes_path.stat().st_size == 0:
        minutes_fh.write(",".join(MINUTE_COLUMNS) + "\n")
        minutes_fh.flush()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(sig: signal.Signals) -> None:
        logger.warning("%s received: stopping every replica", sig.name)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal, sig)
    end_reason = "window_end"
    exit_code = 0
    engine: Optional[ReplicaServer] = None
    try:
        if single:
            engine, engine_rec = await start_engine(spec, out, args.replica_port_base)
            meta["engine"].update({k: engine_rec[k] for k in (
                "args", "started_wallclock", "health_after_s", "ready_wallclock", "warmup_s")})
            meta["engine"]["pid"] = engine_rec["pid"]
            meta["engine_ready_wallclock"] = engine_rec["ready_wallclock"]
            meta["engine_health_after_s"] = engine_rec["health_after_s"]
            _write_meta(out, meta)
            manager.event(None, "engine_ready", {"url": engine_url,
                                                 "health_after_s": engine_rec["health_after_s"]})
        await router.start(args.port)
        manager.event(None, "pool_start", {"t0": t0, "policy": runner.name, "lead": lead,
                                           "gate": args.lead_gate_min, "minute_s": minute_s,
                                           "engine": args.engine})
        launched, drained, cancelled = controller.prestart()
        manager.event(None, "prefix", {"launched": launched, "drained": drained,
                                       "cancelled": cancelled,
                                       "installed": controller.installed[:lead].tolist()})
        for t in range(args.duration_min):
            if await _sleep_until(t0 + t * minute_s, stop):
                end_reason = "signal"
                break
            if manager.fatal is not None:
                raise StageError(manager.fatal)
            now = time.time()
            late = now - (t0 + t * minute_s)
            if late > 1.0:
                logger.warning("tick %d is %.1fs late", t, late)
            router.current_minute = t
            row, previous = controller.tick(t, now)
            if previous is not None:
                minutes_fh.write(previous.as_csv() + "\n")
                minutes_fh.flush()
            manager.event(None, "tick", {"t": t, "admitted": row.admitted,
                                         "installed_t": row.installed_t, "s": row.s,
                                         "target_s": row.target_s, "launched": row.launched_ids,
                                         "drained": row.drained_ids, "cancelled": row.cancelled_ids,
                                         "x_now": row.admitted, "queued": row.queued,
                                         "in_flight": row.in_flight,
                                         "raw_target": controller.targets_raw.get(t)})
        else:
            if await _sleep_until(t0 + args.duration_min * minute_s, stop):
                end_reason = "signal"
        last = controller.finish()
        if last is not None:
            minutes_fh.write(last.as_csv() + "\n")
            minutes_fh.flush()
    except StageError as exc:
        logger.error("pool aborted: %s", exc)
        end_reason, exit_code = f"error: {exc}", 1
    except Exception as exc:  # noqa: BLE001 - the replicas must be stopped whatever happened
        logger.exception("pool crashed")
        end_reason, exit_code = f"crash: {type(exc).__name__}: {exc}", 1
    finally:
        manager.event(None, "pool_end", {"reason": end_reason, **router.snapshot()})
        router.queue.end_of_run = True  # queued work may still go to draining replicas
        await manager.stop_all(args.drain_grace_s, still_busy=lambda: len(router.queue) > 0)
        dropped = router.queue.abandon_all("pool ended")
        if dropped:
            manager.event(None, "queue_flushed", {"abandoned": dropped})
        await router.stop()
        if engine is not None:
            await asyncio.to_thread(engine.stop)
            manager.event(None, "engine_stopped", {"url": engine_url})
        minutes_fh.close()
        meta.update({"finished_wallclock": time.time(), "end_reason": end_reason,
                     "exit_code": exit_code, "replicas_launched": len(manager.replicas()),
                     "requests": router.snapshot()})
        _write_meta(out, meta)
        manager.close()
        logger.info("pool finished: %s (exit %d)", end_reason, exit_code)
    return exit_code


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.policy == "fixed" and args.fixed_replicas is None:
        raise SystemExit("--policy fixed needs --fixed-replicas")
    if args.policy in ("reactive", "velocity") and args.context is None:
        raise SystemExit(f"--policy {args.policy} needs --context")
    if args.policy == "plan" and args.plan is None:
        raise SystemExit("--policy plan needs --plan")
    return asyncio.run(run_pool(args))


if __name__ == "__main__":
    raise SystemExit(main())
