#!/usr/bin/env python3
"""The batch-B stage bodies: what actually gets measured on the box.

``stages.py`` owns the markers, the heartbeat and the budget; this file owns the
commands.  A stage body takes a ``StepContext`` and a monotonic deadline, appends
every subprocess' output to ``logs/stage_<name>.log`` and raises ``StageError``
on a non-zero exit, a deadline overrun or a server that never answers ``/health``.

The ``vllm serve`` arguments come from the detected GPU tier (see ``hardware.py``),
not from constants, and a tier that cannot hold 14B/32B drops them from the model
list rather than OOM-ing at full price.

Under ``--dry-run`` the same code paths run: the server is ``fake_server.py``
instead of ``vllm serve`` and the measurement scripts get their own ``--dry-run``
flag, so every record is tagged and no number is a measurement.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hardware import RUNBOOK_TIER, SMALL_MODEL_SLUGS, GpuTier
from layout import RemoteLayout, iso_now
from serving import ManagedServer, ServerSpec, StageError, free_port, run_step

logger = logging.getLogger("q2.remote.measure")


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    dtype: str

    def path(self, models_dir: Path) -> str:
        return str(models_dir / self.slug)


DEFAULT_MODELS = (
    ModelSpec("qwen2.5-7b", "bfloat16"),
    ModelSpec("qwen2.5-14b", "bfloat16"),
    ModelSpec("qwen2.5-32b-awq", "float16"),
    ModelSpec("qwen2.5-72b-awq", "float16"),  # batch B4; downloaded by bootstrap when the disk allows
)


def select_models(
    tier: GpuTier, models: Sequence[ModelSpec] = DEFAULT_MODELS
) -> tuple[ModelSpec, ...]:
    """Drop the models this card cannot serve, and say which and why."""
    if tier.allow_large_models:
        return tuple(models)
    kept = tuple(model for model in models if model.slug in SMALL_MODEL_SLUGS)
    dropped = [model.slug for model in models if model not in kept]
    if dropped:
        logger.warning("refusing %s on this GPU: %s", ", ".join(dropped), tier.reason)
    return kept or (models[0],)


@dataclass(frozen=True)
class StepContext:
    layout: RemoteLayout
    python: Path
    tier: GpuTier = RUNBOOK_TIER
    models: tuple[ModelSpec, ...] = DEFAULT_MODELS
    dry_run: bool = False
    port: int = 8000
    ramp_band: tuple[float, float] = (60.0, 120.0)
    health_timeout_s: float = 600.0
    cooldown_s: float = 10.0
    shutdown_grace_s: float = 30.0
    coldstart_runs: int = 3
    concurrency_duration_s: float = 90.0
    prefix_tokens: int = 3000
    burst_k: int = 20
    repeats: int = 3

    @property
    def max_model_len(self) -> int:
        return self.tier.max_model_len

    @property
    def gpu_memory_utilization(self) -> float:
        return self.tier.gpu_memory_utilization

    def header(self) -> str:
        return f"{self.tier.header()} dry_run={self.dry_run}"

    def script(self, name: str) -> str:
        return str(self.layout.scripts / name)

    def model_path(self, model: ModelSpec) -> str:
        return model.path(self.layout.models)

    @property
    def vllm_bin(self) -> str:
        candidate = self.python.parent / "vllm"
        return str(candidate) if candidate.exists() else "vllm"


def _step(ctx: StepContext, stage: str, cmd: Sequence[str], deadline: float) -> None:
    run_step(cmd, ctx.layout.stage_log(stage), deadline, cwd=ctx.layout.scripts,
             header=f"stage={stage} {ctx.header()}")


def _server(ctx: StepContext, model: ModelSpec, stage: str) -> ManagedServer:
    """A vLLM server on the box, or ``fake_server.py`` in a dry run."""
    port = free_port() if ctx.dry_run else ctx.port
    log_path = ctx.layout.logs / f"server_{stage}_{model.slug}.log"
    if ctx.dry_run:
        cmd: tuple[str, ...] = (
            str(ctx.python), ctx.script("fake_server.py"), "--port", str(port),
            "--model", ctx.model_path(model), "--time-scale", "0.05", "--log-level", "WARNING",
        )
        label = f"fake_server {model.slug}"
    else:
        cmd = (
            ctx.vllm_bin, "serve", ctx.model_path(model),
            "--host", "localhost", "--port", str(port),
            "--dtype", model.dtype, "--max-model-len", str(ctx.max_model_len),
            "--gpu-memory-utilization", f"{ctx.gpu_memory_utilization:g}",
            "--enable-prefix-caching",
        )
        label = f"vllm serve {model.slug}"
    spec = ServerSpec(cmd, f"http://localhost:{port}", log_path, label,
                      header=f"stage={stage} {ctx.header()}")
    return ManagedServer(spec, ctx.shutdown_grace_s)


@dataclass(frozen=True)
class PatternReport:
    matched: Mapping[str, tuple[str, ...]]
    unmatched: tuple[str, ...]
    counts: Mapping[str, int]

    def as_dict(self, **extra: Any) -> dict[str, Any]:
        return {
            **extra,
            "matched": {name: list(pats) for name, pats in self.matched.items()},
            "unmatched": list(self.unmatched),
            "line_counts_per_pattern": dict(self.counts),
        }


def check_log_patterns(
    lines: Iterable[str], patterns: Mapping[str, Sequence[str]]
) -> PatternReport:
    """Which milestone regexes matched at least one log line, and how often."""
    compiled = {name: [(pat, re.compile(pat)) for pat in pats] for name, pats in patterns.items()}
    hits: dict[str, list[str]] = {name: [] for name in patterns}
    counts: dict[str, int] = {pat: 0 for pats in patterns.values() for pat in pats}
    for line in lines:
        for name, pairs in compiled.items():
            for pat, regex in pairs:
                if regex.search(line):
                    counts[pat] += 1
                    if pat not in hits[name]:
                        hits[name].append(pat)
    return PatternReport(
        matched={name: tuple(pats) for name, pats in hits.items()},
        unmatched=tuple(name for name, pats in hits.items() if not pats),
        counts=counts,
    )


def load_vllm_patterns(scripts_dir: Path) -> dict[str, tuple[str, ...]]:
    """The milestone regexes of ``coldstart_vllm.py`` (imported late: it needs numpy)."""
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from coldstart_vllm import MILESTONES, VLLMLogPatterns

    patterns = VLLMLogPatterns()
    return {name: tuple(getattr(patterns, name)) for name in MILESTONES}


def _startup_log(ctx: StepContext, model: ModelSpec, deadline: float) -> Path:
    """Produce one start-up log to check the milestone regexes against."""
    if ctx.dry_run:
        probe_out = ctx.layout.out / "_patterncheck"
        _step(ctx, "check_patterns", (
            str(ctx.python), ctx.script("coldstart_vllm.py"), "--dry-run", "--runs", "1",
            "--models", ctx.model_path(model), "--out", str(probe_out),
        ), deadline)
        logs = sorted((probe_out / "coldstart" / "logs").glob("*.log"))
        if not logs:
            raise StageError("dry-run coldstart produced no start-up log to check")
        return logs[-1]
    server = _server(ctx, model, "check_patterns")
    try:
        server.start()
        server.wait_health(min(ctx.health_timeout_s, 600.0), deadline)
    except StageError as exc:
        logger.warning("%s; checking the captured log anyway", exc)
    finally:
        server.stop()
    return server.spec.log_path


def stage_check_patterns(ctx: StepContext, deadline: float) -> None:
    """Start the cheapest model once and prove the log regexes still match."""
    model = ctx.models[0]
    log_path = _startup_log(ctx, model, deadline)
    report = check_log_patterns(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        load_vllm_patterns(ctx.layout.scripts),
    )
    out_path = ctx.layout.out / "patterns_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict(schema="q2.patterns.v1", ts=iso_now(), dry_run=ctx.dry_run,
                             model=ctx.model_path(model), log=str(log_path),
                             gpu_tier=ctx.tier.as_dict())
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    logger.info("pattern report -> %s", out_path)
    if report.unmatched:
        raise StageError(
            "unmatched vLLM log milestones: " + ", ".join(report.unmatched)
            + f"; fix VLLMLogPatterns in coldstart_vllm.py before spending GPU time ({out_path})"
        )


def stage_coldstart(ctx: StepContext, deadline: float) -> None:
    """RUNBOOK section 3: one ``coldstart_vllm.py`` invocation per dtype."""
    by_dtype: dict[str, list[str]] = {}
    for model in ctx.models:
        by_dtype.setdefault(model.dtype, []).append(ctx.model_path(model))
    for dtype, paths in by_dtype.items():
        cmd = [
            str(ctx.python), ctx.script("coldstart_vllm.py"),
            "--out", str(ctx.layout.out), "--runs", str(ctx.coldstart_runs),
            "--models", ",".join(paths), "--dtype", dtype,
            "--max-model-len", str(ctx.max_model_len),
            "--gpu-memory-utilization", f"{ctx.gpu_memory_utilization:g}",
        ]
        if ctx.dry_run:
            cmd.append("--dry-run")
        _step(ctx, "coldstart", cmd, deadline)


def stage_concurrency(ctx: StepContext, deadline: float) -> None:
    """RUNBOOK section 4: one server at a time, then the closed-loop sweep."""
    for model in ctx.models:
        with _server(ctx, model, "concurrency") as server:
            server.wait_health(ctx.health_timeout_s, deadline)
            cmd = [
                str(ctx.python), ctx.script("concurrency_curve.py"),
                "--out", str(ctx.layout.out), "--model", ctx.model_path(model),
                "--duration", f"{ctx.concurrency_duration_s:g}",
            ]
            cmd += ["--dry-run", "--levels", "1,5"] if ctx.dry_run \
                else ["--base-url", server.spec.base_url]
            _step(ctx, "concurrency", cmd, deadline)
        time.sleep(ctx.cooldown_s)


def stage_prefix(ctx: StepContext, deadline: float) -> None:
    """RUNBOOK section 5: sequential burst, then a second pass with ``--concurrent``."""
    model = ctx.models[0]
    with _server(ctx, model, "prefix") as server:
        server.wait_health(ctx.health_timeout_s, deadline)
        cmd = [
            str(ctx.python), ctx.script("prefix_warmup.py"),
            "--out", str(ctx.layout.out), "--model", ctx.model_path(model),
            "--prefix-tokens", str(ctx.prefix_tokens), "--burst-k", str(ctx.burst_k),
            "--repeats", str(ctx.repeats),
        ]
        cmd += ["--dry-run"] if ctx.dry_run else ["--base-url", server.spec.base_url]
        _step(ctx, "prefix", cmd, deadline)
        _step(ctx, "prefix", [*cmd, "--concurrent"], deadline)


REPLAY_TRACE = "traces/burst_20260324_0830.csv"
REPLAY_MODEL = "qwen2.5-7b"
REPLAY_WARM_SCALES = (1, 5, 10, 20, 40)
# (label, seconds between the server launch and the burst onset, scales): a lead of 0 is the
# reactive rule (start when the burst starts); 30 s is the replica tier's measured lead
REPLAY_LATE = (("late0", 0.0, (10, 40)), ("late30", 30.0, (10,)))
REPLAY_MAX_OFFSET_S = 360.0  # the first 6 min hold 81% of the burst's requests
REPLAY_PROBE_LEVELS = "1,50,100,200"  # a short concurrency probe: is this card the T3 card?
REPLAY_PROBE_DURATION_S = 30.0
REPLAY_CLIENT_HEAD_S = 5.0  # client start -> burst onset when the server is launched at once


def _replay_client_cmd(ctx: StepContext, model: ModelSpec, base_url: str, out: Path,
                       label: str, scale: int, max_offset: float, t0_delay: float,
                       warmup: int, trace: str | None = None,
                       give_up_s: float = 180.0) -> list[str]:
    """``replay_client.py`` on ``REPLAY_TRACE`` (batch B') or on ``trace`` (the pool stage)."""
    cmd = [str(ctx.python), ctx.script("replay_client.py"),
           "--trace", ctx.script(REPLAY_TRACE) if trace is None else trace,
           "--base-url", base_url, "--model", ctx.model_path(model), "--out", str(out),
           "--label", label, "--scale", str(scale), "--max-offset", f"{max_offset:g}",
           "--t0-delay", f"{t0_delay:g}", "--warmup", str(warmup),
           "--give-up-s", f"{give_up_s:g}"]
    if ctx.dry_run:
        cmd.append("--dry-run")
    return cmd


def _popen_step(ctx: StepContext, stage: str, cmd: Sequence[str]) -> subprocess.Popen[bytes]:
    """Start a client in the background with its output appended to the stage log."""
    log = ctx.layout.stage_log(stage)
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("ab")
    fh.write(f"\n### {iso_now()} popen: {' '.join(cmd)}\n".encode())
    fh.flush()
    return subprocess.Popen(list(cmd), stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(ctx.layout.scripts), start_new_session=True)


def _wait_step(proc: subprocess.Popen[bytes], deadline: float, what: str) -> None:
    while proc.poll() is None:
        if time.monotonic() > deadline:
            proc.kill()
            raise StageError(f"{what} ran past the stage deadline")
        time.sleep(1.0)
    if proc.returncode != 0:
        raise StageError(f"{what} exited {proc.returncode}")


def stage_replay(ctx: StepContext, deadline: float) -> None:
    """Batch B' (F10): one real prompted burst replayed against a warm and a late replica.

    Warm: the server is healthy before the burst and serves every scale in turn.  Late: the client
    is started first with the burst onset ``REPLAY_CLIENT_HEAD_S + lead`` seconds away and the
    server is launched ``REPLAY_CLIENT_HEAD_S`` seconds later, so the replica starts ``lead``
    seconds before the burst; requests that find no server retry and their TTFT is counted from
    the scheduled arrival.  Wall-clock timestamps of the server launch and its first healthy
    answer are written next to the client's records so the realised lead is measured, not assumed.
    """
    model = next((m for m in ctx.models if m.slug == REPLAY_MODEL), ctx.models[0])
    out = ctx.layout.out / "replay"
    out.mkdir(parents=True, exist_ok=True)
    max_offset = 30.0 if ctx.dry_run else REPLAY_MAX_OFFSET_S
    warm_scales = (1, 2) if ctx.dry_run else REPLAY_WARM_SCALES
    late_plan = (("late0", 0.0, (1,)), ("late30", 3.0, (2,))) if ctx.dry_run else REPLAY_LATE
    with _server(ctx, model, "replay") as server:
        server.wait_health(ctx.health_timeout_s, deadline)
        # the clone may not be the card T3 was measured on: probe the concurrency curve first
        probe = [str(ctx.python), ctx.script("concurrency_curve.py"), "--out", str(ctx.layout.out),
                 "--model", ctx.model_path(model), "--duration",
                 f"{5.0 if ctx.dry_run else REPLAY_PROBE_DURATION_S:g}"]
        probe += (["--dry-run", "--levels", "1,5"] if ctx.dry_run
                  else ["--base-url", server.spec.base_url, "--levels", REPLAY_PROBE_LEVELS])
        _step(ctx, "replay", probe, deadline)
        time.sleep(ctx.cooldown_s)
        for scale in warm_scales:
            cmd = _replay_client_cmd(ctx, model, server.spec.base_url, out, f"warm_x{scale}",
                                     scale, max_offset, 0.0, warmup=3)
            _step(ctx, "replay", cmd, deadline)
            time.sleep(ctx.cooldown_s)
    time.sleep(ctx.cooldown_s)
    for label, lead_s, scales in late_plan:
        for scale in scales:
            tag = f"{label}_x{scale}"
            server = _server(ctx, model, f"replay_{tag}")
            cmd = _replay_client_cmd(ctx, model, server.spec.base_url, out, tag, scale,
                                     max_offset, REPLAY_CLIENT_HEAD_S + lead_s, warmup=0)
            client_started = time.time()
            proc = _popen_step(ctx, "replay", cmd)
            try:
                time.sleep(REPLAY_CLIENT_HEAD_S)
                server_started = time.time()
                server.start()
                ready_after = server.wait_health(ctx.health_timeout_s, deadline)
                server_ready = time.time()
                _wait_step(proc, deadline, f"replay client {tag}")
            finally:
                if proc.poll() is None:
                    proc.kill()
                server.stop()
            (out / f"{tag}_server.json").write_text(json.dumps({
                "label": tag, "scale": scale, "planned_lead_s": lead_s,
                "client_started_wallclock": client_started,
                "server_started_wallclock": server_started,
                "server_ready_wallclock": server_ready, "health_after_s": ready_after,
                "burst_onset_wallclock": client_started + REPLAY_CLIENT_HEAD_S + lead_s,
                "realised_lead_s": (client_started + REPLAY_CLIENT_HEAD_S + lead_s)
                - server_ready, "dry_run": ctx.dry_run}, indent=2) + "\n", encoding="utf-8")
            time.sleep(ctx.cooldown_s)


SHORTPROMPT_LEVELS = "1,50,100,200,400"
SHORTPROMPT_DURATION_S = 30.0
SHORTPROMPT_MEDIAN_TOKENS = 45.0  # classroom prompts: 60 characters, about 40-50 tokens
SHORTPROMPT_SIGMA = 0.3
SHORTPROMPT_MAX_TOKENS = 194  # measured reply length, 281 characters / 1.45
SHORTPROMPT_DIR = "shortprompt"


def stage_shortprompt(ctx: StepContext, deadline: float) -> None:
    """Batch B3: the concurrency curve again, with prompts as short as classroom questions.

    T3 and the replay probe used tutoring prompts around 300 tokens (Azure-like lengths); the
    classroom prompts of the genuine exports are about 60 characters, so that calibration was
    conservative by roughly 1.75x (results/q2/replay).  This stage measures every model the
    card can serve at ``SHORTPROMPT_LEVELS`` with ``SHORTPROMPT_MEDIAN_TOKENS``-token prompts and
    ``SHORTPROMPT_MAX_TOKENS`` output tokens, into ``<out>/shortprompt/concurrency``, and then
    builds ``<out>/shortprompt/T3_calibration.csv`` from it with ``summarize.py`` so that
    ``load_service_config`` can read the new capacity directly.
    """
    out = ctx.layout.out / SHORTPROMPT_DIR
    out.mkdir(parents=True, exist_ok=True)
    for model in ctx.models:
        with _server(ctx, model, "shortprompt") as server:
            server.wait_health(ctx.health_timeout_s, deadline)
            cmd = [
                str(ctx.python), ctx.script("concurrency_curve.py"),
                "--out", str(out), "--model", ctx.model_path(model),
                "--duration", f"{5.0 if ctx.dry_run else SHORTPROMPT_DURATION_S:g}",
                "--prompt-median-tokens", f"{SHORTPROMPT_MEDIAN_TOKENS:g}",
                "--prompt-sigma", f"{SHORTPROMPT_SIGMA:g}",
                "--max-tokens", str(SHORTPROMPT_MAX_TOKENS),
            ]
            cmd += (["--dry-run", "--levels", "1,5"] if ctx.dry_run
                    else ["--base-url", server.spec.base_url, "--levels", SHORTPROMPT_LEVELS])
            _step(ctx, "shortprompt", cmd, deadline)
        time.sleep(ctx.cooldown_s)
    lo, hi = ctx.ramp_band
    _step(ctx, "shortprompt", (
        str(ctx.python), ctx.script("summarize.py"),
        "--results", str(out), "--out", str(out), "--ramp-band", f"{lo:g}", f"{hi:g}",
    ), deadline)


LADDER_MODELS = ("qwen2.5-7b", "qwen2.5-32b-awq", "qwen2.5-72b-awq")  # the model ladder (14B is redundant)
LADDER_SHORT_LEVELS = "1,25,50,100,150,200,300,400,600,800"
LADDER_RAG_LEVELS = "1,25,50,100,200,400"
LADDER_RAG_TOKENS = (2000, 4000, 8000)
LADDER_MAX_MODEL_LEN = 12288  # room for an 8k context, the question and the reply
LADDER_DIR = "ladder"
LADDER_COLDSTART_RUNS = 2


def _curve_cmd(ctx: StepContext, model: ModelSpec, out: Path, levels: str,
               extra: Sequence[str] = ()) -> list[str]:
    cmd = [str(ctx.python), ctx.script("concurrency_curve.py"), "--out", str(out),
           "--model", ctx.model_path(model),
           "--duration", f"{5.0 if ctx.dry_run else SHORTPROMPT_DURATION_S:g}",
           "--prompt-median-tokens", f"{SHORTPROMPT_MEDIAN_TOKENS:g}",
           "--prompt-sigma", f"{SHORTPROMPT_SIGMA:g}", "--max-tokens", str(SHORTPROMPT_MAX_TOKENS),
           *extra]
    return cmd + (["--dry-run", "--levels", "1,5"] if ctx.dry_run else ["--levels", levels])


def stage_ladder(ctx: StepContext, deadline: float) -> None:
    """Batch B4: the model ladder under classroom, shared-context and private-context prompts.

    For every model of ``LADDER_MODELS`` whose checkpoint is on disk: (1) the short-prompt curve on
    a finer level grid than batch B3 so that the 2 s knee of the 7B is measured rather than
    extrapolated; (2) RAG-style prompts with a ``LADDER_RAG_TOKENS`` course-material block shared by
    every request (prefix-cache friendly); (3) the same lengths as a distinct context per request
    (no cache reuse).  Servers run with ``LADDER_MAX_MODEL_LEN``.  Each regime lands in its own
    folder under ``<out>/ladder`` with a ``T3_calibration.csv`` from ``summarize.py``; the 72B also
    gets a two-run cold start.
    """
    root = ctx.layout.out / LADDER_DIR
    root.mkdir(parents=True, exist_ok=True)
    wide = replace(ctx, tier=replace(ctx.tier, max_model_len=LADDER_MAX_MODEL_LEN))
    present = [m for m in ctx.models if m.slug in LADDER_MODELS
               and (ctx.dry_run or Path(ctx.model_path(m), "config.json").exists())]
    missing = [slug for slug in LADDER_MODELS if slug not in {m.slug for m in present}]
    if missing:
        logger.warning("ladder: checkpoints missing, skipped: %s", ", ".join(missing))
    regimes: list[tuple[str, list[str]]] = [("short", [])]
    for n in LADDER_RAG_TOKENS:
        regimes.append((f"rag_shared_{n}", ["--shared-prefix-tokens", str(n)]))
        regimes.append((f"rag_unique_{n}", ["--unique-prefix-tokens", str(n)]))
    for model in present:
        with _server(wide, model, "ladder") as server:
            server.wait_health(ctx.health_timeout_s, deadline)
            for name, extra in regimes:
                levels = LADDER_SHORT_LEVELS if name == "short" else LADDER_RAG_LEVELS
                args = list(extra) + ([] if ctx.dry_run else ["--base-url", server.spec.base_url])
                _step(ctx, "ladder", _curve_cmd(ctx, model, root / name, levels, args), deadline)
                time.sleep(ctx.cooldown_s)
        time.sleep(ctx.cooldown_s)
    lo, hi = ctx.ramp_band
    for name, _ in regimes:
        out = root / name
        if (out / "concurrency" / "concurrency_summary.jsonl").exists():
            _step(ctx, "ladder", (str(ctx.python), ctx.script("summarize.py"), "--results", str(out),
                                  "--out", str(out), "--ramp-band", f"{lo:g}", f"{hi:g}"), deadline)
    big = [m for m in present if m.slug == "qwen2.5-72b-awq"]
    if big:
        model = big[0]
        cmd = [str(ctx.python), ctx.script("coldstart_vllm.py"), "--out", str(root / "coldstart"),
               "--runs", str(LADDER_COLDSTART_RUNS), "--models", ctx.model_path(model),
               "--dtype", model.dtype, "--max-model-len", str(ctx.max_model_len),
               "--gpu-memory-utilization", f"{ctx.gpu_memory_utilization:g}"]
        if ctx.dry_run:
            cmd.append("--dry-run")
        _step(ctx, "ladder", cmd, deadline)


def stage_summarize(ctx: StepContext, deadline: float) -> None:
    lo, hi = ctx.ramp_band
    _step(ctx, "summarize", (
        str(ctx.python), ctx.script("summarize.py"),
        "--results", str(ctx.layout.out), "--out", str(ctx.layout.out),
        "--ramp-band", f"{lo:g}", f"{hi:g}",
    ), deadline)


def stage_pool(ctx: StepContext, deadline: float) -> None:
    """Batch B' step two: the policy-driven replica pool.

    The body lives in ``pool_stage.py`` (imported here, not at module level, because it needs
    ``StepContext`` and the helpers above); it reads ``<scripts>/pool_inputs/pool_runs.json``
    and writes ``<out>/pool/calib_multi/`` and ``<out>/pool/runs/<label>/``.
    """
    from pool_stage import run_pool_stage

    run_pool_stage(ctx, deadline)


def stage_calib_single(ctx: StepContext, deadline: float) -> None:
    """Batch B' step two, phase 1: one capped replica calibrated alone on this card."""
    from pool_stage import run_calib_single_stage

    run_calib_single_stage(ctx, deadline)


STAGE_IMPLS: Mapping[str, Callable[[StepContext, float], None]] = {
    "check_patterns": stage_check_patterns,
    "coldstart": stage_coldstart,
    "concurrency": stage_concurrency,
    "prefix": stage_prefix,
    "summarize": stage_summarize,
    "replay": stage_replay,
    "shortprompt": stage_shortprompt,
    "ladder": stage_ladder,
    "pool": stage_pool,
    "calib_single": stage_calib_single,
}
