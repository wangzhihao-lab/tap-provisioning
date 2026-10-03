"""Event loop: one arrival stream, one policy, one trajectory, one set of metrics.

The master seed is split into three independent streams - workload, service and
policy - so that a comparison is *paired*: ``run_all`` generates the arrival stream once and
replays the identical stream for every policy.

A decision taken at minute ``t`` installs the replica count for minute ``t + d``; the first ``d``
minutes come from ``Policy.prefix``, which is where an open-loop policy that already knows the
timetable is allowed to pre-commit.

Lead time and commitment granularity are separate constraints (``ServiceConfig``).  ``lead_min`` is
the delay above; ``commit_block_min`` forces the installed capacity to be piecewise-constant on
blocks of that length, which ``commit_blocks`` applies to every policy's decision trace in the same
place as the integer ceiling.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import PolicyFactory
from .config import RunConfig
from .metrics import RunMetrics, fluid_cost, summarize
from .policies.base import Policy, PolicyContext, SimState, as_int_plan
from .service import ServiceResult, replay
from .workload import ArrivalStream, generate

logger = logging.getLogger(__name__)

DEFAULT_POLICIES: Tuple[str, ...] = ("static", "reactive", "velocity", "forecast_ilp", "lcp",
                                     "calendar", "bdinterp", "offline", "informed",
                                     "stale_calendar")


@dataclass(frozen=True)
class RunContext(PolicyContext):
    """``PolicyContext`` as ``make_context`` builds it, plus the trust knob named for the run.

    ``trust_override`` is the ``name@eps`` value a run was asked for, None when ``trust`` is just
    ``PolicyConfig.trust``; ``tap`` runs at its fixed eps unless one is named.  It lives in a
    subclass because ``policies/base.py`` is vendored byte for byte into ``scripts/q2/simpolicies``
    and its hash is pinned by the manifests the B5 pool runs recorded.
    """

    trust_override: Optional[float] = None


@dataclass(frozen=True)
class RunResult:
    """One policy on one workload; ``diagnostics`` is what the policy's ``diagnostics()`` reported."""

    policy: str
    x: np.ndarray
    service: ServiceResult
    metrics: RunMetrics
    diagnostics: Optional[Dict[str, float]] = None


def derive_rngs(seed: int) -> Tuple[np.random.Generator, np.random.Generator,
                                    np.random.Generator]:
    """Workload / service / policy generators derived from one master seed."""
    children = np.random.SeedSequence(seed).spawn(3)
    return tuple(np.random.default_rng(c) for c in children)  # type: ignore[return-value]


def build_stream(cfg: RunConfig) -> ArrivalStream:
    """Generate the arrival stream of one run (workload RNG stream only)."""
    rng_workload, _, _ = derive_rngs(cfg.seed)
    return generate(cfg.workload, rng_workload, cfg.root)


def fluid_mu(cfg: RunConfig, stream: ArrivalStream) -> float:
    """Fluid capacity of one replica in requests per minute, from the mean output length."""
    mean_out = float(stream.output_tokens.mean()) if stream.n_requests else 128.0
    return cfg.service.requests_per_min(mean_out)


def trigger_grid(cfg: RunConfig, stream: ArrivalStream) -> Optional[np.ndarray]:
    """Every minute a prompted burst can start on: the bells and the later in-session triggers.

    ``WorkloadConfig.triggers_per_session`` prompts fall at ``bell + k * duration / m``; both are
    fixed by the timetable's block structure, so the grid is family knowledge,
    not the instance's amplitudes; it is shifted by ``forecast_offset_min`` like every other
    bell-aware forecast.  ``None`` when the stream carries no session table.
    """
    sessions = stream.sessions
    if "bell_min" not in sessions.columns or "duration_min" not in sessions.columns:
        return None
    m = max(1, int(cfg.workload.triggers_per_session))
    bells = sessions["bell_min"].to_numpy(dtype=float)
    dur = sessions["duration_min"].to_numpy(dtype=float)
    grid = np.concatenate([bells + dur / m * k for k in range(m)])
    shift = int(round(cfg.workload.forecast_offset_min))
    return np.unique(np.floor(grid + 1e-9).astype(int) + shift)


def make_context(cfg: RunConfig, stream: ArrivalStream, trust: Optional[float] = None
                 ) -> PolicyContext:
    """Everything a policy may know before minute 0."""
    return RunContext(
        horizon_min=stream.horizon_min, mu=fluid_mu(cfg, stream),
        lead_min=max(0, int(math.ceil(cfg.service.lead_min - 1e-9))), cost=cfg.cost,
        policy=cfg.policy,
        service=cfg.service, lam_hat_in=stream.lam_hat_in, lam_hat_off=stream.lam_hat_off,
        trust=cfg.policy.trust if trust is None else trust,
        lam_stale_in=stream.lam_stale_in,
        commit_block_min=max(1, int(round(cfg.service.commit_block_min))),
        bells=trigger_grid(cfg, stream), lam_hat_learn_in=stream.lam_hat_learn_in,
        lam_hat_learn_prior_in=stream.lam_hat_learn_prior_in, trust_override=trust)


def _minute_tokens(stream: ArrivalStream) -> np.ndarray:
    """Total tokens arriving in each minute (the token-velocity policy's only input)."""
    minute = np.minimum((stream.t_arrive_s // 60.0).astype(np.int64), stream.horizon_min - 1)
    tok = (stream.prompt_tokens + stream.output_tokens).astype(float)
    return np.bincount(minute, weights=tok, minlength=stream.horizon_min)


def make_policy(name: str, ctx: PolicyContext, stream: ArrivalStream) -> Policy:
    """Instantiate a policy and hand the oracles the realised demand."""
    policy = PolicyFactory(name)(ctx)
    if hasattr(policy, "bind"):
        policy.bind(stream.lam_in, stream.lam_off)
    return policy


def commit_blocks(targets, block_min: int, horizon: Optional[int] = None) -> np.ndarray:
    """Expand one level per commitment block into the per-minute trajectory."""
    out = np.repeat(np.asarray(targets, dtype=float), max(1, int(block_min)))
    return out if horizon is None else out[:horizon]


def trajectory(policy: Policy, stream: ArrivalStream, ctx: PolicyContext) -> np.ndarray:
    """Run the decision loop and return the realised per-minute replica count.

    Without a commitment block (``commit_block_min = 1``, the rented tiers) the loop is the whole
    story: ``decide(t)`` installs a target at minute ``t + d``.  With one, the installed capacity
    has to be constant on blocks of that length, and the level of a block is asked for at the
    block's own decision minute - ``start - d`` - through ``Policy.block_target``, so that it can
    depend on nothing the committing minute does not already know.  Blocks that start before minute
    ``d`` are decided before the loop, next to ``Policy.prefix``, which is the same pre-commitment
    an open-loop policy has always been allowed.  The per-minute loop still runs for every policy
    so that feedback rules keep advancing their own state, and ``SimState.x_now`` stays the
    per-minute target the loop produced rather than the level the block holds - the convention that
    has always applied to the integer ceiling as well.  The always-on base tier
    (``PolicyConfig.min_replicas``) is applied here, to the realised trajectory of every policy: it
    is a property of the deployment, not a decision any policy gets to make.
    """
    horizon = stream.horizon_min
    lead = ctx.lead_min
    block = max(1, int(ctx.commit_block_min))
    plan = np.zeros(horizon + lead + 1, dtype=float)
    policy.start_run()
    prefix = np.asarray(policy.prefix(lead), dtype=float)
    if lead:
        plan[:lead] = np.pad(prefix, (0, max(0, lead - prefix.size)), mode="edge")[:lead]
    levels: Optional[np.ndarray] = None
    if block > 1:
        starts = range(0, horizon, block)
        levels = np.zeros(len(starts), dtype=float)
        for a in starts:  # every block whose decision minute falls before the run starts
            if a >= lead:
                break
            levels[a // block] = policy.block_target(a, min(a + block, horizon))
    state = SimState(stream.lam_in, stream.lam_off, _minute_tokens(stream))
    for t in range(horizon):
        state.t = t
        state.x_now = float(plan[t])
        plan[t + lead] = policy.decide(t, state)
        start = t + lead
        if levels is not None and start < horizon and start % block == 0:
            levels[start // block] = policy.block_target(start, min(start + block, horizon))
    floor = float(ctx.policy.min_replicas)  # the base tier is bought whatever the policy says
    if levels is None:
        return as_int_plan(plan[:horizon], ctx.cost, floor)
    return as_int_plan(commit_blocks(levels, block, horizon), ctx.cost, floor)


def run_policy(name: str, cfg: RunConfig, stream: ArrivalStream, ctx: Optional[PolicyContext] = None,
               offline_total: Optional[float] = None, trust: Optional[float] = None) -> RunResult:
    """Simulate one policy on one arrival stream."""
    ctx = ctx or make_context(cfg, stream, trust)
    policy = make_policy(name, ctx, stream)
    policy.reset()
    x = trajectory(policy, stream, ctx)
    result = replay(stream, x, cfg.service)
    metrics = summarize(name, cfg.seed, x, stream, result, ctx.mu, cfg.cost,
                        cfg.service.billing_block_min, offline_total)
    logger.info("%-12s gpu_h=%.2f starts=%.0f slo_in=%.3f ttft_p95_in=%.2f", name,
                metrics.gpu_hours, metrics.starts, metrics.slo_in, metrics.ttft_p95_in_s)
    report = getattr(policy, "diagnostics", None)  # only policies that report any define it
    return RunResult(name, x, result, metrics, report() if callable(report) else None)


def run_all(cfg: RunConfig, policies: Optional[Sequence[str]] = None,
            stream: Optional[ArrivalStream] = None) -> Dict[str, RunResult]:
    """Replay the same arrival stream through every policy (the paired design)."""
    names: List[str] = list(policies or DEFAULT_POLICIES)
    stream = stream if stream is not None else build_stream(cfg)
    ctx = make_context(cfg, stream)
    base = run_policy("offline", cfg, stream, ctx)
    denom = base.metrics.total_cost
    out: Dict[str, RunResult] = {}
    for name in names:
        out[name] = (base if name == "offline" else run_policy(name, cfg, stream, ctx, denom))
    if "offline" in out:
        out["offline"] = run_policy("offline", cfg, stream, ctx, denom)
    return out


def lp_lower_bound(cfg: RunConfig, stream: ArrivalStream, ctx: Optional[PolicyContext] = None
                   ) -> float:
    """Fluid (LP relaxation) cost of the realised demand: no integer trajectory can beat it."""
    from .policies.offline import solve_offline_lp
    ctx = ctx or make_context(cfg, stream)
    z = solve_offline_lp(stream.lam_in, stream.lam_off, ctx.mu, cfg.cost, ctx.commit_block_min,
                         x_min=float(cfg.policy.min_replicas))
    return fluid_cost(z, stream.lam_in, stream.lam_off, ctx.mu, cfg.cost,
                      cfg.service.billing_block_min).total_cost


__all__ = ["DEFAULT_POLICIES", "RunContext", "RunResult", "build_stream", "commit_blocks", "trigger_grid", "derive_rngs",
           "fluid_mu", "lp_lower_bound", "make_context", "make_policy", "run_all", "run_policy",
           "trajectory"]
