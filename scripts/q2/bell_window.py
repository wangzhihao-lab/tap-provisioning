#!/usr/bin/env python3
"""One bell as a 45-minute window: the sub-stream, the client's length rules, the capped replica.

Shared by ``make_bell_trace.py`` and ``pool_compare.py`` so that both worlds are built from the
same rules: the window is ``[bell - 18, bell + 27)`` minutes of in-class
requests, times from the window start rounded to the millisecond the client records; the trace
encodes lengths the way ``replay_client.py`` decodes them and the simulator side takes its tokens
back through the client's own formulas (``Request.max_tokens``, ``make_prompt``) plus the router's
``ceil(chars / 1.45) + 12``, the client's 320-token clip included; the replica is the restricted 7B
one of ``calib_single`` with K pinned to ``--max-num-seqs 10`` and the TTFT p95 at K set to the
card's measured value at N = 10 (``load_service_config`` would put the SLO there because it
defines K by the SLO crossing); the window is thinned by ``f = mu_cap / mu_72b`` (Bernoulli, fixed
seed), forecasts scaled alike, so the bell needs as many emulated replicas as 72B ones in the paper.
"""

from __future__ import annotations

import logging
import math
import sys
import types
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
for _extra in (PILOT / "src", PILOT / "scripts" / "sim", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from pilot.sim import CostConfig, PolicyConfig, RunConfig, ServiceConfig  # noqa: E402
from pilot.sim.policies.base import PolicyContext  # noqa: E402
from pilot.sim.service import ServiceResult, load_service_config  # noqa: E402
from pilot.sim.sweeps import with_price_ratio  # noqa: E402
from pilot.sim.workload import IN_CLASS, ArrivalStream  # noqa: E402
from tuned_config import apply_policy_values, policy_overrides  # noqa: E402

# ``replay_client`` imports ``requests`` at module level for the box; the laptop only needs its
# length formulas, so a stub stands in while the client is imported and is removed right after.
_STUBBED = "requests" not in sys.modules
if _STUBBED:
    try:
        import requests  # noqa: F401
        _STUBBED = False
    except ModuleNotFoundError:
        sys.modules["requests"] = types.ModuleType("requests")
from replay_client import CHARS_PER_TOKEN, Request, make_prompt  # noqa: E402

if _STUBBED:
    del sys.modules["requests"]

logger = logging.getLogger(__name__)

CASE = "c3_72b_shared4k"
CALIB_CSV = PILOT / "scripts/q2/results/b5_remote/calib_single_6000d/T3_calibration.csv"  # the card the run uses
MODEL_ROW = "qwen2.5-7b"
MAX_CONCURRENCY = 10  # --max-num-seqs of the restricted replica
MIRROR = PILOT / "outputs/validation_remote/box2"
TUNED: Dict[Tuple[str, str], Path] = {
    ("d_rep", "velocity"): MIRROR / "validation" / CASE / "tuning_r30" / "best.json",
    ("d_rep", "reactive"): MIRROR / "validation2" / CASE / "tuning_r30_slo95disp" / "best.json",
    ("d_inst", "velocity"): MIRROR / "validation" / CASE / "tuning8_r30" / "best.json",
    ("d_inst", "reactive"): MIRROR / "validation2" / CASE / "tuning8_r30_slo95disp" / "best.json",
}
TIERS: Tuple[str, ...] = ("d_rep", "d_inst")
LEAD_MIN: Dict[str, float] = {"d_rep": 1.0, "d_inst": 8.0}  # decision -> installed, minutes
GATE_MIN: Dict[str, int] = {"d_rep": 0, "d_inst": 8}  # the pool's --lead-gate-min
PRICE_RATIO = 30.0
C_RUN = 0.0333
BETA_START = 0.0333
SLO_S = 2.0
X_MAX = 4.0  # replicas one RTX 6000D (85.6 GB) holds at GPU_FRAC each
GPU_FRAC = 0.22  # --gpu-memory-utilization of one capped replica on that card
FIT_HEADROOM = 1.1  # thinning keeps the optimum's peak at least this far below X_MAX
MIN_REPLICAS = 1
PROMPT_TOKEN_CAP = 6000
OUTPUT_TOKEN_CAP = 320  # replay_client.Request.max_tokens clips here (design: 1,000)
OUTPUT_TOKEN_FLOOR = 16
ROUTER_TEMPLATE_TOKENS = 12  # the router counts ceil(len(content) / 1.45) + 12 + max_tokens
UNSERVED_TTFT_S = 1e6  # a request that never started: an unbounded wait (as single_bell_replay)
THIN_SEED = 2026  # numpy default_rng seed of the Bernoulli thinning


@dataclass(frozen=True)
class WindowSpec:
    """The replayed window around one bell, in minutes of the run the stream came from."""

    bell_min: int
    before_min: int = 18
    after_min: int = 27
    tail_min: int = 10  # simulator drain after the last arrival
    plan_tail_min: int = 8  # the plan carries the d_inst lead beyond the window

    @property
    def start_min(self) -> int:
        """First minute of the window (T0)."""
        return self.bell_min - self.before_min

    @property
    def end_min(self) -> int:
        """First minute after the window."""
        return self.bell_min + self.after_min

    @property
    def duration_min(self) -> int:
        """Minutes the pool is driven for."""
        return self.before_min + self.after_min

    @property
    def horizon_min(self) -> int:
        """Minutes the simulator runs for (window plus drain)."""
        return self.duration_min + self.tail_min

    @property
    def plan_len(self) -> int:
        """Entries of a plan file: ``duration + 8 + 1``."""
        return self.duration_min + self.plan_tail_min + 1


def _cut(arr: Optional[np.ndarray], lo: int, n: int) -> Optional[np.ndarray]:
    """``arr[lo:lo + n]`` as floats, zero-padded when the source is shorter; None stays None."""
    if arr is None:
        return None
    part = np.asarray(arr, dtype=float)[lo:lo + n]
    return np.pad(part, (0, n - part.size)) if part.size < n else part.copy()


def _window_sessions(stream: ArrivalStream, spec: WindowSpec, idx: np.ndarray) -> pd.DataFrame:
    """Sessions of the bell plus every session with a request in the window, re-timed to T0."""
    sessions = stream.sessions
    if sessions.empty or "session_id" not in sessions.columns:
        return sessions.copy()
    keep = sessions["session_id"].isin(np.unique(stream.session_id[idx]))
    if "bell_min" in sessions.columns:
        keep |= sessions["bell_min"] == spec.bell_min
    part = sessions[keep].reset_index(drop=True).copy()
    for col in ("bell_min", "realised_start_min"):
        if col in part.columns:
            part[col] = part[col] - spec.start_min
    return part


def cut_window(stream: ArrivalStream, spec: WindowSpec) -> ArrivalStream:
    """The window's in-class requests as a stream of their own, times counted from T0: ``lam_in``
    recounted, ``lam_off`` zero (the pool replays no off-class request), forecasts cut to the same
    minutes, sessions reduced to those the window touches, token lengths the generator's."""
    lo_s, hi_s = spec.start_min * 60.0, spec.end_min * 60.0
    keep = (stream.kind == IN_CLASS) & (stream.t_arrive_s >= lo_s) & (stream.t_arrive_s < hi_s)
    idx = np.flatnonzero(keep)
    t = np.round(stream.t_arrive_s[idx] - lo_s, 3)
    order = np.argsort(t, kind="stable")
    idx, t = idx[order], t[order]
    horizon = spec.horizon_min
    minute = np.minimum((t // 60.0).astype(np.int64), horizon - 1)
    lam_in = np.bincount(minute, minlength=horizon).astype(float)
    zeros = np.zeros(horizon, dtype=float)
    lo = spec.start_min
    return ArrivalStream(
        t_arrive_s=t, kind=np.full(t.size, IN_CLASS, dtype=np.int8),
        session_id=stream.session_id[idx], prompt_tokens=stream.prompt_tokens[idx],
        output_tokens=stream.output_tokens[idx], lam_in=lam_in, lam_off=zeros,
        lam_hat_in=_cut(stream.lam_hat_in, lo, horizon), lam_hat_off=zeros.copy(),
        sessions=_window_sessions(stream, spec, idx), horizon_min=horizon,
        lam_stale_in=_cut(stream.lam_stale_in, lo, horizon),
        student_id=None if stream.student_id is None else stream.student_id[idx],
        lam_hat_learn_in=_cut(stream.lam_hat_learn_in, lo, horizon),
        lam_hat_learn_prior_in=_cut(stream.lam_hat_learn_prior_in, lo, horizon))


def thin_window(window: ArrivalStream, f: float, seed: int = THIN_SEED) -> ArrivalStream:
    """Bernoulli thinning: every request kept with probability ``f`` and every forecast scaled by
    ``f`` (``f >= 1`` keeps the stream); ``lam_in`` is recounted, the session table untouched."""
    if f >= 1.0:
        return window
    keep = np.random.default_rng(seed).random(window.n_requests) < f
    t = window.t_arrive_s[keep]
    minute = np.minimum((t // 60.0).astype(np.int64), window.horizon_min - 1)

    def scaled(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if arr is None else np.asarray(arr, dtype=float) * f

    return replace(window, t_arrive_s=t, kind=window.kind[keep], session_id=window.session_id[keep],
                   prompt_tokens=window.prompt_tokens[keep], output_tokens=window.output_tokens[keep],
                   lam_in=np.bincount(minute, minlength=window.horizon_min).astype(float),
                   lam_hat_in=scaled(window.lam_hat_in), lam_stale_in=scaled(window.lam_stale_in),
                   student_id=None if window.student_id is None else window.student_id[keep],
                   lam_hat_learn_in=scaled(window.lam_hat_learn_in),
                   lam_hat_learn_prior_in=scaled(window.lam_hat_learn_prior_in))


# --- the client's length rules ------------------------------------------------------------------
CLASSROOM_BURST_CSV = PILOT / "scripts/q2/traces/burst_20260324_0830.csv"
LENGTH_SEED = 2027  # numpy default_rng seed of the classroom length sampling


def classroom_lengths(window: ArrivalStream, burst_csv: Path = CLASSROOM_BURST_CSV,
                      seed: int = LENGTH_SEED) -> ArrivalStream:
    """Give every request the length of a request of the recorded classroom burst.

    The generator's lengths come from the Azure conversation trace (prompts around a thousand
    tokens); the classroom questions of the platform exports are about 60 characters with replies
    of about 280.  Each request draws one (prompt, reply) pair of the recorded burst with
    replacement, so the replayed stream carries the classroom regime of the paper's short-prompt
    ladder row and the engine's decode stays in its near-linear range (B5, 2026-09-22: with the
    Azure lengths the prefill of the shared engine cut the per-stream rate from 82 to 45 tok/s
    between 10 and 50 streams in flight).  Tokens follow the client's conventions.
    """
    burst = pd.read_csv(burst_csv)
    pick = np.random.default_rng(seed).integers(0, len(burst), size=window.n_requests)
    prompt_chars = burst["prompt_chars"].to_numpy(dtype=np.int64)[pick]
    reply_chars = burst["reply_chars"].to_numpy(dtype=np.int64)[pick]
    prompt = np.array([prompt_tokens_of(int(i), int(c)) for i, c in enumerate(prompt_chars)],
                      dtype=np.int64)
    out = np.array([output_tokens_of(int(c)) for c in reply_chars], dtype=np.int64)
    return replace(window, prompt_tokens=prompt, output_tokens=out)


def output_tokens_of(reply_chars: int) -> int:
    """Output tokens the client asks for (``ignore_eos``): the router counts ``max_tokens``."""
    return Request(0, 0.0, 0, 0, int(reply_chars)).max_tokens


@lru_cache(maxsize=None)
def reply_chars_for(tokens: int) -> int:
    """Characters that ``replay_client`` decodes back into exactly ``tokens`` output tokens: the
    design's ``round(tokens x 1.45)`` nudged by a character where the client's ``ceil(chars / 1.45)``
    disagrees.  ``tokens`` must sit inside the client's ``[16, 320]`` clip."""
    chars = int(round(tokens * CHARS_PER_TOKEN))
    while chars > 0 and output_tokens_of(chars) > tokens:
        chars -= 1
    while output_tokens_of(chars) < tokens:
        chars += 1
    if output_tokens_of(chars) != tokens:
        raise ValueError(f"no reply_chars decodes to {tokens} tokens")
    return chars


def encode_prompt_chars(prompt_tokens: np.ndarray) -> np.ndarray:
    """``round(min(tokens, 6000) x 1.45)`` characters of filler (design 6)."""
    capped = np.minimum(np.asarray(prompt_tokens, dtype=float), PROMPT_TOKEN_CAP)
    return np.rint(capped * CHARS_PER_TOKEN).astype(np.int64)


def encode_reply_chars(output_tokens: np.ndarray) -> np.ndarray:
    """Reply characters that the client turns back into the clipped token count."""
    clipped = np.clip(np.asarray(output_tokens, dtype=np.int64), OUTPUT_TOKEN_FLOOR, OUTPUT_TOKEN_CAP)
    return np.array([reply_chars_for(int(v)) for v in clipped], dtype=np.int64)


def prompt_tokens_of(idx: int, prompt_chars: int) -> int:
    """Prompt tokens as the pool's router counts them: ``ceil(len(content) / 1.45) + 12``."""
    content = make_prompt(int(idx), int(prompt_chars))
    return int(math.ceil(len(content) / CHARS_PER_TOKEN)) + ROUTER_TEMPLATE_TOKENS


def trace_frame(window: ArrivalStream) -> pd.DataFrame:
    """The client's CSV rows for the window: one per request, in dispatch order."""
    return pd.DataFrame({"offset_s": np.round(window.t_arrive_s, 3),
                         "student": window.session_id.astype(np.int64),
                         "prompt_chars": encode_prompt_chars(window.prompt_tokens),
                         "reply_chars": encode_reply_chars(window.output_tokens)})


def client_lengths(prompt_chars: np.ndarray, reply_chars: np.ndarray,
                   idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-request (prompt, output) tokens both worlds account for; row ``i`` is request ``[i]``."""
    ids = np.arange(len(prompt_chars)) if idx is None else np.asarray(idx)
    prompt = np.array([prompt_tokens_of(int(i), int(c)) for i, c in zip(ids, prompt_chars)],
                      dtype=np.int64)
    out = np.array([output_tokens_of(int(c)) for c in reply_chars], dtype=np.int64)
    return prompt, out


def with_client_lengths(window: ArrivalStream, trace: pd.DataFrame) -> ArrivalStream:
    """``window`` with the token lengths the client will send and the router will count."""
    prompt, out = client_lengths(trace["prompt_chars"].to_numpy(), trace["reply_chars"].to_numpy())
    return replace(window, prompt_tokens=prompt, output_tokens=out)


def stream_from_records(rec: pd.DataFrame, horizon_min: int) -> ArrivalStream:
    """The window stream rebuilt from the client's records (or trace rows) alone: ``offset_s``,
    ``student``, ``prompt_chars`` and ``max_tokens`` (records) or ``reply_chars`` (trace); ``idx``
    when present fixes the request numbering."""
    key = "idx" if "idx" in rec.columns else "offset_s"
    rec = rec.sort_values(key, kind="stable").reset_index(drop=True)
    t = rec["offset_s"].to_numpy(dtype=float)
    idx = rec["idx"].to_numpy() if "idx" in rec.columns else np.arange(len(rec))
    if "max_tokens" in rec.columns:
        out = rec["max_tokens"].to_numpy(dtype=np.int64)
        prompt = np.array([prompt_tokens_of(int(i), int(c))
                           for i, c in zip(idx, rec["prompt_chars"])], dtype=np.int64)
    else:
        prompt, out = client_lengths(rec["prompt_chars"].to_numpy(), rec["reply_chars"].to_numpy(),
                                     idx)
    minute = np.minimum((t // 60.0).astype(np.int64), horizon_min - 1)
    lam_in = np.bincount(minute, minlength=horizon_min).astype(float)
    zeros = np.zeros(horizon_min, dtype=float)
    return ArrivalStream(t_arrive_s=t, kind=np.full(t.size, IN_CLASS, dtype=np.int8),
                         session_id=rec["student"].to_numpy(dtype=np.int64), prompt_tokens=prompt,
                         output_tokens=out, lam_in=lam_in, lam_off=zeros, lam_hat_in=lam_in.copy(),
                         lam_hat_off=zeros.copy(), sessions=pd.DataFrame(), horizon_min=horizon_min)


def minute_tokens(stream: ArrivalStream) -> np.ndarray:
    """Tokens arriving per minute (prompt + output), the token-velocity policy's input."""
    minute = np.minimum((stream.t_arrive_s // 60.0).astype(np.int64), stream.horizon_min - 1)
    tok = (stream.prompt_tokens + stream.output_tokens).astype(float)
    return np.bincount(minute, weights=tok, minlength=stream.horizon_min)


# --- the capped replica and the run configuration ----------------------------------------------
def measured_ttft_p95(calib: Path, model: str, concurrency: int) -> float:
    """The card's median ``ttft_p95_s@N=<concurrency>`` for ``model`` (batch-B T3 CSV format)."""
    df = pd.read_csv(calib)
    row = df[(df["model"] == model) & (df["metric"] == f"ttft_p95_s@N={concurrency}")]
    if row.empty:
        raise ValueError(f"{calib}: no ttft_p95_s@N={concurrency} row for {model}")
    return float(row["median"].iloc[0])


def capped_service(lead_min: float, calib: Path = CALIB_CSV, model: str = MODEL_ROW,
                   ttft_at_k_s: Optional[float] = None) -> ServiceConfig:
    """The restricted replica: the T3 card with K pinned to ``--max-num-seqs`` (10) and the TTFT
    p95 at K set to the card's measured value at N = K, not to the SLO ``load_service_config``
    assumes when it defines K by the SLO crossing; ``ttft_at_k_s`` overrides that value."""
    loaded = load_service_config(calib, model, base=ServiceConfig(), slo_in_s=SLO_S)
    at_k = measured_ttft_p95(calib, model, MAX_CONCURRENCY) if ttft_at_k_s is None else ttft_at_k_s
    return replace(loaded, max_concurrency=MAX_CONCURRENCY, ttft_at_k_s=float(at_k),
                   lead_min=float(lead_min))


def fluid_mu(service: ServiceConfig, stream: ArrivalStream) -> float:
    """``mu_cap``: requests per minute of one replica at the stream's mean output length."""
    return service.requests_per_min(float(stream.output_tokens.mean()) if stream.n_requests else 128.0)


def tier_config(base: RunConfig, tier: str, policy: str, mu: float,
                service: Optional[ServiceConfig] = None, x_max: float = X_MAX) -> RunConfig:
    """``base`` on ``tier`` for ``policy``: capped replica, r = 30 prices, x_max 5, tuned knobs.

    ``policy`` is ``velocity``, ``reactive`` or ``plan``; the first two take the validation
    round's tuned values for the tier (``TUNED``), the planners keep the base ``PolicyConfig``.
    """
    if tier not in LEAD_MIN:
        raise KeyError(f"unknown tier '{tier}'; known: {sorted(LEAD_MIN)}")
    service = replace(service or capped_service(LEAD_MIN[tier]), lead_min=LEAD_MIN[tier])
    cost = replace(CostConfig(), c_run=C_RUN, beta_start=BETA_START, slo_in_s=SLO_S, x_max=float(x_max))
    knobs = PolicyConfig(min_replicas=MIN_REPLICAS)
    tuned = TUNED.get((tier, policy))
    if tuned is not None:
        if not tuned.exists():
            raise FileNotFoundError(f"tuned values for {tier}/{policy} missing: {tuned}")
        knobs = apply_policy_values(knobs, policy_overrides(tuned, None))
    cfg = replace(base, service=service, cost=cost, policy=knobs)
    return with_price_ratio(cfg, mu, PRICE_RATIO)


def minute_table(stream: ArrivalStream, result: ServiceResult, n_minutes: int,
                 slo_s: float = SLO_S) -> pd.DataFrame:
    """Per minute of arrival: requests, share within ``slo_s``, TTFT p95, timeouts.  A request that
    never started is an unbounded wait (``UNSERVED_TTFT_S``), so a minute's p95 is below that
    sentinel only when at least 95% of its requests were answered."""
    minute = np.minimum((stream.t_arrive_s // 60.0).astype(np.int64), stream.horizon_min - 1)
    served = np.asarray(result.served, dtype=bool)
    ttft = np.where(served, result.ttft_s, UNSERVED_TTFT_S)
    ok = served & (result.ttft_s <= slo_s)
    rows = []
    for m in range(n_minutes):
        sel = minute == m
        n = int(sel.sum())
        rows.append({"minute": m, "arrivals": n,
                     "share_slo": float(ok[sel].mean()) if n else float("nan"),
                     "ttft_p95_s": float(np.percentile(ttft[sel], 95)) if n else float("nan"),
                     "timeouts": int(np.asarray(result.timed_out)[sel].sum())})
    return pd.DataFrame(rows)


def measured_minutes(rec: pd.DataFrame, duration: int, abandon_s: float,
                     slo_s: float = SLO_S, minute_s: float = 60.0) -> pd.DataFrame:
    """The client's records per minute of scheduled arrival (``minute_s`` long, 6 s in a dry run):
    share within the SLO and TTFT p95, a give-up being an unbounded wait as in ``minute_table``;
    the ``aligned`` p95 also treats a wait beyond ``abandon_s`` as one (the simulator's rule)."""
    minute = np.minimum((rec["offset_s"].to_numpy(dtype=float) // minute_s).astype(np.int64),
                        duration - 1)
    ttft = pd.to_numeric(rec["ttft_s"], errors="coerce").to_numpy(dtype=float)
    raw = np.where(np.isfinite(ttft), ttft, UNSERVED_TTFT_S)
    aligned = np.where(raw > abandon_s, UNSERVED_TTFT_S, raw)
    rows = []
    for m in range(duration):
        sel = minute == m
        n = int(sel.sum())
        rows.append({"t": m, "arrivals_meas": n,
                     "share_meas": float(np.mean(raw[sel] <= slo_s)) if n else float("nan"),
                     "p95_meas": float(np.percentile(raw[sel], 95)) if n else float("nan"),
                     "p95_meas_aligned": (float(np.percentile(aligned[sel], 95)) if n
                                          else float("nan")),
                     "gave_up_meas": int((~np.isfinite(ttft[sel])).sum()),
                     "over_abandon_meas": int((raw[sel] > abandon_s).sum())})
    return pd.DataFrame(rows)


def context_of(ctx: Dict[str, object]) -> PolicyContext:
    """``PolicyContext`` (forecast-free) from a ``context_*.json`` of design 5.6."""
    horizon = int(ctx["horizon_min"])  # type: ignore[arg-type]
    zeros = np.zeros(horizon)
    return PolicyContext(horizon_min=horizon, mu=float(ctx["mu"]),  # type: ignore[arg-type]
                         lead_min=int(ctx["lead_min"]),  # type: ignore[arg-type]
                         cost=CostConfig(**ctx["cost"]),  # type: ignore[arg-type]
                         policy=PolicyConfig(**ctx["policy"]),  # type: ignore[arg-type]
                         service=ServiceConfig(**ctx["service"]),  # type: ignore[arg-type]
                         lam_hat_in=zeros, lam_hat_off=zeros.copy())


__all__ = ["CALIB_CSV", "CASE", "GATE_MIN", "LEAD_MIN", "MAX_CONCURRENCY", "MODEL_ROW",
           "OUTPUT_TOKEN_CAP", "PRICE_RATIO", "PROMPT_TOKEN_CAP", "SLO_S", "THIN_SEED", "TIERS",
           "TUNED", "UNSERVED_TTFT_S", "X_MAX", "GPU_FRAC", "FIT_HEADROOM", "WindowSpec", "capped_service", "client_lengths",
           "context_of", "cut_window", "encode_prompt_chars", "encode_reply_chars", "fluid_mu",
           "measured_minutes", "measured_ttft_p95", "minute_table", "minute_tokens",
           "output_tokens_of", "prompt_tokens_of", "reply_chars_for", "stream_from_records",
           "classroom_lengths", "CLASSROOM_BURST_CSV",
           "thin_window", "tier_config", "trace_frame", "with_client_lengths"]
