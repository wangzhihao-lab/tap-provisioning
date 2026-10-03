#!/usr/bin/env python3
"""Export the busiest bell of the validation case as the replica pool's inputs (B' step 2).

    uv run python scripts/q2/make_bell_trace.py --out scripts/q2/pool_inputs

Rebuilds the validation round's central 72B case (``single_bell_replay.base_config``: seed 10,
three weeks, cell C), takes week 3's busiest bell and the window ``[-18, +27)`` minutes around it,
and writes under ``--out`` everything the box (``replica_pool.py`` through
``remote/measurements.py``) and the local comparison (``pool_compare.py``) need:

``bell_c3_w3.csv`` / ``.json``   the trace the client replays (offset_s, student, prompt_chars,
                                  reply_chars), its provenance and the per-minute demand
``context_<tier>_<policy>.json``  design 5.6: mu, lead, x_max, the cost / policy / service configs
``plan_learnprior.json``, ``plan_offline.json``   ``calendar_learn_prior`` and ``offline`` run on
                                  the window sub-stream itself, x cut to 45 + 8 + 1 entries
``sim_predictions.json``          the simulator's answer for every run of design 4: x per minute,
                                  share within 2 s per minute, replica-minutes, starts, slo_in
``pool_runs.json``                design 5.7: the run list the box executes, in design 4's order

The window sub-stream (``bell_window``) carries the client's token lengths, so the simulator and
the box account for the same requests, and is thinned by ``f = mu_cap / mu_72b`` so that the
bell needs as many emulated replicas as it needs 72B replicas in the paper: ``mu_cap`` is the
restricted replica's fluid capacity at the window's mean output length and ``mu_72b`` the paper's
(``engine.fluid_mu`` of the c3 configuration, as ``single_bell_replay.py`` computes it).  Every
policy, the planners included, is run on the window sub-stream with the tier's lead, one replica
at the window start and ``x_max`` = ``X_MAX``.  ``--mu-cap`` overrides the computed capacity.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
for _extra in (PILOT / "src", PILOT / "scripts" / "sim", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bell_window import (CASE, GATE_MIN, LEAD_MIN, OUTPUT_TOKEN_CAP, PRICE_RATIO,  # noqa: E402
                         PROMPT_TOKEN_CAP, THIN_SEED, TIERS, TUNED, X_MAX, GPU_FRAC, FIT_HEADROOM, CALIB_CSV, WindowSpec,
                         classroom_lengths,
                         capped_service, cut_window, fluid_mu, minute_table, minute_tokens,
                         thin_window, tier_config, trace_frame, with_client_lengths)
from pilot.sim import RunConfig, ServiceConfig, engine  # noqa: E402
from pilot.sim.metrics import summarize  # noqa: E402
from pilot.sim.service import replay  # noqa: E402
from pilot.sim.workload import ArrivalStream  # noqa: E402
from single_bell_replay import base_config, busiest_bell  # noqa: E402

logger = logging.getLogger("make_bell_trace")

TRACE_STEM = "bell_c3_w3"
INPUTS_PREFIX = "pool_inputs"  # how the box addresses the files, relative to scripts/q2
HEAD_S = 240  # five concurrent vLLM launches at pool start (plan rounds) need more than 180 s
ABANDON_S = 60.0  # the router's queue abandonment = the simulator's abandon_in_s; the client gives up at the same age
# vLLM cold starts measured on the run's card (B5 calib_multi, PRO 6000, 2026-09-22): the virtual
# replicas of the single-engine pool wait these many seconds before they serve (cycled)
COLD_START_S: Tuple[float, ...] = (40.1, 29.1, 28.1, 29.1, 28.1)
CALIB_MULTI_LEVELS = "8,16,24,32,40"


def calib_multi_spec(x_max: float) -> Dict[str, object]:
    """The co-residency calibration: every replica the card holds, behind the router."""
    return {"replicas": int(x_max), "levels": CALIB_MULTI_LEVELS, "duration_s": 30}
PLANNERS: Dict[str, str] = {"plan_learnprior": "calendar_learn_prior", "plan_offline": "offline"}
PLAN_LEAD: Dict[str, int] = {"plan_learnprior": 8, "plan_offline": 1}


@dataclass(frozen=True)
class RunSpec:
    """One round of design 4."""

    label: str
    policy: str  # reactive | velocity | plan
    tier: str
    plan: Optional[str] = None  # plan file stem for the plan rounds

    @property
    def context(self) -> str:
        """The context file the round reads."""
        return f"context_{self.tier}_{self.policy}.json"


RUNS: Tuple[RunSpec, ...] = (
    RunSpec("plan_learnprior_d_inst", "plan", "d_inst", "plan_learnprior"),
    RunSpec("velocity_d_inst", "velocity", "d_inst"),
    RunSpec("plan_offline", "plan", "d_rep", "plan_offline"),
    RunSpec("plan_learnprior_d_rep", "plan", "d_rep", "plan_learnprior"),
    RunSpec("velocity_d_rep", "velocity", "d_rep"),
    RunSpec("reactive_d_inst", "reactive", "d_inst"),
    RunSpec("reactive_d_rep", "reactive", "d_rep"),
)


def _listed(values: Sequence[float]) -> List[Optional[float]]:
    """Floats for JSON: NaN becomes null."""
    return [None if (isinstance(v, float) and not np.isfinite(v)) else float(v) for v in values]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write ``payload`` as indented JSON."""
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def context_payload(cfg: RunConfig, mu: float, spec: WindowSpec, tier: str, policy: str
                    ) -> Dict[str, object]:
    """Design 5.6: what the pool's controller needs to run ``policy`` on ``tier``."""
    tuned = TUNED.get((tier, policy))
    return {"mu": mu, "lead_min": int(LEAD_MIN[tier]), "horizon_min": spec.duration_min,
            "x_max": int(cfg.cost.x_max), "min_replicas": int(cfg.policy.min_replicas),
            "cost": asdict(cfg.cost), "policy": asdict(cfg.policy), "service": asdict(cfg.service),
            "meta": {"tier": tier, "policy": policy, "gate_min": GATE_MIN[tier],
                     "price_ratio": PRICE_RATIO, "case": CASE, "seed": cfg.seed,
                     "bell_min": spec.bell_min, "tuned": None if tuned is None else str(tuned),
                     "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}


def window_plans(configs: Dict[Tuple[str, str], RunConfig], window: ArrivalStream,
                 spec: WindowSpec, mu: float) -> Dict[str, np.ndarray]:
    """The planners run on the window sub-stream, x cut to ``plan_len`` entries (last value padded).

    ``calendar_learn_prior`` is run at both tiers' leads; its x is the same whatever the lead (it
    pre-commits the first ``d`` minutes and returns ``plan[t + d]`` afterwards), which is checked
    here, so one file serves the d_inst and the d_rep round.  ``offline`` runs at the d_rep lead.
    """
    out: Dict[str, np.ndarray] = {}
    for stem, name in PLANNERS.items():
        xs = {}
        for tier in (TIERS if stem == "plan_learnprior" else ("d_rep",)):
            cfg = configs[(tier, "plan")]
            ctx = replace(engine.make_context(cfg, window), mu=mu)
            xs[tier] = engine.run_policy(name, cfg, window, ctx).x
        if len(xs) > 1 and not np.array_equal(xs["d_rep"], xs["d_inst"]):
            logger.warning("%s: x differs between the tiers' leads; the d_inst run is written", stem)
        x = np.rint(xs.get("d_inst", xs["d_rep"])).astype(int)[: spec.plan_len]
        out[stem] = np.pad(x, (0, spec.plan_len - x.size), mode="edge")
        logger.info("%s (%s): window x %s, tail %s", stem, name,
                    out[stem][: spec.duration_min].tolist(), out[stem][spec.duration_min:].tolist())
    return out


def prediction(run: RunSpec, cfg: RunConfig, window: ArrivalStream, spec: WindowSpec, mu: float,
               plan: Optional[np.ndarray]) -> Dict[str, object]:
    """The simulator's answer for one round on the window sub-stream."""
    n = spec.duration_min
    if plan is None:
        ctx = replace(engine.make_context(cfg, window), mu=mu)
        res = engine.run_policy(run.policy, cfg, window, ctx)
        x, result, metrics = res.x, res.service, res.metrics
    else:
        x = np.asarray(plan, dtype=float)
        x = np.pad(x, (0, max(0, window.horizon_min - x.size)), mode="edge")[: window.horizon_min]
        result = replay(window, x, cfg.service)
        metrics = summarize(run.label, cfg.seed, x, window, result, mu, cfg.cost,
                            cfg.service.billing_block_min)
    table = minute_table(window, result, n)
    tokens = minute_tokens(window)
    return {"label": run.label, "policy": run.policy, "tier": run.tier,
            "lead_min": int(LEAD_MIN[run.tier]), "gate_min": GATE_MIN[run.tier], "mu": mu,
            "plan": run.plan, "horizon_min": window.horizon_min, "duration_min": n,
            "x": np.rint(x).astype(int).tolist(),
            "arrivals": window.lam_in[:n].astype(int).tolist(),
            "tokens": tokens[:n].astype(int).tolist(),
            "share_slo": _listed(table["share_slo"].tolist()),
            "ttft_p95_s": _listed(table["ttft_p95_s"].tolist()),
            "timeouts_per_minute": table["timeouts"].astype(int).tolist(),
            "replica_minutes": float(x[:n].sum()), "replica_minutes_total": float(x.sum()),
            "starts": float(metrics.starts),
            "starts_after_t0": float(np.maximum(np.diff(x[:n]), 0.0).sum()),
            "slo_in": float(metrics.slo_in), "ttft_p95_in_s": float(metrics.ttft_p95_in_s),
            "gpu_hours": float(metrics.gpu_hours), "timeouts": int(metrics.timeouts),
            "n_requests": int(window.n_requests)}


def trace_meta(stream: ArrivalStream, raw: ArrivalStream, window: ArrivalStream,
               trace: pd.DataFrame, spec: WindowSpec, seed: int, week: int, mu: float,
               mean_out: float, thinning: Dict[str, object], service: ServiceConfig
               ) -> Dict[str, object]:
    """``bell_c3_w3.json``: provenance, window, thinning, demand per minute, length conventions.

    ``raw`` is the thinned window with the generator's lengths, ``window`` the same with the
    client's; ``mean_out`` is the unthinned window's mean output length ``mu`` was computed at.
    """
    sessions = stream.sessions
    at_bell = (sessions[sessions["bell_min"] == spec.bell_min] if "bell_min" in sessions.columns
               else sessions)
    n = spec.duration_min
    return {"case": CASE, "seed": seed, "week": week, "bell_min": spec.bell_min,
            "window_min": [spec.start_min, spec.end_min], "window_from_bell_min": [-spec.before_min,
                                                                                    spec.after_min],
            "duration_min": n, "horizon_min": spec.horizon_min, "n_requests": int(len(trace)),
            "n_sessions_at_bell": int(len(at_bell)),
            "seats_at_bell": int(at_bell["roster"].sum()) if "roster" in at_bell else None,
            "n_sessions_in_window": int(trace["student"].nunique()), "mu_cap_req_min": mu,
            "mean_output_tokens": mean_out,
            "mean_output_tokens_thinned": float(window.output_tokens.mean()),
            "service": {"max_concurrency": service.max_concurrency,
                        "ttft_at_k_s": service.ttft_at_k_s, "ttft_floor_s": service.ttft_floor_s,
                        "decode_tokens_per_s": service.decode_tokens_per_s,
                        "per_stream_tokens_per_s": service.per_stream_tokens_per_s,
                        "source": service.source, "model": service.model},
            "thinning": thinning, "peak_arrivals_per_min": int(window.lam_in[:n].max()),
            "lam_in": window.lam_in[:n].astype(int).tolist(),
            "lam_hat_in": _listed(window.lam_hat_in[:n].tolist()),
            "lam_hat_learn_prior_in": (None if window.lam_hat_learn_prior_in is None
                                       else _listed(window.lam_hat_learn_prior_in[:n].tolist())),
            "lengths": {"chars_per_token": 1.45, "prompt_token_cap": PROMPT_TOKEN_CAP,
                        "prompt_chars_cap": int(trace["prompt_chars"].max()),
                        "output_token_cap": OUTPUT_TOKEN_CAP,
                        "note": "replay_client clips max_tokens to [16, 320]; reply_chars encodes the "
                                "clipped count exactly and the simulator side uses the same lengths "
                                "(design 6 asked for a 1,000-token cap, which the client cannot send)",
                        "prompt_tokens_router": "ceil(len(content) / 1.45) + 12",
                        "output_tokens_clipped": int((raw.output_tokens > OUTPUT_TOKEN_CAP).sum()),
                        "prompt_tokens_clipped": int((raw.prompt_tokens > PROMPT_TOKEN_CAP).sum())},
            "columns": ["offset_s", "student", "prompt_chars", "reply_chars"],
            "student_is": "session_id of the generator", "source": "scripts/q2/make_bell_trace.py",
            "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def pool_runs_payload(spec: WindowSpec, x_max: float = X_MAX, gpu_frac: float = GPU_FRAC,
                      engine_mode: str = "single", cold_start_s: Tuple[float, ...] = COLD_START_S
                      ) -> Dict[str, object]:
    """Design 5.7: the run list for ``remote/measurements.stage_pool``."""
    runs = []
    for run in RUNS:
        row: Dict[str, object] = {"label": run.label, "policy": run.policy,
                                  "lead_gate_min": GATE_MIN[run.tier],
                                  "context": f"{INPUTS_PREFIX}/{run.context}"}
        if run.plan:
            row["plan"] = f"{INPUTS_PREFIX}/{run.plan}.json"
        runs.append(row)
    return {"trace": f"{INPUTS_PREFIX}/{TRACE_STEM}.csv", "max_offset_s": spec.duration_min * 60,
            "duration_min": spec.duration_min, "head_s": HEAD_S, "calib_multi": calib_multi_spec(x_max),
            "abandon_s": ABANDON_S, "give_up_s": ABANDON_S, "max_replicas": int(x_max),
            "gpu_frac": float(gpu_frac), "engine": engine_mode, "cold_start_s": list(cold_start_s),
            "runs": runs}


def build_inputs(stream: ArrivalStream, spec: WindowSpec, base: RunConfig, out: Path,
                 week: int, mu_override: Optional[float] = None,
                 ttft_at_k_s: Optional[float] = None, mu_ref: Optional[float] = None,
                 calib: Path = CALIB_CSV, x_max: float = X_MAX, gpu_frac: float = GPU_FRAC,
                 engine_mode: str = "single", cold_start_s: Tuple[float, ...] = COLD_START_S,
                 length_regime: str = "classroom") -> Dict[str, object]:
    """Write every input file for ``stream``'s window into ``out``; returns the predictions.

    ``mu_override`` replaces the computed ``mu_cap``, ``ttft_at_k_s`` the card's measured TTFT
    p95 at K, ``mu_ref`` the paper's ``mu_72b`` (default: ``engine.fluid_mu`` of ``base``).
    """
    out.mkdir(parents=True, exist_ok=True)
    raw = cut_window(stream, spec)
    if length_regime == "classroom":
        raw = classroom_lengths(raw)
    services = {tier: capped_service(LEAD_MIN[tier], calib=calib, ttft_at_k_s=ttft_at_k_s)
                for tier in TIERS}
    unthinned = with_client_lengths(raw, trace_frame(raw))
    mean_out = float(unthinned.output_tokens.mean())
    mu = float(mu_override) if mu_override else fluid_mu(services["d_rep"], unthinned)
    mu_72b = float(mu_ref) if mu_ref else engine.fluid_mu(base, stream)
    f_paper = mu / mu_72b  # the bell needs as many capped replicas as 72B replicas in the paper
    f_fit = (x_max - FIT_HEADROOM) * mu / float(raw.lam_in.max())  # the optimum fits under x_max
    f = min(1.0, f_paper, f_fit)
    thinned = thin_window(raw, f, THIN_SEED)
    trace = trace_frame(thinned)
    window = with_client_lengths(thinned, trace)
    thinning = {"f": f, "f_paper": f_paper, "f_fit": f_fit, "x_max": float(x_max), "fit_headroom": FIT_HEADROOM,
                "seed": THIN_SEED, "mu_cap_req_min": mu, "mu_72b_req_min": mu_72b,
                "kept": int(window.n_requests), "dropped": int(raw.n_requests - window.n_requests),
                "peak_arrivals_unthinned": int(raw.lam_in.max())}
    logger.info("window [%d, %d) = bell %d %+d/%+d min: K=%d, TTFT p95 at K %.3f s, mean output "
                "%.1f tokens (cap %d), mu_cap %.2f req/min%s; mu_72b %.2f -> f = %.4f: %d of %d "
                "in-class requests kept, peak %d/min (unthinned %d)", spec.start_min, spec.end_min,
                spec.bell_min, -spec.before_min, spec.after_min, services["d_rep"].max_concurrency,
                services["d_rep"].ttft_at_k_s, mean_out, OUTPUT_TOKEN_CAP, mu,
                " (override)" if mu_override else "", mu_72b, f, window.n_requests, raw.n_requests,
                int(window.lam_in.max()), int(raw.lam_in.max()))
    trace.to_csv(out / f"{TRACE_STEM}.csv", index=False, float_format="%.3f")
    write_json(out / f"{TRACE_STEM}.json",
               trace_meta(stream, thinned, window, trace, spec, base.seed, week, mu, mean_out,
                          thinning, services["d_rep"]))
    configs: Dict[Tuple[str, str], RunConfig] = {}
    for tier in TIERS:
        for policy in ("velocity", "reactive", "plan"):
            cfg = tier_config(base, tier, policy, mu, services[tier], x_max=x_max)
            configs[(tier, policy)] = cfg
            write_json(out / f"context_{tier}_{policy}.json",
                       context_payload(cfg, mu, spec, tier, policy))
    plans = window_plans(configs, window, spec, mu)
    for stem, x in plans.items():
        write_json(out / f"{stem}.json", {"label": stem, "lead_min": PLAN_LEAD[stem],
                                          "x": x.astype(int).tolist(),
                                          "meta": {"policy": PLANNERS[stem], "mu": mu,
                                                   "source": "run on the window sub-stream",
                                                   "bell_min": spec.bell_min}})
    predictions: Dict[str, object] = {}
    for run in RUNS:
        cfg = configs[(run.tier, run.policy)]
        pred = prediction(run, cfg, window, spec, mu, None if run.plan is None else plans[run.plan])
        predictions[run.label] = pred
        logger.info("%-24s predicted slo_in %.3f, replica-minutes %.0f (x peak %d), starts after T0 "
                    "%.0f, timeouts %d", run.label, pred["slo_in"], pred["replica_minutes"],
                    max(pred["x"][: spec.duration_min]), pred["starts_after_t0"], pred["timeouts"])
    write_json(out / "sim_predictions.json", predictions)
    write_json(out / "pool_runs.json", pool_runs_payload(spec, x_max, gpu_frac, engine_mode, cold_start_s))
    logger.info("wrote %d files under %s", len(list(out.glob("*"))), out)
    return predictions


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=PILOT / "scripts/q2/pool_inputs")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--weeks", type=int, default=3)
    ap.add_argument("--week", type=int, default=3, help="week whose busiest bell is exported")
    ap.add_argument("--mu-cap", type=float, default=None,
                    help="override the fluid capacity (default: requests_per_min at the window's "
                         "mean output length)")
    ap.add_argument("--ttft-at-k-s", type=float, default=None,
                    help="override the service model's TTFT p95 at K = 10 (default: the 2 s SLO "
                         "convention of load_service_config; measured at N = 10: 0.414 s)")
    ap.add_argument("--calib", type=Path, default=CALIB_CSV,
                    help="T3_calibration.csv of the capped replica on the card the run uses")
    ap.add_argument("--x-max", type=float, default=X_MAX,
                    help="replicas the card holds (4 on an 80-86 GB card at 0.22, 5 on a 96 GB card at 0.19)")
    ap.add_argument("--gpu-frac", type=float, default=GPU_FRAC,
                    help="--gpu-memory-utilization of one capped replica")
    ap.add_argument("--engine", choices=("single", "process"), default="single",
                    help="single: one vLLM engine, replicas are admission units")
    ap.add_argument("--cold-start-s", default=",".join(f"{v:g}" for v in COLD_START_S),
                    help="comma-separated emulated cold starts of the virtual replicas (seconds)")
    ap.add_argument("--length-regime", choices=("classroom", "azure"), default="classroom",
                    help="classroom: request lengths drawn from the recorded classroom burst; azure: the generator's")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("pilot.sim").setLevel(logging.WARNING)
    base = base_config(args.seed, args.weeks, LEAD_MIN["d_inst"], None)
    stream = engine.build_stream(base)
    bell = busiest_bell(stream, args.week)
    build_inputs(stream, WindowSpec(bell), base, args.out, args.week, args.mu_cap,
                 args.ttft_at_k_s, calib=args.calib, x_max=args.x_max, gpu_frac=args.gpu_frac,
                 engine_mode=args.engine,
                 cold_start_s=tuple(float(v) for v in args.cold_start_s.split(",") if v.strip()),
                 length_regime=args.length_regime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
