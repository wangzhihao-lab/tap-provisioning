"""Canonical numbers, part 3: the laptop cell C around TAP, and facts the supplementary quotes.

Imported by ``canonical_numbers.py`` and kept apart from ``canonical_extra.py`` for the size limit:
TAP against its components (section 27c), the seed spread of every row of section 27a (27d), when
TAP's online size correction re-solves (29c) and the facts the supplementary text quotes outside the
tables (section 30).  Sections 27 and 29 read ``paired_cell_c.csv``, the finished runs it points to
and the R1 runs' ``diagnostics.csv``; 29c rebuilds the streams from the runs' arguments.  Rounding
follows section 27: GPU-hours per week to 1 decimal, attainment to 3, percentages to 1.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from canonical_extra import (RUNS, LAPTOP_TIERS, PAIRED_CSV, WEEK_MIN, StreamCache, _paired,
                             _where)

if TYPE_CHECKING:
    from pilot.sim.config import RunConfig
    from pilot.sim.policies.base import PolicyContext
    from pilot.sim.workload import ArrivalStream

logger = logging.getLogger("canonical_laptop")

COMPONENTS: Tuple[Tuple[str, str], ...] = (
    ("tap", "TAP: learned-prior plan with the size correction, LCP fallback (eps = 0.1)"),
    ("calendar_learn_prior", "TAP without the size correction and the fallback (learned-prior plan alone)"),
    ("calendar_adapt", "timing given (the cell C forecast) plus the size correction"),
    ("calendar", "timing given, no correction"),
    ("offline", "offline optimum: knows the demand, no lead time"))
PLANNERS: Tuple[str, ...] = ("tap", "calendar_learn_prior", "calendar_adapt", "calendar")
FEEDBACK: Tuple[str, ...] = ("reactive", "velocity")  # every variant of either rule
R1_RUNS: Tuple[Tuple[str, str, str], ...] = (  # (short, case, tier) of TAP's R1 runs
    ("c2", "c2_32b_shared4k", "inst"), ("c2", "c2_32b_shared4k", "rep"),
    ("c3", "c3_72b_shared4k", "inst"), ("c3", "c3_72b_shared4k", "rep"))
RESOLVE_SEEDS: Tuple[int, ...] = (10, 11, 12, 13, 14)
DAYS: Tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
B5 = "scripts/q2/results/b5_remote"
MULTI_PROCESS = f"{B5}/b5/pool_process_attempt/calib_multi"  # five vLLM processes behind the router
ALONE = f"{B5}/mps_test/r1_alone/concurrency/concurrency_summary.jsonl"  # one process driven alone
TIMING_CELLS = "ABCDEFGHIJKL"  # the letters of SweepConfig.trigger_offset, in its order


def components_table(repo: Path) -> pd.DataFrame:
    """TAP against its components, laptop cell C, whole run: GPU-hours per week, relative to TAP's in
    percent from unrounded means, and in-class attainment (mean and lowest seed)."""
    rows: List[Dict[str, object]] = []
    for (case, tier), part in _paired(repo).query("week == 0").groupby(["case", "tier"], sort=False):
        by = part.set_index("name")
        tap = float(by.loc["tap", "gpu_h_per_week"])
        for policy, role in COMPONENTS:
            r = by.loc[policy]
            gpu = float(r["gpu_h_per_week"])
            rows.append({**_where(str(case), str(tier)), "policy": policy, "role": role,
                         "gpu_h_wk": f"{gpu:.1f}", "gpu_h_vs_tap_pct": f"{100 * (gpu / tap - 1):+.1f}",
                         "slo_in": f"{float(r['slo_in']):.3f}",
                         "slo_in_lowest_seed": f"{float(r['slo_in_min']):.3f}",
                         "run": r["source"], "source": PAIRED_CSV})
    return pd.DataFrame(rows)


def _seed_rows(repo: Path, source: str, policy: str, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One whole-run row per seed of ``policy`` in a finished run, selected as the paired table does."""
    from runs_paired_table import cell_c  # scripts/sim: the exact cell-C selection

    if source not in frames:
        frames[source] = cell_c(repo / source)
    frame = frames[source]
    part = frame[(frame["policy"] == policy) & (frame["week"] == 0)]
    return part.assign(gpu_h_per_week=part["gpu_hours"] / part["weeks"])


def _group(policy: str) -> str:
    """The group of section 27d a paired-table policy belongs to."""
    if policy in PLANNERS:
        return "planners"
    return "feedback rules" if policy in FEEDBACK else "other"


def laptop_seed_spread_tables(repo: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Seed standard deviation (ddof = 1, as section 17) of the six-week GPU-hours per week and
    in-class attainment of every row of section 27a, and the largest of each group over case-tiers.

    The seeds of every row must reproduce the paired table's means, or this raises.
    """
    frames: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []
    spread: List[Dict[str, object]] = []
    for r in _paired(repo).query("week == 0").itertuples(index=False):
        seeds = _seed_rows(repo, str(r.source), str(r.policy), frames)
        gpu, slo = seeds["gpu_h_per_week"], seeds["slo_in"]
        if (len(seeds) != int(r.seeds) or abs(float(gpu.mean()) - float(r.gpu_h_per_week)) > 1e-9
                or abs(float(slo.mean()) - float(r.slo_in)) > 1e-12):
            raise ValueError(f"{r.source}: the seeds of {r.name} do not give the means of {PAIRED_CSV}")
        where = _where(str(r.case), str(r.tier))
        sd_gpu, sd_slo = float(gpu.std(ddof=1)), float(slo.std(ddof=1))
        spread.append({"group": _group(str(r.policy)), "row": f"{r.name} ({where['case']}, {where['tier']})",
                       "gpu": sd_gpu, "slo": sd_slo})
        rows.append({**where, "group": _group(str(r.policy)), "policy": r.name,
                     "gpu_h_wk_sd": f"{sd_gpu:.1f}", "slo_in_sd": f"{sd_slo:.3f}",
                     "seeds": int(len(seeds)), "run": r.source, "source": PAIRED_CSV})
    table = pd.DataFrame(spread)
    summary: List[Dict[str, object]] = []
    for group, members in (("planners", ", ".join(PLANNERS)),
                           ("feedback rules", "every reactive and velocity variant")):
        part = table[table["group"] == group].reset_index(drop=True)
        top_gpu, top_slo = part.loc[part["gpu"].idxmax()], part.loc[part["slo"].idxmax()]
        summary.append({"group": group, "members": members, "rows": int(len(part)),
                        "max_gpu_h_wk_sd": f"{float(top_gpu['gpu']):.1f}", "row_with_max_gpu_sd": top_gpu["row"],
                        "max_slo_in_sd": f"{float(top_slo['slo']):.3f}", "row_with_max_slo_sd": top_slo["row"],
                        "source": f"{PAIRED_CSV} (runs as in section 27a)"})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def term_time(minute: int) -> str:
    """A minute of the run as teaching week, weekday and clock time (minute 0: Monday 00:00, week 1)."""
    week, rem = divmod(int(minute), WEEK_MIN)
    day, clock = divmod(rem, 1440)
    return f"week {week + 1}, {DAYS[day]} {clock // 60:02d}:{clock % 60:02d}"


def replay_resolves(ctx: PolicyContext, lam_in: np.ndarray) -> List[Tuple[int, float]]:
    """Minute and ratio of every size re-solve TAP's advice makes on a stream.

    The rule of ``CalendarAdaptPolicy.observe`` on the advice's own ``ratio`` (trailing window, prior
    mass of the first teaching day, clip): a check once ``REPLAN_EVERY_MIN`` minutes have passed since
    the last re-solve, a re-solve when the ratio moved by more than ``REPLAN_TOL`` from the one the
    plan in force was solved for, none once no slot can still change (``_replan``).  The ratio reads
    only the revealed in-class demand and the forecast, so no plan is solved here.
    """
    from pilot.sim.policies import calendar_adapt as ca
    from pilot.sim.policies.base import SimState
    from pilot.sim.policies.tap import LearnedPriorAdaptPolicy

    advice = LearnedPriorAdaptPolicy(ctx)
    size = int(advice.forecast()[0].size)  # the length of the plan the advice solves
    state = SimState(lam_in, np.zeros_like(lam_in), np.zeros_like(lam_in))
    planned, last = advice.planned_ratio, advice._last_replan  # the policy's own starting state
    events: List[Tuple[int, float]] = []
    for t in range(int(ctx.horizon_min)):
        state.t = t
        if t - last < ca.REPLAN_EVERY_MIN:
            continue
        ratio = advice.ratio(state)
        if abs(ratio / planned - 1.0) > ca.REPLAN_TOL and min(size, t + ctx.lead_min) < size:
            planned, last = ratio, t
            events.append((t, ratio))
    return events


def _r1_diagnostics(repo: Path, short: str, tier: str) -> Optional[pd.DataFrame]:
    """Cell-C rows of an R1 run's ``diagnostics.csv``; None while the run has no DONE marker."""
    name = f"r1_{short}_{tier}"
    path = repo / RUNS / "tap" / name / "diagnostics.csv"
    if not ((repo / RUNS / "logs" / f"{name}.DONE").exists() and path.exists()):
        logger.info("re-solve replay: %s is not finished, skipped", name)
        return None
    diag = pd.read_csv(path, float_precision="round_trip")
    return diag[diag["cell"] == "C"]


def _with_values(args: Sequence[str], values: Dict[str, str]) -> List[str]:
    """A copy of command-line ``args`` with the value after each flag of ``values`` replaced."""
    out = list(args)
    for flag, value in values.items():
        out[out.index(flag) + 1] = value
    return out


def _check_replay(name: str, seed: int, diag: pd.DataFrame, events: List[Tuple[int, float]]) -> None:
    """Raise unless the replay gives the run's own number of re-solves and final ratio."""
    row = diag[diag["seed"] == seed]
    if len(row) != 1:
        raise ValueError(f"{name}: {len(row)} diagnostics rows for cell C, seed {seed}")
    final = events[-1][1] if events else 1.0  # the ratio in force at the end (1.0: never re-solved)
    n_run, ratio_run = int(row["n_replans"].iloc[0]), float(row["final_ratio"].iloc[0])
    if len(events) != n_run or abs(final - ratio_run) > 1e-12:
        raise ValueError(f"{name} seed {seed}: replay gives {len(events)} re-solves ending at {final!r}, "
                         f"diagnostics.csv {n_run} at {ratio_run!r}")


def _add_paths(repo: Path) -> None:
    """Put the simulator package and the sweep / audit scripts on ``sys.path``."""
    for sub in ("src", "scripts/sim", "scripts/audit"):
        if str(repo / sub) not in sys.path:
            sys.path.insert(0, str(repo / sub))


def cell_c_config(repo: Path, short: str, tier: str, lead: float, seed: int) -> RunConfig:
    """The configuration of an R1 run of TAP: the central learned-prior run's arguments
    (``prefix_invariance.sweep_args``, ``CELL_C``) at the run's own tier, lead and seed."""
    _add_paths(repo)
    import prefix_invariance as pi  # the command lines of the central learned-prior runs
    import run_sweep

    base = pi.sweep_args(short)
    calibration = str(repo / base[base.index("--calibration") + 1])  # relative to the repository root
    args = _with_values(base, {"--calibration": calibration, "--lead-tier": LAPTOP_TIERS[tier],
                               "--lead-min": f"{lead:g}"})
    cfg = run_sweep.base_config(run_sweep.build_parser().parse_args(args))
    cfg = replace(cfg, workload=replace(cfg.workload, **pi.CELL_C), seed=seed)
    if abs(float(cfg.service.lead_min) - lead) > 1e-12:
        raise ValueError(f"r1_{short}_{tier}: lead {cfg.service.lead_min} rebuilt, {lead} run")
    return cfg


def _resolve_row(case: str, tier: str, lead: float, name: str,
                 per_seed: Dict[int, List[Tuple[int, float]]]) -> Dict[str, object]:
    """One case-tier of section 29c: the first re-solve of every seed, as minute and term time."""
    first = {seed: events[0] for seed, events in per_seed.items() if events}
    counts = [len(events) for events in per_seed.values()]
    row: Dict[str, object] = {**_where(case, tier), "lead_min": f"{lead:g}", "seeds": len(per_seed),
                              "n_replans_min / max": f"{min(counts)} / {max(counts)}",
                              "replay_equals_diagnostics": f"{len(per_seed)} of {len(per_seed)} seeds "
                                                           "(n_replans, final_ratio)"}
    if first:
        early = min(first, key=lambda s: first[s][0])
        late = max(first, key=lambda s: first[s][0])
        ratios = [ratio for _minute, ratio in first.values()]
        row.update({"first_resolve_minute_min / max": f"{first[early][0]} / {first[late][0]}",
                    "earliest_seed": f"{term_time(first[early][0])} (seed {early})",
                    "latest_seed": f"{term_time(first[late][0])} (seed {late})",
                    "minute_per_seed": "; ".join(f"{s}: {first[s][0]}" for s in sorted(first)),
                    "ratio_at_resolve_min / max": f"{min(ratios):.3f} / {max(ratios):.3f}"})
    row["source"] = (f"{RUNS}/tap/{name}/diagnostics.csv; replay of src/pilot/sim/policies/"
                     "calendar_adapt.py on the stream of scripts/audit/prefix_invariance.py "
                     "(sweep_args, CELL_C)")
    return row


def resolve_time_table(repo: Path) -> Tuple[pd.DataFrame, StreamCache]:
    """When TAP's online size correction re-solves, laptop cell C, per case-tier over seeds 10-14.

    Each R1 run's stream is rebuilt from its configuration (``cell_c_config``) and the correction's
    rule is replayed on it with TAP's forecast (``replay_resolves``).  The replay must reproduce
    ``n_replans`` and ``final_ratio`` of the run's ``diagnostics.csv`` for every seed, or this
    raises.  Also returns each seed's (lam_in, forecast) under the key ``mass_ratio_table`` uses, so
    that table need not rebuild seed 10.  About 25 s per seed (one stream serves every case-tier).
    """
    _add_paths(repo)
    from pilot.sim.engine import build_stream, make_context

    cache: StreamCache = {}
    runs = {(short, case, tier): diag for short, case, tier in R1_RUNS
            if (diag := _r1_diagnostics(repo, short, tier)) is not None}
    found: Dict[Tuple[str, str, str], Dict[int, List[Tuple[int, float]]]] = {key: {} for key in runs}
    for seed in RESOLVE_SEEDS:
        built: Optional[Tuple[Tuple[object, ...], ArrivalStream]] = None  # the stream of this seed
        for (short, case, tier), diag in runs.items():
            cfg = cell_c_config(repo, short, tier, float(diag["lead_min"].iloc[0]), seed)
            key = (cfg.workload, cfg.seed, str(cfg.root))  # the key of mass_ratio_table's cache
            if built is None or built[0] != key:
                stream = build_stream(cfg)
                if stream.lam_hat_learn_prior_in is None:
                    raise ValueError("the stream carries no learned-prior forecast")
                built = (key, stream)
                cache[key] = (stream.lam_in, stream.lam_hat_learn_prior_in)
            events = replay_resolves(make_context(cfg, built[1]), built[1].lam_in)
            _check_replay(f"r1_{short}_{tier}", seed, diag, events)
            found[(short, case, tier)][seed] = events
    rows = [_resolve_row(case, tier, float(runs[(short, case, tier)]["lead_min"].iloc[0]),
                         f"r1_{short}_{tier}", per_seed)
            for (short, case, tier), per_seed in found.items()]
    return pd.DataFrame(rows), cache


def _curve_top(path: Path) -> Dict[str, Any]:
    """The record (parsed JSON) of the highest concurrency level of a ``concurrency_summary.jsonl``."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"{path}: no concurrency record")
    return max(records, key=lambda rec: int(rec["level"]))


def supplementary_facts_table(repo: Path) -> pd.DataFrame:
    """Numbers the supplementary text quotes outside the tables, each read from its source file."""
    from pilot.sim.config import SweepConfig

    rows: List[Dict[str, object]] = []
    meta = json.loads((repo / MULTI_PROCESS / "calib_multi_meta.json").read_text(encoding="utf-8"))
    for rel, what in ((f"{MULTI_PROCESS}/concurrency/concurrency_summary.jsonl",
                       f"{int(meta['replicas'])} co-resident vLLM processes of one model behind the router: "
                       "aggregate output tokens per second"),
                      (ALONE, "one vLLM process of the same model driven alone (MPS test, r1_alone): "
                              "output tokens per second")):
        rec = _curve_top(repo / rel)
        rows.append({"group": "B5 co-residency attempt (7B, one card)",
                     "name": f"{what} at {int(rec['level'])} concurrent streams",
                     "value": f"{float(rec['throughput_tokens_s']):.0f}", "unit": "tokens/s",
                     "setting": f"{Path(str(rec['model'])).name}, prompt median {float(rec['prompt_median_tokens']):.0f} "
                                f"tokens, at most {int(rec['max_tokens'])} output tokens, "
                                f"{float(rec['duration_s']):.0f} s, {int(rec['requests_ok'])} requests",
                     "source": rel})
    grid = SweepConfig().trigger_offset
    if len(grid) != len(TIMING_CELLS):
        raise ValueError(f"SweepConfig.trigger_offset has {len(grid)} cells, expected {len(TIMING_CELLS)}")
    total = {k: math.sqrt(v[1] ** 2 + v[2] ** 2 + v[3] ** 2) for k, v in zip(TIMING_CELLS, grid)}
    rows.append({"group": "timing cells",
                 "name": "total timing spread of cell C, sqrt(session sd^2 + section sd^2 + common sd^2)",
                 "value": f"{total['C']:.2f}", "unit": "min",
                 "setting": "cells G / H / I / J: " + " / ".join(f"{total[k]:.2f}" for k in "GHIJ"),
                 "source": "src/pilot/sim/config.py SweepConfig.trigger_offset"})
    return pd.DataFrame(rows)


__all__ = ["cell_c_config", "components_table", "laptop_seed_spread_tables", "replay_resolves",
           "resolve_time_table", "supplementary_facts_table", "term_time"]
