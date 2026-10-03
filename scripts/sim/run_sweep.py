"""Run one parameter sweep of the provisioning simulator and write its CSV.

    uv run python scripts/sim/run_sweep.py --sweep sigma --quick
    uv run python scripts/sim/run_sweep.py --sweep all --weeks 4 --college college_18
    uv run python scripts/sim/run_sweep.py --sweep epsilon --quick --lead-tier d_rep \
        --price-ratios 6 18 --out outputs/sim_f6_quick

``--quick`` is the reduced grid used to prove the pipeline: one college slice, two teaching weeks,
two seeds and two points per grid.  Each sweep lands in ``<out>/<sweep>/<sweep>.csv``.

The prices can be moved two ways.  ``--p-in`` / ``--p-off`` set them absolutely (``--p-in``
alone keeps ``p_off = p_in / 6``), and ``--price-ratios`` gives the epsilon sweep its second,
*dimensionless* axis: for each ratio the sweep solves ``p_in = ratio * c_run / mu`` per cell, so the
facets of F6 differ only in how expensive an unserved in-class request is relative to capacity.
``--price-ratios none`` switches that axis off and keeps whatever ``--p-in`` left behind.

The baselines can be run at their tuned settings.  ``--tuned best.json`` applies what
``scripts/sim/tune_baselines.py`` selected (the ``policies.*.config`` sections of that file) and
``--policy-config`` applies a flat JSON object of ``PolicyConfig`` fields, inline or as a path, on
top of it.  Without either flag the ``PolicyConfig`` defaults are used, so a sweep run from before
these options existed reproduces unchanged.

Every CSV carries whole-run rows (``week = 0``) and one row per teaching week (``week = k``), so
week 1 - the blind spot of the history-driven policies - can be read apart from the steady state.

Policies that report per-run diagnostics (``tap``: the share of slots in BDINTERP-d's advice phase,
the number of size re-solves and the ratio in force at the end) add one row per run to
``<out>/diagnostics.csv``, labelled with ``--case`` (default: the ``--model`` row key), the tier
and the timing cell.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:  # allow `python scripts/sim/run_sweep.py` without install
    sys.path.insert(0, str(PILOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:  # sibling module, not a package
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot.sim import (  # noqa: E402
    CostConfig,
    PolicyConfig,
    RunConfig,
    ServiceConfig,
    SweepConfig,
    WorkloadConfig,
)
from pilot.sim.config import LEAD_TIERS, with_lead_tier  # noqa: E402
from pilot.sim.engine import DEFAULT_POLICIES  # noqa: E402
from pilot.sim.service import load_service_config  # noqa: E402
from pilot.sim.sweeps import SWEEPS, run_sweep  # noqa: E402
from tuned_config import apply_policy_values, policy_overrides  # noqa: E402

logger = logging.getLogger("run_sweep")

QUICK_COLLEGE = "college_18"
F6_TRUSTS: Tuple[float, ...] = (0.1, 0.5, 1.0)  # policy 7's epsilon values F6 draws
P_OFF_SHARE = 6.0  # ``--p-in`` without ``--p-off`` keeps the default p_in / p_off ratio


def price_ratios(values: Optional[Sequence[str]]) -> Tuple[float, ...]:
    """``--price-ratios``: the epsilon sweep's price axis, ``none`` switching the axis off.

    Absent, the grid of ``SweepConfig`` is used; ``none`` returns the empty tuple, which makes the
    error grid run once at the base ``CostConfig`` prices (the pre-two-axis behaviour).
    """
    if values is None:
        return SweepConfig().price_ratio
    if len(values) == 1 and str(values[0]).strip().lower() == "none":
        return ()
    return tuple(float(v) for v in values)


def cost_config(p_in: Optional[float], p_off: Optional[float], c_run: Optional[float] = None,
                beta_start: Optional[float] = None) -> CostConfig:
    """Base prices with ``--p-in`` / ``--p-off`` applied; ``p_off`` defaults to ``p_in / 6``."""
    cost = CostConfig()
    if c_run is not None:
        cost = replace(cost, c_run=c_run)
    if beta_start is not None:
        cost = replace(cost, beta_start=beta_start)
    if p_in is None and p_off is None:
        return cost
    new_in = cost.p_in if p_in is None else p_in
    new_off = p_off if p_off is not None else (new_in / P_OFF_SHARE if p_in is not None
                                               else cost.p_off)
    return replace(cost, p_in=new_in, p_off=new_off)


def policies_for(sweep: str, chosen: Optional[Sequence[str]]) -> List[str]:
    """``--policies`` if given, else the default list; the error grid feeds F6, so it needs
    policy 7 at the three trust values of that figure instead of at the cell's single one."""
    if chosen:
        return list(chosen)
    names = list(DEFAULT_POLICIES)
    if sweep != "epsilon":
        return names
    at = names.index("bdinterp")
    return names[:at] + [f"bdinterp@{e}" for e in F6_TRUSTS] + names[at + 1:]


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    ap = argparse.ArgumentParser(description="Provisioning simulator sweeps (batch C)")
    ap.add_argument("--sweep", default="sigma", help=f"one of {sorted(SWEEPS)} or 'all'")
    ap.add_argument("--quick", action="store_true", help="reduced grid: 1 college, 2 weeks, 2 seeds")
    ap.add_argument("--out", type=Path, default=PILOT / "outputs" / "sim")
    ap.add_argument("--root", type=Path, default=REPO, help="root the input paths resolve against")
    ap.add_argument("--slots", default=WorkloadConfig().slots_csv)
    ap.add_argument("--college", default=None)
    ap.add_argument("--campus", default=None)
    ap.add_argument("--weeks", type=int, default=4, help="teaching weeks in a run (1-6 tested)")
    ap.add_argument("--adoption", type=float, default=WorkloadConfig().adoption)
    ap.add_argument("--sigma", type=float, default=WorkloadConfig().sigma)
    ap.add_argument("--rho", type=float, default=WorkloadConfig().rho_mean)
    ap.add_argument("--lead-tier", default="d_inst", choices=sorted(LEAD_TIERS),
                    help="provisioning tier: lead time, commitment block and billing block")
    ap.add_argument("--p-in", type=float, default=None,
                    help="override CostConfig.p_in (USD-equivalent per unserved in-class request)")
    ap.add_argument("--p-off", type=float, default=None,
                    help="override CostConfig.p_off; the default with --p-in is p_in / 6")
    ap.add_argument("--offset-cells", nargs="*", default=None, metavar="CELL",
                    help="offset sweep: run only these timing cells, letters A-F in the order of SweepConfig.trigger_offset")
    ap.add_argument("--forecast-scales", type=float, nargs="*", default=None,
                    help="epsilon sweep's forecast-scale grid (default SweepConfig.forecast_scale); "
                         "lets one grid be split across processes")
    ap.add_argument("--price-ratios", nargs="*", default=None,
                    help="epsilon sweep's second axis, p_in*mu/c_run; 'none' switches it off "
                         f"(default {list(SweepConfig().price_ratio)})")
    ap.add_argument("--calibration", type=Path, default=None, help="T3 calibration CSV")
    ap.add_argument("--min-replicas", type=int, default=0,
                    help="always-on base tier applied to every policy and the optimum")
    ap.add_argument("--lead-stat", choices=("median", "max"), default="median",
                    help="replica start-up from the T3 median (warm restart) or max (fresh node)")
    ap.add_argument("--c-run", type=float, default=None, help="USD per replica-minute override")
    ap.add_argument("--beta-start", type=float, default=None, help="USD per replica start override")
    ap.add_argument("--model", default=None, help="model row to read from the calibration CSV")
    ap.add_argument("--no-in-class-priority", action="store_true",
                    help="admission ablation: one first-come queue across in-class and off-class requests")
    ap.add_argument("--lead-min", type=float, default=None,
                    help="override the tier's lead time in minutes (e.g. a measured cold start)")
    ap.add_argument("--slo-in-s", type=float, default=None,
                    help="in-class TTFT SLO in seconds (default 2): sets both the calibration knee K and "
                         "the attainment threshold; RAG-length prompts need a looser target")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--policies", nargs="*", default=None,
                    help=f"default {list(DEFAULT_POLICIES)}; 'name@eps' pins policy 7's trust")
    ap.add_argument("--max-sections", type=int, default=0)
    ap.add_argument("--no-velocity-hysteresis", action="store_true",
                    help="policy 3 acts on every slope again (the pre-hysteresis baseline)")
    ap.add_argument("--tuned", type=Path, default=None,
                    help="best.json from scripts/sim/tune_baselines.py: run the baselines at the "
                         "settings it selected instead of the PolicyConfig defaults")
    ap.add_argument("--policy-config", default=None,
                    help="flat JSON object of PolicyConfig fields (inline or a path), applied "
                         "after --tuned")
    ap.add_argument("--forecast-warm-start", default=None, choices=["stale_calendar"],
                    help="policy 4's week-1 plan; the default is the reactive fallback")
    ap.add_argument("--learn-prior-from", default=None,
                    help="slots CSV of an earlier semester whose realised starts give policy 12a "
                         "(calendar_learn_prior) its cold-start timing prior")
    ap.add_argument("--case", default=None,
                    help="label of the diagnostics rows (default: the --model row key)")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


OFFSET_KEYS: Tuple[str, ...] = ("trigger_offset_mean", "trigger_offset_sd", "trigger_offset_course_sd",
                                "trigger_offset_common_sd", "forecast_offset", "forecast_offset_sd")
DIAG_FIRST: Tuple[str, ...] = ("case", "tier", "lead_min", "cell", "seed", "policy", "phase_share",
                               "n_replans", "final_ratio", "trust")


def cell_letter(row: Mapping[str, object]) -> str:
    """Timing-cell letter (A-L, the order of ``SweepConfig.trigger_offset``) of an offset-sweep row."""
    if any(k not in row for k in OFFSET_KEYS):
        return ""
    cell = tuple(float(row[k]) for k in OFFSET_KEYS)  # type: ignore[arg-type]
    for idx, spec in enumerate(SweepConfig().trigger_offset):
        if tuple(float(v) for v in spec) == cell:
            return chr(ord("A") + idx)
    return ""


def write_diagnostics(frames: Sequence[Tuple[str, Optional[pd.DataFrame]]], args: argparse.Namespace,
                      base: RunConfig) -> Optional[Path]:
    """``<out>/diagnostics.csv``: one row per run of a policy that reports diagnostics."""
    parts = [frame.assign(sweep=name) for name, frame in frames if frame is not None and len(frame)]
    if not parts:
        return None
    diag = pd.concat(parts, ignore_index=True)
    diag["case"] = args.case or args.model or base.service.model
    diag["tier"] = args.lead_tier
    diag["lead_min"] = base.service.lead_min
    diag["cell"] = [cell_letter(r) for r in diag.to_dict("records")]
    first = [c for c in DIAG_FIRST if c in diag.columns]
    diag = diag[first + [c for c in diag.columns if c not in first]]
    path = Path(args.out) / "diagnostics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(path, index=False)
    logger.info("wrote %s (%d diagnostics rows)", path, len(diag))
    return path


def offset_cells(letters: Sequence[str] | None) -> Dict[str, object]:
    """``trigger_offset`` restricted to the named cells (A-F), or nothing when no cells are named."""
    if not letters:
        return {}
    grid = SweepConfig().trigger_offset
    picked = []
    for letter in letters:
        idx = ord(letter.upper()) - ord("A")
        if idx < 0 or idx >= len(grid):
            raise SystemExit(f"unknown timing cell {letter!r}; known: {[chr(ord('A') + i) for i in range(len(grid))]}")
        picked.append(grid[idx])
    return {"trigger_offset": tuple(picked)}


def base_config(args: argparse.Namespace) -> RunConfig:
    """Assemble the base ``RunConfig`` the sweep varies around."""
    weeks = tuple(range(1, max(1, args.weeks) + 1))
    college = args.college if args.college is not None else (QUICK_COLLEGE if args.quick else None)
    workload = WorkloadConfig(slots_csv=args.slots, college=college, campus=args.campus,
                              weeks=(1, 2) if args.quick else weeks, adoption=args.adoption,
                              sigma=args.sigma, rho_mean=args.rho,
                              max_sections=args.max_sections,
                              learn_prior_slots_csv=args.learn_prior_from)
    service = with_lead_tier(ServiceConfig(), args.lead_tier)
    if args.calibration:
        loaded = load_service_config(args.calibration, args.model, base=service,
                                     lead_stat=args.lead_stat,
                                     **({"slo_in_s": args.slo_in_s} if args.slo_in_s else {}))
        service = with_lead_tier(loaded, args.lead_tier)
        if args.lead_tier == "d_rep":  # the replica tier's lead IS the measured start-up time
            service = replace(service, lead_min=loaded.lead_min)
    if args.lead_min is not None:
        service = replace(service, lead_min=float(args.lead_min))
    if args.no_in_class_priority:
        service = replace(service, in_class_priority=False)
    policy = PolicyConfig(velocity_hysteresis=not args.no_velocity_hysteresis,
                          forecast_warm_start=args.forecast_warm_start,
                          min_replicas=args.min_replicas)
    overrides = policy_overrides(args.tuned, args.policy_config)
    if overrides:
        policy = apply_policy_values(policy, overrides)
        logger.info("policy overrides applied: %s", overrides)
    cost = cost_config(args.p_in, args.p_off, args.c_run, args.beta_start)
    if args.slo_in_s:
        cost = replace(cost, slo_in_s=args.slo_in_s)
    return RunConfig(workload=workload, service=service, policy=policy, cost=cost,
                     root=str(args.root))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("pilot.sim").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    base = base_config(args)
    sweep_cfg = SweepConfig(price_ratio=price_ratios(args.price_ratios),
                            **({"forecast_scale": tuple(args.forecast_scales)} if args.forecast_scales else {}),
                            **offset_cells(args.offset_cells))
    names: List[str] = sorted(SWEEPS) if args.sweep == "all" else [args.sweep]
    seeds = args.seeds if args.seeds else ((0, 1) if args.quick else None)
    diagnostics: List[Tuple[str, Optional[pd.DataFrame]]] = []
    for name in names:
        t0 = time.perf_counter()
        result = run_sweep(name, base, sweep_cfg, seeds, policies_for(name, args.policies),
                           quick=args.quick)
        result.write(Path(args.out) / name / f"{name}.csv")
        diagnostics.append((name, result.diagnostics))
        wall = time.perf_counter() - t0
        whole = result.frame[result.frame["week"] == 0]
        summary = (whole.groupby("policy")["cost_ratio"].mean().sort_values()
                   .round(3).to_dict())
        logger.info("sweep %s: %d rows (%d whole-run) in %.1f s -> %s", name, len(result.frame),
                    len(whole), wall, Path(args.out) / name / f"{name}.csv")
        logger.info("mean cost ratio to the offline optimum: %s", summary)
    write_diagnostics(diagnostics, args, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
