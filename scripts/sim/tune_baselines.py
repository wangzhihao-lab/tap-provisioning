"""Fair tuning of the two feedback baselines (policies 2 and 3) on the quick slice.

    uv run python scripts/sim/tune_baselines.py                      # both grids, 2 weeks, 1 college
    uv run python scripts/sim/tune_baselines.py --policies velocity  # one grid only
    uv run python scripts/sim/tune_baselines.py --price-ratio 18 --out outputs/sim/tuning_r18

The token-velocity policy (TokenScale type) and the reactive HPA are the two baselines whose
behaviour is decided entirely by thresholds and windows, and their defaults are the textbook
numbers rather than the best a practitioner would run on *this* workload.  Reporting a 25x cost
ratio for a baseline nobody tuned is a straw man, so each one gets a grid search here and the
formal sweeps then run it at its own best setting.

Protocol (the point of the script, and the thing to keep honest):

* the grid is scored on the **tuning seeds** (0, 1) only, by mean cost ratio to the offline optimum;
* the chosen configuration is then reported on a **held-out seed** (2) that took no part in the
  selection, next to the default configuration on the same seed, so the table shows both the gain
  and whether it generalised;
* every grid contains the current default, which is what makes the before/after column meaningful;
* the two grids are disjoint in the fields they move (``velocity_*`` against ``hpa_*``), so the two
  winners can be applied together.

Outputs (``--out``, default ``outputs/sim/tuning``): ``<policy>_grid.csv`` with one row per
configuration (per-seed ratios, tuning mean, the tuning seeds' GPU-hours, starts, run and switch
cost and attainment, held-out metrics) and ``best.json``, the chosen ``PolicyConfig`` values in the
schema of ``tuned_config``.  ``run_sweep.py --tuned best.json`` applies them; the ``PolicyConfig``
defaults are left untouched.

With ``--slo-floor`` the row is picked by one of two rules (``tuning_select.py``): ``--slo-select
cost`` (the default, unchanged) takes the lowest mean cost ratio among the cells reaching the
floor, ``--slo-select resource`` the lowest mean resource cost ``run_cost + switch_cost``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:  # allow running without installing the package
    sys.path.insert(0, str(PILOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:  # sibling module, not a package
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot.sim import (  # noqa: E402
    ArrivalStream,
    PolicyConfig,
    RunConfig,
    CostConfig,
    ServiceConfig,
    WorkloadConfig,
)
from pilot.sim.config import LEAD_TIERS, with_lead_tier  # noqa: E402
from pilot.sim.engine import build_stream, fluid_mu, make_context, run_policy  # noqa: E402
from pilot.sim.service import load_service_config  # noqa: E402
from pilot.sim.sweeps import with_price_ratio  # noqa: E402
from tuned_config import apply_policy_values  # noqa: E402
from tuning_report import TOP_ROWS, before_after, table  # noqa: E402,F401  (re-exported)
from tuning_select import (  # noqa: E402,F401  (chosen / payload re-exported)
    SLO_SELECT,
    chosen,
    payload,
    resource_columns,
)

logger = logging.getLogger("tune_baselines")

QUICK_COLLEGE = "college_18"
TUNING_SEEDS: Tuple[int, ...] = (0, 1)
HELDOUT_SEED = 2
SCORE = "cost_ratio"  # the objective; lower is better
KEEP = ("cost_ratio", "total_cost", "gpu_hours", "starts", "slo_in", "slo_first_in", "viol_in",
        "run_cost", "switch_cost")


# --------------------------------------------------------------------------- grids
# Both grids contain the current default, so a row of the CSV is the "before" number.  Both were
# widened once after a first pass put the winner on a boundary (reactive at the lowest utilisation
# thresholds and the longest scale-down hold, velocity at the widest hysteresis band), so that the
# selected configuration is interior and "the best this baseline can do here" is not an artefact
# of where the grid stopped.
VELOCITY_WINDOWS: Tuple[int, ...] = (2, 5, 10, 20, 30)
VELOCITY_CLIPS: Tuple[float, ...] = (4.0, 16.0)
# (persist, dwell) of the hysteresis band; the default is (2, 3).  ``None`` is the band switched
# off, i.e. the pre-hysteresis slope-every-minute rule, kept in the grid as the other extreme.
VELOCITY_BANDS: Tuple[Optional[Tuple[int, int]], ...] = (None, (1, 1), (1, 15), (2, 3), (3, 10),
                                                         (5, 20))
HPA_UP_UTIL: Tuple[float, ...] = (0.2, 0.4, 0.7)
HPA_DOWN_UTIL: Tuple[float, ...] = (0.1, 0.2, 0.3)
HPA_COOLDOWN: Tuple[int, ...] = (0, 3)
# The scale-down hold is capped *below the bell spacing* (s = 100 min at this institution, the
# same s the H1 premise of the paper's analysis uses).  The cost is monotone decreasing in this knob
# on a bell-driven workload, so without a cap the search walks off to "never scale down", which is
# the static-peak baseline the paper already reports as policy 1 rather than a threshold rule: a
# hold longer than the spacing cannot release capacity between two bells at all.  Lifting the cap
# to 120 min does win on the two tuning seeds (mean 2.40 against 2.68) and then loses on the
# held-out one (3.80 against 2.17), which is what two tuning seeds and fifty candidates buy.
HPA_HOLD_CAP_MIN = 100
HPA_DOWN_HOLD: Tuple[int, ...] = (5, 15, 45)

KNOBS: Dict[str, Tuple[str, ...]] = {
    "velocity": ("velocity_window_min", "velocity_clip", "velocity_hysteresis",
                 "velocity_persist_min", "velocity_dwell_min"),
    "reactive": ("hpa_up_util", "hpa_down_util", "hpa_cooldown_min", "hpa_down_hold_min"),
}


def velocity_grid() -> List[Dict[str, object]]:
    """Window x extrapolation clip x hysteresis band: 5 x 2 x 6 = 60 configurations."""
    out: List[Dict[str, object]] = []
    for window, clip, band in itertools.product(VELOCITY_WINDOWS, VELOCITY_CLIPS, VELOCITY_BANDS):
        persist, dwell = band if band is not None else (2, 3)
        out.append({"velocity_window_min": window, "velocity_clip": clip,
                    "velocity_hysteresis": band is not None,
                    "velocity_persist_min": persist, "velocity_dwell_min": dwell})
    return out


def reactive_grid() -> List[Dict[str, object]]:
    """Up / down utilisation thresholds x cooldown x scale-down hold: 3 x 3 x 2 x 3 = 54."""
    if max(HPA_DOWN_HOLD) >= HPA_HOLD_CAP_MIN:
        raise ValueError(f"scale-down hold must stay below the bell spacing {HPA_HOLD_CAP_MIN} min")
    return [{"hpa_up_util": up, "hpa_down_util": down, "hpa_cooldown_min": cool,
             "hpa_down_hold_min": hold}
            for up, down, cool, hold in itertools.product(HPA_UP_UTIL, HPA_DOWN_UTIL,
                                                          HPA_COOLDOWN, HPA_DOWN_HOLD)]


GRIDS: Dict[str, Callable[[], List[Dict[str, object]]]] = {"velocity": velocity_grid,
                                                          "reactive": reactive_grid}


def defaults_for(policy: str) -> Dict[str, object]:
    """The knob values policy ``policy`` runs with today (the "before" configuration)."""
    base = PolicyConfig()
    return {k: getattr(base, k) for k in KNOBS[policy]}


# --------------------------------------------------------------------------- instances
@dataclass(frozen=True)
class TuningSpec:
    """The slice, the seeds and the prices every configuration is scored on."""

    college: Optional[str] = QUICK_COLLEGE
    weeks: int = 2
    lead_tier: str = "d_inst"
    adoption: float = WorkloadConfig().adoption
    sigma: float = WorkloadConfig().sigma
    tuning_seeds: Tuple[int, ...] = TUNING_SEEDS
    heldout_seed: int = HELDOUT_SEED
    price_ratio: Optional[float] = None
    max_sections: int = 0  # 0 = the whole slice; a cap keeps a test instance small
    root: str = str(REPO)
    calibration: Optional[str] = None  # T3 CSV: the measured service model (batch B)
    model: Optional[str] = None
    lead_stat: str = "median"
    c_run: Optional[float] = None
    beta_start: Optional[float] = None
    min_replicas: int = 0
    slo_floor: Optional[float] = None  # pick the cheapest cell with slo_in >= this (tuning seeds)
    slo_select: str = "cost"  # "cost" (mean cost ratio) or "resource" (run + switch cost), tuning_select
    slo_in_s: Optional[float] = None  # in-class TTFT SLO (knee + attainment threshold); None = 2 s
    lead_min: Optional[float] = None  # override the tier's lead time (minutes), e.g. a measured cold start
    # trigger timing of the slice: (mean, per-session sd, per-section sd, per-bell common sd) in
    # minutes; all zero = bursts at the bell, the synchronous slice the round tuned on first
    trigger_offset: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def seeds(self) -> Tuple[int, ...]:
        """Tuning seeds first, then the held-out one."""
        return tuple(self.tuning_seeds) + (self.heldout_seed,)

    def to_dict(self) -> Dict[str, object]:
        """Provenance block for ``best.json``."""
        return {"college": self.college or "ALL", "weeks": self.weeks,
                "lead_tier": self.lead_tier, "adoption": self.adoption, "sigma": self.sigma,
                "max_sections": self.max_sections, "price_ratio": self.price_ratio,
                "slo_floor": self.slo_floor, "trigger_offset": list(self.trigger_offset),
                **({"slo_select": self.slo_select} if self.slo_select != "cost" else {})}


@dataclass(frozen=True)
class Instance:
    """One seed: the arrival stream, the base config and the offline optimum that scores it."""

    seed: int
    cfg: RunConfig
    stream: ArrivalStream
    denom: float


def base_config(spec: TuningSpec, seed: int) -> RunConfig:
    """The run config of one seed, before any policy knob is moved."""
    mean, sd, course_sd, common_sd = spec.trigger_offset
    workload = WorkloadConfig(college=spec.college, weeks=tuple(range(1, spec.weeks + 1)),
                              adoption=spec.adoption, sigma=spec.sigma,
                              max_sections=spec.max_sections,
                              trigger_offset_mean_min=mean, trigger_offset_sd_min=sd,
                              trigger_offset_course_sd_min=course_sd,
                              trigger_offset_common_sd_min=common_sd)
    service = with_lead_tier(ServiceConfig(), spec.lead_tier)
    if spec.calibration:
        loaded = load_service_config(Path(spec.calibration), spec.model, base=service,
                                     lead_stat=spec.lead_stat,
                                     **({"slo_in_s": spec.slo_in_s} if spec.slo_in_s else {}))
        service = with_lead_tier(loaded, spec.lead_tier)
        if spec.lead_tier == "d_rep":
            service = replace(service, lead_min=loaded.lead_min)
    if spec.lead_min is not None:
        service = replace(service, lead_min=float(spec.lead_min))
    cost = CostConfig()
    if spec.slo_in_s:
        cost = replace(cost, slo_in_s=float(spec.slo_in_s))
    if spec.c_run is not None:
        cost = replace(cost, c_run=spec.c_run)
    if spec.beta_start is not None:
        cost = replace(cost, beta_start=spec.beta_start)
    policy = PolicyConfig(min_replicas=spec.min_replicas)
    return RunConfig(workload=workload, service=service, cost=cost, policy=policy, seed=seed,
                     root=spec.root)


def build_instances(spec: TuningSpec) -> Dict[int, Instance]:
    """Stream and offline optimum of every seed, generated once and shared by the whole grid."""
    out: Dict[int, Instance] = {}
    for seed in spec.seeds:
        cfg = base_config(spec, seed)
        stream = build_stream(cfg)
        cfg = with_price_ratio(cfg, fluid_mu(cfg, stream), spec.price_ratio)
        ctx = make_context(cfg, stream)
        oracle = run_policy("offline", cfg, stream, ctx)
        out[seed] = Instance(seed, cfg, stream, oracle.metrics.total_cost)
        logger.info("seed %d: %d requests, offline optimum %.1f USD", seed, stream.n_requests,
                    oracle.metrics.total_cost)
    return out


def score(policy: str, knobs: Dict[str, object], inst: Instance) -> Dict[str, float]:
    """Run ``policy`` at ``knobs`` on one instance and keep the metrics the table reports."""
    cfg = replace(inst.cfg, policy=apply_policy_values(inst.cfg.policy, knobs))
    ctx = make_context(cfg, inst.stream)
    run = run_policy(policy, cfg, inst.stream, ctx, inst.denom)
    row = run.metrics.to_row()
    row["cost_ratio"] = run.metrics.total_cost / inst.denom if inst.denom > 0 else float("nan")
    return {k: float(row[k]) for k in KEEP}


def evaluate(policy: str, spec: TuningSpec, instances: Dict[int, Instance]) -> pd.DataFrame:
    """One row per configuration: per-seed ratios, the tuning mean and the held-out metrics."""
    grid = GRIDS[policy]()
    if len(grid) > 60:
        raise ValueError(f"{policy}: grid of {len(grid)} configurations exceeds the cap of 60")
    default = defaults_for(policy)
    rows: List[Dict[str, object]] = []
    for idx, knobs in enumerate(grid):
        row: Dict[str, object] = {"policy": policy, "config_id": idx,
                                  "is_default": knobs == default, **knobs}
        tune = []
        for seed in spec.tuning_seeds:
            metrics = score(policy, knobs, instances[seed])
            row[f"cost_ratio_seed{seed}"] = metrics["cost_ratio"]
            row[f"slo_first_in_seed{seed}"] = metrics["slo_first_in"]
            tune.append(metrics)
        row["cost_ratio_tune_mean"] = sum(m["cost_ratio"] for m in tune) / len(tune)
        row["cost_ratio_tune_max"] = max(m["cost_ratio"] for m in tune)
        row["slo_in_tune_mean"] = sum(m["slo_in"] for m in tune) / len(tune)
        row.update(resource_columns(list(zip(spec.tuning_seeds, tune))))
        held = score(policy, knobs, instances[spec.heldout_seed])
        row.update({f"{k}_heldout": v for k, v in held.items()})
        rows.append(row)
    frame = ranked(pd.DataFrame(rows))
    logger.info("%s: %d configurations scored", policy, len(frame))
    return frame


def ranked(frame: pd.DataFrame) -> pd.DataFrame:
    """Order a grid by the tuning objective and number the rows.

    Exact ties are common - a knob can fail to bind at all on a slice - so the order within a tie
    is pinned rather than left to the sort: the default cell wins whenever it is among the tied
    ones (a tie is no evidence for moving off it), otherwise the lowest-index cell of the grid,
    which is reproducible and, for the clip and the window, the value closest to the default.
    """
    out = (frame.sort_values(["cost_ratio_tune_mean", "is_default", "config_id"],
                             ascending=[True, False, True])
           .reset_index(drop=True))
    out.insert(2, "rank", out.index + 1)
    return out


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    ap = argparse.ArgumentParser(description="Tune the feedback baselines on the quick slice")
    ap.add_argument("--policies", nargs="*", default=sorted(GRIDS),
                    help=f"grids to run; default {sorted(GRIDS)}")
    ap.add_argument("--out", type=Path, default=PILOT / "outputs" / "sim" / "tuning")
    ap.add_argument("--root", type=Path, default=REPO, help="root the input paths resolve against")
    ap.add_argument("--college", default=QUICK_COLLEGE, help="'ALL' for the institution slice")
    ap.add_argument("--weeks", type=int, default=2)
    ap.add_argument("--adoption", type=float, default=WorkloadConfig().adoption)
    ap.add_argument("--sigma", type=float, default=WorkloadConfig().sigma)
    ap.add_argument("--lead-tier", default="d_inst", choices=sorted(LEAD_TIERS))
    ap.add_argument("--max-sections", type=int, default=0, help="0 = the whole slice")
    ap.add_argument("--tuning-seeds", type=int, nargs="*", default=list(TUNING_SEEDS))
    ap.add_argument("--heldout-seed", type=int, default=HELDOUT_SEED)
    ap.add_argument("--price-ratio", type=float, default=None,
                    help="re-price every cell so that p_in*mu/c_run equals this; default: the "
                         "base CostConfig prices")
    ap.add_argument("--calibration", default=None, help="T3 calibration CSV (batch B)")
    ap.add_argument("--model", default=None, help="model row of the calibration CSV")
    ap.add_argument("--lead-stat", choices=("median", "max"), default="median")
    ap.add_argument("--c-run", type=float, default=None)
    ap.add_argument("--beta-start", type=float, default=None)
    ap.add_argument("--min-replicas", type=int, default=0)
    ap.add_argument("--slo-in-s", type=float, default=None, help="in-class TTFT SLO seconds (default 2)")
    ap.add_argument("--lead-min", type=float, default=None, help="override the tier's lead time (minutes)")
    ap.add_argument("--slo-floor", type=float, default=None,
                    help="choose the cheapest cell with slo_in >= this on the tuning seeds")
    ap.add_argument("--slo-select", choices=SLO_SELECT, default="cost",
                    help="with --slo-floor, 'cheapest' by mean cost ratio (default) or by mean "
                         "run + switch cost ('resource')")
    ap.add_argument("--trigger-offset", type=float, nargs=4, default=(0.0, 0.0, 0.0, 0.0),
                    metavar=("MEAN", "SD", "COURSE_SD", "COMMON_SD"),
                    help="trigger timing of the slice in minutes (default: bursts at the bell); the "
                         "measured independent dispersion is -8.7 5.3 4.0 0")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def spec_from(args: argparse.Namespace) -> TuningSpec:
    """Assemble the tuning spec from the parsed arguments."""
    college = None if str(args.college).upper() == "ALL" else args.college
    return TuningSpec(college=college, weeks=max(1, args.weeks), lead_tier=args.lead_tier,
                      adoption=args.adoption, sigma=args.sigma,
                      tuning_seeds=tuple(args.tuning_seeds), heldout_seed=args.heldout_seed,
                      price_ratio=args.price_ratio, max_sections=args.max_sections,
                      root=str(args.root), calibration=args.calibration, model=args.model,
                      lead_stat=args.lead_stat, c_run=args.c_run, beta_start=args.beta_start,
                      min_replicas=args.min_replicas, slo_floor=args.slo_floor,
                      slo_select=args.slo_select,
                      slo_in_s=args.slo_in_s, lead_min=args.lead_min,
                      trigger_offset=tuple(float(v) for v in args.trigger_offset))


def run(spec: TuningSpec, policies: Iterable[str], out: Path) -> Dict[str, object]:
    """Score every grid, write the CSVs and ``best.json``, and return the payload."""
    out.mkdir(parents=True, exist_ok=True)
    instances = build_instances(spec)
    sections: Dict[str, Dict[str, object]] = {}
    for policy in policies:
        if policy not in GRIDS:
            raise KeyError(f"no grid for '{policy}'; known: {sorted(GRIDS)}")
        t0 = time.perf_counter()
        frame = evaluate(policy, spec, instances)
        frame.to_csv(out / f"{policy}_grid.csv", index=False)
        sections[policy] = chosen(frame, policy, KNOBS[policy], spec.slo_floor, spec.slo_select)
        print(table(frame, policy, spec, KNOBS[policy]))
        logger.info("%s: %.1f s -> %s", policy, time.perf_counter() - t0,
                    out / f"{policy}_grid.csv")
    body = payload(spec, sections, SCORE)
    (out / "best.json").write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    print(before_after(sections))
    logger.info("wrote %s", out / "best.json")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("pilot.sim").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    run(spec_from(args), list(args.policies), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
