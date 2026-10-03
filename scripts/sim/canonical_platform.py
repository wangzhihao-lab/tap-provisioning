"""Canonical numbers, part 7: the two simulation platforms on the base cell (section 34).

The cloud boxes (x86) and this laptop (arm64) draw different arrival streams from the same simulator
and arguments (results/runs/JOBS_README.md, Platform), so the paper's cell-C numbers come
from two sets of streams: the laptop's (sections 7 and 27a) and the cloud's (sections 1-3).  For c2
and c3 at both tiers (cell C, r = 30, seeds 10-14, six weeks) this pairs every row of section 27a
with the cloud row of the same policy run with the same configuration and summarises the gaps.

* Laptop rows: ``results/runs/paired_cell_c.csv`` (whole run), named as in section 27a.
* Cloud rows: ``cells_long.csv``, main timetable: the rows of sections 1-3 (baselines tuned on
  their own tier) and, at the instance tier, the reactive and velocity rows tuned at the replica
  tier, which section 27a runs there too; each recomputed from the run ``assemble``
  (``pilot.validation.tables``) takes it from and checked against ``cells_long.csv``.  The latter
  are the first round's (``r30_d_inst``, ``r30_d_inst_slo95``), older than the follow-up round's
  planner-kernel change, so only their feedback rows enter, after two checks: the first-round and
  re-run ``r30_d_rep`` agree on offline, reactive and velocity, and ``r30_d_inst``'s offline rows
  equal ``r30_d_inst_t8``'s (the same streams).
* Configuration: each row's run command (laptop originals: ``runs_groups.laptop_originals``,
  checked by ``runs_s3_jobs.evidence``; rerun jobs: the START line of their log; cloud:
  the boxes' phase files, ``runs_groups.cloud_originals``) parsed by run_sweep.py's parser.
  Two rows pair when case, tier and policy agree and so does all the policy reads: sweep, workload,
  service, cost, seeds and policy configuration, the tuned knobs compared only for the rule that
  reads them (``tune_baselines.KNOBS``), the learned-timing prior only for ``calendar_learn_prior``.
  Cloud rows of one configuration (an SLO-floor pick equal to the cost pick) must agree seed for
  seed and are listed once.
"""

from __future__ import annotations

import argparse
import logging
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from canonical_cells_laptop import sweep_frame
from canonical_extra import LAPTOP_TIERS, PAIRED_CSV, WEEKS, _paired
from canonical_sensitivity import RUNS, KEYS, SEEDS

if TYPE_CHECKING:
    from pilot.sim.config import RunConfig
    from pilot.validation.runs import RunFile

logger = logging.getLogger("canonical_platform")

CELLS_LONG = "results/sim/validation/tables/cells_long.csv"
REMOTE = "outputs/validation_remote"
CASES_HERE: Tuple[str, ...] = ("c2_32b_shared4k", "c3_72b_shared4k")
RATIO = 30
CLOUD_GROUPS: Tuple[Tuple[str, str], ...] = (("d_inst", "d_inst"), ("d_rep", "d_rep"),
                                             ("d_inst", "d_rep"))  # (tier, tuned_at)
FEEDBACK: Tuple[str, ...] = ("reactive", "velocity")
PLANNERS: Tuple[str, ...] = ("calendar", "calendar_learn_prior", "calendar_learn", "calendar_adapt",
                             "stale_calendar", "forecast_ilp")
GROUPS: Tuple[str, ...] = ("offline", "planners", "feedback rules", "other")
PRIOR_READERS: Tuple[str, ...] = ("calendar_learn_prior", "tap")  # read --learn-prior-from
OUTCOME: Tuple[str, ...] = ("gpu_hours", "starts", "run_cost", "switch_cost", "penalty_cost",
                            "total_cost", "x_mean", "x_peak", "n_in", "n_off", "slo_in", "slo_off",
                            "timeouts")
Parsed = Tuple[argparse.Namespace, "RunConfig"]  # run_sweep.py's parse of one command


@dataclass(frozen=True, eq=False)
class CloudRow:
    """One cloud row of cell C: its cells_long.csv label, the run it comes from, its seeds."""

    case: str
    tier: str  # d_inst | d_rep
    tuned_at: str
    label: str  # the policy column of cells_long.csv
    policy: str  # the policy the run ran (the label without an SLO-floor suffix)
    csv: str  # the run's offset.csv, relative to the repository root
    rows: pd.DataFrame  # whole-run rows, one per seed
    gpu: float  # GPU-hours per week, mean over the seeds
    slo: float  # in-class attainment, mean over the seeds


@dataclass(frozen=True, eq=False)
class PlatformTables:
    """Section 34: every pair, the largest gaps per group, the rows without a counterpart."""

    pairs: pd.DataFrame
    summary: pd.DataFrame
    unpaired: pd.DataFrame
    cloud_checked: int  # cloud rows recomputed from their run and found equal to cells_long.csv
    round1_checked: int  # rows behind the two checks of the first-round runs


def group_of(policy: str) -> str:
    """The group a policy is summarised in."""
    if policy == "offline":
        return "offline"
    return "planners" if policy in PLANNERS else "feedback rules" if policy in FEEDBACK else "other"


def _whole_c(repo: Path, csv: str) -> pd.DataFrame:
    """Whole-run cell-C rows of seeds 10-14 of a run (``csv`` relative to the repository)."""
    frame = sweep_frame(repo, csv)
    whole = frame[(frame["cell"] == "C") & (frame["week"] == 0) & frame["seed"].isin(SEEDS)]
    if (whole["weeks"] != WEEKS).any():
        raise ValueError(f"{csv}: a run of other than {WEEKS} weeks")
    return whole


def _run_sweep_argv(tokens: Sequence[str]) -> Optional[List[str]]:
    """The run_sweep.py arguments of one command: its own, or those a floor-eval passes after --."""
    script = next((i for i, t in enumerate(tokens) if t.endswith("run_sweep.py")), None)
    if script is not None:
        return list(tokens[script + 1:])
    if "floor-eval" in tokens and "--" in tokens:
        return list(tokens[list(tokens).index("--") + 1:])
    return None


def laptop_argv(repo: Path, needed: Set[str]) -> Dict[str, Tuple[str, ...]]:
    """The argument list of every laptop run in ``needed`` (offset.csv paths relative to the repository root)."""
    from runs_groups import flag_value, laptop_originals  # scripts/sim
    from runs_s3_jobs import evidence  # scripts/sim

    out: Dict[str, Tuple[str, ...]] = {}
    for orig in laptop_originals():
        if orig.csv in needed:
            tuned = flag_value(orig.argv, "--tuned")
            if tuned is None:
                raise ValueError(f"{orig.out}: rebuilt without --tuned")
            evidence(orig, tuned)  # raises unless its CSV and log agree with the rebuilt arguments
            out[orig.csv] = orig.argv
    logs = repo / RUNS / "logs"
    for done in sorted(logs.glob("*.DONE")):
        for line in (logs / f"{done.stem}.log").read_text(encoding="utf-8").splitlines():
            if not line.startswith("START "):
                continue
            for part in line.split(" ", 3)[3].split(" && "):  # START <date> <time> <command>
                argv = _run_sweep_argv(shlex.split(part))
                if argv is None:
                    continue
                csv = f"{flag_value(argv, '--out')}/offset/offset.csv"
                if csv in needed and out.setdefault(csv, tuple(argv)) != tuple(argv):
                    raise ValueError(f"{csv}: two argument lists (rerun job {done.stem})")
    lacking = sorted(needed - set(out))
    if lacking:
        raise ValueError(f"laptop runs without a known command: {lacking}")
    return out


def _parse(pilot: Path, argv: Sequence[str], cache: Dict[Tuple[str, ...], Parsed]) -> Parsed:
    """run_sweep.py's parse of ``argv`` (once per list), the files it reads at parse time made
    absolute: the runs started in the repository root."""
    from runs_groups import run_config  # scripts/sim

    key = tuple(argv)
    if key not in cache:
        tokens = list(argv)
        for i, tok in enumerate(tokens[:-1]):
            if tok in ("--calibration", "--tuned") and not Path(tokens[i + 1]).is_absolute():
                tokens[i + 1] = str(pilot / tokens[i + 1])
        cache[key] = run_config(tokens)
    return cache[key]


def config_key(parsed: Parsed, policy: str) -> Tuple[object, ...]:
    """What ``policy`` reads of a run: sweep, workload, service, cost, policy config and seeds, the
    tuned knobs only of the rule that reads them, the learned-timing prior only for its readers."""
    from runs_groups import TUNED_KNOBS  # scripts/sim
    from pilot.sim import PolicyConfig
    from tune_baselines import KNOBS  # scripts/sim

    args, cfg = parsed
    base, reads = PolicyConfig(), KNOBS.get(policy, ())
    knobs = replace(cfg.policy, **{k: getattr(base, k) for k in TUNED_KNOBS if k not in reads})
    workload = (cfg.workload if policy in PRIOR_READERS
                else replace(cfg.workload, learn_prior_slots_csv=None))
    return (args.sweep, workload, cfg.service, cfg.cost, knobs, tuple(args.seeds or ()), cfg.root)


def _sources(group: Sequence[RunFile], frames: Mapping[str, pd.DataFrame]) -> Dict[str, RunFile]:
    """The run each cells_long.csv label of a group comes from, as ``assemble`` stacks them."""
    from pilot.validation.tables import SLO_SUFFIXES

    base = [r for r in group if r.spec.variant == "main"]
    if len(base) != 1:
        raise ValueError(f"{[r.run for r in group]}: not one base run")
    have = set(frames[base[0].run]["policy"])
    out: Dict[str, RunFile] = {p: base[0] for p in have}
    for run in group:
        policies = set(frames[run.run]["policy"])
        if run.spec.is_learn_only:
            out.update({p: run for p in policies - have})
            have |= policies
        elif run.spec.variant in SLO_SUFFIXES:
            suffix = SLO_SUFFIXES[run.spec.variant]
            out.update({p + suffix: run for p in policies - {"offline"}})
    return out


def _cloud_row(case: str, tier: str, tuned_at: str, label: str, csv: str,
               frame: pd.DataFrame, cells: pd.DataFrame) -> CloudRow:
    """One cloud row from its run, checked against cells_long.csv (unrounded means)."""
    policy = label.split(" (")[0]
    rows = frame[frame["policy"] == policy].sort_values("seed").reset_index(drop=True)
    if tuple(int(s) for s in rows["seed"]) != SEEDS:
        raise ValueError(f"{csv}: {policy} lacks one whole-run row per seed of {SEEDS}")
    row = CloudRow(case, tier, tuned_at, label, policy, csv, rows,
                   float((rows["gpu_hours"] / rows["weeks"]).mean()), float(rows["slo_in"].mean()))
    ref = cells[(cells["slots"] == "main") & (cells["ratio"] == RATIO) & (cells["case"] == case)
                & (cells["tier"] == tier) & (cells["tuned_at"] == tuned_at) & (cells["cell"] == "C")
                & (cells["policy"] == label)]
    if (len(ref) != 1 or abs(float(ref["gpu_h_wk_mean"].iloc[0]) - row.gpu) > 1e-9
            or abs(float(ref["slo_in_mean"].iloc[0]) - row.slo) > 1e-9):
        raise ValueError(f"{case} {tier} tuned at {tuned_at}, {label}: {csv} does not reproduce "
                         f"{CELLS_LONG}")
    return row


def cloud_rows(repo: Path) -> List[CloudRow]:
    """Every cloud row of cell C this section compares, each checked against cells_long.csv."""
    from pilot.validation import discover

    cells = pd.read_csv(repo / CELLS_LONG, float_precision="round_trip")
    runs = [r for r in discover(repo / REMOTE) if r.case in CASES_HERE and r.spec.ratio == RATIO]
    out: List[CloudRow] = []
    for case in CASES_HERE:
        for tier, tuned_at in CLOUD_GROUPS:
            group = sorted((r for r in runs if r.case == case
                            and r.spec.group == (RATIO, tier, tuned_at, "main")), key=lambda r: r.run)
            csvs = {r.run: str(r.path.relative_torepo) for r in group}
            frames = {run: _whole_c(repo, f"{csv}") for run, csv in csvs.items()}
            for label, src in sorted(_sources(group, frames).items()):
                if tuned_at != tier and label.split(" (")[0] not in FEEDBACK:
                    continue  # first-round planners: older planner code (module docstring)
                out.append(_cloud_row(case, tier, tuned_at, label, csvs[src.run], frames[src.run],
                                      cells))
    return out


def _equal(a: pd.DataFrame, b: pd.DataFrame, policies: Sequence[str], where: str) -> int:
    """Rows of ``policies`` in two runs (same cells, seeds, weeks); raises unless every outcome
    column is equal.  Returns the rows compared."""
    cols = [*KEYS, "policy", *OUTCOME]
    order = [*KEYS, "policy"]
    x = a[a["policy"].isin(policies)][cols].sort_values(order).reset_index(drop=True)
    y = b[b["policy"].isin(policies)][cols].sort_values(order).reset_index(drop=True)
    if x.empty or x.shape != y.shape or not ((x == y) | (x.isna() & y.isna())).all().all():
        raise ValueError(f"{where}: the {'/'.join(policies)} rows differ")
    return len(x)


def round1_checks(repo: Path) -> int:
    """The first-round runs against the re-run and the follow-up round (module docstring)."""
    from runs_groups import CASES  # scripts/sim

    checked = 0
    for case in (CASES["c2"], CASES["c3"]):
        path = f"{REMOTE}/{case.box}/{{}}/{case.name}/{{}}/offset/offset.csv"  # tree, run
        old, new, inst, t8 = (sweep_frame(repo, path.format(tree, run)) for tree, run in (
            ("validation", "r30_d_rep"), ("validation2", "r30_d_rep"),
            ("validation", "r30_d_inst"), ("validation", "r30_d_inst_t8")))
        checked += _equal(old, new, ("offline", *FEEDBACK), f"{case.name}: first-round and re-run r30_d_rep")
        checked += _equal(inst, t8, ("offline",), f"{case.name}: r30_d_inst and r30_d_inst_t8")
    return checked


@dataclass(frozen=True, eq=False)
class Pair:
    """A laptop row of section 27a and the cloud row(s) of the same configuration."""

    case: str
    tier: str  # d_inst | d_rep
    name: str  # the laptop row's name in section 27a
    policy: str
    laptop_gpu: float
    laptop_slo: float
    laptop_csv: str  # relative to the repository root
    cloud: Tuple[CloudRow, ...]  # equal seed for seed; the first carries the numbers

    @property
    def d_gpu(self) -> float:
        """Cloud minus laptop GPU-hours per week."""
        return self.cloud[0].gpu - self.laptop_gpu

    @property
    def d_pct(self) -> float:
        """The same in percent of the laptop value."""
        return 100.0 * (self.cloud[0].gpu / self.laptop_gpu - 1.0)

    @property
    def d_slo(self) -> float:
        """Cloud minus laptop in-class attainment."""
        return self.cloud[0].slo - self.laptop_slo


def _check_same(hits: Sequence[CloudRow]) -> None:
    """Raise unless cloud rows of one configuration agree seed for seed in every outcome."""
    for other in hits[1:]:
        x, y = hits[0].rows[list(OUTCOME)], other.rows[list(OUTCOME)]
        if not ((x == y) | (x.isna() & y.isna())).all().all():
            raise ValueError(f"{hits[0].label} ({hits[0].csv}) and {other.label} ({other.csv}): "
                             "one configuration, different rows")


def _why(policy: str, same_policy: Sequence[CloudRow], parsed: Parsed) -> str:
    """Why a laptop row has no cloud counterpart."""
    from tune_baselines import KNOBS  # scripts/sim

    if not same_policy:
        return f"no cloud run has {policy}"
    if policy in KNOBS:
        knobs = ", ".join(f"{k} {getattr(parsed[1].policy, k)}" for k in KNOBS[policy])
        return f"no cloud row of {policy} at this tier has its tuned knobs ({knobs})"
    return f"no cloud row of {policy} at this tier has the same run configuration"


def _pair_row(p: Pair, labels: Mapping[str, str]) -> Dict[str, object]:
    """A row of the pair table: both platforms' values and the gaps (from unrounded means)."""
    top = p.cloud[0]
    return {"case": labels.get(p.case, p.case), "tier": p.tier, "group": group_of(p.policy),
            "policy": p.name, "cloud_row": " = ".join(c.label for c in p.cloud),
            "cloud_tuned_at": " / ".join(dict.fromkeys(c.tuned_at for c in p.cloud)),
            "laptop_gpu_h_wk": f"{p.laptop_gpu:.2f}", "cloud_gpu_h_wk": f"{top.gpu:.2f}",
            "d_gpu_h_wk": f"{p.d_gpu:+.2f}", "d_gpu_pct": f"{p.d_pct:+.2f}",
            "laptop_slo_in": f"{p.laptop_slo:.4f}", "cloud_slo_in": f"{top.slo:.4f}",
            "d_slo_in": f"{p.d_slo:+.4f}", "laptop_source": f"{p.laptop_csv}",
            "cloud_source": "; ".join(f"{c.csv}" for c in p.cloud)}


def _summary(pairs: Sequence[Pair], labels: Mapping[str, str]) -> pd.DataFrame:
    """Per case-tier and over all four, per group: the largest absolute gaps and their rows."""
    scopes = [(labels.get(c, c), t, [p for p in pairs if (p.case, p.tier) == (c, t)])
              for c, t in dict.fromkeys((p.case, p.tier) for p in pairs)]
    scopes.append(("overall", "both", list(pairs)))
    rows: List[Dict[str, object]] = []
    for case, tier, part in scopes:
        for group in GROUPS:
            sub = [p for p in part if group_of(p.policy) == group]
            if not sub:
                continue
            rec: Dict[str, object] = {"case": case, "tier": tier, "group": group, "pairs": len(sub)}
            for col, fmt, gap in (("gpu_h_wk", ".2f", lambda p: p.d_gpu),
                                  ("gpu_pct", ".2f", lambda p: p.d_pct),
                                  ("slo_in", ".4f", lambda p: p.d_slo)):
                top = max(sub, key=lambda p: abs(gap(p)))
                where = f"{labels.get(top.case, top.case)} {top.tier}: " if case == "overall" else ""
                rec[f"max_abs_d_{col}"] = f"{abs(gap(top)):{fmt}}"
                rec[f"{col}_row"] = f"{where}{top.name}"
            rows.append(rec)
    return pd.DataFrame(rows)


def platform_tables(repo: Path, labels: Mapping[str, str]) -> PlatformTables:
    """Section 34: pair section 27a's rows with the cloud rows of the same configuration."""
    from runs_groups import cloud_originals  # scripts/sim

    round1 = round1_checks(repo)
    clouds = cloud_rows(repo)
    laptop = _paired(repo).query("week == 0")
    laptop = laptop[laptop["case"].isin(CASES_HERE)]
    lap_argv = laptop_argv(repo, set(laptop["source"]))
    cloud_argv = {o.csv: o.argv for o in cloud_originals()}
    missing = sorted({c.csv for c in clouds} - set(cloud_argv))
    if missing:
        raise ValueError(f"cloud runs without a phase-file command: {missing}")
    pilot = repo
    cache: Dict[Tuple[str, ...], Parsed] = {}
    pairs: List[Pair] = []
    unpaired: List[Dict[str, object]] = []
    used: Set[int] = set()
    for r in laptop.itertuples(index=False):
        tier = LAPTOP_TIERS[str(r.tier)]
        mine = _parse(pilot, lap_argv[str(r.source)], cache)
        same = [c for c in clouds if (c.case, c.tier, c.policy) == (r.case, tier, r.policy)]
        hits = [c for c in same if config_key(_parse(pilot, cloud_argv[c.csv], cache), c.policy)
                == config_key(mine, str(r.policy))]
        if not hits:
            unpaired.append({"platform": "laptop", "case": labels.get(str(r.case), str(r.case)),
                             "tier": tier, "policy": r.name, "why": _why(str(r.policy), same, mine),
                             "source": f"{r.source}"})
            continue
        _check_same(hits)
        used.update(id(c) for c in hits)
        pairs.append(Pair(str(r.case), tier, str(r.name), str(r.policy), float(r.gpu_h_per_week),
                          float(r.slo_in), str(r.source), tuple(hits)))
    for c in clouds:
        if id(c) not in used:
            unpaired.append({"platform": "cloud", "case": labels.get(c.case, c.case), "tier": c.tier,
                             "policy": f"{c.label} (tuned at {c.tuned_at})",
                             "why": "no laptop row of this policy has the same configuration",
                             "source": f"{c.csv}"})
    pairs.sort(key=lambda p: (CASES_HERE.index(p.case), p.tier, GROUPS.index(group_of(p.policy))))
    logger.info("platform: %d pairs, %d rows unpaired (%s)", len(pairs), len(unpaired), PAIRED_CSV)
    return PlatformTables(pd.DataFrame([_pair_row(p, labels) for p in pairs]), _summary(pairs, labels),
                          pd.DataFrame(unpaired), len(clouds), round1)


__all__ = ["CloudRow", "Pair", "PlatformTables", "cloud_rows", "config_key", "group_of",
           "laptop_argv", "platform_tables", "round1_checks"]
