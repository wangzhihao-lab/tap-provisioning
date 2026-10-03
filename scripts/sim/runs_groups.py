"""Original runs and stream groups of the rerun round.

An *original* is a run already on disk: its local directory, the argument list it was produced
with, and where it ran.  Cloud runs are read from the command lists the boxes executed
(``outputs/validation_remote/box*/jobs*/phase*.txt``), with their ``outputs/validation*/`` paths
mapped onto the local mirror; laptop runs are rebuilt from the chain scripts that produced them
(``validation_local/sensitivity_sweeps/run_sensitivity_chain.sh``, ``validation_local/priority_ablation/
run_ablation*.sh``) and checked against the columns their CSVs recorded.

A *stream key* is everything that decides the offline rows of a run: the sweep, the workload
(without the learned-timing prior, which only feeds the learned planners), the service and cost
models, the policy config without the knobs the baseline tuning moves, the seeds and the root.
Two runs with the same key and an overlapping timing cell replay the same arrival stream on the
same service, so their offline rows must agree - on the same platform (the cloud boxes' x86 and
this laptop's arm64 generate different streams from identical arguments; see JOBS_README.md).
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
for _p in (PILOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_sweep  # noqa: E402
from pilot.sim import PolicyConfig, SweepConfig  # noqa: E402
from tune_baselines import KNOBS  # noqa: E402

REMOTE = "outputs/validation_remote"
LOCAL = "outputs/validation_local"
CAL = "scripts/q2/results/b4_remote/b4/ladder/rag_shared_4000/T3_calibration.csv"
SLOTS_SEM2 = "data/timetable/slots_2025-2026-2.csv"  # 2025-26 semester 2: the held-out timetable
PRIOR_SEM2 = SLOTS_SEM2          # the central prior: the earlier semester of the 2026-27 runs
PRIOR_SYM = "data/timetable/slots_2026-2027-1.csv"                 # held-out runs: the other semester
PRIOR_FWD = "data/timetable/slots_2025-2026-1.csv"     # held-out runs: the semester before
CELLS_AF: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")  # the cells of every cloud offset run
TUNED_KNOBS: Tuple[str, ...] = tuple(k for knobs in KNOBS.values() for k in knobs)
OFFSET_KEYS: Tuple[str, ...] = ("trigger_offset_mean", "trigger_offset_sd", "trigger_offset_course_sd",
                                "trigger_offset_common_sd", "forecast_offset", "forecast_offset_sd")


@dataclass(frozen=True)
class Case:
    """One model x prompt case of the validation round."""

    name: str
    model: str  # row key of the calibration CSV, kept verbatim
    p_in_r30: str
    p_in_r100: Optional[str]
    rep_lead: str  # the replica tier's lead time in minutes
    box: str  # where its cloud runs were made


CASES: Dict[str, Case] = {
    "c1": Case("c1_7b_shared4k", "qwen2.5-7b", "0.00087511", None, "0.6",
               "box3"),
    "c2": Case("c2_32b_shared4k", "qwen2.5-32b-awq", "0.002500756",
               "0.008335854", "1", "box1"),
    "c3": Case("c3_72b_shared4k", "qwen2.5-72b-awq", "0.003758436",
               "0.012528121", "1.5", "box2"),
}


@dataclass(frozen=True)
class Original:
    """A run on disk: local directory, the arguments it was made with, the platform it ran on."""

    out: str
    script: str  # "run_sweep" | "tune_baselines"
    argv: Tuple[str, ...]
    platform: str  # "cloud:<box>" | "laptop"

    @property
    def csv(self) -> str:
        """The sweep CSV of a ``run_sweep`` original."""
        return f"{self.out}/offset/offset.csv"

    @property
    def laptop(self) -> bool:
        """Made on this machine (so its streams are reproducible here bit for bit)."""
        return self.platform == "laptop"


# --------------------------------------------------------------------------- argument lists
def flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    """The value after ``flag`` (the last occurrence), or None."""
    idx = [i for i, tok in enumerate(argv) if tok == flag]
    return argv[idx[-1] + 1] if idx and idx[-1] + 1 < len(argv) else None


def drop_flags(argv: Sequence[str], flags: Sequence[str]) -> List[str]:
    """``argv`` without ``flags`` and their values (a value never starts with ``--``)."""
    out: List[str] = []
    skip = False
    for tok in argv:
        if tok.startswith("--"):
            skip = tok in flags
            if skip:
                continue
        elif skip:
            continue
        out.append(tok)
    return out


def set_flag(argv: Sequence[str], flag: str, *values: str) -> List[str]:
    """``argv`` with ``flag`` replaced by (or extended with) ``flag values``."""
    return [*drop_flags(argv, [flag]), flag, *values]


def _map_path(path: str, box: str) -> str:
    """A box's ``outputs/validation*/...`` path -> its local mirror under validation_remote."""
    for tree in ("validation2", "validation"):
        prefix = f"outputs/{tree}/"
        if path.startswith(prefix):
            return f"{REMOTE}/{box}/{tree}/{path[len(prefix):]}"
    return path


def cloud_originals() -> List[Original]:
    """Every command a box ran, from its phase files, with paths on the local mirror."""
    out: Dict[str, Original] = {}
    for phase in sorted((PILOT / REMOTE).glob("box*/jobs*/phase*.txt")):
        box = phase.parts[-3]
        for line in phase.read_text(encoding="utf-8").splitlines():
            tokens = shlex.split(line)
            script = next((t for t in tokens if t.endswith(".py")), None)
            if script is None:
                continue
            argv = tokens[tokens.index(script) + 1:]
            for flag in ("--out", "--tuned"):
                if flag in argv:
                    j = argv.index(flag) + 1
                    argv[j] = _map_path(argv[j], box)
            target = flag_value(argv, "--out")
            if target and target not in out:
                out[target] = Original(target, Path(script).stem, tuple(argv), f"cloud:{box}")
    return list(out.values())


def laptop_originals() -> List[Original]:
    """The sensitivity-round and admission-ablation sweeps of 2026-09-21, rebuilt from their scripts."""
    common = ("--sweep offset --weeks 6 --seeds 10 11 12 13 14 --adoption 1.0 --min-replicas 1 "
              "--c-run 0.0333 --beta-start 0.0333 --lead-stat median").split()
    out: List[Original] = []
    for key, grid in (("c2", ("1", "2", "3", "4", "6", "8", "12", "15")),
                      ("c3", ("1.5", "2", "3", "4", "6", "8", "12", "15"))):
        case = CASES[key]
        tuned = f"{REMOTE}/box2/validation2/{case.name}"
        trep, tinst = f"{tuned}/tuning_r30_slo95disp/best.json", f"{tuned}/tuning8_r30_slo95disp/best.json"
        head = [*common, "--learn-prior-from", PRIOR_SEM2, "--policies", "offline", "calendar",
                "calendar_learn_prior", "reactive", "velocity", "--calibration", CAL, "--model",
                case.model, "--slo-in-s", "2", "--p-in", case.p_in_r30]
        inst = ["--lead-tier", "d_inst", "--lead-min", "8", "--tuned", tinst]
        runs: Dict[str, List[str]] = {
            f"lead_d{d}": ["--offset-cells", "C", "--lead-tier", "d_inst", "--lead-min", d, "--tuned",
                           trep if float(d) <= 2 else tinst] for d in grid}
        runs["cells_GL"] = ["--offset-cells", *"GHIJKL", *inst]
        runs["res"] = ["--offset-cells", "C", "--lead-tier", "d_res", "--tuned", tinst]
        for a in ("0.25", "0.5", "0.75"):
            runs[f"adopt{a}"] = ["--offset-cells", "C", *inst, "--adoption", a]
        for name, prior in (("heldout_sym", PRIOR_SYM), ("heldout_fwd", PRIOR_FWD)):
            runs[name] = ["--offset-cells", "C", *inst, "--slots", SLOTS_SEM2, "--learn-prior-from", prior]
        for name, extra in runs.items():
            target = f"{LOCAL}/sensitivity_sweeps/{case.name}_{name}"
            out.append(Original(target, "run_sweep", tuple([*head, *extra, "--out", target]), "laptop"))
        abl = [*common[:2], "--offset-cells", "C", *common[2:], "--lead-tier", "d_inst", "--lead-min",
               "8", "--learn-prior-from", PRIOR_SEM2, "--policies", "calendar_learn_prior",
               "calendar_adapt", "offline", "reactive", "--calibration", CAL, "--model", case.model,
               "--slo-in-s", "2", "--p-in", case.p_in_r30, "--tuned", tinst]
        for mode, extra in (("on", []), ("off", ["--no-in-class-priority"]),
                            ("ungraded", ["--p-off", case.p_in_r30])):
            target = f"{LOCAL}/priority_ablation/{case.name}_{mode}"
            out.append(Original(target, "run_sweep", tuple([*abl, *extra, "--out", target]), "laptop"))
    return out


def originals() -> Dict[str, Original]:
    """All originals of cases c1-c3 that exist on disk, by local directory."""
    found: Dict[str, Original] = {}
    for orig in cloud_originals() + laptop_originals():
        if not any(CASES[k].name in orig.out for k in CASES):
            continue
        marker = orig.csv if orig.script == "run_sweep" else f"{orig.out}/best.json"
        if (PILOT / marker).exists():
            found[orig.out] = orig
    return found


# --------------------------------------------------------------------------- stream keys
def run_config(argv: Sequence[str]):
    """``(args, RunConfig)`` of a ``run_sweep.py`` argument list, parsed as the script would."""
    args = run_sweep.build_parser().parse_args(list(argv))
    return args, run_sweep.base_config(args)


def stream_key(argv: Sequence[str]) -> Tuple:
    """What decides the offline rows (module docstring); cells are compared separately."""
    args, cfg = run_config(argv)
    base = PolicyConfig()
    policy = replace(cfg.policy, **{k: getattr(base, k) for k in TUNED_KNOBS})
    return (args.sweep, replace(cfg.workload, learn_prior_slots_csv=None), cfg.service, cfg.cost,
            policy, tuple(args.seeds or ()), cfg.root)


def cells_of(argv: Sequence[str], csv: Optional[Path] = None) -> Tuple[str, ...]:
    """Timing cells of an offset run: ``--offset-cells``, else those its CSV holds."""
    args, _ = run_config(argv)
    if args.offset_cells:
        return tuple(c.upper() for c in args.offset_cells)
    if csv is None or not csv.exists():
        return CELLS_AF
    frame = pd.read_csv(csv, usecols=list(OFFSET_KEYS), float_precision="round_trip").drop_duplicates()
    grid = SweepConfig().trigger_offset
    return tuple(sorted(chr(ord("A") + grid.index(tuple(float(v) for v in row)))
                        for row in frame.itertuples(index=False)))


def check_laptop_reconstruction(orig: Original) -> List[str]:
    """Columns of a rebuilt laptop original's CSV that disagree with its rebuilt arguments."""
    _, cfg = run_config(orig.argv)
    row = pd.read_csv(PILOT / orig.csv, nrows=1, float_precision="round_trip").iloc[0]
    want = {"lead_min": cfg.service.lead_min, "commit_block_min": cfg.service.commit_block_min,
            "billing_block_min": cfg.service.billing_block_min, "adoption": cfg.workload.adoption,
            "p_in": cfg.cost.p_in, "p_off": cfg.cost.p_off, "weeks": cfg.workload.n_weeks}
    return [k for k, v in want.items() if float(row[k]) != float(v)]


__all__ = ["CAL", "CASES", "CELLS_AF", "Case", "Original", "PRIOR_FWD", "PRIOR_SEM2", "PRIOR_SYM",
           "SLOTS_SEM2", "cells_of", "check_laptop_reconstruction", "cloud_originals",
           "drop_flags", "flag_value", "laptop_originals", "originals", "run_config", "set_flag",
           "stream_key"]
