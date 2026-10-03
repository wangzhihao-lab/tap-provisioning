"""Archived runs of the replan-boundary check and the commands that wrote them (R1, 2026-09-25).

Inventory: every sweep CSV under ``outputs/validation_local`` (outside ``replan_check/``) whose
whole-run rows (week 0) hold ``tap``, ``calendar_adapt`` or ``forecast_ilp``.  For each, the
``run_sweep.py`` command that wrote it is read from where it was recorded:

- ``runs/logs/*.log`` and ``runs/cells_laptop/logs/*.log``: the runner's ``START`` line
  (``tap/``, ``billing/``, ``paired/``, ``cells_laptop/``); the log must hold one START and one
  ``EXIT 0``;
- ``f6_32b/run_chain.sh`` with the three chain invocations of 2026-09-21 (``d_rep 1 30 6``,
  ``d_rep 1 100 10``, ``d_inst 8 30``; ``chain.log`` must show START and ``EXIT 0`` for each);
- ``priority_ablation/run_ablation.sh`` and ``run_ablation_grading.sh`` (the ``run`` calls at
  their end; ``chain.log`` as above), cross-checked against ``runs_groups.laptop_originals``.

``check_run`` asserts per run: ``--out`` is the CSV's run directory, and the CSV's policies, seeds,
timing cells (or price ratio), tier, blocks, adoption, weeks and prices (offset sweeps) are those
the parsed command gives; the ``policy overrides applied`` line of the run's log equals its
``--tuned`` file; the CSV was written between the command's START and EXIT.
"""

from __future__ import annotations

import ast
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
for _p in (PILOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_sweep  # noqa: E402
from runs_groups import flag_value, laptop_originals  # noqa: E402
from tuned_config import read_tuned  # noqa: E402

LOCAL = "outputs/validation_local"
CHECK = f"{LOCAL}/runs/replan_check"
AFFECTED = ("tap", "calendar_adapt", "forecast_ilp")
SCRIPT = "scripts/sim/run_sweep.py"
F6_CHAINS = (("d_rep", "1", ("30", "6")), ("d_rep", "1", ("100", "10")), ("d_inst", "8", ("30",)))
STAMP = re.compile(r"^(START|EXIT) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
WALL = re.compile(r"in ([\d.]+) s ->")
APPLIED = re.compile(r"policy overrides applied: (\{.*\})")


@dataclass(frozen=True)
class Command:
    """An archived ``run_sweep.py`` command and where it was read from."""

    argv: Tuple[str, ...]
    source: str
    log: Optional[str]
    start: Optional[str]
    end: Optional[str]
    wall_s: Optional[float]


def expand(text: str, env: Dict[str, str]) -> str:
    """Shell ``$NAME`` / ``${NAME}`` substitution; an unknown name raises."""
    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if name not in env:
            raise KeyError(f"unset shell variable {name!r} in {text[:80]!r}")
        return env[name]
    return re.sub(r"\$\{(\w+)\}|\$(\w+)", sub, text)


def script_vars(text: str) -> Dict[str, str]:
    """Top-level ``NAME="value"`` / ``NAME=value`` assignments of a shell script."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r'^([A-Z_]+)=(?:"([^"]*)"|(\S+))\s*$', line.strip())
        if m:
            out[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return out


def sweep_line(text: str) -> str:
    """The ``run_sweep.py`` arguments of a script's one call, continuations joined, redirect cut."""
    joined = re.sub(r"\\\n\s*", " ", text)
    lines = [ln for ln in joined.splitlines() if SCRIPT in ln]
    if len(lines) != 1:
        raise ValueError(f"expected one run_sweep call, found {len(lines)}")
    return lines[0].split(SCRIPT, 1)[1].split(" > ", 1)[0].strip()


def chain_ok(chain_log: Path, start: str, exit_ok: str) -> Tuple[str, str]:
    """START and ``EXIT 0`` times of one chain step (both must be there)."""
    text = chain_log.read_text(encoding="utf-8")
    t0 = re.findall(rf"^\[([^\]]+)\] {re.escape(start)}$", text, re.M)
    t1 = re.findall(rf"^\[([^\]]+)\] {re.escape(exit_ok)}$", text, re.M)
    if len(t0) != 1 or len(t1) != 1:
        raise ValueError(f"{chain_log}: {start!r} / {exit_ok!r} found {len(t0)} / {len(t1)} times")
    return t0[0], t1[0]


def wall_of(log: Path) -> Optional[float]:
    """Seconds of the sweep from run_sweep's summary line."""
    hits = WALL.findall(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else []
    return float(hits[-1]) if hits else None


def from_runner_logs() -> Dict[str, Command]:
    """Commands of the rerun runner jobs, by ``--out``."""
    out: Dict[str, Command] = {}
    for log in sorted([*PILOT.glob(f"{LOCAL}/runs/logs/*.log"),
                       *PILOT.glob(f"{LOCAL}/runs/cells_laptop/logs/*.log")]):
        stamps = [STAMP.match(ln) for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()]
        starts = [m for m in stamps if m and m.group(1) == "START"]
        exits = [m for m in stamps if m and m.group(1) == "EXIT"]
        if not starts:
            continue
        if len(starts) != 1 or len(exits) != 1 or not exits[0].group(3).startswith("0 "):
            raise ValueError(f"{log}: {len(starts)} START / {len(exits)} EXIT lines or non-zero exit")
        wall = float(re.search(r"wall_s=(\d+)", exits[0].group(3)).group(1))  # type: ignore[union-attr]
        for part in starts[0].group(3).split(" && "):
            tokens = shlex.split(part)
            if SCRIPT not in tokens:
                continue
            argv = tuple(tokens[tokens.index(SCRIPT) + 1:])
            target = flag_value(argv, "--out")
            if target in out:
                raise ValueError(f"--out {target} in two logs")
            out[target] = Command(argv, f"START line of {log.relative_to(PILOT)}", str(log.relative_to(PILOT)),
                                  starts[0].group(2), exits[0].group(2), wall)
    return out


def from_f6_chain() -> Dict[str, Command]:
    """The F6 32B chain runs of 2026-09-21, by ``--out``."""
    base = PILOT / LOCAL / "f6_32b"
    text = (base / "run_chain.sh").read_text(encoding="utf-8")
    env, line = script_vars(text), sweep_line(text)
    out: Dict[str, Command] = {}
    for tier, lead, ratios in F6_CHAINS:
        for r in ratios:
            argv = tuple(shlex.split(expand(line, {**env, "TIER": tier, "LEAD": lead, "R": r})))
            t0, t1 = chain_ok(base / "logs" / "chain.log", f"START r{r} {tier}", f"EXIT 0 r{r} {tier}")
            log = base / "logs" / f"r{r}_{tier}.log"
            out[flag_value(argv, "--out")] = Command(  # type: ignore[index]
                argv, f"f6_32b/run_chain.sh {tier} {lead} ... {r}", str(log.relative_to(PILOT)), t0, t1,
                wall_of(log))
    return out


def from_ablation_scripts() -> Dict[str, Command]:
    """The admission-priority and grading ablations of 2026-09-21, by ``--out``."""
    base = PILOT / LOCAL / "priority_ablation"
    out: Dict[str, Command] = {}
    for script, modes in (("run_ablation.sh", (("on", ""), ("off", "--no-in-class-priority"))),
                          ("run_ablation_grading.sh", (("ungraded", ""),))):
        text = (base / script).read_text(encoding="utf-8")
        if script == "run_ablation.sh" and 'flag="--no-in-class-priority"' not in text:
            raise ValueError("run_ablation.sh no longer sets the off flag this generator assumes")
        env, line = script_vars(text), sweep_line(text)
        for m in re.finditer(r"^run (\S+) (\S+) (\S+) (\S+)\s*$", text, re.M):
            case, model, pin, tuned = m.groups()
            for mode, flag in modes:
                local = {**env, "case": case, "model": model, "pin": pin, "tuned": tuned,
                         "flag": flag, "mode": mode}
                argv = tuple(shlex.split(expand(line, local)))
                label = f"priority-{mode}" if mode != "ungraded" else "ungraded"
                t0, t1 = chain_ok(base / "logs" / "chain.log", f"START {case} {label}",
                                  f"EXIT 0 {case} {label}")
                log = base / "logs" / f"{case}_{mode}.log"
                out[flag_value(argv, "--out")] = Command(  # type: ignore[index]
                    argv, f"priority_ablation/{script} run {case}", str(log.relative_to(PILOT)), t0, t1,
                    wall_of(log))
    rebuilt = {o.out: o.argv for o in laptop_originals() if "priority_ablation" in o.out}
    for target, cmd in out.items():
        if target not in rebuilt or parsed(cmd.argv) != parsed(rebuilt[target]):
            raise ValueError(f"{target}: script command disagrees with runs_groups.laptop_originals")
    return out


def parsed(argv: Sequence[str]) -> Dict[str, object]:
    """``run_sweep``'s namespace of an argument list (order-free comparison)."""
    return vars(run_sweep.build_parser().parse_args(list(argv)))


def inventory() -> List[Tuple[str, Dict[str, int]]]:
    """Sweep CSVs whose week-0 rows hold an affected policy, with the run counts."""
    found = []
    for path in sorted((PILOT / LOCAL).rglob("*.csv")):
        rel = str(path.relative_to(PILOT))
        if rel.startswith(CHECK) or path.name == "diagnostics.csv":
            continue
        head = pd.read_csv(path, nrows=0)
        if not {"policy", "week"} <= set(head.columns):
            continue
        w0 = pd.read_csv(path, usecols=["policy", "week"]).query("week == 0")["policy"]
        counts = {p: int((w0 == p).sum()) for p in AFFECTED if (w0 == p).any()}
        if counts:
            found.append((rel, counts))
    return found


def check_run(csv: str, cmd: Command) -> List[str]:
    """What in the CSV disagrees with the command (empty when everything matches)."""
    args = run_sweep.build_parser().parse_args(list(cmd.argv))
    cfg = run_sweep.base_config(args)
    frame = pd.read_csv(PILOT / csv, float_precision="round_trip")
    bad: List[str] = []
    if str(Path(csv).parents[1]) != args.out.as_posix():
        bad.append(f"--out {args.out} is not the run directory of {csv}")
    if Path(csv).parent.name != args.sweep:
        bad.append(f"--sweep {args.sweep} does not match {csv}")
    if set(frame["policy"]) != set(args.policies or ()):
        bad.append(f"policies {sorted(set(frame['policy']))} != --policies {args.policies}")
    if sorted(set(frame["seed"])) != sorted(args.seeds or ()):
        bad.append(f"seeds {sorted(set(frame['seed']))} != --seeds {args.seeds}")
    if args.sweep == "offset":
        cells = sorted({run_sweep.cell_letter(r) for r in frame.to_dict("records")})
        if not args.offset_cells or cells != sorted(c.upper() for c in args.offset_cells):
            bad.append(f"cells {cells} != --offset-cells {args.offset_cells}")
    if args.sweep == "epsilon":
        ratios = sorted(set(frame["price_ratio"]))
        if ratios != sorted(float(r) for r in args.price_ratios or ()):
            bad.append(f"price ratios {ratios} != --price-ratios {args.price_ratios}")
    want = {"lead_min": cfg.service.lead_min, "commit_block_min": cfg.service.commit_block_min,
            "billing_block_min": cfg.service.billing_block_min, "adoption": cfg.workload.adoption,
            "weeks": cfg.workload.n_weeks}
    if args.sweep != "epsilon":
        want.update({"p_in": cfg.cost.p_in, "p_off": cfg.cost.p_off})
    for col, value in want.items():
        if set(frame[col].astype(float)) != {float(value)}:
            bad.append(f"column {col} {sorted(set(frame[col]))[:3]} != {value}")
    if cmd.log and args.tuned:
        hits = APPLIED.findall((PILOT / cmd.log).read_text(encoding="utf-8", errors="replace"))
        if not hits or ast.literal_eval(hits[-1]) != read_tuned(PILOT / args.tuned):
            bad.append(f"log {cmd.log} overrides differ from {args.tuned}")
    if cmd.start and cmd.end:
        mtime = datetime.fromtimestamp((PILOT / csv).stat().st_mtime).replace(microsecond=0)
        t0, t1 = (datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in (cmd.start, cmd.end))
        if not t0 <= mtime <= t1 + timedelta(seconds=5):  # the logs stamp whole seconds
            bad.append(f"{csv} written {mtime:%F %T}, outside {cmd.start} .. {cmd.end}")
    return bad


def commands() -> Dict[str, Command]:
    """Every archived command of the three sources, by ``--out``."""
    out: Dict[str, Command] = {}
    for part in (from_runner_logs(), from_f6_chain(), from_ablation_scripts()):
        clash = set(out) & set(part)
        if clash:
            raise ValueError(f"--out recorded twice: {sorted(clash)}")
        out.update(part)
    return out


__all__ = ["AFFECTED", "CHECK", "Command", "LOCAL", "SCRIPT", "check_run", "commands", "expand",
           "inventory", "parsed", "script_vars", "sweep_line"]
