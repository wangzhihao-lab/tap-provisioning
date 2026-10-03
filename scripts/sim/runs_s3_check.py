"""Check of one s3-consistency re-run (``runs_s3_jobs.py``) against the run it repeats.

    runs_s3_check.py --new NEW.csv --orig ORIG.csv --policies offline reactive velocity
                          --old-tuned OLD/best.json --new-tuned NEW/best.json --report R.json
                          [--log JOB.log]

The re-run repeats a laptop sensitivity run (``validation_local/sensitivity_sweeps``,
``priority_ablation``) with the same arguments except ``--policies``, ``--tuned`` (the
resource-rule ``best.json`` of the same case and tier) and ``--out``.  On this laptop that gives:

(a) the offline rows are the original's: the same (timing cell, seed, week) keys, once each, and
    equal values in every column the original carries;
(b) a feedback rule whose configuration is the same in both ``best.json`` files
    (``tuning_select.changed_policies``) reproduces the original's rows in the same sense;
(c) a rule whose configuration moved covers the same keys; its whole-run means per cell are
    reported next to the original's, not compared.

The new CSV must hold exactly ``--policies``.  With ``--log`` the last ``policy overrides applied``
line of the job's log (run_sweep.py writes it) must equal the new ``best.json``.  Any failure exits
3, which marks the job FAILED and stops the runner from starting further jobs: a mismatch means the
premise of the batch (same stream, same code, only the tuned knobs move) does not hold.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runs_check import HALT, KEYS, PILOT, read  # noqa: E402
from runs_groups import OFFSET_KEYS  # noqa: E402
from pilot.sim import SweepConfig  # noqa: E402
from tuned_config import read_tuned  # noqa: E402
from tuning_select import changed_policies, configs_of  # noqa: E402

RULES = ("reactive", "velocity")  # the feedback rules a best.json tunes
SUMMARY = ("gpu_hours", "starts", "slo_in", "cost_ratio", "total_cost")
APPLIED = re.compile(r"policy overrides applied: (\{.*\})")


def with_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the timing-cell letter of every row (the order of ``SweepConfig``)."""
    grid = [tuple(float(v) for v in spec) for spec in SweepConfig().trigger_offset]
    letters = [chr(ord("A") + grid.index(tuple(float(v) for v in row)))
               for row in frame[list(OFFSET_KEYS)].itertuples(index=False)]
    return frame.assign(cell=letters)


def rows_of(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    """One policy's rows; a repeated (cell, seed, week) key is a mapping error and raises."""
    rows = frame[frame["policy"] == policy]
    if rows.duplicated(list(KEYS)).any():
        raise ValueError(f"{policy}: repeated (timing cell, seed, week) rows")
    return rows


def compare_rows(new: pd.DataFrame, orig: pd.DataFrame, policy: str,
                 values: bool) -> Dict[str, object]:
    """Keys of one policy in both runs and, with ``values``, every column of the original."""
    a, b = rows_of(new, policy), rows_of(orig, policy)
    keys = list(KEYS)
    m = a.merge(b, on=keys, how="outer", suffixes=("_new", "_orig"), indicator=True)
    both = m[m["_merge"] == "both"]
    out: Dict[str, object] = {"rows_new": int(len(a)), "rows_orig": int(len(b)),
                              "only_new": int((m["_merge"] == "left_only").sum()),
                              "only_orig": int((m["_merge"] == "right_only").sum()),
                              "cells": sorted(set(with_cell(a)["cell"])) if len(a) else []}
    ok = len(b) > 0 and out["only_new"] == 0 and out["only_orig"] == 0
    if values:
        cols = [c for c in b.columns if c not in keys]
        missing = [c for c in cols if c not in a.columns]
        diff: Dict[str, Dict[str, float]] = {}
        for col in (c for c in cols if c not in missing):
            x, y = both[f"{col}_new"], both[f"{col}_orig"]
            same = (x == y) | (x.isna() & y.isna())
            if not bool(same.all()):
                gap = (pd.to_numeric(x, errors="coerce") - pd.to_numeric(y, errors="coerce")).abs()
                diff[col] = {"rows": int((~same).sum()), "max_abs": float(gap.max())}
        out.update({"columns": len(cols), "missing_columns": missing, "diff": diff})
        ok = ok and not missing and not diff
    out["equal" if values else "same_keys"] = bool(ok)
    return out


def whole_run(frame: pd.DataFrame, policy: str) -> Dict[str, Dict[str, float]]:
    """Per timing cell: seed means of the whole-run (week 0) ``SUMMARY`` columns."""
    rows = with_cell(frame[(frame["policy"] == policy) & (frame["week"] == 0)])
    return {str(cell): {k: float(grp[k].mean()) for k in SUMMARY}
            for cell, grp in rows.groupby("cell")}


def applied(log: Path) -> Optional[Dict[str, object]]:
    """The last ``policy overrides applied`` dict of a job log, or None."""
    hits = APPLIED.findall(log.read_text(encoding="utf-8", errors="replace"))
    return ast.literal_eval(hits[-1]) if hits else None


def run_check(new_csv: str, orig_csv: str, policies: Sequence[str], old_tuned: str,
              new_tuned: str, report: str, log: Optional[str]) -> int:
    """The whole check; writes ``report`` and returns the exit code."""
    new, orig = read(new_csv), read(orig_csv)
    old_cfg = json.loads((PILOT / old_tuned).read_text(encoding="utf-8"))
    new_cfg = json.loads((PILOT / new_tuned).read_text(encoding="utf-8"))
    changed = changed_policies(old_cfg, new_cfg)
    failures: List[str] = []
    found = sorted(set(new["policy"]))
    if found != sorted(policies):
        failures.append(f"new run holds policies {found}, expected {sorted(policies)}")
    offline = compare_rows(new, orig, "offline", values=True)
    lines = [f"offline   {'equal' if offline['equal'] else 'DIFFERS'} rows {offline['rows_new']}/"
             f"{offline['rows_orig']} cells {offline['cells']}; changed rules {changed or 'none'}"]
    if not offline["equal"]:
        failures.append("offline rows differ from the original's")
    rules: Dict[str, Dict[str, object]] = {}
    for rule in (p for p in policies if p in RULES):
        moved = rule in changed
        res = compare_rows(new, orig, rule, values=not moved)
        res["config"] = "changed" if moved else "unchanged"
        if moved:
            res["whole_run_new"], res["whole_run_orig"] = whole_run(new, rule), whole_run(orig, rule)
        ok = bool(res["same_keys"] if moved else res["equal"])
        if not ok:
            failures.append(f"{rule} ({res['config']} config) does not match the original")
        lines.append(f"{rule:9s} {res['config']} config: {'same keys' if moved else 'equal'} {ok} "
                     f"rows {res['rows_new']}/{res['rows_orig']}")
        rules[rule] = res
    got: Optional[Dict[str, object]] = None
    if log is not None:
        want, got = read_tuned(PILOT / new_tuned), applied(PILOT / log)
        if got != want:
            failures.append(f"run_sweep applied {got}, the new best.json holds {want}")
    body = {"new": new_csv, "orig": orig_csv, "old_tuned": old_tuned, "new_tuned": new_tuned,
            "changed_rules": changed, "old_config": configs_of(old_cfg),
            "new_config": configs_of(new_cfg), "applied_overrides": got, "offline": offline,
            "rules": rules, "failures": failures, "status": "FAILED" if failures else "verified"}
    target = PILOT / report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    for line in lines + [f"FAIL {f}" for f in failures]:
        print(line)
    print(f"status {body['status']} -> {report}")
    return HALT if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--new", "--orig", "--old-tuned", "--new-tuned", "--report"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--log", default=None, help="the job's log (run_sweep's override line)")
    args = ap.parse_args(argv)
    try:
        return run_check(args.new, args.orig, args.policies, args.old_tuned, args.new_tuned,
                         args.report, args.log)
    except (OSError, ValueError, KeyError) as exc:
        print(f"FAIL check could not run: {exc!r}")
        return HALT


if __name__ == "__main__":
    raise SystemExit(main())
