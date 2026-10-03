"""Checks of the rerun round.

    runs_check.py offline --new NEW.csv --orig ORIG.csv [--orig ...] --report R.json [--stream-only]
    runs_check.py floor-eval --old OLD/best.json --new NEW/best.json --note N.json
                                  --pair PAIR.csv --orig ORIG.csv -- <run_sweep.py arguments>

``offline``: the offline rows of a new run against those of the originals of its stream group,
matched on (timing cell, seed, week), every column both CSVs carry (``--stream-only``: only the
columns the arrival stream and the service decide, for a run at another price).  An original made
on this laptop (``validation_local``) must agree exactly - otherwise the check exits 3, which stops
the runner's queue; a cloud original is compared and reported but cannot agree (JOBS_README.md,
"Platform").  An original with no matching row is a mapping error and also exits 3.  When the new
run has TAP rows, the report adds TAP's whole-run numbers per cell and its diagnostics.

``floor-eval``: re-run the six-week floor-baseline evaluation with the resource-rule ``best.json``
only if it selected another configuration than the original one; ``--note`` records the decision.

``columns``: every column two tables share must be equal (the Fig. 1 replay with TAP added against
the one on disk); exits 1 otherwise - that job fails, the queue goes on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runs_groups import OFFSET_KEYS, flag_value  # noqa: E402
from pilot.sim import SweepConfig  # noqa: E402
from tuning_select import changed_policies, configs_of  # noqa: E402

HALT = 3
KEYS = (*OFFSET_KEYS, "seed", "week")
SKIP = ("sweep", "policy")
STREAM_COLS = ("n_in", "n_off", "mu", "sigma2_measured", "sigma2_pooled", "sync_sessions", "eta",
               "eta_norm", "p_in", "p_off", "price_ratio_real", "lead_min", "commit_block_min",
               "billing_block_min", "adoption", "rho", "weeks", "college")
REPORT_COLS = ("gpu_hours", "starts", "slo_in", "n_in", "mu", "total_cost")


def read(path: str) -> pd.DataFrame:
    """A sweep CSV, floats parsed exactly (pandas' default parser is not round-trip)."""
    return pd.read_csv(PILOT / path, float_precision="round_trip")


def cell_letters(frame: pd.DataFrame) -> List[str]:
    """Timing-cell letters present in rows of an offset sweep."""
    grid = SweepConfig().trigger_offset
    rows = frame[list(OFFSET_KEYS)].drop_duplicates().itertuples(index=False)
    return sorted(chr(ord("A") + grid.index(tuple(float(v) for v in r))) for r in rows)


def compare(new: pd.DataFrame, orig: pd.DataFrame, stream_only: bool) -> Dict[str, object]:
    """Offline rows of ``new`` against ``orig``: matched rows, differing columns, their size."""
    a, b = new[new["policy"] == "offline"], orig[orig["policy"] == "offline"]
    keys = [k for k in KEYS if k in a.columns and k in b.columns]
    m = a.merge(b, on=keys, suffixes=("_new", "_orig"))
    cols = sorted(c[:-4] for c in m.columns
                  if c.endswith("_new") and f"{c[:-4]}_orig" in m.columns and c[:-4] not in SKIP)
    if stream_only:
        cols = [c for c in cols if c in STREAM_COLS]
    diff: Dict[str, Dict[str, float]] = {}
    for col in cols:
        x, y = m[f"{col}_new"], m[f"{col}_orig"]
        same = (x == y) | (x.isna() & y.isna())
        if bool(same.all()):
            continue
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        rel = ((xn - yn).abs() / yn.abs().replace(0.0, np.nan)).max()
        diff[col] = {"rows": int((~same).sum()), "max_abs": float((xn - yn).abs().max()),
                     "max_rel": float(rel) if pd.notna(rel) else float("nan")}
    return {"rows_matched": int(len(m)), "rows_new": int(len(a)), "columns": len(cols),
            "cells": cell_letters(m) if len(m) else [], "equal": not diff, "diff": diff}


def tap_summary(new: pd.DataFrame, out_dir: Path) -> Optional[Dict[str, object]]:
    """TAP per cell: whole-run GPU-h and starts per week, in-class attainment, diagnostics."""
    tap = new[(new["policy"] == "tap") & (new["week"] == 0)]
    if tap.empty:
        return None
    grid = SweepConfig().trigger_offset
    tap = tap.assign(cell=[chr(ord("A") + grid.index(tuple(float(v) for v in r)))
                           for r in tap[list(OFFSET_KEYS)].itertuples(index=False)])
    out: Dict[str, object] = {}
    for cell, grp in tap.groupby("cell"):
        weeks = grp["weeks"].astype(float)
        out[cell] = {"seeds": int(len(grp)), "gpu_hours_per_week": float((grp["gpu_hours"] / weeks).mean()),
                     "starts_per_week": float((grp["starts"] / weeks).mean()),
                     "slo_in": float(grp["slo_in"].mean()), "slo_in_min": float(grp["slo_in"].min())}
    diag_path = out_dir / "diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        for cell, grp in diag[diag["policy"] == "tap"].groupby("cell"):
            if cell in out:
                out[cell].update({"phase_share_min": float(grp["phase_share"].min()),
                                  "phase_share_mean": float(grp["phase_share"].mean()),
                                  "n_replans": [int(v) for v in grp["n_replans"]],
                                  "final_ratio_mean": float(grp["final_ratio"].mean())})
    return out


def run_offline(new_csv: str, origs: Sequence[str], report: str, stream_only: bool) -> int:
    """The ``offline`` subcommand; returns the exit code."""
    new = read(new_csv)
    results, status, laptop = [], "verified", False
    for path in origs:
        platform = "laptop" if "validation_local" in path else "cloud"
        res = {"path": path, "platform": platform, **compare(new, read(path), stream_only)}
        if res["rows_matched"] == 0 or (platform == "laptop" and not res["equal"]):
            status = "FAILED"
        laptop |= platform == "laptop"
        results.append(res)
        worst = {k: v["max_rel"] for k, v in res["diff"].items() if k in REPORT_COLS}
        print(f"{platform:6s} {'EQUAL' if res['equal'] else 'differs'} rows {res['rows_matched']:4d} "
              f"cells {''.join(res['cells'])} {path}" + (f"  max rel {worst}" if worst else ""))
    if status != "FAILED" and not laptop:
        status = "unverified-cloud-only"
    body = {"new": new_csv, "status": status, "stream_only": stream_only, "originals": results,
            "tap": tap_summary(new, (PILOT / new_csv).parents[1])}
    target = PILOT / report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"status {status} -> {report}")
    return HALT if status == "FAILED" else 0


def run_floor_eval(old: str, new: str, note: str, pair: Optional[str], orig: Optional[str],
                   sweep_argv: Sequence[str]) -> int:
    """The ``floor-eval`` subcommand; returns the exit code."""
    before = json.loads((PILOT / old).read_text(encoding="utf-8"))
    after = json.loads((PILOT / new).read_text(encoding="utf-8"))
    changed = changed_policies(before, after)
    body: Dict[str, object] = {"old": old, "new": new, "changed": changed,
                               "old_config": configs_of(before), "new_config": configs_of(after),
                               "selection": {p: s.get("selection") for p, s in after["policies"].items()},
                               "rerun": bool(changed)}
    code = 0
    if changed:
        code = subprocess.call([sys.executable, str(HERE / "run_sweep.py"), *sweep_argv], cwd=PILOT)
        body["run_sweep_exit"] = code
        out_csv = f"{flag_value(sweep_argv, '--out')}/offset/offset.csv"
        if code == 0:
            report = f"{flag_value(sweep_argv, '--out')}/invariant.json"
            code = run_offline(out_csv, [p for p in (pair, orig) if p], report, False)
            body["invariant"] = report
    (PILOT / note).parent.mkdir(parents=True, exist_ok=True)
    (PILOT / note).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"changed: {changed or 'none'}; rerun {'done' if changed else 'not needed'} -> {note}")
    return code


def run_columns(new_csv: str, orig_csv: str, report: str) -> int:
    """The ``columns`` subcommand; returns the exit code."""
    new, orig = read(new_csv), read(orig_csv)
    shared = [c for c in orig.columns if c in new.columns]
    same = len(new) == len(orig) and all(
        bool(((new[c] == orig[c]) | (new[c].isna() & orig[c].isna())).all()) for c in shared)
    body = {"new": new_csv, "orig": orig_csv, "rows": [len(new), len(orig)], "shared": shared,
            "added": [c for c in new.columns if c not in orig.columns], "equal": same}
    (PILOT / report).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"{'EQUAL' if same else 'DIFFERENT'} on {len(shared)} shared columns; added {body['added']}")
    return 0 if same else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: List[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    off = sub.add_parser("offline")
    off.add_argument("--new", required=True)
    off.add_argument("--orig", action="append", default=[])
    off.add_argument("--report", required=True)
    off.add_argument("--stream-only", action="store_true")
    fe = sub.add_parser("floor-eval")
    for flag in ("--old", "--new", "--note"):
        fe.add_argument(flag, required=True)
    fe.add_argument("--pair", default=None)
    fe.add_argument("--orig", default=None)
    col = sub.add_parser("columns")
    for flag in ("--new", "--orig", "--report"):
        col.add_argument(flag, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "offline":
        return run_offline(args.new, args.orig, args.report, args.stream_only)
    if args.cmd == "columns":
        return run_columns(args.new, args.orig, args.report)
    return run_floor_eval(args.old, args.new, args.note, args.pair, args.orig, rest)


if __name__ == "__main__":
    raise SystemExit(main())
