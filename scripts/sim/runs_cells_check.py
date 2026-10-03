"""Check of one cells-laptop job (``runs/cells_laptop/JOBS.txt``) against its R1 TAP run.

    runs_cells_check.py --new NEW/offset/offset.csv --pair R1/offset/offset.csv
                             --policies offline calendar ... --report NEW/invariant.json
                             [--cells A B C D E F] [--seeds 10 11 12 13 14] [--weeks 6]
                             [--cloud CLOUD/offset/offset.csv]

Each job repeats one cloud run of the validation round on this laptop with the cloud run's
arguments: the same policies, seeds, weeks, prices, calibration, lead tier and ``--tuned`` file
(on the local mirror), ``--offset-cells A B C D E F`` written out because the default grid now
runs to cell L, and ``--out`` under ``cells_laptop``.  The run passes when

(a) it holds exactly ``--policies``, each with one row per (timing cell, seed, week) of
    ``--cells`` x ``--seeds`` x weeks 0..``--weeks`` and no other row;
(b) its offline rows are those of the R1 TAP run of the same case and tier
    (``runs/tap/r1_<case>_<tier>``): the same keys, once each, and equal values in every
    column of the R1 CSV (``runs_s3_check.compare_rows``), so it replays TAP's streams.

Any failure exits 3: the job is marked FAILED and ``runs_runner.py`` starts nothing further.
``--cloud`` records how far the cloud original's offline rows are; they cannot agree (JOBS_README.md,
Platform), so that comparison never fails the check.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runs_check import HALT, PILOT, REPORT_COLS, compare, read  # noqa: E402
from runs_s3_check import compare_rows, with_cell  # noqa: E402

CELLS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SEEDS: Tuple[int, ...] = (10, 11, 12, 13, 14)
WEEKS = 6
Key = Tuple[str, str, int, int]  # (policy, timing cell, seed, week)


def grid_failures(frame: pd.DataFrame, policies: Sequence[str], cells: Sequence[str],
                  seeds: Sequence[int], weeks: int) -> List[str]:
    """What keeps ``frame`` from holding one row per policy x cell x seed x week and no other."""
    failures: List[str] = []
    found = sorted(set(frame["policy"]))
    if found != sorted(policies):
        failures.append(f"policies {found}, expected {sorted(policies)}")
    rows = with_cell(frame)
    have: List[Key] = [(str(p), str(c), int(s), int(w)) for p, c, s, w in
                       zip(rows["policy"], rows["cell"], rows["seed"], rows["week"])]
    want = set(itertools.product(policies, cells, seeds, range(weeks + 1)))
    if len(have) != len(set(have)):
        failures.append(f"{len(have) - len(set(have))} repeated (policy, cell, seed, week) rows")
    missing, extra = sorted(want - set(have)), sorted(set(have) - want)
    if missing:
        failures.append(f"{len(missing)} rows missing, e.g. {missing[:3]}")
    if extra:
        failures.append(f"{len(extra)} rows outside the grid, e.g. {extra[:3]}")
    if (rows.loc[rows["week"] == 0, "weeks"] != weeks).any():
        failures.append(f"whole-run rows that cover other than {weeks} weeks")
    return failures


def run_check(new_csv: str, pair_csv: str, policies: Sequence[str], cells: Sequence[str],
              seeds: Sequence[int], weeks: int, report: str, cloud_csv: Optional[str]) -> int:
    """The whole check; writes ``report`` and returns the exit code."""
    new, pair = read(new_csv), read(pair_csv)
    grid = grid_failures(new, policies, cells, seeds, weeks)
    offline = compare_rows(new, pair, "offline", values=True)
    failures = grid + ([] if offline["equal"] else ["offline rows differ from the R1 TAP run's"])
    seen = "".join(cast(List[str], offline["cells"]))
    lines = [f"grid      {'FAILED' if grid else 'complete'}: {len(set(new['policy']))} "
             f"policies x cells {''.join(cells)} x seeds {len(seeds)} x weeks 0-{weeks} "
             f"({len(new)} rows)",
             f"offline   {'EQUAL' if offline['equal'] else 'DIFFERS'} rows {offline['rows_new']}/"
             f"{offline['rows_orig']} cells {seen} vs {pair_csv}"]
    body: Dict[str, object] = {"new": new_csv, "pair": pair_csv, "policies": sorted(policies),
                               "cells": list(cells), "seeds": list(seeds), "weeks": weeks,
                               "offline_vs_pair": offline}
    if cloud_csv is not None:
        res = {"path": cloud_csv, "platform": "cloud", **compare(new, read(cloud_csv), False)}
        diff = cast(Dict[str, Dict[str, float]], res["diff"])
        worst = {k: v["max_rel"] for k, v in diff.items() if k in REPORT_COLS}
        lines.append(f"cloud     {'equal' if res['equal'] else 'differs'} rows "
                     f"{res['rows_matched']} {cloud_csv}" + (f"  max rel {worst}" if worst else ""))
        body["offline_vs_cloud"] = res
    body.update({"failures": failures, "status": "FAILED" if failures else "verified"})
    target = PILOT / report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    for line in lines + [f"FAIL {f}" for f in failures]:
        print(line)
    print(f"status {body['status']} -> {report}")
    return HALT if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--new", "--pair", "--report"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--weeks", type=int, default=WEEKS)
    ap.add_argument("--cloud", default=None, help="the cloud original, for the record only")
    args = ap.parse_args(argv)
    try:
        return run_check(args.new, args.pair, args.policies, args.cells, args.seeds, args.weeks,
                         args.report, args.cloud)
    except (OSError, ValueError, KeyError) as exc:
        print(f"FAIL check could not run: {exc!r}")
        return HALT


if __name__ == "__main__":
    raise SystemExit(main())
