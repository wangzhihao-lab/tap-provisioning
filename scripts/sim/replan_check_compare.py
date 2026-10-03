"""Row-by-row check of one replan-boundary re-run (the re-solve boundary check) against its original.

    replan_check_compare.py stamp   --out RUN_DIR
    replan_check_compare.py compare --new NEW.csv --orig ORIG.csv --policies tap [...] --report R.json
    replan_check_compare.py audit   --new NEW.json --orig OLD.json --report R.json

The re-run (``replan_check_jobs.py``) repeats an archived ``run_sweep.py`` command with the fixed
integer re-solve, keeping every argument except ``--policies`` (the affected policies plus
``offline``) and ``--out``.  ``stamp`` records a hash of the simulator source before the run;
``compare`` then checks, on (cell, seed, week) keys - the timing cell of an offset sweep, the
error cell and price ratio of an epsilon sweep:

- the offline rows equal the original's in every column (same keys, same values).  Otherwise the
  stream or the configuration was not reproduced: status ``failed``, exit 3 (the runner stops);
- the new CSV holds exactly ``offline`` and ``--policies``, and each of them covers the
  original's keys (a mismatch is a mapping error: ``failed``, exit 3);
- every row of an affected policy, every column the original carries: identical or not; a
  differing row is listed with the old and new value of every differing column, and each column's
  largest absolute and relative difference is reported (``gpu_hours_per_week`` and
  ``starts_per_week`` normalise the whole-run rows, week 0, by the number of teaching weeks);
- TAP's ``diagnostics.csv`` rows (phase_share, n_replans, final_ratio) the same way.

Status ``identical`` or ``differs`` exits 0: a difference is a finding, not a failure.  A crash of
the check itself writes a ``failed`` report and exits 1 (the job is marked FAILED, the queue goes
on; the run's CSV is intact and the compare command can be repeated by hand).  ``audit`` compares
a ``scripts/audit/tap_prefix_invariance.py`` JSON with the copy saved before the re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
for _p in (PILOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_sweep import OFFSET_KEYS, cell_letter  # noqa: E402

HALT = 3
CRASH = 1
KEY_CANDIDATES = (*OFFSET_KEYS, "error_knob", "error_value", "price_ratio", "seed", "week")
DIAG_KEYS = (*OFFSET_KEYS, "error_knob", "error_value", "price_ratio", "seed")
REPORT_METRICS = ("gpu_hours_per_week", "starts_per_week", "gpu_hours", "starts", "slo_in",
                  "slo_off", "total_cost", "cost_ratio", "run_cost", "switch_cost", "penalty_cost",
                  "x_peak", "unserved_in")
CODE_FILES = ("src/pilot/**/*.py", "scripts/sim/run_sweep.py", "scripts/sim/tuned_config.py",
              "scripts/audit/*.py")
STAMP = "code_stamp.json"


def sha256(path: Path) -> str:
    """Hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hash() -> Dict[str, object]:
    """One hash over the simulator source a run executes (``CODE_FILES``)."""
    files = sorted({p for pat in CODE_FILES for p in PILOT.glob(pat) if p.is_file()})
    digest = hashlib.sha256()
    for p in files:
        digest.update(f"{p.relative_to(PILOT)}\0{sha256(p)}\n".encode())
    return {"tree_sha256": digest.hexdigest(), "files": len(files),
            "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def rel(path: Path) -> str:
    """``path`` relative to the repository root when it lies under it, else absolute."""
    try:
        return str(path.resolve().relative_to(PILOT))
    except ValueError:
        return str(path)


def read(path: str) -> pd.DataFrame:
    """A sweep CSV, floats parsed exactly (pandas' default parser is not round-trip)."""
    return pd.read_csv(PILOT / path, float_precision="round_trip")


def fnum(value: object) -> object:
    """JSON-safe scalar (NaN and inf -> None, numpy -> Python)."""
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def pair(new: object, orig: object) -> Dict[str, object]:
    """Old value, new value and their difference (None for non-numeric values)."""
    try:
        diff: Optional[float] = float(new) - float(orig)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        diff = None
    return {"orig": fnum(orig), "new": fnum(new), "diff": fnum(diff)}


def with_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add per-week GPU-hours and starts (whole-run rows divided by the number of weeks)."""
    if not {"gpu_hours", "starts", "weeks", "week"} <= set(frame.columns):
        return frame
    per = np.where(frame["week"] == 0, frame["weeks"].astype(float), 1.0)
    return frame.assign(gpu_hours_per_week=frame["gpu_hours"] / per,
                        starts_per_week=frame["starts"] / per)


def label(row: Dict[str, object]) -> str:
    """Timing-cell letter of an offset row, ``knob=value[@r<ratio>]`` of an epsilon row."""
    if "error_knob" in row:
        out = f"{row['error_knob']}={float(row['error_value']):g}"  # type: ignore[arg-type]
        ratio = row.get("price_ratio")
        return out + (f"@r{float(ratio):g}" if ratio is not None and pd.notna(ratio) else "")
    return cell_letter(row) or "?"


def indexed(frame: pd.DataFrame, keys: Sequence[str], what: str) -> pd.DataFrame:
    """``frame`` indexed by its key columns (as exact strings; the columns stay); repeats raise."""
    idx = pd.MultiIndex.from_arrays([frame[k].astype(str) for k in keys], names=list(keys))
    out = frame.set_axis(idx, axis=0)
    if out.index.duplicated().any():
        raise ValueError(f"{what}: repeated keys {list(keys)}")
    return out


def compare_rows(new: pd.DataFrame, orig: pd.DataFrame, keys: Sequence[str],
                 what: str) -> Dict[str, object]:
    """Keys and every original column of two row sets; differing rows listed column by column."""
    a, b = indexed(new, keys, f"{what} new"), indexed(orig, keys, f"{what} original")
    common = b.index.intersection(a.index, sort=False)
    only_new, only_orig = a.index.difference(b.index), b.index.difference(a.index)
    cols = [c for c in b.columns if c not in keys]
    missing = [c for c in cols if c not in a.columns]
    cols = [c for c in cols if c not in missing]
    A, B = a.loc[common], b.loc[common]
    eq = pd.DataFrame({c: same(A[c], B[c]) for c in cols}, index=common)
    row_eq = eq.all(axis=1).to_numpy() if cols else np.ones(len(common), bool)
    diff_cols: Dict[str, Dict[str, object]] = {}
    for c in cols:
        bad = ~eq[c].to_numpy()
        if not bad.any():
            continue
        xn = pd.to_numeric(A[c][bad], errors="coerce")
        yn = pd.to_numeric(B[c][bad], errors="coerce")
        ab = (xn - yn).abs()
        relgap = ab / yn.abs().replace(0.0, np.nan)
        diff_cols[c] = {"rows": int(bad.sum()), "max_abs": fnum(ab.max()),
                        "max_rel": fnum(relgap.max())}
    rows: List[Dict[str, object]] = []
    for pos in np.flatnonzero(~row_eq):
        ra, rb = A.iloc[pos], B.iloc[pos]
        record: Dict[str, object] = {k: fnum(rb[k]) for k in keys}
        record["cell"] = label(rb.to_dict())
        changed = [c for c in cols if not bool(eq.iloc[pos][c])]
        shown = dict.fromkeys([*(m for m in REPORT_METRICS if m in cols), *changed])
        record["columns_differing"] = changed
        record["values"] = {c: pair(ra[c], rb[c]) for c in shown}
        rows.append(record)
    w0 = (B["week"].astype(int) == 0).to_numpy() if "week" in B.columns else np.zeros(len(B), bool)
    return {"rows_new": int(len(a)), "rows_orig": int(len(b)), "rows_compared": int(len(common)),
            "rows_identical": int(row_eq.sum()), "rows_differ": int((~row_eq).sum()),
            "week0_compared": int(w0.sum()), "week0_differ": int((~row_eq & w0).sum()),
            "only_new": [list(k) for k in only_new], "only_orig": [list(k) for k in only_orig],
            "columns": len(cols), "missing_columns": missing, "diff_columns": diff_cols,
            "differing_rows": rows}


def same(x: pd.Series, y: pd.Series) -> pd.Series:
    """Element-wise exact equality, NaN equal to NaN."""
    return (x == y) | (x.isna() & y.isna())


def covers(res: Dict[str, object]) -> bool:
    """Same key set on both sides, something compared, no original column missing."""
    return (int(res["rows_orig"]) > 0 and not res["only_new"] and not res["only_orig"]
            and not res["missing_columns"])


def compare_diagnostics(new_dir: Path, orig_dir: Path, policies: Sequence[str]
                        ) -> Tuple[Optional[Dict[str, object]], List[str]]:
    """The per-run diagnostics rows of the affected policies (TAP), when the original has any."""
    orig_path, new_path = orig_dir / "diagnostics.csv", new_dir / "diagnostics.csv"
    if not orig_path.exists():
        return None, []
    orig = pd.read_csv(orig_path, float_precision="round_trip")
    orig = orig[orig["policy"].isin(policies)]
    if orig.empty:
        return None, []
    if not new_path.exists():
        return {"orig": rel(orig_path)}, ["new run wrote no diagnostics.csv"]
    new = pd.read_csv(new_path, float_precision="round_trip")
    out: Dict[str, object] = {"orig": rel(orig_path), "new": rel(new_path), "policies": {}}
    failures: List[str] = []
    for policy in sorted(set(orig["policy"])):
        keys = [k for k in DIAG_KEYS if k in orig.columns]
        res = compare_rows(new[new["policy"] == policy], orig[orig["policy"] == policy], keys,
                           f"diagnostics {policy}")
        out["policies"][policy] = res  # type: ignore[index]
        if not covers(res):
            failures.append(f"diagnostics of {policy} do not cover the original's runs")
    return out, failures


def run_compare(new_csv: str, orig_csv: str, policies: Sequence[str], report: str) -> int:
    """The ``compare`` subcommand; writes ``report`` and returns the exit code."""
    new, orig = with_derived(read(new_csv)), with_derived(read(orig_csv))
    keys = [k for k in KEY_CANDIDATES if k in orig.columns]
    failures: List[str] = []
    want = sorted({"offline", *policies})
    found = sorted(set(new["policy"]))
    if found != want:
        failures.append(f"new run holds policies {found}, expected {want}")
    body: Dict[str, object] = {"new": new_csv, "orig": orig_csv, "policies": list(policies),
                               "keys": keys, "new_sha256": sha256(PILOT / new_csv),
                               "orig_sha256": sha256(PILOT / orig_csv)}
    stamp_path = (PILOT / new_csv).parents[1] / STAMP
    start = json.loads(stamp_path.read_text()) if stamp_path.exists() else None
    end = code_hash()
    body["code"] = {"start": start, "end": end,
                    "unchanged": (start["tree_sha256"] == end["tree_sha256"]) if start else None}
    offline = compare_rows(new[new["policy"] == "offline"], orig[orig["policy"] == "offline"], keys,
                           "offline")
    body["offline"] = {**offline, "equal": covers(offline) and offline["rows_differ"] == 0}
    if not body["offline"]["equal"]:  # type: ignore[index]
        failures.append("offline rows differ from the original's (stream or configuration not reproduced)")
    per: Dict[str, Dict[str, object]] = {}
    for policy in policies:
        res = compare_rows(new[new["policy"] == policy], orig[orig["policy"] == policy], keys, policy)
        if not covers(res):
            failures.append(f"{policy}: keys or columns do not cover the original's")
        per[policy] = res
    body["affected"] = per
    diag, diag_fail = compare_diagnostics((PILOT / new_csv).parents[1], (PILOT / orig_csv).parents[1],
                                          policies)
    body["diagnostics"] = diag
    failures += diag_fail
    differs = any(int(r["rows_differ"]) for r in per.values()) or bool(
        diag and any(int(r["rows_differ"]) for r in diag.get("policies", {}).values()))
    body["failures"] = failures
    body["status"] = "failed" if failures else ("differs" if differs else "identical")
    write(report, body)
    print(f"offline {'equal' if body['offline']['equal'] else 'DIFFERS'} "  # type: ignore[index]
          f"rows {offline['rows_compared']}/{offline['rows_orig']}; code unchanged during run: "
          f"{body['code']['unchanged']}")  # type: ignore[index]
    for policy, res in per.items():
        print(f"{policy:15s} rows {res['rows_compared']}/{res['rows_orig']} identical "
              f"{res['rows_identical']} differ {res['rows_differ']} (week 0: {res['week0_differ']})")
    for line in failures:
        print(f"FAIL {line}")
    print(f"status {body['status']} -> {report}")
    return HALT if failures else 0


def run_audit(new_json: str, orig_json: str, report: str) -> int:
    """The ``audit`` subcommand: verdict, re-solves, phase share and cut counts, old against new."""
    new = json.loads((PILOT / new_json).read_text(encoding="utf-8"))
    old = json.loads((PILOT / orig_json).read_text(encoding="utf-8"))
    fields = ("eps", "phase_share", "n_replans", "advice_alone_equals_advice_in_tap", "cuts")
    seeds: Dict[str, Dict[str, object]] = {}
    for seed in sorted(set(new["seeds"]) | set(old["seeds"]), key=int):
        a, b = new["seeds"].get(seed, {}), old["seeds"].get(seed, {})
        diff = {f: {"orig": b.get(f), "new": a.get(f)} for f in fields if a.get(f) != b.get(f)}
        seeds[seed] = {"same": not diff, "diff": diff, "seconds": {"orig": b.get("seconds"),
                                                                    "new": a.get("seconds")}}
    same_all = new["verdict"] == old["verdict"] and all(s["same"] for s in seeds.values())
    body = {"new": new_json, "orig": orig_json, "verdict": {"orig": old["verdict"], "new": new["verdict"]},
            "seeds": seeds, "status": "identical" if same_all else "differs",
            "code": code_hash()}
    write(report, body)
    print(f"verdict {old['verdict']!r} -> {new['verdict']!r}; status {body['status']} -> {report}")
    return 0


def write(report: str, body: Dict[str, object]) -> None:
    """Write a report JSON (parents created)."""
    target = PILOT / report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=2, default=fnum) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stamp")
    st.add_argument("--out", required=True)
    cmp_ = sub.add_parser("compare")
    for flag in ("--new", "--orig", "--report"):
        cmp_.add_argument(flag, required=True)
    cmp_.add_argument("--policies", nargs="+", required=True)
    aud = sub.add_parser("audit")
    for flag in ("--new", "--orig", "--report"):
        aud.add_argument(flag, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "stamp":
        target = PILOT / args.out / STAMP
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(code_hash(), indent=2) + "\n", encoding="utf-8")
        print(f"code stamp -> {rel(target)}")
        return 0
    try:
        if args.cmd == "audit":
            return run_audit(args.new, args.orig, args.report)
        return run_compare(args.new, args.orig, args.policies, args.report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        write(args.report, {"new": args.new, "orig": args.orig, "status": "failed",
                            "failures": [f"check could not run: {exc!r}"],
                            "traceback": traceback.format_exc()})
        print(f"FAIL check could not run: {exc!r}")
        return CRASH


if __name__ == "__main__":
    raise SystemExit(main())
