"""Summary of the replan-boundary check (``replan_check/``), readable while the queue still runs.

    uv run python scripts/sim/replan_check_summary.py

Reads ``replan_check/runs.json`` (the job list ``replan_check_jobs.py`` wrote), the runner's
markers in ``replan_check/logs`` and every finished job's ``report.json``, and writes
``results/runs/replan_check.md`` plus two tables next to it: ``replan_check.csv`` (one row
per run and affected policy, the audits last) and ``replan_check_rows.csv`` (every differing row).
A job without a DONE or FAILED marker is ``pending`` (``running`` when its log has a START line and
no EXIT); a DONE job reports its compare status (identical / differs / failed).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
REPO = PILOT
CHECK = PILOT / "outputs" / "validation_local" / "runs" / "replan_check"
LOGS = CHECK / "logs"
DOCS = REPO / "results" / "runs"
ROW_COLS = ("gpu_hours_per_week", "starts_per_week", "slo_in", "slo_off", "total_cost", "cost_ratio")
MAXES = ("gpu_hours_per_week", "starts_per_week", "slo_in", "slo_off", "total_cost", "cost_ratio")


def job_state(job: Dict[str, object]) -> Tuple[str, Optional[Dict[str, object]]]:
    """``(state, report)``: pending / running / failed / the report's status."""
    job_id = str(job["job_id"])
    report_path = PILOT / str(job["report"])
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    done, failed = LOGS / f"{job_id}.DONE", LOGS / f"{job_id}.FAILED"
    if failed.exists() and (not done.exists() or failed.stat().st_mtime > done.stat().st_mtime):
        return "failed", report  # a re-run after a failure leaves the old marker: the newer one counts
    if done.exists():
        return (str(report["status"]) if report else "failed"), report
    log = LOGS / f"{job_id}.log"
    running = log.exists() and "START " in log.read_text(errors="replace") and \
        "EXIT " not in log.read_text(errors="replace")
    return ("running" if running else "pending"), None


def fmt(value: object, digits: int = 4) -> str:
    """Compact number for the markdown tables."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def run_rows(job: Dict[str, object], state: str, report: Optional[Dict[str, object]]
             ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Summary rows (one per affected policy) and differing rows of one run job."""
    summary: List[Dict[str, object]] = []
    detail: List[Dict[str, object]] = []
    affected = (report or {}).get("affected", {})
    diag = ((report or {}).get("diagnostics") or {}).get("policies", {})
    for policy in job["policies"]:  # type: ignore[union-attr]
        res = affected.get(policy) if report else None
        row: Dict[str, object] = {"group": job["group"], "run": job["name"], "policy": policy,
                                  "week0_runs": job["week0_runs"].get(policy),  # type: ignore[union-attr]
                                  "status": state}
        if res is not None and state in ("identical", "differs", "failed"):
            dres = diag.get(policy)
            differ = int(res["rows_differ"]) + (int(dres["rows_differ"]) if dres else 0)
            row.update({"status": state if state == "failed" else ("differs" if differ else "identical"),
                        "rows_compared": res["rows_compared"], "rows_identical": res["rows_identical"],
                        "rows_differ": res["rows_differ"], "week0_compared": res["week0_compared"],
                        "week0_differ": res["week0_differ"],
                        "diag_rows": dres["rows_compared"] if dres else None,
                        "diag_differ": dres["rows_differ"] if dres else None,
                        "offline_equal": report["offline"]["equal"],  # type: ignore[index]
                        "code_unchanged": report["code"]["unchanged"]})  # type: ignore[index]
            for col in MAXES:
                stats = res["diff_columns"].get(col)
                row[f"max_abs_{col}"] = stats["max_abs"] if stats else 0.0
            tc = res["diff_columns"].get("total_cost")
            row["max_rel_total_cost"] = tc["max_rel"] if tc else 0.0
            for rec in res["differing_rows"]:
                item = {"group": job["group"], "run": job["name"], "policy": policy, "kind": "sweep row",
                        "cell": rec["cell"], "seed": rec.get("seed"), "week": rec.get("week"),
                        "n_columns": len(rec["columns_differing"])}
                for col in ROW_COLS:
                    v = rec["values"].get(col, {})
                    item.update({f"{col}_orig": v.get("orig"), f"{col}_new": v.get("new"),
                                 f"{col}_diff": v.get("diff")})
                detail.append(item)
            for rec in (dres or {}).get("differing_rows", []):
                item = {"group": job["group"], "run": job["name"], "policy": policy,
                        "kind": "diagnostics", "cell": rec["cell"], "seed": rec.get("seed"),
                        "n_columns": len(rec["columns_differing"])}
                for col in ("phase_share", "n_replans", "final_ratio"):
                    v = rec["values"].get(col, {})
                    item.update({f"{col}_orig": v.get("orig"), f"{col}_new": v.get("new")})
                detail.append(item)
        if state == "failed" and report:
            row["note"] = "; ".join(report.get("failures", []))
        summary.append(row)
    return summary, detail


def audit_row(job: Dict[str, object], state: str, report: Optional[Dict[str, object]]) -> Dict[str, object]:
    """One summary row of a TAP prefix audit."""
    row: Dict[str, object] = {"group": "audit", "run": job["name"], "policy": "tap", "status": state}
    if report and state in ("identical", "differs"):
        seeds = report["seeds"]
        row.update({"rows_compared": len(seeds),
                    "rows_differ": sum(not s["same"] for s in seeds.values()),
                    "note": f"verdict {report['verdict']['orig']!r} -> {report['verdict']['new']!r}"})
        diffs = {k: s["diff"] for k, s in seeds.items() if not s["same"]}
        if diffs:
            row["note"] = f"{row['note']}; changed seeds {json.dumps(diffs)[:400]}"
    return row


def markdown(summary: pd.DataFrame, detail: pd.DataFrame, jobs: List[Dict[str, object]],
             states: Counter, stamps: Counter) -> str:
    """The markdown report."""
    runs = summary[summary["group"] != "audit"]
    lines = [
        "# Replan-boundary check (R1): archived runs re-run on the fixed code",
        "",
        f"Written {datetime.now():%Y-%m-%d %H:%M} by `scripts/sim/replan_check_summary.py` from "
        "`outputs/validation_local/runs/replan_check/` (job list `JOBS.txt`, `runs.json`; "
        "reports `<group>/<run>/report.json`).",
        "",
        f"Jobs: {len(jobs)}; " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())) + ".",
        "Simulator source hash of the finished jobs (start of run): "
        + (", ".join(f"`{h[:16]}` x{n}" for h, n in stamps.items()) or "none yet") + ".",
        "",
        "Each job repeats the archived `run_sweep.py` command with `--policies` cut to the affected policies",
        "plus `offline` and `--out` under `replan_check/`; offline rows must reproduce exactly (else the job",
        "fails and the queue halts); every row of an affected policy (every cell/error cell x seed x week",
        "0-6, all columns) is compared exactly. Originals of 2026-09-21 (`f6_32b`, `priority_ablation`)",
        "predate other code changes as well, so a difference there is not attributable to R1 alone.",
        "",
        "## Whole-run policy runs (week 0)",
        "",
        "| policy | archived | compared | identical | differ | pending runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, grp in runs.groupby("policy"):
        done = grp[grp["status"].isin(["identical", "differs"])]
        comp = int(done["week0_compared"].fillna(0).sum()) if "week0_compared" in done else 0
        diff = int(done["week0_differ"].fillna(0).sum()) if "week0_differ" in done else 0
        lines.append(f"| {policy} | {int(grp['week0_runs'].sum())} | {comp} | {comp - diff} | {diff} | "
                     f"{int(grp.loc[~grp['status'].isin(['identical', 'differs']), 'week0_runs'].sum())} |")
    lines += ["", "## Per run", "",
              "| group | run | policy | status | rows | differ | wk-0 differ | max abs GPU-h/wk | "
              "max abs starts/wk | max abs slo_in | max rel total_cost | TAP diag differ | note |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary.to_dict("records"):
        lines.append(f"| {r['group']} | {r['run']} | {r['policy']} | {r['status']} | {fmt(r.get('rows_compared'))} | "
                     f"{fmt(r.get('rows_differ'))} | {fmt(r.get('week0_differ'))} | "
                     f"{fmt(r.get('max_abs_gpu_hours_per_week'))} | {fmt(r.get('max_abs_starts_per_week'))} | "
                     f"{fmt(r.get('max_abs_slo_in'))} | {fmt(r.get('max_rel_total_cost'))} | "
                     f"{fmt(r.get('diag_differ'))} | {fmt(r.get('note'))} |")
    lines += ["", "## Differing rows", ""]
    if detail.empty:
        lines.append("None so far.")
    else:
        lines += ["| group | run | policy | kind | cell | seed | week | GPU-h/wk old -> new | starts/wk old -> new | "
                  "slo_in old -> new | slo_off old -> new | total_cost old -> new | re-solves old -> new |",
                  "|---|---|---|---|---|---:|---:|---|---|---|---|---|---|"]
        for r in detail.to_dict("records"):
            def arrow(col: str) -> str:
                return f"{fmt(r.get(f'{col}_orig'), 6)} -> {fmt(r.get(f'{col}_new'), 6)}"
            lines.append(f"| {r['group']} | {r['run']} | {r['policy']} | {r['kind']} | {r['cell']} | "
                         f"{fmt(r.get('seed'))} | {fmt(r.get('week'))} | {arrow('gpu_hours_per_week')} | "
                         f"{arrow('starts_per_week')} | {arrow('slo_in')} | {arrow('slo_off')} | "
                         f"{arrow('total_cost')} | {arrow('n_replans')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Entry point."""
    manifest = json.loads((CHECK / "runs.json").read_text(encoding="utf-8"))
    jobs: List[Dict[str, object]] = manifest["jobs"]
    summary: List[Dict[str, object]] = []
    detail: List[Dict[str, object]] = []
    states: Counter = Counter()
    stamps: Counter = Counter()
    for job in jobs:
        state, report = job_state(job)
        states[state] += 1
        start = ((report or {}).get("code") or {}).get("start") if job["kind"] == "run" else None
        if start:
            stamps[start["tree_sha256"]] += 1
        if job["kind"] == "audit":
            summary.append(audit_row(job, state, report))
            continue
        rows, more = run_rows(job, state, report)
        summary += rows
        detail += more
    frame, rows_frame = pd.DataFrame(summary), pd.DataFrame(detail)
    DOCS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DOCS / "replan_check.csv", index=False)
    rows_frame.to_csv(DOCS / "replan_check_rows.csv", index=False)
    (DOCS / "replan_check.md").write_text(markdown(frame, rows_frame, jobs, states, stamps), encoding="utf-8")
    print(f"{len(jobs)} jobs: {dict(states)}; {len(rows_frame)} differing rows -> "
          f"{(DOCS / 'replan_check.md').relative_to(REPO)} (+ .csv, _rows.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
