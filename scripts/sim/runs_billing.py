"""Billing sensitivity table: ``results/runs/billing_sensitivity.md``.

    uv run python scripts/sim/runs_billing.py

Two ways of billing a replica start, both on the laptop's streams of cell C, seeds 10-14:

``order``  (sensitivity) a replica is billed from the order on: each start pays its whole lead
           time, ``beta = c d`` with ``d`` the slots the engine executes (instance tier 8, 72B
           replica tier 2); planners re-solve at that price and the baselines are retuned at it.
``main``   (the paper's) billed from the first serving minute plus one replica-minute per start,
           ``beta = c``.

Billed GPU-hours per week = ``(gpu_hours + starts * d / 60) / weeks`` (``d = 1`` under ``main``),
USD per week = ``(run_cost + switch_cost) / weeks``; both are means over the seeds, as is the
in-class attainment (whole run).  Rows under ``main`` come from the round's own laptop runs: TAP
from the R1 job, calendar planners and the floor-tuned baselines from the laptop original of the
same stream group (or the section-3 rerun where the resource rule chose another configuration),
the cost-tuned baselines and ``fixed`` from ``ref_main``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
OUT = PILOT / "outputs" / "validation_local" / "runs"
LOCAL = PILOT / "outputs" / "validation_local"
DOC = PILOT / "results" / "runs" / "billing_sensitivity.md"
CELL_C = (-8.7, 5.3, 4.0, 0.0, -8.7, 6.6)
OFFSET_KEYS = ("trigger_offset_mean", "trigger_offset_sd", "trigger_offset_course_sd",
               "trigger_offset_common_sd", "forecast_offset", "forecast_offset_sd")
BLOCKS: Tuple[Tuple[str, str, str, int, str], ...] = (
    ("c2_32b_shared4k", "inst", "32B, instance tier (d = 8)", 8, "lead_d8"),
    ("c3_72b_shared4k", "inst", "72B, instance tier (d = 8)", 8, "lead_d8"),
    ("c3_72b_shared4k", "rep", "72B, replica tier (1.5 min -> d = 2)", 2, "lead_d1.5"))


def rows(path: Path, policies: Dict[str, str]) -> List[Dict[str, object]]:
    """Per-policy whole-run means over seeds of cell C, relabelled by ``policies``."""
    if not path.exists():
        return []
    frame = pd.read_csv(path, float_precision="round_trip")
    cell = frame[list(OFFSET_KEYS)].apply(tuple, axis=1) == CELL_C
    frame = frame[cell & (frame["week"] == 0)]
    out = []
    for policy, label in policies.items():
        grp = frame[frame["policy"] == policy]
        if len(grp):
            out.append({"label": label, "source": str(path.relative_to(PILOT)), "frame": grp})
    return out


def measure(entry: Dict[str, object], d_slots: int) -> Dict[str, object]:
    """Billed GPU-h / week, USD / week and attainment of one row group."""
    grp: pd.DataFrame = entry["frame"]  # type: ignore[assignment]
    weeks = grp["weeks"].astype(float)
    billed = ((grp["gpu_hours"] + grp["starts"] * d_slots / 60.0) / weeks).mean()
    usd = ((grp["run_cost"] + grp["switch_cost"]) / weeks).mean()
    return {"label": entry["label"], "billed": float(billed), "usd": float(usd),
            "slo": float(grp["slo_in"].mean()), "slo_min": float(grp["slo_in"].min()),
            "seeds": int(len(grp)), "source": entry["source"]}


def floor_rows(case: str, tier: str, original: Path) -> List[Dict[str, object]]:
    """Floor-tuned baselines at beta = c: the section-3 rerun when it exists, else the original."""
    note = OUT / "tuning_resource" / f"{case}_{tier}" / "compare.json"
    rerun = OUT / "tuning_resource" / f"{case}_{tier}" / "eval" / "offset" / "offset.csv"
    if note.exists() and json.loads(note.read_text())["rerun"] and rerun.exists():
        return rows(rerun, {"reactive": "reactive (floor)", "velocity": "velocity (floor)"})
    return rows(original, {"reactive": "reactive (floor)", "velocity": "velocity (floor)"})


def block(case: str, tier: str, d_slots: int, lap: str) -> Tuple[List[Dict], List[Dict], List[str]]:
    """The ``order`` and ``main`` rows of one case x tier, plus notes on missing inputs."""
    root = OUT / "billing" / f"{case}_{tier}"
    base = {"tap": "TAP", "calendar_learn_prior": "calendar_learn_prior", "calendar": "calendar",
            "reactive": "reactive (cost)", "velocity": "velocity (cost)", "fixed": "fixed"}
    order = rows(root / "eval_cost" / "offset" / "offset.csv", base) + rows(
        root / "eval_floor" / "offset" / "offset.csv",
        {"reactive": "reactive (floor)", "velocity": "velocity (floor)"})
    original = LOCAL / "sensitivity_sweeps" / f"{case}_{lap}" / "offset" / "offset.csv"
    main = (rows(OUT / "tap" / f"r1_{case[:2]}_{tier}" / "offset" / "offset.csv", {"tap": "TAP"})
            + rows(original, {"calendar_learn_prior": "calendar_learn_prior", "calendar": "calendar"})
            + rows(root / "ref_main" / "offset" / "offset.csv",
                   {"reactive": "reactive (cost)", "velocity": "velocity (cost)", "fixed": "fixed"})
            + floor_rows(case, tier, original))
    missing = [label for label in (*base.values(), "reactive (floor)", "velocity (floor)")
               if label not in {r["label"] for r in order}]
    return ([measure(r, d_slots) for r in order], [measure(r, 1) for r in main],
            [f"order: missing {', '.join(missing)}"] if missing else [])


def table(measured: List[Dict[str, object]], title: str) -> List[str]:
    """Markdown rows with the percentage against TAP."""
    tap = next((m for m in measured if m["label"] == "TAP"), None)
    lines = [f"**{title}**", "", "| policy | billed GPU-h / week | USD / week | vs TAP | in-class SLO "
             "(mean) | 3 dp | lowest seed | seeds | source |", "|---|---|---|---|---|---|---|---|---|"]
    for m in measured:
        rel = f"{(m['billed'] / tap['billed'] - 1) * 100:+.1f} %" if tap and m is not tap else "-"
        lines.append(f"| {m['label']} | {m['billed']:.2f} | {m['usd']:.2f} | {rel} | {m['slo']!r} | "
                     f"{m['slo']:.3f} | {m['slo_min']:.4f} | {m['seeds']} | `{m['source']}` |")
    return lines + [""]


def main(argv: Optional[List[str]] = None) -> int:
    """Write the table; a missing input is reported in the file, not silently skipped."""
    out = [f"# Billing from the order vs from the first serving minute ({pd.Timestamp.now():%Y-%m-%d})",
           "", (__doc__ or "").split("\n\n", 1)[1].strip(), ""]
    for case, tier, title, d_slots, lap in BLOCKS:
        order, main_rows, notes = block(case, tier, d_slots, lap)
        out += [f"## {title}", ""]
        out += table(order, f"Billed from the order (beta = c x {d_slots}, retuned baselines)")
        out += table(main_rows, "Main caliber (beta = c), same laptop streams")
        out += [f"- {n}" for n in notes] + ([""] if notes else [])
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
