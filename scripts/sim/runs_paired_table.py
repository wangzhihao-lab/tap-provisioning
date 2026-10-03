"""Cell C on the laptop's streams, every central policy next to TAP.

    uv run python scripts/sim/runs_paired_table.py

Collects, per case x tier, the cell-C rows (seeds 10-14, six weeks) of every policy variant from
runs made on this laptop only: TAP from the R1 job, the rest from the laptop originals of the same
stream group, from the ``p1_*`` paired runs and from section 3's rerun.  Every source must carry
offline rows equal to the R1 job's, which is checked here again, so the table is paired throughout.
A reactive / velocity variant is labelled by the ``best.json`` it ran with.  "floor (dispersed
slice)" is section 3's selection (resource rule on the laptop's tuning week): both rules from its
rerun when its ``compare.json`` says a configuration changed, where a rule it kept must have the
laptop original's rows, and from the laptop original otherwise.  Where a configuration changed, the
laptop original's rows stay as "floor (dispersed slice, previous selection)", the configuration the
penalty-inclusive cost ratio selected on the cloud's tuning week.

Writes ``results/runs/paired_cell_c.csv`` (long: whole run and each week) and
``paired_cell_c.md`` (whole run: GPU-hours per week and in-class attainment, unrounded).
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
DOCS = PILOT / "results" / "runs"
KEYS = ("trigger_offset_mean", "trigger_offset_sd", "trigger_offset_course_sd",
        "trigger_offset_common_sd", "forecast_offset", "forecast_offset_sd")
CELL_C = (-8.7, 5.3, 4.0, 0.0, -8.7, 6.6)
FLOOR = "floor (dispersed slice)"
PREVIOUS = "floor (dispersed slice, previous selection)"
PREVIOUS_NOTE = ("laptop original; previous selection: penalty-inclusive cost ratio on the cloud "
                 "tuning week")
FEEDBACK: Tuple[str, ...] = ("reactive", "velocity")  # the rules the floor tuning configures
GROUPS: Tuple[Tuple[str, str, str], ...] = (("c2_32b_shared4k", "inst", "lead_d8"),
                                            ("c2_32b_shared4k", "rep", "lead_d1"),
                                            ("c3_72b_shared4k", "inst", "lead_d8"),
                                            ("c3_72b_shared4k", "rep", "lead_d1.5"))


def cell_c(path: Path) -> pd.DataFrame:
    """Cell-C rows of a sweep CSV (floats parsed exactly)."""
    frame = pd.read_csv(path, float_precision="round_trip")
    return frame[frame[list(KEYS)].apply(tuple, axis=1) == CELL_C]


def original(case: str, lead: str) -> Path:
    """The laptop original of a case x tier (the floor variants of the cloud's selection)."""
    return LOCAL / "sensitivity_sweeps" / f"{case}_{lead}" / "offset" / "offset.csv"


def rerun(case: str, tier: str) -> Path:
    """Section 3's six-week rerun of the floor variants."""
    return OUT / "tuning_resource" / f"{case}_{tier}" / "eval" / "offset" / "offset.csv"


def decision(case: str, tier: str) -> Optional[Dict[str, object]]:
    """Section 3's ``compare.json`` of a case x tier, or None if it is not there."""
    path = OUT / "tuning_resource" / f"{case}_{tier}" / "compare.json"
    return json.loads(path.read_text()) if path.exists() else None


def floor_sources(case: str, tier: str, lead: str) -> List[Tuple[Path, Dict[str, str], str]]:
    """The floor variants of the dispersed tuning: section 3's selection and, where it changed a
    configuration, the previous selection (nothing while section 3 has not decided)."""
    decided = decision(case, tier)
    if decided is None:
        return []
    if not decided["rerun"]:
        return [(original(case, lead), {p: FLOOR for p in FEEDBACK},
                 "laptop original; section-3 resource rule kept the configuration")]
    changed = [p for p in FEEDBACK if p in decided["changed"]]  # type: ignore[operator]
    out = [(rerun(case, tier), {p: FLOOR}, "section-3 resource rule (rerun; configuration "
            + ("changed)" if p in changed else "kept, rows equal the laptop original's)"))
           for p in FEEDBACK]
    if changed:
        out.append((original(case, lead), {p: PREVIOUS for p in changed}, PREVIOUS_NOTE))
    return out


def kept_rows_differ(case: str, tier: str, lead: str) -> List[str]:
    """Problems: a rule whose configuration section 3 kept has other cell-C rows in the rerun than
    in the laptop original (every column both carry)."""
    decided = decision(case, tier)
    orig, new = original(case, lead), rerun(case, tier)
    if decided is None or not decided["rerun"] or not (orig.exists() and new.exists()):
        return []  # nothing to compare, or a missing file collect() reports
    a, b = cell_c(orig), cell_c(new)
    cols = [c for c in a.columns if c in b.columns and c != "sweep"]
    problems: List[str] = []
    for spec in (p for p in FEEDBACK if p not in decided["changed"]):  # type: ignore[operator]
        x = a.loc[a["policy"] == spec, cols].sort_values(["seed", "week"]).reset_index(drop=True)
        y = b.loc[b["policy"] == spec, cols].sort_values(["seed", "week"]).reset_index(drop=True)
        if x.empty or x.shape != y.shape or not ((x == y) | (x.isna() & y.isna())).all().all():
            problems.append(f"{spec}: section 3 kept its configuration, but the rows of "
                            f"{new.relative_to(PILOT)} differ from {orig.relative_to(PILOT)}")
    return problems


def sources(case: str, tier: str, lead: str) -> List[Tuple[Path, Dict[str, str], str]]:
    """``(csv, {policy spec: variant label}, note)`` of every laptop run of one case x tier."""
    short = case[:2]
    out: List[Tuple[Path, Dict[str, str], str]] = [
        (OUT / "tap" / f"r1_{short}_{tier}" / "offset" / "offset.csv", {"tap": "-"}, "R1 job"),
        (original(case, lead), {"offline": "-", "calendar": "-", "calendar_learn_prior": "-"},
         "laptop original")]
    if tier == "inst":
        out.append((LOCAL / "priority_ablation" / f"{case}_on" / "offset" / "offset.csv",
                    {"calendar_adapt": "-"}, "laptop original"))
    out += floor_sources(case, tier, lead)
    for run in sorted((OUT / "paired" / f"{case}_{tier}").glob("*/offset/offset.csv")):
        tag = run.parents[1].name
        specs = pd.read_csv(run, usecols=["policy"])["policy"].unique()
        label = {"tuning8_r30": "cost (tier's own)", "tuning_r30": "cost (replica-tier tuning)",
                 "tuning_r30_slo95": "floor (synchronous slice)"}.get(tag, tag)
        out.append((run, {p: (label if p in ("reactive", "velocity") else "-") for p in specs
                          if p != "offline"}, f"p1 run, --tuned {tag}"))
    return out


def collect(case: str, tier: str, lead: str) -> Tuple[pd.DataFrame, List[str]]:
    """Long table of one case x tier and the problems found on the way."""
    problems: List[str] = kept_rows_differ(case, tier, lead)
    if decision(case, tier) is None:
        problems.append("no section-3 compare.json: the floor variants of the dispersed tuning "
                        "are left out")
    rows: List[pd.DataFrame] = []
    anchor: Optional[pd.DataFrame] = None
    for path, wanted, note in sources(case, tier, lead):
        if not path.exists():
            problems.append(f"missing {path.relative_to(PILOT)}")
            continue
        frame = cell_c(path)
        off = frame[frame["policy"] == "offline"].sort_values(["seed", "week"]).reset_index(drop=True)
        cols = ["seed", "week", "gpu_hours", "starts", "slo_in", "n_in", "mu", "total_cost"]
        if anchor is None:
            anchor = off[cols]
        elif not off[cols].equals(anchor):
            problems.append(f"offline rows of {path.relative_to(PILOT)} differ from the R1 job's")
            continue
        for spec, variant in wanted.items():
            part = frame[frame["policy"] == spec]
            if part.empty:
                problems.append(f"{spec} not in {path.relative_to(PILOT)}")
                continue
            rows.append(part.assign(variant=variant, source=str(path.relative_to(PILOT)), note=note))
    if not rows:
        return pd.DataFrame(), problems
    frame = pd.concat(rows, ignore_index=True)
    per = frame["weeks"].where(frame["week"] == 0, 1).astype(float)  # whole-run rows -> per week
    frame = frame.assign(gpu_h_per_week=frame["gpu_hours"] / per, starts_per_week=frame["starts"] / per,
                         usd_per_week=(frame["run_cost"] + frame["switch_cost"]) / per)
    agg = (frame.groupby(["policy", "variant", "week", "source", "note"], as_index=False)
           .agg(seeds=("seed", "nunique"), gpu_h_per_week=("gpu_h_per_week", "mean"),
                starts_per_week=("starts_per_week", "mean"), usd_per_week=("usd_per_week", "mean"),
                slo_in=("slo_in", "mean"), slo_in_min=("slo_in", "min")))
    agg.insert(0, "tier", tier)
    agg.insert(0, "case", case)
    return agg, problems


def main() -> int:
    """Write the CSV and the markdown table; report what is missing rather than skip it."""
    parts, lines = [], ["# Cell C on the laptop's streams: every central policy next to TAP", "",
                        (__doc__ or "").split("\n\n")[2].strip(), ""]
    for case, tier, lead in GROUPS:
        agg, problems = collect(case, tier, lead)
        lines += [f"## {case}, {'instance tier (d = 8)' if tier == 'inst' else 'replica tier'}", ""]
        if not agg.empty:
            parts.append(agg)
            whole = agg[agg["week"] == 0]
            tap = whole.loc[whole["policy"] == "tap", "gpu_h_per_week"]
            lines += ["| policy | variant | GPU-h / week | vs TAP | in-class SLO (mean of seeds) | lowest "
                      "seed | starts / week | seeds | source |", "|---|---|---|---|---|---|---|---|---|"]
            for r in whole.sort_values("gpu_h_per_week").itertuples(index=False):
                rel = f"{(r.gpu_h_per_week / float(tap.iloc[0]) - 1) * 100:+.2f} %" if len(tap) else "-"
                lines.append(f"| {r.policy} | {r.variant} | {r.gpu_h_per_week!r} | {rel} | {r.slo_in!r} | "
                             f"{r.slo_in_min!r} | {r.starts_per_week:.2f} | {r.seeds} | `{r.source}` |")
        lines += [f"- {p}" for p in problems] + [""]
    DOCS.mkdir(parents=True, exist_ok=True)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(DOCS / "paired_cell_c.csv", index=False)
    (DOCS / "paired_cell_c.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOCS / 'paired_cell_c.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
