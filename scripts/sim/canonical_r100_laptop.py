"""Canonical numbers, part 6b: TAP at r = 100 against r = 30 on the laptop's streams (section 33).

TAP and the offline optimum of every R7 job (``runs/tap/r7_<case>_<tier>``: cells A-F at
r = 100) next to the R1 job of the same case and tier, read as ``canonical_cells_laptop`` reads
them.  The two jobs are set side by side only if their run commands differ in ``--p-in`` and
``--out`` alone and their offline rows agree in every stream column (``runs_check.STREAM_COLS``
without the prices) on the same (timing cell, seed, week) keys.  No laptop baseline was run at
r = 100: every other laptop sweep at r = 100 is listed with the reason it does not pair, and one
that paired with an R7 job would have to be added as a source, so finding one raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from canonical_cells_laptop import (TapJob, job_argv, job_stats, platform_of, sweep_frame,
                                    tap_jobs, tap_row, to_end)
from canonical_sensitivity import KEYS, LOCAL, OFFSET_KEYS, TAP_JOBS, Run, check_offline

# The R1 boundary check (2026-09-25) re-ran archived jobs into this folder; its outputs are copies of
# runs already listed elsewhere, not runs of their own, so no scan below may count them.
REPLAN_CHECK = f"{LOCAL}/runs/replan_check"

PRICE_COLUMNS: Tuple[str, ...] = ("p_in", "p_off", "price_ratio_real")


def same_streams(repo: Path, a: TapJob, b: TapJob) -> None:
    """Raise unless two jobs ran the same command but for the price and the output folder, and
    their offline rows agree in every column the price does not move."""
    from runs_check import STREAM_COLS  # scripts/sim
    from runs_groups import drop_flags  # scripts/sim

    x_argv, y_argv = (drop_flags(job_argv(repo, j.name), ["--p-in", "--out"]) for j in (a, b))
    if x_argv != y_argv:
        raise ValueError(f"{a.name} and {b.name}: the run commands differ beyond --p-in and --out")

    cols = [c for c in STREAM_COLS if c not in PRICE_COLUMNS]
    x, y = (j.frame[j.frame["policy"] == "offline"] for j in (a, b))
    m = x.merge(y, on=list(KEYS), how="outer", suffixes=("", "__other"), indicator=True)
    bad = [c for c in cols if not ((m[c] == m[f"{c}__other"])
                                   | (m[c].isna() & m[f"{c}__other"].isna())).all()]
    if bad or not (m["_merge"] == "both").all():
        raise ValueError(f"{a.folder} and {b.folder}: other streams (keys, or {bad} differ)")


def r100_candidates(repo: Path, jobs: Sequence[TapJob]) -> pd.DataFrame:
    """Every laptop sweep at r = 100 besides the TAP jobs and why it does not pair; one that
    pairs with a TAP job raises (it would have to be added as a source)."""
    own = {j.folder for j in jobs}
    rows: List[Dict[str, object]] = []
    for path in sorted((repo / LOCAL).glob("**/*.csv")):
        folder = str(path.parents[1].relative_to(repo))
        if path.parent.name != path.stem or folder in own or folder.startswith(REPLAN_CHECK):
            continue  # sweep outputs are <run>/<sweep>/<sweep>.csv; replan_check holds copies
        head = pd.read_csv(path, nrows=0).columns
        if "price_ratio_real" not in head or "policy" not in head:
            continue
        if not (pd.read_csv(path, usecols=["price_ratio_real"])["price_ratio_real"].round()
                == 100).any():
            continue
        frame = pd.read_csv(path, float_precision="round_trip")
        csv = str(path.relative_to(repo))
        if any(k not in head for k in OFFSET_KEYS):
            verdict = "no trigger-offset columns: no timing cell to pair"
        else:
            frame = sweep_frame(repo, csv)
            for job in jobs:
                base = job.frame[job.frame["cell"].isin(set(frame["cell"]))]
                try:
                    check_offline(Run(job.case, job.name, TAP_JOBS, job.name), base, {csv: frame})
                except ValueError:
                    continue
                raise ValueError(f"{csv} pairs with {job.name}: add it to section 33's sources")
            verdict = "offline rows equal no R7 job's in any cell"
        rows.append({"run": csv, "sweep": path.stem, "platform": platform_of(folder),
                     "price_ratio": ", ".join(f"{v:g}" for v in sorted(set(
                         frame["price_ratio_real"].round()))),
                     "seeds": ", ".join(str(int(s)) for s in sorted(frame["seed"].unique())),
                     "policies": ", ".join(sorted(set(frame["policy"]) - {"offline"})),
                     "verdict": verdict})
    return pd.DataFrame(rows)


def ratio100_tables(repo: Path, labels: Mapping[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Section 33: TAP of every R7 job next to the R1 job of its case and tier, per cell, and the
    laptop sweeps at r = 100 that could have held a baseline."""
    r30 = {(j.case, j.tier): j for j in tap_jobs(repo, "r1")}
    r100 = tap_jobs(repo, "r7")
    rows: List[Dict[str, object]] = []
    for job in r100:
        base: Optional[TapJob] = r30.get((job.case, job.tier))
        if base is None:
            raise ValueError(f"{job.name}: no R1 job of the same case and tier")
        same_streams(repo, job, base)
        at100, at30 = job_stats(job), job_stats(base)
        for cell in at100:
            a, b = at100[cell]["tap"], at30[cell]["tap"]
            for j, by in ((job, at100[cell]), (base, at30[cell])):
                row = tap_row(j, cell, by, labels)
                ratio = ", ".join(f"{v:g}" for v in
                                  sorted(set(j.frame["price_ratio_real"].round())))
                row = {**{k: row[k] for k in ("case", "tier", "lead_min", "cell")}, "r": ratio,
                       **row}
                if j is job:
                    row.update({"tap_gpu_h_wk_vs_r30_pct": f"{100.0 * (a.gpu / b.gpu - 1.0):+.1f}",
                                "slo_in_vs_r30": f"{a.slo_in - b.slo_in:+.4f}"})
                rows.append(row)
    return to_end(pd.DataFrame(rows)), r100_candidates(repo, r100)


__all__ = ["r100_candidates", "ratio100_tables"]
