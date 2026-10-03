"""Week-by-week seat-start profiles of the school timetable.

Reads the slot table written by ``run_t2.py`` and, for every weekday bell, builds the
seat-starts S_w(bell) of teaching week w = 1..18.  Two outputs under ``results/t2/``:

* ``between_week_dispersion.csv``  per-bell median / max / min over weeks 5-16, the largest
  absolute deviation from the median (A^Delta), the largest week-to-week step, and the
  week 1/4/17/18 values (the freshman step lands in week 5).
* ``week_to_week_error.csv``  per week: total seat-starts and the summed absolute error over
  bells of two calendar-oblivious predictors, previous-week (seasonal naive) and the
  weeks 5-16 median profile.  A calendar-aware predictor has error 0 on this table by
  construction; the cross-semester (stale calendar) error is in ``cross_semester.md``.

Run from the repository root: ``python scripts/t2/week_dynamics.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SLOTS = ROOT / "data" / "timetable" / "slots_2026-2027-1.csv"
OUT_DIR = ROOT / "results" / "t2"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
BELLS = {1: "08:30", 3: "10:20", 5: "14:00", 7: "15:50", 9: "18:40"}  # period_start -> bell
N_WEEKS = 18
STEADY = range(5, 17)  # weeks 5..16 inclusive


def parse_weeks(spec: str) -> Set[int]:
    """Expand a week specification such as ``1-18`` or ``1-2,4-18`` into a set."""
    weeks: Set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            weeks.update(range(int(lo), int(hi) + 1))
        else:
            weeks.add(int(part))
    return weeks


def weekly_profiles(slots: pd.DataFrame) -> pd.DataFrame:
    """Return a (bell x week) table of seat-starts, bells labelled ``Mon 08:30`` etc."""
    rows: Dict[str, List[float]] = {}
    for wd_i, wd in enumerate(WEEKDAYS, start=1):
        for period, bell in BELLS.items():
            sub = slots[(slots["weekday"] == wd_i) & (slots["period_start"] == period)]
            series = [0.0] * N_WEEKS
            for _, row in sub.iterrows():
                seats = float(row["enrollment"])
                for w in parse_weeks(row["weeks"]):
                    if 1 <= w <= N_WEEKS:
                        series[w - 1] += seats
            rows[f"{wd} {bell}"] = series
    table = pd.DataFrame(rows, index=[f"w{w}" for w in range(1, N_WEEKS + 1)]).T
    return table


def dispersion(table: pd.DataFrame) -> pd.DataFrame:
    """Per-bell steady-state statistics (weeks 5-16) plus the regime-boundary weeks."""
    steady_cols = [f"w{w}" for w in STEADY]
    st = table[steady_cols]
    med = st.median(axis=1)
    out = pd.DataFrame(
        {
            "median_w5_16": med,
            "max_w5_16": st.max(axis=1),
            "min_w5_16": st.min(axis=1),
            "max_abs_dev_w5_16": (st.sub(med, axis=0)).abs().max(axis=1),
            "max_step_w5_16": st.diff(axis=1).abs().max(axis=1),
            "range_w5_16": st.max(axis=1) - st.min(axis=1),
            "week1": table["w1"],
            "week4": table["w4"],
            "week17": table["w17"],
            "week18": table["w18"],
        }
    )
    out["rel_dev"] = (out["max_abs_dev_w5_16"] / out["median_w5_16"]).round(3)
    return out


def week_errors(table: pd.DataFrame) -> pd.DataFrame:
    """Summed absolute error over bells of two calendar-oblivious predictors, per week."""
    steady_cols = [f"w{w}" for w in STEADY]
    med = table[steady_cols].median(axis=1)
    rows = []
    for w in range(1, N_WEEKS + 1):
        cur = table[f"w{w}"]
        prev = table[f"w{w - 1}"] if w > 1 else cur * 0.0
        rows.append(
            {
                "week": w,
                "total_seat_starts": cur.sum(),
                "err_prev_week": (cur - prev).abs().sum(),
                "err_median_profile": (cur - med).abs().sum(),
                "worst_bell_prev_week": (cur - prev).abs().idxmax(),
                "worst_bell_abs_prev_week": (cur - prev).abs().max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    slots = pd.read_csv(SLOTS)
    table = weekly_profiles(slots)
    disp = dispersion(table)
    errs = week_errors(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    disp.to_csv(OUT_DIR / "between_week_dispersion.csv", index_label="bell")
    errs.to_csv(OUT_DIR / "week_to_week_error.csv", index=False)
    table.to_csv(OUT_DIR / "weekly_bell_profiles.csv", index_label="bell")
    a_max = disp["median_w5_16"].max()
    a_delta = disp["max_abs_dev_w5_16"].max()
    logger.info("A_max(typ) = %.0f at %s", a_max, disp["median_w5_16"].idxmax())
    logger.info("A^Delta = %.0f at %s (%.3f of A_max)", a_delta, disp["max_abs_dev_w5_16"].idxmax(), a_delta / a_max)
    logger.info("rel_dev median %.3f, p90 %.3f, max %.3f", disp["rel_dev"].median(), disp["rel_dev"].quantile(0.9), disp["rel_dev"].max())
    med_sum = disp["median_w5_16"].sum()
    a_fam = disp["max_w5_16"].max()
    logger.info("A (family bound, max over weeks 5-16) = %.0f at %s; sum of bell medians = %.0f; P2M_bell = %.3f (typ %.3f)",
                a_fam, disp["max_w5_16"].idxmax(), med_sum, len(disp) * a_fam / med_sum, len(disp) * a_max / med_sum)
    logger.info("R2 budgets: sum range = %.0f (%.3f of sum medians), sum max_abs_dev = %.0f (%.3f); max range %.0f at %s",
                disp["range_w5_16"].sum(), disp["range_w5_16"].sum() / med_sum, disp["max_abs_dev_w5_16"].sum(),
                disp["max_abs_dev_w5_16"].sum() / med_sum, disp["range_w5_16"].max(), disp["range_w5_16"].idxmax())
    w4, w5 = errs.loc[errs["week"] == 4, "total_seat_starts"].item(), errs.loc[errs["week"] == 5, "total_seat_starts"].item()
    logger.info("week 4 -> 5 step: %.0f -> %.0f (%+.1f%%); week-1 peak bell %.0f at %s",
                w4, w5, 100 * (w5 / w4 - 1), table["w1"].max(), table["w1"].idxmax())
    logger.info("\n%s", errs.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))


if __name__ == "__main__":
    main()
