"""Aggregate the timetable into the load statistics T2 and F1 need.

Everything here is an aggregate over sections; no personal column ever enters these frames.
The headline quantity for the paper is ``same_bell``: how many students start a class at the same
bell, on a typical weekday, across all campuses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .parse import BellSchedule, TimetableConfig

logger = logging.getLogger(__name__)

ALL_CAMPUS = "ALL"


@dataclass(frozen=True)
class StatsConfig:
    """Windows and thresholds of the T2 statistics."""

    typical_lo: int = 5
    typical_hi: int = 16
    teaching_weekdays: Tuple[int, ...] = (1, 2, 3, 4, 5)
    teaching_periods: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    top_k: Tuple[int, ...] = (5, 10)
    quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90, 0.99)


@dataclass(frozen=True)
class T2Result:
    """Every frame and scalar the T2 table and the F1 panels are built from."""

    slot_load: pd.DataFrame
    typical_week: pd.DataFrame
    week_totals: pd.DataFrame
    session_starts: pd.DataFrame
    same_bell: pd.DataFrame
    n_periods: pd.DataFrame
    enrollment: Dict[str, float]
    concentration: Dict[str, float]
    counts: Dict[str, int]
    peak: Dict[str, float] = field(default_factory=dict)


def _with_all_campus(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Append campus-agnostic rows (campus == ``ALL``) to a per-campus aggregate."""
    total = frame.drop(columns=["campus"]).groupby(list(keys), as_index=False).sum(numeric_only=True)
    total.insert(0, "campus", ALL_CAMPUS)
    return pd.concat([frame, total], ignore_index=True)


def _dedup_cells(week_rows: pd.DataFrame, period_column: str) -> pd.DataFrame:
    """One row per (campus, weekday, period, week, section): a section is counted once per cell."""
    frame = week_rows.rename(columns={period_column: "period"})
    keys = ["campus", "weekday", "period", "week", "section_id"]
    return frame.drop_duplicates(subset=keys)[keys + ["enrollment"]]


def slot_load(week_rows: pd.DataFrame) -> pd.DataFrame:
    """Sections and seats in every campus x weekday x period x week cell.

    Args:
        week_rows: output of ``expand_weeks`` (one row per section x slot x week).

    Returns:
        DataFrame campus, weekday, period, week, n_sections, seats, with ``ALL`` campus totals.
    """
    if week_rows.empty:
        return pd.DataFrame(columns=["campus", "weekday", "period", "week", "n_sections", "seats"])
    cells = _dedup_cells(week_rows.explode("periods"), "periods")
    grouped = cells.groupby(["campus", "weekday", "period", "week"], as_index=False).agg(
        n_sections=("section_id", "nunique"), seats=("enrollment", "sum"))
    out = _with_all_campus(grouped, ["weekday", "period", "week"])
    logger.info("slot load: %d cells over %d weeks", len(out), out["week"].nunique())
    return out


def typical_week(load: pd.DataFrame, cfg: StatsConfig | None = None) -> pd.DataFrame:
    """Median seats and sections per campus x weekday x period over the stable teaching weeks."""
    cfg = cfg or StatsConfig()
    window = load[(load["week"] >= cfg.typical_lo) & (load["week"] <= cfg.typical_hi)]
    if window.empty:
        logger.warning("no weeks in [%d, %d]; using every week", cfg.typical_lo, cfg.typical_hi)
        window = load
    weeks = sorted(window["week"].unique())
    full = _reindex_weeks(window, weeks)
    out = full.groupby(["campus", "weekday", "period"], as_index=False).agg(
        seats=("seats", "median"), n_sections=("n_sections", "median"))
    return out


def _reindex_weeks(window: pd.DataFrame, weeks: Sequence[int]) -> pd.DataFrame:
    """Fill absent (cell, week) combinations with zeros so the median is not biased upward."""
    cells = window[["campus", "weekday", "period"]].drop_duplicates()
    grid = cells.merge(pd.DataFrame({"week": list(weeks)}), how="cross")
    return grid.merge(window, on=["campus", "weekday", "period", "week"], how="left").fillna(
        {"n_sections": 0, "seats": 0})


def week_totals(load: pd.DataFrame) -> pd.DataFrame:
    """Seat-periods per campus per week (the series whose maximum is the peak week)."""
    return load.groupby(["campus", "week"], as_index=False).agg(
        seats=("seats", "sum"), n_sections=("n_sections", "sum"))


def peak_week(totals: pd.DataFrame, campus: str = ALL_CAMPUS) -> Dict[str, float]:
    """Week with the most seat-periods, and its ratio to the median week."""
    series = totals[totals["campus"] == campus]
    if series.empty:
        return {}
    top = series.loc[series["seats"].idxmax()]
    median = float(series["seats"].median())
    return {"peak_week": int(top["week"]), "peak_week_seat_periods": float(top["seats"]),
            "median_week_seat_periods": median,
            "peak_over_median_week": float(top["seats"]) / median if median else float("nan")}


def session_start_load(week_rows: pd.DataFrame) -> pd.DataFrame:
    """Sections and seats whose slot *starts* in each campus x weekday x period x week cell."""
    if week_rows.empty:
        return pd.DataFrame(columns=["campus", "weekday", "period", "week", "n_sections", "seats"])
    cells = _dedup_cells(week_rows, "period_start")
    grouped = cells.groupby(["campus", "weekday", "period", "week"], as_index=False).agg(
        n_sections=("section_id", "nunique"), seats=("enrollment", "sum"))
    return _with_all_campus(grouped, ["weekday", "period", "week"])


def session_starts(start_load: pd.DataFrame, slots: pd.DataFrame, cfg: StatsConfig | None = None,
                   campus: str = ALL_CAMPUS) -> pd.DataFrame:
    """Session-start histogram over weekday x period: how many students begin a class at that bell.

    ``n_sections``/``seats`` count the sections that start in that cell in at least one week of the
    semester; ``*_typ`` are the medians over the stable teaching weeks, i.e. a typical week.
    """
    cfg = cfg or StatsConfig()
    frame = start_load[start_load["campus"] == campus]
    if frame.empty:
        return pd.DataFrame(columns=["weekday", "period", "n_sections", "seats",
                                     "n_sections_typ", "seats_typ"])
    starts = slots if campus == ALL_CAMPUS else slots[slots["campus"] == campus]
    starts = starts.rename(columns={"period_start": "period"}).drop_duplicates(
        subset=["section_id", "weekday", "period"])
    total = starts.groupby(["weekday", "period"], as_index=False).agg(
        n_sections=("section_id", "nunique"), seats=("enrollment", "sum"))
    window = frame[(frame["week"] >= cfg.typical_lo) & (frame["week"] <= cfg.typical_hi)]
    weeks = sorted(window["week"].unique()) or sorted(frame["week"].unique())
    filled = _reindex_weeks(window if len(window) else frame, weeks)
    typ = filled.groupby(["weekday", "period"], as_index=False).agg(
        n_sections_typ=("n_sections", "median"), seats_typ=("seats", "median"))
    return total.merge(typ, on=["weekday", "period"], how="outer").fillna(0).sort_values(
        ["weekday", "period"], ignore_index=True)


def same_bell(start_load: pd.DataFrame, bells: BellSchedule | None = None,
              cfg: StatsConfig | None = None, campus: str = ALL_CAMPUS) -> pd.DataFrame:
    """Sections and seats starting at each bell, all campuses pooled.

    For every (weekday, bell) the per-week value is taken over the stable teaching weeks (median);
    ``*_typ`` is then the median over Mon-Fri and ``*_max`` the Mon-Fri maximum.  ``off_block``
    marks a start time that is not one of the six block bells (a slot starting mid-block).
    """
    bells = bells or BellSchedule()
    cfg = cfg or StatsConfig()
    frame = start_load[(start_load["campus"] == campus)
                       & (start_load["weekday"].isin(cfg.teaching_weekdays))].copy()
    if frame.empty:
        return pd.DataFrame(columns=["bell", "n_sections_typ", "seats_typ", "n_sections_max",
                                     "seats_max", "weekday_max", "off_block"])
    frame["bell"] = frame["period"].map(bells.start)
    block_bells = set(bells.bells())
    window = frame[(frame["week"] >= cfg.typical_lo) & (frame["week"] <= cfg.typical_hi)]
    window = window if len(window) else frame
    weeks = sorted(window["week"].unique())
    cells = window[["bell", "weekday"]].drop_duplicates().merge(
        pd.DataFrame({"week": weeks}), how="cross")
    per_week = cells.merge(
        window.groupby(["bell", "weekday", "week"], as_index=False).agg(
            n_sections=("n_sections", "sum"), seats=("seats", "sum")),
        on=["bell", "weekday", "week"], how="left").fillna({"n_sections": 0, "seats": 0})
    per_day = per_week.groupby(["bell", "weekday"], as_index=False).agg(
        n_sections=("n_sections", "median"), seats=("seats", "median"))
    out = per_day.groupby("bell", as_index=False).agg(
        n_sections_typ=("n_sections", "median"), seats_typ=("seats", "median"),
        n_sections_max=("n_sections", "max"), seats_max=("seats", "max"))
    idx = per_day.loc[per_day.groupby("bell")["seats"].idxmax(), ["bell", "weekday"]]
    out = out.merge(idx.rename(columns={"weekday": "weekday_max"}), on="bell", how="left")
    out["off_block"] = ~out["bell"].isin(block_bells)
    if bool(out["off_block"].any()):
        logger.warning("%d start time(s) fall mid-block: %s", int(out["off_block"].sum()),
                       out.loc[out["off_block"], "bell"].tolist())
    return out.sort_values("bell", ignore_index=True)


def enrollment_summary(sections: pd.DataFrame, cfg: StatsConfig | None = None) -> Dict[str, float]:
    """Quantiles and totals of the section enrollment (上课人数)."""
    cfg = cfg or StatsConfig()
    series = sections["enrollment"].astype(float)
    out = {f"p{int(q * 100)}": float(series.quantile(q)) for q in cfg.quantiles}
    out.update({"max": float(series.max()), "mean": float(series.mean()),
                "total": float(series.sum()), "n_zero": float((series == 0).sum()),
                "n_lt10": float((series < 10).sum())})
    return out


def n_periods_distribution(slots: pd.DataFrame) -> pd.DataFrame:
    """How long a single meeting is, in periods."""
    counts = slots["n_periods"].value_counts().sort_index()
    return pd.DataFrame({"n_periods": counts.index, "n_slots": counts.to_numpy(),
                         "share": (counts / counts.sum()).to_numpy()})


def concentration(typ: pd.DataFrame, cfg: StatsConfig | None = None,
                  campus: str = ALL_CAMPUS) -> Dict[str, float]:
    """Peak-to-mean of typical-week seats over the teaching grid, and the top-k cell shares."""
    cfg = cfg or StatsConfig()
    grid = typ[(typ["campus"] == campus) & typ["weekday"].isin(cfg.teaching_weekdays)
               & typ["period"].isin(cfg.teaching_periods)]
    n_cells = len(cfg.teaching_weekdays) * len(cfg.teaching_periods)
    seats = grid["seats"].sort_values(ascending=False).to_numpy()
    total = float(seats.sum())
    if total <= 0:
        return {}
    mean = total / n_cells
    out = {"grid_cells": float(n_cells), "peak_seats": float(seats[0]), "mean_seats": mean,
           "peak_to_mean": float(seats[0]) / mean, "seat_periods_week": total}
    for k in cfg.top_k:
        out[f"share_top{k}"] = float(seats[:k].sum()) / total
    return out


def counts(sections: pd.DataFrame, slots: pd.DataFrame) -> Dict[str, int]:
    """Row counts of the parsed export."""
    weekend = slots[slots["weekday"] >= 6]
    return {"n_sections": len(sections), "n_courses": int(sections["course_code"].nunique()),
            "n_colleges": int(sections["college"].nunique()),
            "n_campuses": int(sections.loc[sections["campus"] != "", "campus"].nunique()),
            "n_blank_campus": int((sections["campus"] == "").sum()),
            "n_ambiguous": int(sections["ambiguous"].sum()),
            "n_general_elective": int(sections["general_elective"].sum()),
            "n_slots": len(slots), "n_weekend_slots": len(weekend),
            "n_weekend_sections": int(weekend["section_id"].nunique()),
            "n_blank_room": int((slots["room"] == "").sum())}


def compute(sections: pd.DataFrame, slots: pd.DataFrame, week_rows: pd.DataFrame,
            cfg: StatsConfig | None = None, bells: BellSchedule | None = None) -> T2Result:
    """Run every T2 statistic over one parsed timetable."""
    cfg = cfg or StatsConfig()
    bells = bells or BellSchedule()
    load = slot_load(week_rows)
    typ = typical_week(load, cfg)
    totals = week_totals(load)
    starts = session_start_load(week_rows)
    result = T2Result(
        slot_load=load, typical_week=typ, week_totals=totals,
        session_starts=session_starts(starts, slots, cfg),
        same_bell=same_bell(starts, bells, cfg),
        n_periods=n_periods_distribution(slots), enrollment=enrollment_summary(sections, cfg),
        concentration=concentration(typ, cfg), counts=counts(sections, slots),
        peak=peak_week(totals))
    logger.info("T2: %d sections, peak-to-mean %.2f", result.counts["n_sections"],
                result.concentration.get("peak_to_mean", float("nan")))
    return result


def _rows(result: T2Result, tt: TimetableConfig, cfg: StatsConfig) -> List[Tuple[str, object, str, str]]:
    c, e, k, p = result.counts, result.enrollment, result.concentration, result.peak
    win = f"median over weeks {cfg.typical_lo}-{cfg.typical_hi}"
    top = result.same_bell.sort_values("seats_typ", ascending=False)
    bell_rows: List[Tuple[str, object, str, str]] = []
    for row in result.same_bell.itertuples():
        bell_rows.append((f"Seats starting at bell {row.bell}", round(row.seats_typ), "students",
                          f"typical weekday, {win}; Mon-Fri max {row.seats_max:.0f} "
                          f"(weekday {int(row.weekday_max)})"))
    head = top.iloc[0] if len(top) else None
    return [
        ("Course sections", c["n_sections"], "sections", "one row per section in the export"),
        ("Distinct courses", c["n_courses"], "course codes", ""),
        ("Colleges", c["n_colleges"], "colleges", ""),
        ("Campuses", c["n_campuses"], "campuses", f"{c['n_blank_campus']} section(s) with no campus"),
        ("Meeting slots", c["n_slots"], "section-slots", "section x weekday x period block"),
        ("Weekend slots", c["n_weekend_slots"], "section-slots",
         f"{c['n_weekend_sections']} sections; Sat/Sun make-up sessions"),
        ("University-wide elective sections", c["n_general_elective"], "sections", "temporary-class labels"),
        ("Ambiguous week lists", c["n_ambiguous"], "sections",
         f"{c['n_ambiguous'] / max(c['n_sections'], 1):.1%} of sections; slots get the week union"),
        ("Teaching weeks", tt.weeks, "weeks", f"week 1 Monday {tt.week1_monday}"),
        ("Periods per day", tt.periods, "periods", "45 min each, six bell blocks"),
        ("Enrollment p10", round(e["p10"]), "students", "per section"),
        ("Enrollment median", round(e["p50"]), "students", "per section"),
        ("Enrollment p90", round(e["p90"]), "students", "per section"),
        ("Enrollment p99", round(e["p99"]), "students", "per section"),
        ("Enrollment max", round(e["max"]), "students", "largest single section"),
        ("Total enrolled seats", round(e["total"]), "seats", "sum over sections, students double-counted"),
        ("Sections with 0 enrolled", round(e["n_zero"]), "sections", "placeholder rows"),
        ("Typical meeting length", _mode_periods(result.n_periods), "periods",
         "modal n_periods over slots"),
        ("Seat-periods in a typical week", round(k.get("seat_periods_week", 0)), "seat-periods",
         f"{win}, Mon-Fri periods 1-10, all campuses"),
        ("Peak-to-mean of seats", round(k.get("peak_to_mean", float("nan")), 2), "ratio",
         "teaching grid Mon-Fri x periods 1-10, typical week"),
        ("Share of seat-periods in top 5 cells", round(k.get("share_top5", float("nan")), 4), "share",
         "top 5 weekday x period cells of the typical week"),
        ("Share of seat-periods in top 10 cells", round(k.get("share_top10", float("nan")), 4), "share",
         "top 10 weekday x period cells"),
        ("Peak week", p.get("peak_week", ""), "week index", "week with the most seat-periods"),
        ("Peak week over median week", round(p.get("peak_over_median_week", float("nan")), 3), "ratio",
         "seat-periods"),
        ("Seats starting at the same bell (max)", round(head.seats_typ) if head is not None else "",
         "students", f"bell {head.bell if head is not None else ''}, typical weekday, {win}"),
        ("Sections starting at the same bell (max)",
         round(head.n_sections_typ) if head is not None else "", "sections",
         f"bell {head.bell if head is not None else ''}, typical weekday"),
    ] + bell_rows


def _mode_periods(n_periods: pd.DataFrame) -> int:
    if n_periods.empty:
        return 0
    return int(n_periods.loc[n_periods["n_slots"].idxmax(), "n_periods"])


def t2_table(result: T2Result, tt: TimetableConfig | None = None,
             cfg: StatsConfig | None = None) -> pd.DataFrame:
    """The T2 table: one row per statistic (statistic, value, unit, note)."""
    rows = _rows(result, tt or TimetableConfig(), cfg or StatsConfig())
    frame = pd.DataFrame(rows, columns=["statistic", "value", "unit", "note"])
    frame["value"] = pd.Series([r[1] for r in rows], dtype=object)  # keep ints out of float format
    return frame


def write_t2(result: T2Result, out_dir: str | Path, tt: TimetableConfig | None = None,
             cfg: StatsConfig | None = None) -> Path:
    """Write ``T2_table.csv`` and the prose-ready ``T2_summary.md``; returns the CSV path."""
    tt, cfg = tt or TimetableConfig(), cfg or StatsConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = t2_table(result, tt, cfg)
    csv_path = out / "T2_table.csv"
    table.to_csv(csv_path, index=False)
    (out / "T2_summary.md").write_text(_summary_md(result, table, tt, cfg), encoding="utf-8")
    logger.info("wrote %s and T2_summary.md", csv_path)
    return csv_path


def _summary_md(result: T2Result, table: pd.DataFrame, tt: TimetableConfig,
                cfg: StatsConfig) -> str:
    c, k = result.counts, result.concentration
    bell = result.same_bell.sort_values("seats_typ", ascending=False)
    lines = [
        "# T2 - institution-wide timetable statistics", "",
        f"Semester {tt.weeks} teaching weeks, week 1 Monday {tt.week1_monday}; "
        f"{tt.periods} periods per day in six bell blocks.", "",
        f"The export holds {c['n_sections']} course sections over {c['n_courses']} distinct "
        f"courses, {c['n_colleges']} colleges and {c['n_campuses']} campuses, which expand to "
        f"{c['n_slots']} weekly meeting slots and "
        f"{int(result.slot_load.loc[result.slot_load['campus'] == ALL_CAMPUS, 'n_sections'].sum())}"
        " section-period-weeks.",
        f"{c['n_ambiguous']} section(s) "
        f"({c['n_ambiguous'] / max(c['n_sections'], 1):.1%}) have a week list that cannot be "
        "aligned to their slots; those slots take the union of the listed weeks.", "",
        "## Synchronized starts", "",
    ]
    if len(bell):
        head = bell.iloc[0]
        lines += [
            f"On a typical weekday {round(head.seats_typ)} students across all campuses start a "
            f"class at the {head.bell} bell ({round(head.n_sections_typ)} sections); the Mon-Fri "
            f"maximum for that bell is {head.seats_max:.0f} students.",
            "", "| bell | sections (typical) | seats (typical) | seats (Mon-Fri max) |",
            "|---|---|---|---|",
        ]
        lines += [f"| {r.bell} | {r.n_sections_typ:.0f} | {r.seats_typ:.0f} | {r.seats_max:.0f} |"
                  for r in bell.sort_values("bell").itertuples()]
    lines += [
        "", "## Concentration", "",
        f"Over the Mon-Fri x period 1-10 grid of a typical week (median over weeks "
        f"{cfg.typical_lo}-{cfg.typical_hi}) the campus-wide seat count has a peak-to-mean of "
        f"{k.get('peak_to_mean', float('nan')):.2f}; the busiest 5 of the "
        f"{int(k.get('grid_cells', 0))} cells carry {k.get('share_top5', float('nan')):.1%} and the "
        f"busiest 10 carry {k.get('share_top10', float('nan')):.1%} of the weekly seat-periods.",
        "", "## Table", "",
        "| statistic | value | unit | note |", "|---|---|---|---|",
    ]
    lines += [f"| {r.statistic} | {r.value} | {r.unit} | {r.note} |" for r in table.itertuples()]
    lines.append("")
    return "\n".join(lines)
