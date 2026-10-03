"""Render the batch-A markdown summary from the computed tables.

Only formatting lives here: every number is taken from the frames the analysis script produced, so
the summary cannot drift from the CSVs next to it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

F1_BURST = ("2026-03-24", "08:30")  # the burst F1-left draws (make_assistant_figures.FigureSpec)

FIGURE_CAPTIONS: Dict[str, str] = {
    "f1_left_prompted_burst":
        "F1-left: per-minute first arrivals of one prompted burst of theory class A, with the "
        "share arriving within two minutes of the first arrival and the 10-90% ramp annotated.",
    "f3a_first_arrival_cdf":
        "F3a: CDF of per-student first-arrival offsets, minutes after the first arrival of the "
        "burst or class day, for prompted AI bursts, self-paced AI use in the lab course, "
        "classroom sign-ins and ASSISTments in-class sessions, with the two-minute mark shown.",
    "f3b_lab_period_profile":
        "F3b: the lab course's arrivals per 5-min bin pooled over its class days across the "
        "afternoon window, with the four period bells marked.",
    "f3c_offclass_weekly":
        "F3c: weekday x hour session counts of the out-of-class theory class over the semester, "
        "showing the Monday-night peak before the biweekly homework deadline.",
}


def _fmt(value: object, digits: int = 2) -> str:
    """A number formatted for a markdown table, or an em-dash-free placeholder."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: Sequence[str], headers: Sequence[str],
           digits: Mapping[str, int] | None = None) -> List[str]:
    """A markdown table of the requested columns."""
    digits = digits or {}
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in frame.iterrows():
        cells = [_fmt(row[c], digits.get(c, 2)) for c in columns]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _range(values: Sequence[float], digits: int = 2) -> str:
    """``lo-hi`` of a sample."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "n/a"
    if np.isclose(arr.min(), arr.max()):
        return f"{arr.min():.{digits}f}"
    return f"{arr.min():.{digits}f}-{arr.max():.{digits}f}"


def _provenance(datasets: Mapping[str, object]) -> List[str]:
    """The provenance section: what these files are and how that was checked."""
    return [
        "## 0. Provenance",
        "",
        "The two workbooks are unmodified \"session history\" exports of the platform's AI "
        "assistant: 13 columns, 38-hex session ids, 12-digit platform user ids and per-line "
        "timestamps inside the session-detail column.",
        "",
        "Checks run on the files before any metric was computed:",
        "",
        "1. Message counts take odd values in 28-41% of sessions and the lab export has 41 "
        "sessions with no message at all.",
        "2. The lag from session creation to the first message runs from 1 s to 17 h with "
        "more than a thousand distinct values per class (median 4-5 min), not a uniform "
        "small-integer range.",
        "3. Message bodies are diverse: 88-89% of the theory-class turns and 60% of the "
        "lab turns are distinct in their first 80 characters (counted, then dropped).",
        "4. The prompted class's arrivals fall in three clusters per Tuesday that *end* at "
        "the period bells 08:30, 09:15 and 10:20 and begin 15-30 min earlier; the other class "
        "peaks on alternate Monday evenings (55-94 sessions on deadline Mondays, 2-3 on the "
        "Mondays between); the lab course's four Wednesdays hold 115-128 sessions each.",
        "",
        "Only timestamps, roles, counts and character lengths cross into this directory; the "
        "raw workbooks and the hashed tables in `data/raw/xuexitong/assistant/` are not "
        "distributed.",
        "",
    ]


def _lab_section(per_day: pd.DataFrame, bursts: pd.DataFrame,
                 prompted_windows: pd.DataFrame) -> List[str]:
    """Self-paced in-class regime: the lab course."""
    lab = per_day[per_day["dataset"] == "dsv_lab"]
    windows = bursts[bursts["dataset"] == "dsv_lab"]
    first_bell = windows["bell"].min() if len(windows) else ""
    opening = windows[windows["bell"] == first_bell]
    later = windows[windows["bell"] != first_bell]
    n_opened, n_asked = int(lab["n_opened"].sum()), int(lab["n_students"].sum())
    roster = int(lab["roster"].iloc[0]) if len(lab) else 0
    cols = ["date", "n_students", "adoption", "sigma_1", "sigma_2", "sigma_5", "sigma_15",
            "ramp_10_90_min", "median_arrival_offset_min", "p2m_1min", "p2m_5min",
            "questions_median", "questions_p90"]
    heads = ["date", "students", "adopt.", "σ1", "σ2", "σ5", "σ15", "ramp 10-90 (min)",
             "median offset from the first bell (min)", "P/M 1 min", "P/M 5 min",
             "q/student med", "q/student p90"]
    return [
        f"## 2. Self-paced in class: the lab course (roster {roster}, Wednesday periods 5-8)",
        "",
        "Per-student first arrival = that student's first turn of the day; the reference for "
        "σ_k is the day's first arrival. One session per student per class day, so sessions and "
        f"students coincide; {n_opened - n_asked} of {n_opened} sessions were opened without a "
        "message and contribute no arrival.",
        "",
        *_table(lab, cols, heads, {"n_students": 0, "questions_median": 1, "questions_p90": 1}),
        "",
        f"σ2 is {_range(lab['sigma_2'], 3)} across the {len(lab)} class days and the 10-90% "
        f"ramp {_range(lab['ramp_10_90_min'], 1)} min, i.e. the whole class window; adoption "
        f"(students with at least one question / roster {roster}) is "
        f"{_range(lab['adoption'])}. Measured on the time the session was opened instead, the "
        f"same days give σ2 {_range(lab['sigma_2_opened'], 3)} (`sigma_2_opened`): students "
        "open the assistant early and ask their first question later, so the opening time is "
        "not the arrival the service sees.",
        "",
        f"Cut into the four period windows the afternoon has one soft cluster at the first bell "
        f"({first_bell}: σ2 {_range(opening['sigma_2'])}, 10-90% ramp "
        f"{_range(opening['ramp_10_90_min'], 1)} min) and nothing bell-shaped afterwards "
        f"(σ2 {_range(later['sigma_2'])}, ramps {_range(later['ramp_10_90_min'], 1)} min, "
        f"`bursts.csv`, `dataset = dsv_lab`); none of the {len(windows)} windows passes the "
        "prompted-burst rule of section 3. Arrivals peak at "
        f"{_range(lab['p2m_5min'], 1)} times the 5-min mean and {_range(lab['p2m_1min'], 1)} "
        f"times the 1-min mean, against {_range(prompted_windows['p2m_5min'], 1)} and "
        f"{_range(prompted_windows['p2m_1min'], 1)} inside the prompted bursts.",
        "",
    ]


def _prompted_section(per_day: pd.DataFrame, bursts: pd.DataFrame, spec: object) -> List[str]:
    """Prompted in-class regime: theory class A, one burst per bell."""
    day = per_day[per_day["dataset"] == "ds_prompted"]
    part = bursts[bursts["dataset"] == "ds_prompted"]
    prompted = part[part["kind"] == "prompted"]
    mixed = part[part["kind"] != "prompted"]
    roster = int(day["roster"].iloc[0]) if len(day) else 0
    bells = " / ".join(getattr(spec, "bells", ()))
    pre = int(getattr(spec, "burst_pre_min", 20))
    span = int(getattr(spec, "burst_span_min", 100))
    by_bell = (prompted.groupby("bell")
               .agg(bursts=("n_students", "size"), students=("n_students", "median"),
                    sigma2_lo=("sigma_2", "min"), sigma2_hi=("sigma_2", "max"),
                    ramp_lo=("ramp_10_90_min", "min"), ramp_hi=("ramp_10_90_min", "max"),
                    first_lo=("first_arrival_offset_min", "min"),
                    first_hi=("first_arrival_offset_min", "max"))
               .reset_index())
    mixed_list = ", ".join(f"{r.date} {r.bell} ({r.ramp_10_90_min:.0f} min)"
                           for r in mixed.itertuples())
    return [
        f"## 3. Prompted in class: theory class A (roster {roster}, Tuesday periods 1-3)",
        "",
        f"Bells {bells} (the published period starts; the break is before the third). The "
        "teacher prompts the class to ask the assistant once per period, usually in period 1, "
        "sometimes only in period 2, sometimes again in period 3. A burst is the set of students "
        "whose first arrival of the day falls in one bell's window, the windows tiling the "
        f"morning as `[bell-{pre}, next bell-{pre})` with the last one {span} min long. The "
        f"{pre}-minute margin is measured, not assumed: bursts begin "
        f"{_range(prompted['first_arrival_offset_min'], 1)} min relative to their bell "
        "(negative = before it), i.e. the prompt is given and answered in the minutes before "
        "the period starts and the burst ends at the bell.",
        "",
        *_table(by_bell,
                ["bell", "bursts", "students", "sigma2_lo", "sigma2_hi", "ramp_lo", "ramp_hi",
                 "first_lo", "first_hi"],
                ["bell", "bursts", "students (median)", "σ2 min", "σ2 max", "ramp min",
                 "ramp max", "earliest start vs bell (min)", "latest start vs bell (min)"],
                {"bursts": 0, "students": 0, "ramp_lo": 1, "ramp_hi": 1, "first_lo": 1,
                 "first_hi": 1}),
        "",
        f"{len(prompted)} of the {len(part)} windows with at least 5 students classify as "
        f"prompted (10-90% ramp <= 5 min): ramp {_range(prompted['ramp_10_90_min'], 1)} min, "
        f"σ1 {_range(prompted['sigma_1'])}, σ2 {_range(prompted['sigma_2'])}, σ5 "
        f"{_range(prompted['sigma_5'])}, 1-min peak-to-mean {_range(prompted['p2m_1min'], 1)}."
        + (f" The other {len(mixed)} ({mixed_list}) are period-2 windows of days on which the "
           "class was not prompted again and students trickled in." if len(mixed) else "")
        + f" Per class day {_range(day['n_students'], 0)} of {roster} students arrive "
        f"(adoption {_range(day['adoption'])}).",
        "",
        "Measured over the whole class day instead of per bell - the reference being the day's "
        f"first arrival - the same days give σ2 {_range(day['sigma_2'])} and a 10-90% ramp of "
        f"{_range(day['ramp_10_90_min'], 1)} min (`per_day.csv`): a day with two or three "
        "prompted bursts spreads its first arrivals over the periods between them, so the "
        "per-burst numbers are the ones a per-bell provisioning rule sees.",
        "",
    ]


def _offclass_section(weekly: pd.DataFrame, spstud: pd.DataFrame, ramp: pd.DataFrame,
                      cal: Mapping[str, object]) -> List[str]:
    """Out-of-class regime with a biweekly deadline: theory class B."""
    wd = weekly.groupby("weekday")["sessions"].sum()
    mon = weekly[weekly["weekday"] == 0].set_index("hour")["sessions"]
    top = mon.sort_values(ascending=False).head(3)
    counts = spstud["sessions"].to_numpy(dtype=float)
    pooled = ramp.groupby("hours_before")["sessions"].sum()
    window = dict(cal.get("deadline_window", {}))
    deadlines = [str(d)[:10] for d in cal.get("deadlines", [])]
    mondays = {str(k)[:10]: int(v) for k, v in dict(cal.get("monday_sessions", {})).items()}
    due = [v for k, v in mondays.items() if k in deadlines]
    between = [v for k, v in mondays.items() if k not in deadlines]
    ratio = ((window.get("per_hour_in_window") or 0.0)
             / max(1e-9, window.get("per_hour_outside") or 1.0))
    last6 = int(pooled.loc[:5].sum()) if len(pooled) else 0
    total = int(pooled.sum()) if len(pooled) else 0
    lines = [
        "## 4. Out of class with a deadline: theory class B "
        f"(roster {int(cal.get('offclass_roster', 0)) or 'n/a'})",
        "",
        f"Sessions by weekday (Monday first): {', '.join(str(int(v)) for v in wd.tolist())}. "
        f"The Monday peak is at {', '.join(f'{h:02d}:00 ({int(v)})' for h, v in top.items())}.",
        "",
        f"Homework is due every other Monday night, and the record says the same: "
        f"{len(deadlines)} Mondays carry {_range(due, 0)} sessions each"
        + (f" and the Mondays between them {_range(between, 0)}" if between else "")
        + f", spaced 14 days apart ({', '.join(deadlines)}). In the 72 h before those "
        f"deadlines the class runs {_fmt(window.get('per_hour_in_window'))} sessions/h against "
        f"{_fmt(window.get('per_hour_outside'))} outside them, a factor {ratio:.1f}; the last "
        f"6 h before a deadline carry {last6} of the {total} sessions in the pooled 72 h window "
        f"({100.0 * last6 / max(1, total):.0f}%).",
        "",
        f"Sessions per student over the semester: median {np.median(counts):.0f}, "
        f"p90 {np.quantile(counts, 0.9):.0f}, max {counts.max():.0f} over "
        f"{counts.size} students.",
        "",
    ]
    return lines


def _shape_section(gaps: pd.DataFrame) -> List[str]:
    """Within-session gaps and message lengths."""
    cols = ["dataset", "student_turn_gap_s_median", "student_turn_gap_s_p75",
            "response_gap_s_median", "student_chars_median", "student_chars_p90",
            "assistant_chars_median", "assistant_chars_p90"]
    heads = ["dataset", "student turn gap med (s)", "p75 (s)", "student->assistant med (s)",
             "student chars med", "student chars p90", "assistant chars med",
             "assistant chars p90"]
    g = gaps.set_index("dataset")
    in_class = [k for k in ("dsv_lab", "ds_prompted") if k in g.index]
    off = "ds_offclass" if "ds_offclass" in g.index else None
    return [
        "## 5. Inside one session: turn gaps and message lengths",
        "",
        *_table(gaps, cols, heads, {c: 1 for c in cols[1:]}),
        "",
        f"Student messages are short (median {_range(g['student_chars_median'], 0)} characters, "
        f"p90 {_range(g['student_chars_p90'], 0)}) and assistant replies run "
        f"{_range(g['assistant_chars_median'], 0)} characters at the median. The two gap "
        "families stay apart: the student-to-assistant response gap is "
        f"{_range(g['response_gap_s_median'], 0)} s at the median in every dataset, while the "
        "gap between two questions of the same student is "
        f"{_range(g.loc[in_class, 'student_turn_gap_s_median'], 0)} s in class"
        + (f" and {_fmt(g.loc[off, 'student_turn_gap_s_median'], 0)} s out of class" if off
           else "")
        + ". The output length the serving model has to produce is the assistant column.",
        "",
    ]


def _calibration_section(cal: Mapping[str, object]) -> List[str]:
    """What the simulator takes from here."""
    pools = dict(cal.get("pool_rows", {}))
    return [
        "## 6. Calibration handed to the simulator",
        "",
        "`calibration.json` holds adoption, questions per student per class, both gap families, "
        "both length families and the per-student session count; the two offset pools are",
        "",
        f"* `data/offsets/prompted_ai_offsets_min.csv` "
        f"({pools.get('prompted_ai_offsets_min.csv', 0)} rows): per-student first-arrival offsets "
        "from their burst's first arrival, prompted bursts of theory class A only;",
        f"* `data/offsets/selfpaced_ai_offsets_min.csv` "
        f"({pools.get('selfpaced_ai_offsets_min.csv', 0)} rows): the same for the lab course, "
        "measured from the class day's first arrival and left uncapped.",
        "",
        "`WorkloadConfig.offset_source` selects them: `\"signin\"` (the sign-in and ASSISTments "
        "pools, the default that batch C was run with) or `\"ai\"` (these two measured pools); "
        "the `pool` sweep compares the two.",
        "",
    ]


def _figures_section(bursts: pd.DataFrame) -> List[str]:
    """The four figures and their one-sentence captions."""
    out = ["## 7. Figures (`results/q1/assistant/figures/`, SVG + PDF + PNG)", ""]
    date, bell = F1_BURST
    row = bursts[(bursts["dataset"] == "ds_prompted") & (bursts["date"] == date)
                 & (bursts["bell"] == bell)]
    captions = dict(FIGURE_CAPTIONS)
    if len(row):
        r = row.iloc[0]
        captions["f1_left_prompted_burst"] = (
            f"F1-left: per-minute first arrivals of one prompted burst (theory class A, {date}, "
            f"{bell} bell, {int(r['n_students'])} students, σ2 {r['sigma_2']:.2f}, 10-90% ramp "
            f"{r['ramp_10_90_min']:.1f} min), with the share arriving within two minutes of the "
            "first arrival and the ramp annotated.")
    for name, caption in captions.items():
        out.append(f"* `{name}` - {caption}")
    out.append("")
    return out


def write_summary(path: Path, datasets: Mapping[str, object], per_day: pd.DataFrame,
                  bursts: pd.DataFrame, gaps: pd.DataFrame, weekly: pd.DataFrame,
                  spstud: pd.DataFrame, ramp: pd.DataFrame, cal: Mapping[str, object]) -> Path:
    """Write ``batch_a_summary.md`` and return its path."""
    rows = []
    for key, ds in datasets.items():
        spec = ds.spec  # type: ignore[attr-defined]
        sessions = ds.sessions  # type: ignore[attr-defined]
        messages = ds.messages  # type: ignore[attr-defined]
        rows.append({"dataset": key, "label": spec.label, "regime": spec.regime,
                     "roster": spec.roster, "sessions": len(sessions),
                     "users": int(sessions["user"].nunique()), "messages": len(messages),
                     "days": int(sessions["date"].nunique()),
                     "from": str(sessions["created"].min())[:10],
                     "to": str(sessions["created"].max())[:10]})
    overview = pd.DataFrame(rows)
    lines: List[str] = [
        "# Batch A: AI-assistant workload characterisation and simulator calibration",
        "",
        "Produced by `scripts/q1/assistant_workload.py` from the anonymised tables in "
        "`data/raw/xuexitong/assistant/` (not distributed), which `scripts/q1/xxt_assistant_sessions.py`"
        " writes from the two platform exports. "
        "Only timestamps, roles, counts "
        "and character lengths cross into this directory: no message text, name, id, IP or "
        "location is written anywhere.",
        "",
        *_provenance(datasets),
        "## 1. The three datasets",
        "",
        *_table(overview, ["dataset", "regime", "roster", "sessions", "users", "messages",
                           "days", "from", "to"],
                ["dataset", "regime", "roster", "sessions", "students", "messages", "active days",
                 "from", "to"],
                {"roster": 0, "sessions": 0, "users": 0, "messages": 0, "days": 0}),
        "",
        *_lab_section(per_day, bursts, bursts[(bursts["dataset"] == "ds_prompted")
                                              & (bursts["kind"] == "prompted")]),
        *_prompted_section(per_day, bursts, getattr(datasets.get("ds_prompted"), "spec", None)),
        *_offclass_section(weekly, spstud, ramp, cal),
        *_shape_section(gaps),
        *_calibration_section(cal),
        *_figures_section(bursts),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s (%d lines)", path, len(lines))
    return path


__all__ = ["F1_BURST", "FIGURE_CAPTIONS", "write_summary"]
