"""Canonical numbers, part 5b: tables of the laptop sensitivity runs (sections 14 and 18-23).

Formats the blocks of ``canonical_sensitivity.py`` (one timing cell of one run: the original's
offline and calendar-planner rows, the s3 re-run's SLO-floor reactive and velocity rules at the
resource-rule configurations, the TAP job's ``tap`` rows, all on one laptop stream) for
``canonical_numbers.py`` and ``canonical_tables.py``.  Means are over seeds 10-14; GPU-hours and USD
per week to 1 decimal, attainment to 3, and to 4 in the floor tables, percentages from unrounded
means (the change against TAP signed, to 1 decimal and to the nearest integer).  A rule reaches the
floor when its unrounded mean in-class attainment is at least 0.95.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from canonical_extra import SLO_FLOOR
from canonical_sensitivity import ABLATION_MODES, FEEDBACK, Block

SWEEP_POLICIES: Tuple[str, ...] = ("offline", "tap", "calendar", "calendar_learn_prior", *FEEDBACK)
ABLATION_POLICIES: Tuple[str, ...] = ("offline", "tap", "calendar_learn_prior", "calendar_adapt",
                                      "reactive")
SWEEP_COLUMNS: Tuple[str, ...] = ("gpu_h_wk", "usd_wk", "slo_in", "slo_off", "excess_usd_wk",
                                  "n_seeds", "source")
LEAD_COLUMNS: Tuple[str, ...] = ("gpu_h_wk", "slo_in", "excess_usd_wk", "n_seeds", "source")
FLOOR_ROWS: Tuple[str, str] = ("cheapest SLO-floor-tuned feedback rule at >= 0.95 (unrounded)",
                               "SLO-floor-tuned reactive rule (always listed)")
FLOOR_COLUMNS: Tuple[str, ...] = (
    "row", "feedback_rule", "gpu_h_wk", "slo_in", "slo_in_4dp", "lowest_seed_4dp", "reaches_0.95",
    "tap_gpu_h_wk", "tap_slo_in", "tap_slo_in_4dp", "tap_reaches_0.95", "extra_over_tap_pct",
    "extra_over_tap_pct_int", "calendar_learn_prior_gpu_h_wk",
    "extra_over_calendar_learn_prior_pct", "source")


def _rows(blocks: Sequence[Block], columns: Sequence[str]) -> pd.DataFrame:
    """One row per block and policy of ``SWEEP_POLICIES`` with the named columns."""
    rows: List[Dict[str, object]] = []
    for b in blocks:
        for policy in (p for p in SWEEP_POLICIES if p in b.stats):
            s = b.stats[policy]
            cells: Dict[str, object] = {
                "gpu_h_wk": f"{s.gpu:.1f}", "usd_wk": f"{s.usd:.1f}", "slo_in": f"{s.slo_in:.3f}",
                "slo_off": f"{s.slo_off:.3f}", "excess_usd_wk": f"{s.excess:.1f}",
                "n_seeds": s.seeds, "source": s.source}
            rows.append({"case": b.case, **b.key, "policy": policy,
                         **{c: cells[c] for c in columns}})
    return pd.DataFrame(rows)


def lead_sweep_table(blocks: Sequence[Block]) -> pd.DataFrame:
    """Section 18: GPU-hours per week, in-class SLO and excess over the offline optimum per lead."""
    return _rows(blocks, LEAD_COLUMNS)


def sweep_table(blocks: Sequence[Block]) -> pd.DataFrame:
    """Sections 19-22: GPU-hours and USD per week, both attainments and the excess per block."""
    return _rows(blocks, SWEEP_COLUMNS)


def ablation_table(blocks: Sequence[Block]) -> pd.DataFrame:
    """Section 14, with GPU-hours against the same policy in the main setting (unrounded means)."""
    rows: List[Dict[str, object]] = []
    main: Dict[Tuple[str, str], float] = {}
    for b in blocks:
        for policy in (p for p in ABLATION_POLICIES if p in b.stats):
            s = b.stats[policy]
            if b.key["setting"] == ABLATION_MODES[0][1]:
                main[(b.case, policy)] = s.gpu
            base = main.get((b.case, policy))
            rows.append({"case": b.case, "setting": b.key["setting"], "policy": policy,
                         "gpu_h_wk": f"{s.gpu:.1f}",
                         "gpu_h_wk_vs_main_pct": f"{100 * (s.gpu / base - 1):.1f}" if base else "",
                         "usd_wk": f"{s.usd:.1f}", "slo_in": f"{s.slo_in:.3f}",
                         "slo_off": f"{s.slo_off:.3f}",
                         "timeouts_per_million_requests": int(round(s.timeouts)),
                         "n_seeds": s.seeds, "source": f"{s.source}/offset/offset.csv"})
    return pd.DataFrame(rows)


def _yes(slo: float) -> str:
    return "yes" if slo >= SLO_FLOOR else "no"


def _pct(value: float, base: float, fmt: str) -> str:
    """``value`` against ``base`` in percent, formatted with ``fmt``."""
    return format(100.0 * (value / base - 1.0), fmt)


def floor_table(blocks: Sequence[Block]) -> pd.DataFrame:
    """Per block: TAP at the floor, the cheapest floor-tuned feedback rule whose unrounded
    attainment is >= 0.95 (or none) and the floor-tuned reactive rule, against TAP and
    calendar_learn_prior."""
    rows: List[Dict[str, object]] = []
    for b in blocks:
        tap, plan = b.stats["tap"], b.stats["calendar_learn_prior"]
        ok = [p for p in FEEDBACK if p in b.stats and b.stats[p].slo_in >= SLO_FLOOR]
        cheapest = min(ok, key=lambda p: b.stats[p].gpu) if ok else None
        for label, policy in zip(FLOOR_ROWS, (cheapest, "reactive")):
            row: Dict[str, object] = dict.fromkeys(FLOOR_COLUMNS, "")
            row.update({"row": label, "feedback_rule": policy or "none reaches 0.95",
                        "tap_gpu_h_wk": f"{tap.gpu:.1f}", "tap_slo_in": f"{tap.slo_in:.3f}",
                        "tap_slo_in_4dp": f"{tap.slo_in:.4f}", "tap_reaches_0.95": _yes(tap.slo_in),
                        "calendar_learn_prior_gpu_h_wk": f"{plan.gpu:.1f}"})
            sources = [tap.source, plan.source]
            if policy is not None:
                s = b.stats[policy]
                row.update({"gpu_h_wk": f"{s.gpu:.1f}", "slo_in": f"{s.slo_in:.3f}",
                            "slo_in_4dp": f"{s.slo_in:.4f}", "lowest_seed_4dp": f"{s.slo_min:.4f}",
                            "reaches_0.95": _yes(s.slo_in),
                            "extra_over_tap_pct": _pct(s.gpu, tap.gpu, "+.1f"),
                            "extra_over_tap_pct_int": _pct(s.gpu, tap.gpu, "+.0f"),
                            "extra_over_calendar_learn_prior_pct": _pct(s.gpu, plan.gpu, "+.1f")})
                sources.append(s.source)
            row["source"] = "; ".join(dict.fromkeys(sources))
            rows.append({"case": b.case, **b.key, **row})
    return pd.DataFrame(rows)


def lead_crossing_table(blocks: Sequence[Block]) -> pd.DataFrame:
    """Per case and rule of section 18: the smallest lead time at which the unrounded attainment
    falls below 0.95, its attainment there and whether it stays below at every longer lead."""
    rows: List[Dict[str, object]] = []
    for case in dict.fromkeys(b.case for b in blocks):
        curve = sorted(((float(b.key["lead_min"]), b) for b in blocks if b.case == case),
                       key=lambda item: item[0])
        for policy in ("tap", *FEEDBACK):
            slo = [(d, b.stats[policy].slo_in) for d, b in curve]
            below = [d for d, s in slo if s < SLO_FLOOR]
            row: Dict[str, object] = {"case": case, "policy": policy,
                                      "leads_min": ", ".join(f"{d:g}" for d, _ in slo),
                                      "first_lead_min_below_0.95": "none", "slo_in_4dp_there": "",
                                      "below_0.95_at_every_longer_lead": ""}
            if below:
                longer = [s for d, s in slo if d > below[0]]
                stays = "yes" if all(s < SLO_FLOOR for s in longer) else "no"
                row.update({"first_lead_min_below_0.95": f"{below[0]:g}",
                            "slo_in_4dp_there": f"{dict(slo)[below[0]]:.4f}",
                            "below_0.95_at_every_longer_lead": stays if longer else "-"})
            row["source"] = "section 18 (unrounded means of its rows)"
            rows.append(row)
    return pd.DataFrame(rows)


def adoption_gap_table(blocks: Sequence[Block]) -> pd.DataFrame:
    """Section 23: per adoption level, GPU-hours of each SLO-floor feedback rule over the learned
    planner's (nearest integer) and TAP's (1 decimal and nearest integer), unrounded means."""
    rows: List[Dict[str, object]] = []
    for b in blocks:
        plan, tap = b.stats["calendar_learn_prior"], b.stats["tap"]
        for policy in (p for p in FEEDBACK if p in b.stats):
            fb = b.stats[policy]
            rows.append({"case": b.case, "section_adoption": b.key["section_adoption"],
                         "feedback_rule": policy, "feedback_gpu_h_wk": f"{fb.gpu:.1f}",
                         "feedback_slo_in": f"{fb.slo_in:.3f}",
                         "planner_gpu_h_wk": f"{plan.gpu:.1f}",
                         "planner_slo_in": f"{plan.slo_in:.3f}",
                         "feedback_extra_pct": _pct(fb.gpu, plan.gpu, ".0f"),
                         "reaches_floor": _yes(fb.slo_in),
                         "feedback_slo_in_4dp": f"{fb.slo_in:.4f}",
                         "tap_gpu_h_wk": f"{tap.gpu:.1f}", "tap_slo_in": f"{tap.slo_in:.3f}",
                         "feedback_extra_over_tap_pct": _pct(fb.gpu, tap.gpu, "+.1f"),
                         "feedback_extra_over_tap_pct_int": _pct(fb.gpu, tap.gpu, "+.0f"),
                         "source": "; ".join(dict.fromkeys([plan.source, fb.source, tap.source]))})
    return pd.DataFrame(rows)


__all__ = ["ABLATION_POLICIES", "FLOOR_COLUMNS", "FLOOR_ROWS", "SWEEP_POLICIES", "ablation_table",
           "adoption_gap_table", "floor_table", "lead_crossing_table", "lead_sweep_table",
           "sweep_table"]
