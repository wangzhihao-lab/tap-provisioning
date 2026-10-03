"""Canonical numbers, part 4: the billing sensitivity of the rerun round.

Imported by ``canonical_numbers.py`` (section 31) and ``canonical_tables.py`` (``ext_billing.tex``).
Every row is measured by ``runs_billing.py`` (``block``, ``measure``) from the runs' own
``offset.csv`` on the laptop's streams of cell C, seeds 10-14, under two calibers of a replica
start:

``order``  billed from the order: each start pays its whole lead, ``beta = c d`` with ``d`` the
           one-minute slots the engine executes (8 at the instance tier, 2 at the 72B replica
           tier); the planners re-solve and the baselines are retuned at that price.
``main``   the paper's: billed from the first serving minute plus one replica-minute per start,
           ``beta = c``.

Billed GPU-hours per week = ``(gpu_hours + starts * d / 60) / weeks`` (``d = 1`` under ``main``)
and USD per week = ``(run_cost + switch_cost) / weeks``, means over the seeds.  Rounding: billed
GPU-hours and USD per week to 1 decimal, attainment to 3 (to 4, with the lowest seed, within 0.01
of 0.95), the change against TAP in percent to 1 decimal and to the nearest integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import runs_billing as cfb
from canonical_extra import SLO_FLOOR, VARIANT_WORDS, _where

SEEDS = 5  # seeds 10-14 of every run
NEAR_FLOOR = 0.01  # rows this close to 0.95 also carry 4 decimals and the lowest seed
_COST, _FLOOR = VARIANT_WORDS["cost (tier's own)"], VARIANT_WORDS["floor (dispersed slice)"]
NAMES: Dict[str, str] = {  # runs_billing's labels -> the policy names of section 27a
    "TAP": "tap", "calendar_learn_prior": "calendar_learn_prior", "calendar": "calendar",
    "reactive (cost)": f"reactive, {_COST}", "velocity (cost)": f"velocity, {_COST}",
    "fixed": "fixed", "reactive (floor)": f"reactive, {_FLOOR}",
    "velocity (floor)": f"velocity, {_FLOOR}"}
FEEDBACK: Tuple[str, ...] = ("reactive (cost)", "velocity (cost)", "reactive (floor)",
                             "velocity (floor)")


@dataclass(frozen=True)
class Measured:
    """One row of ``runs_billing.measure``, unrounded, with its case, tier and caliber."""

    case: str
    tier: str
    caliber: str
    d_slots: int  # replica-minutes billed per start (1 under main)
    label: str
    billed: float
    usd: float
    slo: float
    slo_min: float
    source: str  # the offset.csv, relative to the repository


def measured(repo: Path) -> List[Measured]:
    """Both calibers of every case-tier of ``runs_billing.BLOCKS``, in its row order.

    Raises unless each case-tier and caliber has exactly the rows of ``NAMES``, five seeds each.
    """
    if cfb.PILOT.resolve() != repo.resolve():
        raise ValueError(f"runs_billing reads {cfb.PILOT}, not {repo}")
    out: List[Measured] = []
    for case, tier, _title, d_slots, lap in cfb.BLOCKS:
        order, main, notes = cfb.block(case, tier, d_slots, lap)
        if notes:
            raise ValueError(f"{case}_{tier}: {'; '.join(notes)}")
        for caliber, found, d in (("order", order, d_slots), ("main", main, 1)):
            labels = [str(m["label"]) for m in found]
            if sorted(labels) != sorted(NAMES):
                raise ValueError(f"{case}_{tier}, {caliber}: rows {labels}, expected {list(NAMES)}")
            for m in found:
                if int(m["seeds"]) != SEEDS:
                    raise ValueError(f"{m['source']}: {m['label']} has {m['seeds']} seeds")
                out.append(Measured(case, tier, caliber, d, str(m["label"]), float(m["billed"]),
                                    float(m["usd"]), float(m["slo"]), float(m["slo_min"]),
                                    f"{m['source']}"))
    return out


def _pct(value: float, base: float) -> Tuple[str, str]:
    """``value`` against ``base`` in percent, to 1 decimal and to the nearest integer."""
    rel = 100.0 * (value / base - 1.0)
    return f"{rel:+.1f}", f"{rel:+.0f}"


def _row(m: Measured, tap: Measured) -> Dict[str, object]:
    """One row of section 31a."""
    near = abs(m.slo - SLO_FLOOR) <= NEAR_FLOOR
    pct, pct_int = ("-", "-") if m is tap else _pct(m.billed, tap.billed)
    return {**_where(m.case, m.tier), "caliber": m.caliber,
            "beta": f"c x {m.d_slots}" if m.caliber == "order" else "c",
            "policy": NAMES[m.label], "billed_gpu_h_wk": f"{m.billed:.1f}",
            "usd_wk": f"{m.usd:.1f}", "slo_in": f"{m.slo:.3f}",
            "slo_in_4dp_if_within_0.01_of_0.95": f"{m.slo:.4f}" if near else "",
            "lowest_seed_4dp_if_within_0.01_of_0.95": f"{m.slo_min:.4f}" if near else "",
            "vs_tap_pct": pct, "vs_tap_pct_int": pct_int, "source": m.source}


def _floor_row(group: List[Measured], tap: Measured) -> Dict[str, object]:
    """One row of section 31b: TAP at the floor and the cheapest feedback row reaching it."""
    ok = [m for m in group if m.label in FEEDBACK and m.slo >= SLO_FLOOR]
    best: Optional[Measured] = min(ok, key=lambda m: m.billed) if ok else None
    pick = best if best is not None else tap  # without a candidate, tap only supplies the keys
    pct, pct_int = _pct(pick.billed, tap.billed)
    cells: Dict[str, str] = {
        "feedback_billed_gpu_h_wk": f"{pick.billed:.1f}", "feedback_slo_in": f"{pick.slo:.3f}",
        "feedback_slo_in_4dp": f"{pick.slo:.4f}",
        "feedback_lowest_seed_4dp": f"{pick.slo_min:.4f}",
        "extra_over_tap_pct": pct, "extra_over_tap_pct_int": pct_int}
    if best is None:
        cells = dict.fromkeys(cells, "")
    sources = list(dict.fromkeys([tap.source, pick.source]))  # the files of the two rows
    return {**_where(tap.case, tap.tier), "caliber": tap.caliber,
            "tap_billed_gpu_h_wk": f"{tap.billed:.1f}", "tap_slo_in": f"{tap.slo:.3f}",
            "tap_slo_in_4dp": f"{tap.slo:.4f}",
            "tap_reaches_0.95": "yes" if tap.slo >= SLO_FLOOR else "no",
            "cheapest_feedback_at_0.95": NAMES[best.label] if best is not None else "none",
            **cells, "source": "; ".join(sources)}


def billing_tables(repo: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Section 31a (every row of both calibers) and 31b (the floor per case-tier and caliber)."""
    groups: Dict[Tuple[str, str, str], List[Measured]] = {}
    for m in measured(repo):
        groups.setdefault((m.case, m.tier, m.caliber), []).append(m)
    rows: List[Dict[str, object]] = []
    floor: List[Dict[str, object]] = []
    for group in groups.values():
        tap = next(m for m in group if m.label == "TAP")
        rows += [_row(m, tap) for m in group]
        floor.append(_floor_row(group, tap))
    return pd.DataFrame(rows), pd.DataFrame(floor)


__all__ = ["FEEDBACK", "NAMES", "Measured", "billing_tables", "measured"]
