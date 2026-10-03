"""Draft figures of the validation round, drawn from the result tables and runs named below.

    uv run python scripts/sim/validation_figures.py            # -> results/sim/validation/figures/

F4   main result: GPU-hours per week against in-class SLO, one marker per policy, one panel per
     central case x provisioning tier (cell C, whole run, r = 30).
F4b  the same policies by teaching week at the instance tier (SLO), weeks 1 and 5 marked.
F5   the six timing cells: SLO and GPU-hours per week per cell for the planners that differ.
F7   the lead time: in-class violation and excess cost per week against d, with Lemma 1's bound.
F8   the information ladder: GPU-hours and SLO per policy in the order of what each one knows.

F4, F4b and F8 read cell C of the laptop's runs, TAP included, from
``results/runs/paired_cell_c.csv``, and F5 and F8's rung at the bell (cell B) the six
timing cells of the laptop's runs, TAP included, from ``results/runs/cells_laptop.csv``
(means over seeds).  The cloud tables ``validation_report.py`` writes, ``cells_long.csv`` and
``weekly_learn.csv``, hold no TAP rows and are drawn on request only: ``--f4-source cloud`` draws
F4 and F4b from them as before, into ``f4_main_cloud`` and ``f4b_weekly_cloud``,
``--f5-source cloud`` F5, into ``f5_timing_cells_cloud``, and ``--f8-source cloud`` F8, into
``f8_information_ladder_cloud``.
F7 reads the lead-time sweeps under ``outputs/validation_local/sensitivity_sweeps``
(one run per d: F7 draws whichever have landed) for the planners, their re-runs with the
SLO-floor configurations under ``runs/s3_consistency`` for the feedback rules and TAP's runs
under ``runs/tap``, each only where its offline rows are the sweep's, and Lemma 1's weekly
bound at the executed blind time from ``results/theory/validation_<case>/sim/`` and
``sim_dgrid/instantiation.csv``.  Each figure is drawn at the size it is printed at, with every
text at 8 pt: see ``validation_figures_style``.  Every policy has the colour, marker and dash of
``pilot.policy_style``, and every legend its order.
PDF + PNG (F8 also SVG, text editable).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator, MaxNLocator,  # noqa: E402
                               NullLocator)
from matplotlib.transforms import Bbox  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style, save  # noqa: E402
from pilot.policy_style import GREEN, GREY, INDIGO, rank  # noqa: E402
from pilot.policy_style import style as policy_style  # noqa: E402

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_figures_style as st  # noqa: E402

logger = logging.getLogger("validation_figures")

CASES: Dict[str, str] = {"c2_32b_shared4k": "32B-AWQ, shared 4k context",
                         "c3_72b_shared4k": "72B-AWQ, shared 4k context"}
LEAD_LABEL: Dict[str, Dict[str, str]] = {"c2_32b_shared4k": {"d_rep": "1.0 min", "d_inst": "8 min"},
                                         "c3_72b_shared4k": {"d_rep": "1.5 min", "d_inst": "8 min"}}
LABELS: Dict[str, str] = {
    "tap": "TAP", "offline": "Offline optimum", "informed": "Informed optimum",
    "calendar": "Timetable + known timing", "calendar_adapt": "Timetable + online size correction",
    "calendar_learn_prior": "Timetable + learned timing (prior)", "calendar_learn": "Timetable + learned timing (no prior)",
    "stale_calendar": "Last semester's timetable", "forecast_ilp": "History + ILP",
    "guard": "Bell-aware guard", "lcp": "LCP", "velocity": "Token velocity", "reactive": "Reactive (HPA)",
    "fixed": "Base tier only",
    "velocity (SLO>=0.95, disp)": "Token velocity, SLO-floor tuned", "reactive (SLO>=0.95, disp)": "Reactive, SLO-floor tuned",
}
# F4 draws its markers in this order: TAP last, on top of the cluster at the optimum it sits in.
F4_POLICIES = ("offline", "calendar", "calendar_adapt", "calendar_learn_prior", "calendar_learn", "stale_calendar",
               "forecast_ilp", "guard", "lcp", "velocity", "velocity (SLO>=0.95, disp)", "reactive",
               "reactive (SLO>=0.95, disp)", "fixed", "tap")
F5_POLICIES = ("tap", "calendar", "calendar_learn_prior", "velocity", "reactive", "guard", "offline")
# Each feedback rule is tuned twice, once for cost and once against the SLO floor, and neither
# tuning wins every timing cell.  F5 draws, cell by cell, whichever of the two attains more in
# class, and reads the GPU-hours below off that same tuning.
F5_BEST_OF: Tuple[str, ...] = ("velocity", "reactive")
TUNED_SUFFIX = " (SLO>=0.95, disp)"
F5_BEST_LABELS: Dict[str, str] = {"velocity": "Token velocity, best tuning",
                                  "reactive": "Reactive, best tuning"}
# F7's runs name the two feedback rules plainly, but they are the SLO-floor tuning of the nearest
# measured tier: each series is drawn in the colour and marker of its twin in F4, hollow, and
# named as there.  (csv policy, key into ``style_of`` and ``st.SHORT_LABELS``).
F7_SERIES: Tuple[Tuple[str, str], ...] = (("tap", "tap"),
                                          ("calendar_learn_prior", "calendar_learn_prior"),
                                          ("calendar", "calendar"),
                                          ("reactive", "reactive (SLO>=0.95, disp)"),
                                          ("velocity", "velocity (SLO>=0.95, disp)"),
                                          ("offline", "offline"))
# Where each series of F7 comes from at a lead time d: the planners and the offline optimum from
# the sensitivity round's sweep (``sensitivity_sweeps/<case>_lead_d<d>``), the feedback rules from its
# re-run with the resource rule's SLO-floor configurations (the run of the same name under
# ``F7_FLOOR_RUNS``), TAP from its own runs (``tap_run``).  A run joins the sweep's point only if
# it replays the sweep's stream, i.e. holds the sweep's offline rows.
F7_SWEEP_POLICIES: Tuple[str, ...] = ("calendar_learn_prior", "calendar", "offline")
F7_FLOOR_POLICIES: Tuple[str, ...] = ("reactive", "velocity")
F7_FLOOR_RUNS = PILOT / "outputs/validation_local/runs/s3_consistency"
F7_TAP_RUNS = PILOT / "outputs/validation_local/runs/tap"
# Cell C of ``SweepConfig.trigger_offset``, by the columns a run records it in: TAP's runs at the
# measured tiers hold all six timing cells.
F7_OFFSET_KEYS: Tuple[str, ...] = ("trigger_offset_mean", "trigger_offset_sd",
                                   "trigger_offset_course_sd", "trigger_offset_common_sd",
                                   "forecast_offset", "forecast_offset_sd")
F7_CELL_C: Tuple[float, ...] = (-8.7, 5.3, 4.0, 0.0, -8.7, 6.6)
F7_BOUND_LABEL = "Lemma 1 bound (worst case)"
# Lemma 1 at the simulated streams' sigma_2 (Section VI-A) and r = 30, read from the validation
# round's instantiation (``sim``) and, for the blind times it does not hold, from the lead-time
# grid's (``sim_dgrid``); a blind time both hold must have one value.
F7_BOUND_SIGMA2 = 0.57
F7_BOUND_DIRS: Tuple[str, ...] = ("sim", "sim_dgrid")
F7_BOUND_DASH: Tuple[float, Tuple[float, float]] = (0, (4, 2))  # neutral grey, under the data
# The three planners do not move with d and sit within a marker of each other in both rows, TAP
# on top of the plan it is built on, in the same indigo: TAP is drawn one marker to the left of
# each d and that plan one to the right, the timetable with known timing between them.  Factors
# on d (the axis is logarithmic); an empty map draws every series at its d.
F7_DODGE_FACTOR = 1.09
F7_DODGE: Dict[str, float] = {"tap": 1.0 / F7_DODGE_FACTOR, "calendar_learn_prior": F7_DODGE_FACTOR}
# The two tiers the validation round measured: the cold start of a replica, and eight minutes for
# an instance.
F7_TIERS: Dict[str, Tuple[float, float]] = {"c2_32b_shared4k": (1.0, 8.0),
                                            "c3_72b_shared4k": (1.5, 8.0)}
F7_X_TICKS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 15.0)
F7_XLIM: Tuple[float, float] = (0.85, 18.0)
F7_SEEDS: Tuple[int, ...] = (10, 11, 12, 13, 14)
F7_WEEKS = 6  # gpu_hours and total_cost are six-week totals
# TAP knows what the plan with learned timing and the prior knows, and the counts it has served.
LADDER = ("fixed", "velocity", "reactive (SLO>=0.95, disp)", "guard", "stale_calendar", "forecast_ilp",
          "calendar@B", "calendar_learn", "calendar_learn_prior", "tap", "calendar", "informed",
          "offline")
LADDER_LABELS: Dict[str, str] = {**LABELS, "calendar@B": "Timetable at the bell (no timing)"}
# The six cells are named in the caption of F5; at 8 pt only the letter fits under a panel.
CELL_ORDER: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SLO_FLOOR = 0.95

# Printed sizes: F4 and F4b are the two subfloats of one figure* (Fig. 5 of the paper), F5, F7
# and F8 are one column wide.  At 8 pt F8's value labels come off the bars, where they would sit
# on the bar ends, and the ladder gets its own shorter names; the text quotes every one of the
# values anyway.
#
# F4 and F4b make one grid of six equal panels, two rows (32B, 72B) by three columns (the two
# tiers of F4, then F4b), set side by side at their natural size.  Both files place their axes
# from these lengths in cm instead of through a layout engine, which would size each file's
# panels to its own labels: every panel then has one size and the rows sit at the same heights.
F4_PAIR_CM = 17.8      # width of F4 plus F4b: \textwidth (18.14 cm) less the \hfil between them
F4_LEFT_CM = 0.88      # left of the first column: the y label, the y tick labels and the ticks
F4_GAP_CM = 0.40       # between the two columns of F4: the x tick labels that overhang
F4_RIGHT_CM = 0.20     # right of the last column: the last x tick label
F4_PANEL_CM: Tuple[float, float] = ((F4_PAIR_CM - 2 * F4_LEFT_CM - F4_GAP_CM - 2 * F4_RIGHT_CM) / 3,
                                    2.8)
F4_TOP_CM = 0.08       # above the top row: the top y tick label
F4_ROW_GAP_CM = 0.55   # between the rows: the top row's x tick labels
F4_XLABEL_CM = 0.90    # below the bottom row: its x tick labels and the x label
F4_LEGEND_CM = 1.38    # the legend band at the bottom, four rows in both files
F4_BOTTOM_CM = F4_LEGEND_CM + F4_XLABEL_CM  # lower edge of the bottom row
F4_SIZE_CM: Tuple[float, float] = (F4_LEFT_CM + 2 * F4_PANEL_CM[0] + F4_GAP_CM + F4_RIGHT_CM,
                                   F4_BOTTOM_CM + 2 * F4_PANEL_CM[1] + F4_ROW_GAP_CM + F4_TOP_CM)
F4B_SIZE_CM: Tuple[float, float] = (F4_LEFT_CM + F4_PANEL_CM[0] + F4_RIGHT_CM, F4_SIZE_CM[1])
F4_Y_TICKS: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
F5_SIZE_CM: Tuple[float, float] = (st.COLUMN_CM, 8.0)
# With TAP, F5's legend takes a fourth row out of a page of the same height.  The rows of panels
# sit 0.02 of the figure apart (0.06 in F7) so that a panel stays about as long as its y label,
# which the constrained layout does not count: at 0.06 "In-class SLO attainment" runs to the top
# edge.  The attainment keeps its ticks at quarters, which the shorter axis would thin to halves.
F5_HSPACE = 0.02
F5_SLO_TICKS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
F7_SIZE_CM: Tuple[float, float] = (st.COLUMN_CM, 7.5)
F8_SIZE_CM: Tuple[float, float] = (st.COLUMN_CM, 6.0)
F8_VALUE_LABELS = False
# From 0 the ladder's GPU-hours are one bar to the eye: the rungs from the history forecast to
# the offline optimum span 203 to 230 h.  The axis starts at a round value just below the
# shortest bar, which the caption says, and drops to the next ten below it if the numbers ever
# go lower.
F8_GPU_XMIN = 160.0
F8_LABELS: Dict[str, str] = {
    "fixed": "Base tier only", "velocity": "Token velocity",
    "reactive (SLO>=0.95, disp)": "Reactive, SLO-tuned", "guard": "Bell-aware guard",
    "stale_calendar": "Last semester's timetable", "forecast_ilp": "History + ILP",
    "calendar@B": "Timetable at the bell", "calendar_learn": "Timetable + learned timing",
    "calendar_learn_prior": "Timetable + learned, prior", "tap": "TAP",
    "calendar": "Timetable + known timing", "informed": "Informed optimum",
    "offline": "Offline optimum",
}


def style_of(policy: str, scatter: bool = False) -> Dict[str, object]:
    """Colour, marker and dash of ``policy`` from ``pilot.policy_style``; SLO-tuned twins hollow.

    A scatter keeps the colour and the marker and drops the dash.
    """
    kw = policy_style(policy).line_kw()
    if scatter:
        kw["linestyle"] = "none"
    return kw


def drawing_order(policy: str) -> Tuple[bool, int]:
    """Sort key for drawing the lines: the legend's order reversed, so that its first entry ends
    on top of the lines it runs along, and the offline optimum last of all, since its small star
    hides nothing while any other marker would hide the star."""
    return (policy == "offline", -rank(policy))


def edge_pt(policy: str, solid_pt: float) -> float:
    """Marker edge width: a hollow marker is drawn by its edge alone, so it gets 1.0 pt."""
    return 1.0 if policy_style(policy).hollow else solid_pt


def load_cells(tables: Path) -> pd.DataFrame:
    frame = pd.read_csv(tables / "cells_long.csv")
    return frame[(frame["ratio"] == 30) & (frame["tier"] == frame["tuned_at"]) & (frame["slots"] == "main")]


# F4 and F4b, the paper's main-results figure, read cell C of the laptop's runs (the rerun round,
# section 3a), and F5 the six timing cells of the laptop's runs: one row per
# case, tier, policy variant and week (and, in the six-cell table, timing cell), week 0 the whole
# run and weeks 1-6 the teaching weeks.
CELL_C_LAPTOP = REPO / "results/runs/paired_cell_c.csv"
CELLS_LAPTOP = REPO / "results/runs/cells_laptop.csv"
F4_SOURCES: Tuple[str, ...] = ("laptop", "cloud")
LAPTOP_TIERS: Dict[str, str] = {"rep": "d_rep", "inst": "d_inst"}
# The two tunings of each feedback rule that F4, F5 and F8 draw, as the cloud tables had them: for
# cost at the tier's own tuning, and against the SLO floor on the dispersed slice (the hollow
# twin).  The instance tier's run at the replica tier's tuning and the floor tuned on the
# synchronous slice are left out.  (tier, variant) -> suffix that makes the policy key.
LAPTOP_VARIANTS: Dict[Tuple[str, str], str] = {
    ("d_rep", "cost (replica-tier tuning)"): "", ("d_inst", "cost (tier's own)"): "",
    ("d_rep", "floor (dispersed slice)"): TUNED_SUFFIX,
    ("d_inst", "floor (dispersed slice)"): TUNED_SUFFIX}


def laptop_keys(raw: pd.DataFrame) -> pd.DataFrame:
    """The rows of a laptop table that the figures draw, keyed by tier and policy as the cloud tables.

    ``tier`` becomes ``d_rep`` or ``d_inst``.  A row of variant ``-`` keeps its policy's name, a
    feedback rule's variant becomes its policy key through ``LAPTOP_VARIANTS``, and the rule's
    other variants are left out.
    """
    tier = raw["tier"].map(LAPTOP_TIERS)
    suffix = pd.Series(["" if v == "-" else LAPTOP_VARIANTS.get((t, v))
                        for t, v in zip(tier, raw["variant"])], index=raw.index, dtype=object)
    kept = raw[suffix.notna()]
    return kept.assign(tier=tier[kept.index], policy=kept["policy"] + suffix[kept.index])


def load_laptop_cell_c(path: Path = CELL_C_LAPTOP) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cell C of the laptop's runs as the two cloud tables F4 and F4b read: ``(cells, weekly)``.

    ``cells`` holds the whole run (week 0) under the column names of ``cells_long.csv``,
    ``weekly`` the in-class attainment of weeks 1-6 as the columns ``w1`` .. ``w6`` of
    ``weekly_learn.csv``.  A feedback rule's variant becomes its policy key through
    ``LAPTOP_VARIANTS``; its other variants are left out.  The values are taken as they are.
    """
    kept = laptop_keys(pd.read_csv(path)).assign(cell="C", ratio=30)
    keys = ["case", "tier", "policy", "week"]
    if kept.duplicated(keys).any():
        raise ValueError(f"{path}: more than one row of "
                         f"{kept.loc[kept.duplicated(keys), keys].to_dict('records')}")
    cells = kept[kept["week"] == 0].rename(columns={"gpu_h_per_week": "gpu_h_wk_mean",
                                                    "slo_in": "slo_in_mean"})
    weekly = kept[kept["week"] > 0].pivot(index=["tier", "ratio", "case", "policy"],
                                          columns="week", values="slo_in")
    weekly.columns = [f"w{int(w)}" for w in weekly.columns]
    weekly = weekly.reset_index()
    weekly.insert(3, "metric", "slo_in")
    logger.info("laptop cell C of %s, %d whole-run rows (seeds %s), %d weekly rows", path,
                len(cells), sorted(set(cells["seeds"])), len(weekly))
    return cells.reset_index(drop=True), weekly


def f4_tables(source: str, tables: Path, cell_c: Path = CELL_C_LAPTOP
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The ``(cells, weekly)`` tables that F4 and F4b draw.

    ``laptop``: cell C of the laptop's runs, TAP included (the paper's main-results figure).
    ``cloud``: the cloud tables ``validation_report.py`` writes, which hold no TAP rows (the
    figure before).
    """
    if source == "laptop":
        return load_laptop_cell_c(cell_c)
    if source == "cloud":
        return load_cells(tables), pd.read_csv(tables / "weekly_learn.csv")
    raise ValueError(f"unknown F4 source {source!r}: one of {F4_SOURCES}")


def load_laptop_cells(path: Path = CELLS_LAPTOP) -> pd.DataFrame:
    """The six timing cells of the laptop's runs as the rows of ``cells_long.csv`` that F5 reads.

    The whole run (week 0) of every case, tier, timing cell and policy key (``laptop_keys``),
    with ``gpu_h_per_week`` and ``slo_in`` under the names ``gpu_h_wk_mean`` and
    ``slo_in_mean``.  The values are taken as they are.
    """
    kept = laptop_keys(pd.read_csv(path)).assign(ratio=30)
    keys = ["case", "tier", "cell", "policy", "week"]
    if kept.duplicated(keys).any():
        raise ValueError(f"{path}: more than one row of "
                         f"{kept.loc[kept.duplicated(keys), keys].to_dict('records')}")
    cells = kept[kept["week"] == 0].rename(columns={"gpu_h_per_week": "gpu_h_wk_mean",
                                                    "slo_in": "slo_in_mean"})
    logger.info("laptop timing cells %s of %s, %d whole-run rows (seeds %s)",
                "".join(sorted(set(cells["cell"]))), path, len(cells), sorted(set(cells["seeds"])))
    return cells.reset_index(drop=True)


def f5_cells(source: str, tables: Path, six_cells: Path = CELLS_LAPTOP) -> pd.DataFrame:
    """The cells F5 draws, in the columns of ``cells_long.csv``.

    ``laptop``: the six timing cells of the laptop's runs, TAP included (the paper's figure).
    ``cloud``: the cloud tables ``validation_report.py`` writes, which hold no TAP rows (the
    figure before).
    """
    if source == "laptop":
        return load_laptop_cells(six_cells)
    if source == "cloud":
        return load_cells(tables)
    raise ValueError(f"unknown F5 source {source!r}: one of {F4_SOURCES}")


# F8's rung at the bell is the timetable plan of cell B: the arrivals of cell C, the plan placed at
# the bell.  The laptop ladder reads it from the six timing cells of the laptop's runs, whose cell
# C rows are those of ``CELL_C_LAPTOP`` that the other rungs are read from.
# (policy, cell) of each such rung.
F8_CELL_RUNGS: Tuple[Tuple[str, str], ...] = (("calendar", "B"),)


def f8_cells(source: str, tables: Path, cell_c: Path = CELL_C_LAPTOP,
             six_cells: Path = CELLS_LAPTOP) -> pd.DataFrame:
    """The cells F8 draws, in the columns of ``cells_long.csv``.

    ``laptop``: cell C of the laptop's runs, TAP included, in the variants F4 draws (a feedback
    rule tuned for cost on its tier, or against the SLO floor on the dispersed slice), and the
    rungs of ``F8_CELL_RUNGS`` from the six timing cells of the laptop's runs.  ``cloud``: the
    cloud tables alone, which hold no TAP rows (the figure before).
    """
    if source == "cloud":
        return load_cells(tables)
    if source != "laptop":
        raise ValueError(f"unknown F8 source {source!r}: one of {F4_SOURCES}")
    six = load_laptop_cells(six_cells)
    extra = pd.concat([six[(six["policy"] == p) & (six["cell"] == c)] for p, c in F8_CELL_RUNGS])
    logger.info("F8: %s from %s", ", ".join(f"{p}@{c}" for p, c in F8_CELL_RUNGS), six_cells)
    cells = pd.concat([load_laptop_cell_c(cell_c)[0], extra], ignore_index=True)
    keys = ["case", "tier", "cell", "policy"]
    if cells.duplicated(keys).any():
        raise ValueError(f"F8: more than one row of "
                         f"{cells.loc[cells.duplicated(keys), keys].to_dict('records')}")
    return cells


# Inset span as a fraction of the offline optimum's GPU-hours.  At 72B the stale timetable sits
# 9.8% above the optimum, on the corner of a +8% box, where a connector would start under its
# marker: the 72B box is widened to take it in.
F4_ZOOM_X: Dict[str, Tuple[float, float]] = {"c2_32b_shared4k": (-0.01, 0.08),
                                             "c3_72b_shared4k": (-0.01, 0.13)}
F4_ZOOM_Y = (0.90, 1.005)   # inset span in in-class SLO attainment
F4_INSET_SIZE: Tuple[float, float] = (0.40, 0.45)  # the inset's plot box, in panel fraction
# Lower-left corners of the inset in panel fraction, tried in this order.  Every panel is empty
# right of the cluster and above its caption.  The inset's y tick labels go on its right, where
# neither connector line runs, and still inside the panel.  The last corner, further left, is for
# an instance-tier panel where a floor-tuned rule sits under the connectors of the others.
F4_INSET_AT: Tuple[Tuple[float, float], ...] = ((0.45, 0.34), (0.45, 0.38), (0.42, 0.34),
                                                (0.42, 0.38), (0.33, 0.34))
F4_CLEAR_PT = 2.0  # room kept between the inset, tick labels included, and any marker or text


def f4_inset_axes(ax: plt.Axes, at: Tuple[float, float], lo: float, hi: float) -> plt.Axes:
    """An inset at ``at`` with its limits, ticks and printed tick size, and nothing drawn in it."""
    axins = ax.inset_axes((*at, *F4_INSET_SIZE))
    axins.set_xlim(lo, hi)
    axins.set_ylim(*F4_ZOOM_Y)
    axins.set_yticks([0.90, 0.95, 1.00])
    # two x ticks: three labels of three digits would touch at 8 pt in a box 2 cm wide
    axins.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    axins.yaxis.tick_right()
    st.thin_axes(axins, spine_pt=0.6)
    return axins


def inset_place(ax: plt.Axes, lo: float, hi: float) -> Tuple[float, float]:
    """First corner of ``F4_INSET_AT`` at which the inset covers no marker and no text of the panel.

    The inset is measured with its tick labels; a corner at which a connector line would also
    touch a marker outside the zoomed box is passed over while a cleaner one exists.
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    clear = F4_CLEAR_PT * fig.dpi / 72.0
    probe = f4_inset_axes(ax, F4_INSET_AT[0], lo, hi)
    tight, frame = probe.get_tightbbox(renderer), probe.get_window_extent(renderer)
    probe.remove()
    zoom = Bbox(ax.transData.transform([(lo, F4_ZOOM_Y[0]), (hi, F4_ZOOM_Y[1])]))
    discs = st.marker_discs(ax)
    texts = [t.get_window_extent(renderer).padded(clear) for t in ax.texts]
    panel = ax.get_window_extent(renderer).padded(-clear)
    usable: List[Tuple[float, float]] = []
    for at in F4_INSET_AT:
        box = Bbox(ax.transAxes.transform([at, (at[0] + F4_INSET_SIZE[0], at[1] + F4_INSET_SIZE[1])]))
        full = Bbox([[box.x0 - (frame.x0 - tight.x0), box.y0 - (frame.y0 - tight.y0)],
                     [box.x1 + (tight.x1 - frame.x1), box.y1 + (tight.y1 - frame.y1)]])
        if (not panel.containsx(full.x0) or not panel.containsx(full.x1)
                or not panel.containsy(full.y0) or not panel.containsy(full.y1)
                or any(st.box_gap(full, x, y) < r + clear for x, y, r in discs)
                or any(full.overlaps(t) for t in texts)):
            continue
        usable.append(at)
        if not any(st.segment_gap(x, y, a, b) < r + 0.5 * clear
                   for a, b in st.connector_segments(zoom, box)
                   for x, y, r in discs if st.box_gap(zoom, x, y) > 0.0):
            return at
    if usable:
        logger.warning("F4: a connector touches a marker at every free corner; using %s", usable[0])
        return usable[0]
    logger.warning("F4: every candidate inset covers a marker or a text; using the first")
    return F4_INSET_AT[0]


def f4_inset(ax: plt.Axes, part: pd.DataFrame, offline: float,
             span: Tuple[float, float]) -> None:
    """Zoom on the cluster around the offline optimum: same markers, no legend, 8 pt ticks."""
    lo, hi = offline * (1.0 + span[0]), offline * (1.0 + span[1])
    at = inset_place(ax, lo, hi)
    logger.info("F4 inset at %s", at)
    axins = f4_inset_axes(ax, at, lo, hi)
    for policy in F4_POLICIES:
        if policy not in part.index:
            continue
        x, y = float(part.loc[policy, "gpu_h_wk_mean"]), float(part.loc[policy, "slo_in_mean"])
        if not (lo <= x <= hi and F4_ZOOM_Y[0] <= y <= F4_ZOOM_Y[1]):
            continue
        axins.plot(x, y, markersize=3.2, markeredgewidth=edge_pt(policy, 0.6),
                   **style_of(policy, scatter=True))
    axins.axhline(SLO_FLOOR, color=GREY, linewidth=0.6, linestyle=(0, (2, 2)))
    # the frame and its connectors under the markers, the connectors fainter than the frame
    frame = ax.indicate_inset_zoom(axins, edgecolor=GREY, linewidth=0.5, alpha=0.9, zorder=1.5)
    zoom = Bbox(ax.transData.transform([(lo, F4_ZOOM_Y[0]), (hi, F4_ZOOM_Y[1])]))
    box = Bbox(ax.transAxes.transform([at, (at[0] + F4_INSET_SIZE[0], at[1] + F4_INSET_SIZE[1])]))
    for line, keep in zip(frame.connectors, st.connectors_shown(zoom, box)):
        line.set_visible(keep)
        line.set_linewidth(0.4)
        line.set_alpha(0.5)


def f4_grid(fig: plt.Figure, ncols: int) -> List[List[plt.Axes]]:
    """The two rows of ``ncols`` panels of one subfloat of Fig. 5; each row shares its y axis."""
    width, height = F4_PANEL_CM
    grid: List[List[plt.Axes]] = []
    for r in range(2):
        bottom = F4_BOTTOM_CM + (1 - r) * (height + F4_ROW_GAP_CM)
        row: List[plt.Axes] = []
        for c in range(ncols):
            box = (F4_LEFT_CM + c * (width + F4_GAP_CM), bottom, width, height)
            row.append(st.axes_cm(fig, box, sharey=row[0] if row else None))
        grid.append(row)
    return grid


def f4_ylabel(fig: plt.Figure, text: str) -> None:
    """One y label for the two rows at the left edge, centred on the rows and not the figure."""
    centre_cm = F4_BOTTOM_CM + F4_PANEL_CM[1] + 0.5 * F4_ROW_GAP_CM
    fig.supylabel(text, x=0.04 / (fig.get_figwidth() * st.CM_PER_IN),
                  y=centre_cm / (fig.get_figheight() * st.CM_PER_IN), fontsize=st.FONT_PT)


def figure_f4(cells: pd.DataFrame, out: Path, cfg: FigureConfig, name: str = "f4_main") -> None:
    """Subfloat (a) of Fig. 5: cost against attainment, one panel per case (row) and tier (column)."""
    with st.printed(cfg, F4_SIZE_CM) as pcfg:
        fig = plt.figure(figsize=(pcfg.width_in, pcfg.height_in))
        axes = f4_grid(fig, 2)
        handles: Dict[str, object] = {}
        zoomed: List[Tuple[plt.Axes, pd.DataFrame, Tuple[float, float]]] = []
        for r, case in enumerate(CASES):
            for c, tier in enumerate(("d_rep", "d_inst")):
                ax = axes[r][c]
                part = cells[(cells["case"] == case) & (cells["tier"] == tier)
                             & (cells["cell"] == "C")].set_index("policy")
                for policy in F4_POLICIES:
                    if policy not in part.index:
                        continue
                    row = part.loc[policy]
                    h = ax.plot(row["gpu_h_wk_mean"], row["slo_in_mean"], markersize=3.6,
                                markeredgewidth=edge_pt(policy, 0.7),
                                **style_of(policy, scatter=True), label=st.SHORT_LABELS[policy])[0]
                    handles.setdefault(policy, h)
                ax.axhline(SLO_FLOOR, color=GREY, linewidth=0.7, linestyle=(0, (2, 2)))
                ax.set_ylim(0.0, 1.03)
                ax.set_yticks(F4_Y_TICKS)
                ax.tick_params(labelleft=c == 0)  # the right column shares the left one's ticks
                # up to five round bins: no x tick then sits on the corner of the y labels
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
                ax.set_xlim(*ax.get_xlim())  # the autoscaled view, fixed before the inset is placed
                ax.text(0.02, 0.03, f"{st.SHORT_CASE[case]}, {st.SHORT_TIER[tier]},"
                                    f" {LEAD_LABEL[case][tier]}", transform=ax.transAxes,
                        fontsize=pcfg.small_font_pt, va="bottom")
                if r == 1:
                    ax.set_xlabel("GPU-hours per week")
                if "offline" in part.index:
                    zoomed.append((ax, part, F4_ZOOM_X[case]))
        st.thin(fig)
        for ax, part, span in zoomed:  # once the ticks have their printed size
            f4_inset(ax, part, float(part.loc["offline", "gpu_h_wk_mean"]), span)
        keys = sorted(handles, key=rank)
        st.legend_band(fig, [handles[k] for k in keys], [st.SHORT_LABELS[k] for k in keys],
                       ncol=4, top_cm=F4_LEGEND_CM, handlelength=0.6)
        f4_ylabel(fig, "In-class SLO attainment")
        save(fig, out, name)


def figure_f4b(weekly: pd.DataFrame, out: Path, cfg: FigureConfig,
               name: str = "f4b_weekly") -> None:
    """Subfloat (b) of Fig. 5: the same policies week by week at the instance tier, one row per case."""
    policies = ("tap", "calendar", "calendar_learn_prior", "calendar_learn", "stale_calendar",
                "forecast_ilp", "velocity", "reactive")
    weeks = [f"w{k}" for k in range(1, 7)]
    with st.printed(cfg, F4B_SIZE_CM) as pcfg:
        fig = plt.figure(figsize=(pcfg.width_in, pcfg.height_in))
        axes = [row[0] for row in f4_grid(fig, 1)]
        handles: Dict[str, object] = {}
        for ax, case in zip(axes, CASES):
            part = weekly[(weekly["case"] == case) & (weekly["tier"] == "d_inst")
                          & (weekly["ratio"] == 30)
                          & (weekly["metric"] == "slo_in")].set_index("policy")
            for policy in sorted(policies, key=drawing_order):
                if policy not in part.index:
                    continue
                h = ax.plot(range(1, 7), part.loc[policy, weeks].to_numpy(dtype=float),
                            markersize=3.0, linewidth=1.2, **style_of(policy),
                            label=st.SHORT_LABELS[policy])[0]
                handles.setdefault(policy, h)
            for w in (1, 5):  # the cold start and the cohort step, which the caption names
                ax.axvspan(w - 0.5, w + 0.5, color="0.92", zorder=0)
            ax.set_xticks(range(1, 7))
            ax.set_ylim(0.0, 1.03)
            ax.set_yticks(F4_Y_TICKS)
            ax.text(0.02, 0.03, f"{st.SHORT_CASE[case]}, instance tier", transform=ax.transAxes,
                    fontsize=pcfg.small_font_pt, va="bottom")
        axes[-1].set_xlabel("Teaching week")
        st.thin(fig)
        keys = sorted(handles, key=rank)
        leg = st.legend_band(fig, [handles[k] for k in keys], [st.SHORT_LABELS[k] for k in keys],
                             ncol=2, top_cm=F4_LEGEND_CM, handlelength=1.5)
        st.phase_dashes(leg, [policy_style(k).dash for k in keys])
        f4_ylabel(fig, "In-class SLO attainment")
        save(fig, out, name)


def best_tuning(part: pd.DataFrame, base: str) -> pd.DataFrame:
    """Per timing cell, the whole row of the tuning of ``base`` that attains more in class.

    A feedback rule is tuned twice, for cost and against the SLO floor; the row kept for a cell
    is the one with the higher ``slo_in_mean``, so its GPU-hours are that same tuning's.
    """
    names = (base, base + TUNED_SUFFIX)
    rows: List[pd.Series] = []
    for cell in CELL_ORDER:
        cand = part[(part["cell"] == cell) & part["policy"].isin(names)].dropna(subset=["slo_in_mean"])
        rows.append(cand.loc[cand["slo_in_mean"].idxmax()].rename(cell) if len(cand)
                    else pd.Series(dtype=float, name=cell))
    out = pd.DataFrame(rows).reindex(list(CELL_ORDER))
    for col in ("policy", "slo_in_mean", "gpu_h_wk_mean"):
        if col not in out.columns:
            out[col] = np.nan
    logger.info("F5 %s: %s", base, ", ".join(f"{c}={str(out.loc[c, 'policy']).split(' (')[0]}"
                                             f"{'+floor' if TUNED_SUFFIX in str(out.loc[c, 'policy']) else ''}"
                                             for c in CELL_ORDER))
    return out


def f5_series(part: pd.DataFrame, policy: str) -> pd.DataFrame:
    """The cell-by-cell series F5 draws for one policy, indexed by ``CELL_ORDER``."""
    if policy in F5_BEST_OF:
        return best_tuning(part, policy)
    return part[part["policy"] == policy].set_index("cell").reindex(list(CELL_ORDER))


def figure_f5(cells: pd.DataFrame, out: Path, cfg: FigureConfig,
              name: str = "f5_timing_cells") -> None:
    """One column wide: the six timing cells, attainment on top and GPU-hours below."""
    labels: Dict[str, str] = {**st.SHORT_LABELS, **F5_BEST_LABELS}
    with st.printed(cfg, F5_SIZE_CM) as pcfg:
        fig, axes = st.grid(pcfg, 2, 2, sharex=True, hspace=F5_HSPACE, wspace=0.08)
        handles: Dict[str, object] = {}
        for c, case in enumerate(CASES):
            part = cells[(cells["case"] == case) & (cells["tier"] == "d_inst")]
            series = {policy: f5_series(part, policy) for policy in F5_POLICIES}
            for r, metric in enumerate(("slo_in_mean", "gpu_h_wk_mean")):
                ax = axes[r, c]
                for policy in sorted(F5_POLICIES, key=drawing_order):
                    sel = series[policy]
                    if sel[metric].isna().all():
                        continue
                    h = ax.plot(range(6), sel[metric].to_numpy(dtype=float), markersize=3.0,
                                linewidth=1.2, **style_of(policy), label=labels[policy])[0]
                    handles.setdefault(policy, h)
                if r == 0:
                    ax.axhline(SLO_FLOOR, color=GREY, linewidth=0.7, linestyle=(0, (2, 2)))
                    ax.set_ylim(0.0, 1.03)
                    ax.set_yticks(F5_SLO_TICKS)
                    ax.text(0.02, 0.03, f"{st.SHORT_CASE[case]}, instance tier",
                            transform=ax.transAxes, fontsize=pcfg.small_font_pt, va="bottom")
                else:
                    ax.set_xticks(range(6))
                    ax.set_xticklabels(CELL_ORDER)
                    ax.set_xlabel("Timing cell")
        axes[0, 0].set_ylabel("In-class SLO attainment")
        axes[1, 0].set_ylabel("GPU-hours per week")
        st.thin(fig)
        # two columns, four rows: three columns run the legend off the page even with the
        # feedback rules cut to "Velocity, best tuning" and the handles to 1.2 (9.1 cm against
        # the column's 8.8), so the fourth row comes out of the panels and the gap between their
        # rows (F5_HSPACE), and the page keeps its size
        keys = sorted(handles, key=rank)
        leg = st.legend_below(fig, [handles[k] for k in keys], [labels[k] for k in keys], ncol=2,
                              handlelength=1.8)
        st.phase_dashes(leg, [policy_style(k).dash for k in keys])
        save(fig, out, name)


def tap_run(case: str, lead: float) -> str:
    """TAP's run at lead time ``lead``: the R1 job at a measured tier, the R3 one elsewhere."""
    short = case.split("_")[0]
    rep, inst = F7_TIERS[case]
    if lead in (rep, inst):
        return f"r1_{short}_{'rep' if lead == rep else 'inst'}"
    return f"r3_{short}_d{lead:g}"


def cell_c_rows(path: Path) -> Optional[pd.DataFrame]:
    """Cell-C rows of one run, floats parsed exactly; None if the file cannot be read."""
    try:
        raw = pd.read_csv(path, float_precision="round_trip")
    except (OSError, ValueError) as exc:  # not there, or a run still being written
        logger.warning("F7: cannot read %s: %s", path, exc)
        return None
    return raw[np.logical_and.reduce([raw[k] == v for k, v in zip(F7_OFFSET_KEYS, F7_CELL_C)])]


def offline_pairs(run: pd.DataFrame, sweep: pd.DataFrame) -> Tuple[bool, str]:
    """Whether ``run`` replays the stream of ``sweep``, and why: the offline rows of both, one
    per seed and week, are equal in every column both carry (bar the sweep's name)."""
    a, b = (f[f["policy"] == "offline"] for f in (run, sweep))
    if a.empty or len(a) != len(b):
        return False, f"{len(a)} offline rows against the sweep's {len(b)}"
    cols = [c for c in b.columns if c in a.columns and c not in ("sweep", "policy")]
    a, b = (f[cols].sort_values(["seed", "week"]).reset_index(drop=True) for f in (a, b))
    same = ((a == b) | (a.isna() & b.isna())).all()
    if not bool(same.all()):
        return False, "offline rows differ in " + ", ".join(c for c in cols if not same[c])
    return True, f"{len(a)} offline rows equal in {len(cols)} columns"


def lead_means(run: pd.DataFrame, policies: Sequence[str]) -> pd.DataFrame:
    """Whole-run means over seeds of ``policies`` in one run of one lead time.

    ``week == 0`` is the whole-run row, ``gpu_hours`` and ``total_cost`` are six-week totals,
    and the excess is taken against the offline optimum of the same run and seed before the
    seeds are averaged.
    """
    rows = run[(run["week"] == 0) & run["seed"].isin(F7_SEEDS)]
    base = rows[rows["policy"] == "offline"].set_index("seed")["total_cost"]
    rows = rows[rows["policy"].isin(policies)]
    rows = rows.assign(excess_usd_wk=(rows["total_cost"] - rows["seed"].map(base)) / F7_WEEKS,
                       gpu_h_wk=rows["gpu_hours"] / F7_WEEKS, viol_in=1.0 - rows["slo_in"])
    return rows.groupby(["lead_min", "policy"], as_index=False).agg(
        viol_in=("viol_in", "mean"), excess_usd_wk=("excess_usd_wk", "mean"),
        gpu_h_wk=("gpu_h_wk", "mean"), n_seeds=("seed", "nunique"))


def lead_table(sweeps: Path, case: str, floor_runs: Path = F7_FLOOR_RUNS,
               tap_runs: Path = F7_TAP_RUNS) -> pd.DataFrame:
    """Whole-run means over seeds of every series of F7 for ``case``, per lead time and policy.

    One sweep per lead time d lives in ``<case>_lead_d<d>/offset/offset.csv``; the ones that have
    not landed are left out of the curves.  The planners and the offline optimum are the sweep's,
    the feedback rules its SLO-floor re-run's under ``floor_runs``, TAP the run ``tap_run`` names
    under ``tap_runs``; a run whose offline rows are not the sweep's replays another stream, and
    its point is left out.
    """
    frames: List[pd.DataFrame] = []
    for path in sorted(sweeps.glob(f"{case}_lead_d*/offset/offset.csv")):
        sweep = cell_c_rows(path)
        if sweep is None or "offline" not in set(sweep.loc[sweep["week"] == 0, "policy"]):
            logger.warning("F7: %s has no whole-run rows yet", path)
            continue
        lead = float(sweep["lead_min"].iloc[0])
        frames.append(lead_means(sweep, F7_SWEEP_POLICIES))
        for policies, other in ((F7_FLOOR_POLICIES, floor_runs / path.parents[1].name),
                                (("tap",), tap_runs / tap_run(case, lead))):
            run = cell_c_rows(other / "offset" / "offset.csv")
            if run is None:
                continue
            paired, why = offline_pairs(run, sweep)
            if not paired:
                logger.warning("F7 %s d=%g: %s of %s left out: %s", case, lead,
                               "/".join(policies), other, why)
                continue
            logger.info("F7 %s d=%g: %s from %s/%s, paired (%s)", case, lead, "/".join(policies),
                        other.parent.name, other.name, why)
            frames.append(lead_means(run, policies))
    if not frames:
        logger.warning("F7: no lead-time run under %s for %s", sweeps, case)
        return pd.DataFrame(columns=["lead_min", "policy", "viol_in", "excess_usd_wk",
                                     "gpu_h_wk", "n_seeds"])
    table = pd.concat(frames, ignore_index=True).sort_values(["policy", "lead_min"])
    found = table.groupby("lead_min")["n_seeds"].min()
    logger.info("F7 %s: d = %s min (%d run%s), seeds per point %s", case,
                ", ".join(f"{d:g}" for d in found.index), len(found),
                "" if len(found) == 1 else "s", sorted(set(found)))
    if (found < len(F7_SEEDS)).any():
        logger.warning("F7 %s: %s has fewer than %d seeds", case,
                       ", ".join(f"d={d:g}" for d in found[found < len(F7_SEEDS)].index),
                       len(F7_SEEDS))
    for policy, sel in table.groupby("policy"):
        logger.info("F7 %s %-20s d: violation / excess USD per week  %s", case, policy, "  ".join(
            f"{r.lead_min:g}: {r.viol_in:.4f}/{r.excess_usd_wk:.2f}" for r in sel.itertuples()))
    return table


def blind_min(lead: float) -> int:
    """Lemma 1's blind time at lead time ``lead``: the engine executes the lead as ceil(d)
    one-minute slots, and a slot is revealed only after it ends (72B's 1.5 min: 3 min)."""
    return math.ceil(lead) + 1


def lemma_bound(theory_root: Path, case: str, leads: Sequence[float],
                ratio: int = 30) -> pd.DataFrame:
    """Lemma 1's weekly lower bound in USD at each of ``leads``, at its executed blind time."""
    parts: List[pd.DataFrame] = []
    for sub in F7_BOUND_DIRS:
        path = theory_root / f"validation_{case}" / sub / "instantiation.csv"
        if not path.exists():
            logger.warning("F7: no instantiation %s", path)
            continue
        t = pd.read_csv(path)
        t = t[(t["r"] == ratio) & np.isclose(t["sigma2"], F7_BOUND_SIGMA2)]
        parts.append(t[["d_min", "lower_week_usd"]].assign(source=sub))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["d_min", "lower_week_usd", "source"])
    spread = rows.groupby("d_min")["lower_week_usd"].agg(lambda s: s.max() - s.min())
    if (spread > 1e-9).any():
        logger.warning("F7 %s: the instantiations differ at d_min %s; %s's value is drawn", case,
                       ", ".join(f"{d:g}" for d in spread[spread > 1e-9].index), F7_BOUND_DIRS[0])
    first = rows.drop_duplicates("d_min").set_index("d_min")  # in the order of F7_BOUND_DIRS
    out: List[Dict[str, object]] = []
    for lead in leads:
        blind = float(blind_min(lead))
        if blind not in first.index:
            logger.warning("F7 %s: no Lemma 1 row at blind time %g min (d = %g)", case, blind, lead)
            continue
        out.append({"lead_min": float(lead), "blind_min": blind,
                    "lower_week_usd": float(first.loc[blind, "lower_week_usd"]),
                    "source": first.loc[blind, "source"]})
    bound = pd.DataFrame(out, columns=["lead_min", "blind_min", "lower_week_usd", "source"])
    logger.info("F7 %s Lemma 1: %s", case, ", ".join(
        f"d={r.lead_min:g} (blind {r.blind_min:g}, {r.source}) {r.lower_week_usd:.2f}"
        for r in bound.itertuples()))
    return bound


def figure_f7(sweeps: Path, theory_root: Path, out: Path, cfg: FigureConfig) -> None:
    """One column wide, two rows: violation and excess cost per week against the lead time d."""
    with st.printed(cfg, F7_SIZE_CM) as pcfg:
        fig, axes = st.grid(pcfg, 2, 2, sharex=True, sharey="row", hspace=0.06, wspace=0.08)
        handles: Dict[str, object] = {}
        bound_handle: Optional[object] = None
        row_max: Dict[int, float] = {}
        for c, case in enumerate(CASES):
            table = lead_table(sweeps, case)
            bound = lemma_bound(theory_root, case, sorted(set(table["lead_min"])))
            for r, metric in enumerate(("viol_in", "excess_usd_wk")):
                ax = axes[r, c]
                for name, key in sorted(F7_SERIES, key=lambda s: drawing_order(s[1])):
                    if r == 1 and name == "offline":  # the excess is zero by construction
                        continue
                    sel = table[table["policy"] == name].sort_values("lead_min")
                    if sel.empty:
                        continue
                    ys = sel[metric].to_numpy(dtype=float)
                    xs = sel["lead_min"].to_numpy(dtype=float) * F7_DODGE.get(name, 1.0)
                    h = ax.plot(xs, ys, markersize=3.0, linewidth=1.2, **style_of(key),
                                label=st.SHORT_LABELS[key])[0]
                    handles.setdefault(name, h)
                    row_max[r] = max(row_max.get(r, 0.0), float(np.nanmax(ys)))
                if r == 1 and len(bound):
                    h = ax.plot(bound["lead_min"], bound["lower_week_usd"], color=GREY,
                                linestyle=F7_BOUND_DASH, linewidth=1.0, zorder=1.9,
                                label=F7_BOUND_LABEL)[0]
                    bound_handle = bound_handle or h
                    row_max[r] = max(row_max.get(r, 0.0), float(bound["lower_week_usd"].max()))
                for tier in F7_TIERS[case]:  # the two lead times the round measured
                    ax.axvline(tier, color=GREY, alpha=0.35, linewidth=0.7, zorder=0)
                ax.set_xscale("log")
                ax.set_xlim(*F7_XLIM)
                ax.xaxis.set_major_locator(FixedLocator(list(F7_X_TICKS)))
                ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in F7_X_TICKS]))
                ax.xaxis.set_minor_locator(NullLocator())
                if r == 0:
                    ax.text(0.04, 0.94, st.SHORT_CASE[case], transform=ax.transAxes,
                            fontsize=pcfg.small_font_pt, va="top")
                else:
                    ax.set_xlabel("Lead time $d$ (min)")
        # both quantities are zero at their best, so both rows are anchored there; the pad keeps
        # a marker sitting on zero, or on the row maximum, off the spine.  sharey does the twin.
        for r, top in row_max.items():
            axes[r, 0].set_ylim(-0.035 * top, 1.10 * top)
        axes[0, 0].set_ylabel("In-class violation")
        axes[1, 0].set_ylabel("Excess USD per week")
        st.thin(fig)
        series = dict(F7_SERIES)
        keys = sorted((n for n in series if n in handles), key=lambda n: rank(series[n]))
        if not keys:
            logger.warning("F7: nothing to draw yet")
        entries = [handles[n] for n in keys] + ([bound_handle] if bound_handle else [])
        names = ([st.SHORT_LABELS[series[n]] for n in keys]
                 + ([F7_BOUND_LABEL] if bound_handle else []))
        dashes = [policy_style(series[n]).dash for n in keys] + [F7_BOUND_DASH]
        if entries:
            leg = st.legend_below(fig, entries, names, ncol=2, handlelength=1.8)
            st.phase_dashes(leg, dashes)
        save(fig, out, "f7_tiers")


def figure_f8(cells: pd.DataFrame, out: Path, cfg: FigureConfig, case: str = "c3_72b_shared4k",
              size_cm: Sequence[float] = F8_SIZE_CM, font_pt: float = st.FONT_PT,
              value_labels: bool = F8_VALUE_LABELS, name: str = "f8_information_ladder") -> None:
    """The information ladder, one column wide, every text at ``font_pt``."""
    part = cells[(cells["case"] == case) & (cells["tier"] == "d_inst")]
    rows: List[Dict[str, object]] = []
    for key in LADDER:
        policy, cell = (key.split("@") + ["C"])[:2] if "@" in key else (key, "C")
        sel = part[(part["policy"] == policy) & (part["cell"] == cell)]
        if sel.empty:
            logger.warning("F8: no row of %s in cell %s; the rung is left out", policy, cell)
            continue
        rows.append({"key": key, "gpu": float(sel["gpu_h_wk_mean"].iloc[0]),
                     "slo": float(sel["slo_in_mean"].iloc[0])})
        logger.info("F8 rung %-28s %7.3f GPU-h/wk  SLO %.4f", key, rows[-1]["gpu"], rows[-1]["slo"])
    table = pd.DataFrame(rows)
    with st.printed(cfg, size_cm, font_pt=font_pt) as f8:
        fig, axes = st.grid(f8, 1, 2, sharey=True, wspace=0.01, hspace=0.0)
        y = np.arange(len(table))[::-1]
        axes[0].barh(y, table["gpu"], color=INDIGO, height=0.62)
        axes[1].barh(y, table["slo"], color=GREEN, height=0.62)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([F8_LABELS.get(k, LADDER_LABELS[k]) for k in table["key"]],
                                fontsize=font_pt)
        axes[0].set_ylim(-0.7, len(table) - 0.3)
        axes[0].set_xlabel("GPU-hours per week")
        axes[1].set_xlabel("In-class SLO")
        axes[1].axvline(SLO_FLOOR, color=GREY, linewidth=0.7, linestyle=(0, (2, 2)))
        axes[1].set_xlim(0, 1.02)
        axes[1].set_xticks([0.0, 0.5, 1.0])
        if value_labels:
            for ax, col, fmt in ((axes[0], "gpu", "{:.0f}"), (axes[1], "slo", "{:.3f}")):
                for yi, v in zip(y, table[col]):
                    ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=font_pt)
        axes[0].set_xlim(min(F8_GPU_XMIN, np.floor(table["gpu"].min() / 10.0) * 10.0),
                         table["gpu"].max() * 1.10)
        # round steps only: the default ones include 150, which leaves no tick near the optimum
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, steps=[1, 2, 2.5, 5, 10]))
        st.thin(fig, font_pt)
        save(fig, out, name, exts=("pdf", "png", "svg"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validation-round draft figures")
    ap.add_argument("--tables", type=Path, default=REPO / "results/sim/validation/tables")
    ap.add_argument("--theory", type=Path, default=REPO / "results/theory")
    ap.add_argument("--sweeps", type=Path, default=PILOT / "outputs/validation_local/sensitivity_sweeps",
                    help="lead-time runs for F7; the ones that have landed are the ones drawn")
    ap.add_argument("--out", type=Path, default=REPO / "results/sim/validation/figures")
    ap.add_argument("--only", choices=("all", "f4", "f4b", "f5", "f7", "f8"), default="all",
                    help="draw one figure instead of all five")
    ap.add_argument("--f4-source", choices=F4_SOURCES, default="laptop",
                    help="cell C of F4 and F4b: the laptop's runs (the paper's figure), or the "
                         "cloud tables as before, written to f4_main_cloud and f4b_weekly_cloud")
    ap.add_argument("--f5-source", choices=F4_SOURCES, default="laptop",
                    help="the six timing cells of F5: the laptop's runs (the paper's figure), or "
                         "the cloud tables as before, written to f5_timing_cells_cloud")
    ap.add_argument("--f8-source", choices=F4_SOURCES, default="laptop",
                    help="cell C of F8: the laptop's runs (the paper's figure; the rung at the bell "
                         "from the laptop's six timing cells), or the cloud tables as before, "
                         "written to f8_information_ladder_cloud")
    ap.add_argument("--cell-c", type=Path, default=CELL_C_LAPTOP,
                    help="the laptop's cell-C table (scripts/sim/runs_paired_table.py)")
    ap.add_argument("--cells", type=Path, default=CELLS_LAPTOP,
                    help="the laptop's six-cell table (scripts/sim/runs_cells_table.py)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = apply_style(FigureConfig())
    want = {args.only} if args.only != "all" else {"f4", "f4b", "f5", "f7", "f8"}
    if want & {"f4", "f4b"}:
        f4_cells, f4_weekly = f4_tables(args.f4_source, args.tables, args.cell_c)
        tag = "" if args.f4_source == "laptop" else "_" + args.f4_source
        if "f4" in want:
            figure_f4(f4_cells, args.out, cfg, name="f4_main" + tag)
        if "f4b" in want:
            figure_f4b(f4_weekly, args.out, cfg, name="f4b_weekly" + tag)
    if "f5" in want:
        tag = "" if args.f5_source == "laptop" else "_" + args.f5_source
        figure_f5(f5_cells(args.f5_source, args.tables, args.cells), args.out, cfg,
                  name="f5_timing_cells" + tag)
    if "f7" in want:
        figure_f7(args.sweeps, args.theory, args.out, cfg)
    if "f8" in want:
        tag = "" if args.f8_source == "laptop" else "_" + args.f8_source
        figure_f8(f8_cells(args.f8_source, args.tables, args.cell_c, args.cells), args.out, cfg,
                  name="f8_information_ladder" + tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
