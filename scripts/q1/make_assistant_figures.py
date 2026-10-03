"""Batch A figures: F1-left, F3a, F3b and F3c, from the tables under results/q1/assistant.

    uv run python scripts/q1/make_assistant_figures.py

F1-left  one prompted burst of theory class A at 1-min resolution, arrivals per minute against
         minutes from the bell, annotated with sigma_2 and the 10-90% ramp.
F3a      CDFs of per-student first-arrival offsets: prompted AI bursts, self-paced AI use, the
         classroom sign-ins and ASSISTments in-class, on a log axis from 0.1 to 200 min.
F3b      the lab course's pooled 5-min arrival profile over its four periods, with the bells.
F3c      the out-of-class class's weekday x hour session counts, one line per weekday.

House style comes from ``pilot.figures.apply_style`` (serif, >= 8 pt, the fixed colour cycle,
``svg.fonttype = "none"``); every figure is written as SVG, PDF and PNG.

The numbers come from the CSVs ``assistant_workload.py`` wrote, never from the raw exports, so the
figures and the summary cannot disagree.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:  # allow `python scripts/q1/make_assistant_figures.py`
    sys.path.insert(0, str(PILOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pilot import fig3_layout  # noqa: E402  (the page grid F3a-F3c share with F3d)
from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style, save  # noqa: E402  (pins the Agg backend first)
from pilot.policy_style import GREY  # noqa: E402  (the one neutral grey of reference lines)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator,  # noqa: E402
                               MultipleLocator, NullLocator)

logger = logging.getLogger("make_assistant_figures")
EXTS: Tuple[str, ...] = ("svg", "pdf", "png")
K_MARK = 2.0  # the sigma_k window the paper reports

CM_PER_IN = 2.54
FONT_PT = 8.0       # every glyph at the printed size; mathtext sub/superscripts render at 0.7x
                    # that, so the panels spell sigma-2 and the log ticks out in plain text
PRINT_DPI = 600     # the PNG is only a proof copy of the PDF that LaTeX includes
SIGMA2 = "σ2"  # plain-text sigma-2: a mathtext subscript would drop below 8 pt
# Printed sizes in the manuscript.  F1(left) is one of the two panels of a one-column figure
# (\includegraphics[height=4.6cm]); F3a-F3c are 0.24\textwidth of the two-column layout, on the
# page grid of ``pilot.fig3_layout`` that F3d (scripts/t2/plots.py) uses too.
F1_LEFT_CM: Tuple[float, float] = (4.1, 4.6)
F3_PANEL_CM: Tuple[float, float] = fig3_layout.PAGE_CM
# The F3a legend carries the four pool names only.  At 4.3 cm a "name, sigma2 = 0.84" entry is as
# wide as the axes, and the four of them cover the lower half of every curve; sigma_2 is the height
# of each curve at the 2-min line the panel already draws, and F1(left) and the text give it too.
F3A_SIGMA_IN_LEGEND = False
PAD_IN = 0.012      # constrained-layout padding: panels this small have no margin to spare
# The four offset pools of F3a, in palette order.  F3b draws the lab course (``dsv_lab``), whose
# first arrivals are the "Self-paced AI" pool, so its bars take that curve's colour.
F3A_POOLS: Tuple[Tuple[str, str], ...] = (
    ("Prompted AI", "prompted_ai_offsets_min.csv"),
    ("Self-paced AI", "selfpaced_ai_offsets_min.csv"),
    ("Sign-ins", "signin_offsets_min.csv"),
    ("ASSISTments", "assistments_inclass_offsets_min.csv"),
)
F3B_POOL = "Self-paced AI"
F3A_XLIM: Tuple[float, float] = (0.1, 200.0)   # minutes; offsets below 0.1 are drawn at 0.1
EDGE_FACTOR = 1.2     # on the 4.3 cm log axis, 1.2x is ~2 pt: a marker nearer the frame is cut
LINE_PT = 1.0         # series lines of F3a/F3c
LIGHT_LINE_PT = 1.2   # the two light palette colours read at print size only from 1.2 pt up
LIGHT: Tuple[str, ...] = ("#88CCEE", "#DDCC77")
MARK_PT = 3.0


def printed_style(size_cm: Tuple[float, float], font_pt: float = FONT_PT) -> FigureConfig:
    """House style at the printed size: the PDF page is ``size_cm``, every text ``font_pt``.

    ``savefig.bbox`` is pinned to ``standard`` on purpose.  A tight bounding box would change the
    page size, LaTeX would rescale the panel to the width the manuscript asks for, and the text
    would land at some other point size than the one set here.
    """
    cfg = apply_style(FigureConfig(width_in=size_cm[0] / CM_PER_IN,
                                   height_in=size_cm[1] / CM_PER_IN,
                                   base_font_pt=font_pt, small_font_pt=font_pt, dpi=PRINT_DPI))
    plt.rcParams["savefig.bbox"] = "standard"
    return cfg


def panel(cfg: FigureConfig) -> Tuple[plt.Figure, plt.Axes]:
    """One axes filling the printed panel, with the tick geometry the small panels need."""
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in), layout="constrained")
    fig.get_layout_engine().set(w_pad=PAD_IN, h_pad=PAD_IN, wspace=0.0, hspace=0.0)
    ax.tick_params(length=2.0, width=0.6, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    return fig, ax


def f3_panel(cfg: FigureConfig, key: str) -> Tuple[plt.Figure, plt.Axes]:
    """One Fig. 3 panel: the frame of panel ``key`` on the shared grid, no layout engine."""
    fig = plt.figure(figsize=(cfg.width_in, cfg.height_in))
    ax = fig3_layout.add_axes(fig, key)
    ax.tick_params(length=2.0, width=0.6, pad=1.5)
    ax.xaxis.labelpad = fig3_layout.X_LABELPAD_PT
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    return fig, ax


def plain_log_ticks(ax: plt.Axes, axis: str, ticks: Sequence[float]) -> None:
    """Label a log axis in plain digits; the default ``10^k`` ticks carry a 5.6 pt exponent."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    labels = [f"{t:g}" for t in ticks]
    target.set_major_locator(FixedLocator(list(ticks)))
    target.set_major_formatter(FixedFormatter(labels))
    target.set_minor_locator(NullLocator())


@dataclass(frozen=True)
class FigureSpec:
    """Which burst F1-left draws and how wide its window is."""

    dataset: str = "ds_prompted"
    date: str = "2026-03-24"
    bell: str = "08:30"
    lo_min: float = -12.0    # minutes from the bell
    hi_min: float = 18.0
    lab_bells: Tuple[str, ...] = ("14:00", "14:45", "15:50", "16:35")


def _style(cfg: FigureConfig, i: int) -> Dict[str, object]:
    """Colour, marker and dash pattern of series ``i`` of the house palette."""
    return {"color": cfg.palette[i % len(cfg.palette)],
            "marker": cfg.markers[i % len(cfg.markers)],
            "linestyle": cfg.linestyles[i % len(cfg.linestyles)]}


def _line_pt(color: object, base: float = LINE_PT) -> float:
    """Width of a series line: ``base``, raised to 1.2 pt for the two light palette colours."""
    return max(base, LIGHT_LINE_PT) if str(color).upper() in LIGHT else base


def f1_left(assistant: Path, out: Path, cfg: FigureConfig, spec: FigureSpec) -> None:
    """One prompted burst at 1-min resolution, annotated with its synchrony and ramp."""
    series = pd.read_csv(assistant / "arrivals_per_min.csv")
    bursts = pd.read_csv(assistant / "bursts.csv")
    day = series[(series["dataset"] == spec.dataset) & (series["date"] == spec.date)]
    burst = bursts[(bursts["dataset"] == spec.dataset) & (bursts["date"] == spec.date)
                   & (bursts["bell"] == spec.bell)]
    if day.empty or burst.empty:
        raise ValueError(f"no burst {spec.date} {spec.bell} in {assistant}")
    row = burst.iloc[0]
    win = day[(day["minute_from_bell"] >= spec.lo_min) & (day["minute_from_bell"] <= spec.hi_min)]
    x = win["minute_from_bell"].to_numpy(dtype=float)
    y = win["first_arrivals"].to_numpy(dtype=float)
    fig, ax = panel(cfg)
    ax.bar(x, y, width=1.0, align="edge", color=cfg.palette[0], edgecolor="white", linewidth=0.3,
           label="first arrivals per minute")
    ax.axvline(0.0, color="0.35", linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(0.8, y.max() * 0.60, f"bell {spec.bell}", color="0.3", fontsize=cfg.small_font_pt,
            va="top")
    note = (f"{int(row['n_students'])} of 125 students\n"
            f"{SIGMA2} = {row['sigma_2']:.2f}\n"
            f"10-90% ramp {row['ramp_10_90_min']:.1f} min")
    ax.text(0.97, 0.97, note, transform=ax.transAxes, ha="right", va="top", linespacing=1.15,
            fontsize=cfg.small_font_pt, color="0.15")
    ax.set_xlim(spec.lo_min, spec.hi_min)
    ax.set_ylim(0, y.max() * 1.45)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("Minutes from the bell")
    ax.set_ylabel("Arrivals per minute")
    save(fig, out, "f1_left_prompted_burst", EXTS)


def _cdf(ax: plt.Axes, values: np.ndarray, label: str, cfg: FigureConfig, i: int) -> None:
    """Empirical CDF of one offset pool on a log x axis, with a sparse marker."""
    vals = np.sort(values[np.isfinite(values)])
    if vals.size == 0:
        logger.warning("%s: empty pool", label)
        return
    y = np.arange(1, vals.size + 1) / vals.size
    share = float((vals <= K_MARK).mean())
    name = f"{label}, {SIGMA2}={share:.2f}" if F3A_SIGMA_IN_LEGEND else label
    style = _style(cfg, i)
    x = np.maximum(vals, F3A_XLIM[0])
    marks = np.arange(0, vals.size, max(1, vals.size // 8))   # on the eighths of every pool
    # a marker centred on the frame would be drawn cut in half, so only those inside the view
    marks = marks[(x[marks] > F3A_XLIM[0] * EDGE_FACTOR) & (x[marks] < F3A_XLIM[1] / EDGE_FACTOR)]
    ax.plot(x, y, label=name, markevery=marks.tolist(), markersize=MARK_PT,
            linewidth=_line_pt(style["color"]), **style)


def f3a(assistant: Path, offsets: Path, out: Path, cfg: FigureConfig) -> None:
    """First-arrival CDFs of the four offset pools, with the 2-minute mark."""
    fig, ax = f3_panel(cfg, "a")
    for i, (label, name) in enumerate(F3A_POOLS):
        path = offsets / name
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        _cdf(ax, pd.read_csv(path)["offset_min"].to_numpy(dtype=float), label, cfg, i)
    ax.axvline(K_MARK, color=GREY, linewidth=0.8, linestyle=(0, (1, 2)))
    # at the foot of its line, in the strip under the self-paced plateau (0.2) that no curve enters
    # right of 2 min; at the top the label would sit on the prompted curve
    ax.annotate("2 min", xy=(K_MARK, 0.0), xycoords=("data", "axes fraction"), xytext=(1.5, 1.5),
                textcoords="offset points", ha="left", va="bottom", color="0.35",
                fontsize=cfg.small_font_pt)
    ax.set_xscale("log")
    ax.set_xlim(*F3A_XLIM)
    ax.set_ylim(0, 1.06)
    plain_log_ticks(ax, "x", (0.1, 1, 10, 100))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_xlabel("Minutes after first arrival")
    ax.set_ylabel("Share arrived")
    # in the legend band under the x-axis label, not inside the frame: no empty region of a 4.3 cm
    # axes holds four entries at 8 pt, and one row is wider than the page
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, fig3_layout.band_top(fig)), ncol=2,
               fontsize=cfg.small_font_pt, frameon=False, borderpad=0.0, labelspacing=0.15,
               handlelength=1.0, handletextpad=0.3, columnspacing=0.5, borderaxespad=0.0)
    save(fig, out, "f3a_first_arrival_cdf", EXTS)


def f3b(assistant: Path, out: Path, cfg: FigureConfig, spec: FigureSpec) -> None:
    """The lab course's pooled 5-min profile over its four periods, bells marked."""
    profile = pd.read_csv(assistant / "period_profile.csv")
    x = profile["tod_min"].to_numpy(dtype=float)
    y = profile["first_arrivals"].to_numpy(dtype=float)
    fig, ax = f3_panel(cfg, "b")
    colour = cfg.palette[[label for label, _ in F3A_POOLS].index(F3B_POOL)]
    ax.bar(x, y, width=5.0, align="edge", color=colour, edgecolor="white", linewidth=0.3)
    # the four bells are the tick labels of this panel: a time written next to its line would be
    # about as wide as the 45 min to the next line and would run over it
    bell_min = []
    for i, bell in enumerate(spec.lab_bells):
        h, m = (int(v) for v in bell.split(":"))
        bell_min.append(h * 60 + m)
        ax.axvline(bell_min[-1], color=GREY, linewidth=0.9,
                   linestyle=(0, (4, 2)) if i % 2 == 0 else (0, (1, 2)))
    ax.set_xticks(bell_min)
    ax.set_xticklabels(list(spec.lab_bells), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(x.min(), x.max() + 5.0)
    ax.set_ylim(0, y.max() * 1.10)
    ax.yaxis.set_major_locator(MultipleLocator(40))
    ax.set_xlabel("Bell (4 class days pooled)")
    ax.set_ylabel("Arrivals per 5 min")
    save(fig, out, "f3b_lab_period_profile", EXTS)


def f3c(assistant: Path, out: Path, cfg: FigureConfig) -> None:
    """Weekday x hour session counts of the out-of-class class, one line per weekday."""
    weekly = pd.read_csv(assistant / "weekly_profile.csv")
    names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    fig, ax = f3_panel(cfg, "c")
    peak = 0.0
    for wd, part in weekly.groupby("weekday"):
        part = part.sort_values("hour")
        y = part["sessions"].to_numpy(dtype=float)
        peak = max(peak, float(y.max()))
        style = _style(cfg, int(wd))
        width = _line_pt(style["color"], 1.3 if wd == 0 else LINE_PT)  # Monday, the deadline day
        ax.plot(part["hour"].to_numpy(dtype=float), y, label=names[int(wd)], linewidth=width,
                markevery=3, markersize=MARK_PT, **style)
    # the legend sits in the band below, so the note takes the empty upper left; the arrow leaves
    # the end of "deadline" (0.77 of the note's width, mid x-height of its first line), runs above
    # Monday's evening rise and stops above the peak, its head clear of the Monday line
    ax.annotate("deadline\nMon 23:59", xy=(20.5, peak * 1.12), xytext=(9.0, peak * 1.3),
                va="top", fontsize=cfg.small_font_pt, color="0.15", linespacing=1.15,
                arrowprops={"arrowstyle": "->", "color": GREY, "linewidth": 0.7,
                            "relpos": (0.79, 0.75), "patchA": None, "shrinkA": 1.5})
    ax.set_xlim(8, 23.5)
    ax.set_ylim(0, peak * 1.42)
    ax.set_xticks(np.arange(8, 24, 3))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Sessions in semester")
    fig3_layout.fit_ylabel(ax)
    # the seven days in the band below the axis, like the legends of (a) and (d): inside, on the
    # shared frame, they would crowd the deadline note
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, fig3_layout.band_top(fig)), ncol=4,
               fontsize=cfg.small_font_pt, frameon=False, borderpad=0.0, labelspacing=0.15,
               handlelength=0.9, handletextpad=0.25, columnspacing=0.45, borderaxespad=0.0)
    save(fig, out, "f3c_offclass_weekly", EXTS)


def run(root: Path, out: Optional[Path] = None, spec: Optional[FigureSpec] = None) -> Path:
    """Draw all four figures at their printed sizes and return the output directory."""
    spec = spec or FigureSpec()
    assistant = root / "results/q1/assistant"
    offsets = root / "data/offsets"
    out = out or (assistant / "figures")
    f1_left(assistant, out, printed_style(F1_LEFT_CM), spec)
    cfg3 = printed_style(F3_PANEL_CM)
    f3a(assistant, offsets, out, cfg3)
    f3b(assistant, out, cfg3, spec)
    f3c(assistant, out, cfg3)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    run(args.root, args.out)


if __name__ == "__main__":
    main()
