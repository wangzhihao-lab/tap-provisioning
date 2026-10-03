"""Figures built from the timetable aggregates (F1 material and the T2 sanity panels).

Style contract: serif, >= 8 pt, the fixed eight-colour cycle, no titles inside the axes, editable
text in the SVG.  Every figure is written as PDF + PNG + SVG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from .parse import BellSchedule  # noqa: E402
from .stats import ALL_CAMPUS, StatsConfig  # noqa: E402

logger = logging.getLogger(__name__)

WEEKDAY_LABELS: Tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEKDAY_SHORT: Tuple[str, ...] = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
CAMPUS_LABELS: Dict[str, str] = {
    "": "unspecified", ALL_CAMPUS: "All campuses",
}


@dataclass(frozen=True)
class PlotConfig:
    """Figure geometry and the fixed colour cycle."""

    width_in: float = 3.5
    wide_in: float = 7.16
    height_in: float = 2.4
    dpi: int = 300
    base_font_pt: float = 9.0
    small_font_pt: float = 8.0
    palette: Tuple[str, ...] = (
        "#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255",
    )
    linestyles: Tuple[str, ...] = ("-", "--", "-.", ":")
    heat_cmap: str = "YlGnBu"
    formats: Tuple[str, ...] = ("pdf", "png", "svg")


def configure_style(cfg: PlotConfig | None = None) -> PlotConfig:
    """Apply the publication rcParams; returns the config used."""
    cfg = cfg or PlotConfig()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": cfg.base_font_pt,
        "axes.labelsize": cfg.base_font_pt,
        "axes.titlesize": cfg.base_font_pt,
        "xtick.labelsize": cfg.small_font_pt,
        "ytick.labelsize": cfg.small_font_pt,
        "legend.fontsize": cfg.small_font_pt,
        "figure.dpi": cfg.dpi,
        "savefig.dpi": cfg.dpi,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=list(cfg.palette)),
        "axes.grid": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    return cfg


def save_figure(fig: "plt.Figure", out_dir: str | Path, name: str,
                cfg: PlotConfig | None = None) -> List[Path]:
    """Write ``name`` as PDF, PNG and SVG under ``out_dir``."""
    cfg = cfg or PlotConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in cfg.formats:
        path = out / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    logger.info("wrote %s.{%s}", name, ",".join(cfg.formats))
    return paths


def _campus_label(campus: str) -> str:
    return CAMPUS_LABELS.get(campus, campus)


def _grid(typ: pd.DataFrame, campus: str, weekdays: Sequence[int],
          periods: Sequence[int]) -> np.ndarray:
    frame = typ[typ["campus"] == campus]
    pivot = frame.pivot_table(index="period", columns="weekday", values="seats", aggfunc="sum")
    pivot = pivot.reindex(index=list(periods), columns=list(weekdays)).fillna(0.0)
    return pivot.to_numpy(dtype=float)


def heatmap_typical_week(typ: pd.DataFrame, out_dir: str | Path, cfg: PlotConfig | None = None,
                         stats_cfg: StatsConfig | None = None, name: str = "F_timetable_heatmap",
                         ) -> List[Path]:
    """Typical-week seats per weekday x period, one panel per campus plus the all-campus total."""
    cfg = configure_style(cfg)
    stats_cfg = stats_cfg or StatsConfig()
    order = typ[typ["campus"] != ALL_CAMPUS].groupby("campus")["seats"].sum().sort_values(
        ascending=False)
    campuses = list(order.index) + [ALL_CAMPUS]
    active = typ.loc[typ["seats"] > 0, "weekday"].unique()
    weekdays = sorted(set(stats_cfg.teaching_weekdays) | set(int(w) for w in active))
    periods = sorted(typ["period"].unique())
    grids = [_grid(typ, c, weekdays, periods) for c in campuses]
    vmax = max((g.max() for g in grids), default=1.0) or 1.0
    ncols = min(len(campuses), 5)
    nrows = int(np.ceil(len(campuses) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cfg.wide_in, 1.35 * nrows + 0.8),
                             squeeze=False, constrained_layout=True)
    flat = [ax for row in axes for ax in row]
    mesh = None
    for ax, campus, grid in zip(flat, campuses, grids):
        mesh = ax.pcolormesh(grid, cmap=cfg.heat_cmap, vmin=0.0, vmax=vmax, edgecolors="white",
                             linewidth=0.3)
        names = WEEKDAY_SHORT if len(weekdays) > 5 else WEEKDAY_LABELS
        ax.set_xticks(np.arange(len(weekdays)) + 0.5, [names[w - 1] for w in weekdays])
        ax.set_yticks(np.arange(len(periods)) + 0.5, [str(p) for p in periods])
        ax.invert_yaxis()
        ax.tick_params(length=0)
        ax.set_xlabel(_campus_label(campus))
        if ax is flat[0]:
            ax.set_ylabel("period")
    for ax in flat[len(campuses):]:
        ax.set_visible(False)
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.02)
        bar.set_label("students in class (typical week)", fontsize=cfg.small_font_pt)
    return save_figure(fig, out_dir, name, cfg)


def session_start_bars(starts: pd.DataFrame, out_dir: str | Path, cfg: PlotConfig | None = None,
                       bells: BellSchedule | None = None, stats_cfg: StatsConfig | None = None,
                       name: str = "F_session_starts") -> List[Path]:
    """Seats whose class starts at each period, grouped by weekday (Mon-Fri), periods 1..12."""
    cfg = configure_style(cfg)
    bells = bells or BellSchedule()
    stats_cfg = stats_cfg or StatsConfig()
    periods = list(range(1, len(bells.windows) + 1))
    weekdays = list(stats_cfg.teaching_weekdays)
    width = 0.8 / len(weekdays)
    fig, ax = plt.subplots(figsize=(cfg.wide_in, 2.6), constrained_layout=True)
    table = starts.pivot_table(index="period", columns="weekday", values="seats_typ",
                               aggfunc="sum").reindex(index=periods, columns=weekdays).fillna(0.0)
    base = np.arange(len(periods))
    for i, weekday in enumerate(weekdays):
        ax.bar(base + (i - (len(weekdays) - 1) / 2) * width, table[weekday].to_numpy(), width,
               color=cfg.palette[i], label=WEEKDAY_LABELS[weekday - 1], linewidth=0)
    labels = [f"{p}\n{bells.start(p)}" if bells.start(p) in bells.bells() else str(p)
              for p in periods]
    ax.set_xticks(base, labels)
    ax.set_xlabel("period (bell time of the block start)")
    ax.set_ylabel("students starting a class")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncols=len(weekdays), frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=0.0, columnspacing=1.6, handlelength=1.4)
    return save_figure(fig, out_dir, name, cfg)


def enrollment_cdf(sections: pd.DataFrame, out_dir: str | Path, cfg: PlotConfig | None = None,
                   name: str = "F_enrollment_cdf") -> List[Path]:
    """CDF of section enrollment, by section and weighted by students."""
    cfg = configure_style(cfg)
    values = np.sort(sections["enrollment"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in), constrained_layout=True)
    if values.size:
        by_section = np.arange(1, values.size + 1) / values.size
        total = values.sum()
        by_student = np.cumsum(values) / total if total else np.zeros_like(values)
        ax.step(values, by_section, where="post", color=cfg.palette[0],
                linestyle=cfg.linestyles[0], label="sections")
        ax.step(values, by_student, where="post", color=cfg.palette[5],
                linestyle=cfg.linestyles[1], label="students")
    ax.set_xlabel("section enrollment (students)")
    ax.set_ylabel("cumulative share")
    ax.set_ylim(0, 1.02)
    ax.set_xscale("symlog", linthresh=10, linscale=0.4)
    ax.set_xlim(0, float(values.max()) * 1.2 if values.size else 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    return save_figure(fig, out_dir, name, cfg)


def make_all(sections: pd.DataFrame, typ: pd.DataFrame, starts: pd.DataFrame,
             out_dir: str | Path, cfg: PlotConfig | None = None,
             bells: BellSchedule | None = None) -> List[Path]:
    """Draw the three timetable figures into ``out_dir``."""
    cfg = configure_style(cfg)
    paths = heatmap_typical_week(typ, out_dir, cfg)
    paths += session_start_bars(starts, out_dir, cfg, bells)
    paths += enrollment_cdf(sections, out_dir, cfg)
    return paths
