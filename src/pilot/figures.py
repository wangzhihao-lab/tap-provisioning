"""Q1 figures: Times-like serif, >= 8 pt, fixed colour cycle, no titles inside the axes.

Every figure is written as both PDF (for LaTeX) and PNG (for quick viewing), with
``svg.fonttype = "none"`` so text stays editable if a figure is later exported as SVG.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FigureConfig  # noqa: E402

logger = logging.getLogger(__name__)

_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times", "DejaVu Serif"]


def apply_style(cfg: Optional[FigureConfig] = None) -> FigureConfig:
    """Install the house style in matplotlib's rcParams and return the config used."""
    cfg = cfg or FigureConfig()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF,
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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
        "legend.frameon": False,
    })
    return cfg


def save(fig: plt.Figure, out_dir: Path, name: str,
         exts: Sequence[str] = ("pdf", "png")) -> None:
    """Write ``name`` in each of ``exts`` into ``out_dir`` (PDF and PNG by default).

    ``svg`` is worth asking for when the figure is to be redrawn by hand: ``apply_style`` pins
    ``svg.fonttype = "none"``, so the text of an SVG stays editable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    logger.info("wrote %s.%s", out_dir / name, "/.".join(exts))


def _style_of(cfg: FigureConfig, i: int) -> Dict[str, object]:
    return {"color": cfg.palette[i % len(cfg.palette)],
            "marker": cfg.markers[i % len(cfg.markers)],
            "linestyle": cfg.linestyles[i % len(cfg.linestyles)]}


def ramp_cdf(bursts: Dict[str, pd.DataFrame], out_dir: Path, cfg: FigureConfig,
             gate_min: Optional[float] = None) -> None:
    """Empirical CDF of the 10-90% ramp time of each detected burst, one curve per source."""
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    for i, (name, frame) in enumerate(sorted(bursts.items())):
        vals = np.sort(pd.to_numeric(frame.get("ramp_min"), errors="coerce").dropna().to_numpy())
        if vals.size == 0:
            logger.warning("%s: no ramp times to plot", name)
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        every = max(1, vals.size // 12)
        ax.plot(vals, y, label=name, markevery=every, markersize=3.5, linewidth=1.2,
                **_style_of(cfg, i))
    if gate_min is not None:
        ax.axvline(gate_min, color="0.45", linewidth=0.8, linestyle=(0, (1, 2)))
        ax.text(gate_min, 0.04, f" gate {gate_min:g} min", color="0.35",
                fontsize=cfg.small_font_pt)
    ax.set_xlim(0, cfg.ramp_xlim_min)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Burst ramp time, 10-90% (min)")
    ax.set_ylabel("Fraction of bursts")
    ax.legend(loc="lower right")
    save(fig, out_dir, "ramp_cdf")


def sigma_hist(bursts: Dict[str, pd.DataFrame], out_dir: Path, cfg: FigureConfig,
               k_minutes: int = 2) -> None:
    """Distribution of the synchronization fraction sigma_k across bursts."""
    col = f"sync_{k_minutes}min"
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    bins = np.linspace(0.0, 1.0, cfg.sigma_bins + 1)
    for i, (name, frame) in enumerate(sorted(bursts.items())):
        vals = pd.to_numeric(frame.get(col), errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            logger.warning("%s: no %s values to plot", name, col)
            continue
        style = _style_of(cfg, i)
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.3, label=name,
                color=style["color"], linestyle=style["linestyle"])
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Synchronization fraction $\\sigma_{{{k_minutes}}}$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    save(fig, out_dir, "sigma_hist")


def predictability(summary: pd.DataFrame, out_dir: Path, cfg: FigureConfig,
                   gate_reduction: Optional[float] = None,
                   columns: Sequence[str] = ("mase_ewma", "mase_calendar",
                                             "mase_calendar_first")) -> None:
    """MASE of each predictor against the seasonal-naive benchmark, grouped by source."""
    labels = {"mase_ewma": "Seasonal EWMA", "mase_calendar": "Calendar",
              "mase_calendar_first": "Calendar, first week"}
    sources = list(summary["source"])
    x = np.arange(len(sources), dtype=float)
    width = min(0.8 / max(1, len(columns)), cfg.max_bar_width)
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    top = 1.0
    for i, col in enumerate(columns):
        vals = pd.to_numeric(summary.get(col), errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (i - (len(columns) - 1) / 2) * width, vals, width * 0.9,
               label=labels.get(col, col), color=cfg.palette[i % len(cfg.palette)],
               edgecolor="white", linewidth=0.4)
        top = max(top, float(np.nanmax(vals)) if np.isfinite(vals).any() else top)
    ax.axhline(1.0, color="0.25", linewidth=0.9)
    ax.text(-0.55, 1.01, "seasonal naive", ha="left", va="bottom", color="0.25",
            fontsize=cfg.small_font_pt)
    if gate_reduction is not None:
        ax.axhline(1.0 - gate_reduction, color="0.45", linewidth=0.8, linestyle=(0, (1, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=0)
    ax.set_ylabel("MASE (lower is better)")
    ax.set_xlim(-0.6, len(sources) - 0.4)
    # headroom so the legend never sits on top of a bar
    ax.set_ylim(0, top * cfg.bar_headroom)
    ax.legend(loc="upper right", ncol=1)
    save(fig, out_dir, "predictability")
