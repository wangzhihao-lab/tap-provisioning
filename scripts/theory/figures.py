"""The feasibility figure of the analysis: where A1, A2 and H2 hold.

One panel per synchrony level, axes ``beta/c`` (log) against the price ratio ``r = p_in mu / c``
(log), because those are the two dimensionless quantities every premise of the sketch reduces to:

* **A1** ``p_off mu > c`` is ``r > 1 / (p_off/p_in)`` - a horizontal line, independent of both the
  switching cost and the burst length;
* **A2** ``kappa m > beta`` at the worst tier the lemma covers (``d >= ell``, so ``m = ell``) is
  ``r > 1 + (beta/c)/ell``;
* **H2** ``(p_off mu - c) ell >= beta`` is ``r >= (1 + (beta/c)/ell) / (p_off/p_in)``, which
  dominates the other two at the graded price ratio, so it is the premise that decides the region.

The shaded band is where all three hold at once, i.e. where the R1 lower bound is not vacuous and
lemma H is available to pin OPT.  The calibration point is drawn as a vertical line at its
``beta/c`` with a marker at every ``r`` of the scan, so one look says which scan points have
theoretical support.

House style comes from ``pilot.figures.apply_style``; the figure is written as SVG (editable text),
PDF (for LaTeX) and PNG.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PILOT = Path(__file__).resolve().parents[2]
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:  # sibling module, not a package
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style  # noqa: E402

from lemmas import Calibration, ell_of  # noqa: E402

logger = logging.getLogger(__name__)

BETA_OVER_C_RANGE: Tuple[float, float] = (0.1, 100.0)
R_RANGE: Tuple[float, float] = (1.0, 1000.0)
GRID_POINTS = 400
PANEL_WIDTH_IN = 2.45  # three panels inside the 7.16 in two-column width
PANEL_HEIGHT_IN = 2.9
LEGEND_ROOM = 0.17  # bottom fraction of the canvas the shared legend occupies
FORMATS: Tuple[str, ...] = ("svg", "pdf", "png")
NAME = "instantiation_feasibility"


def _curves(x: np.ndarray, ell: float, p_off_ratio: float) -> Dict[str, np.ndarray]:
    """The three premises as lower bounds on ``r``, over the ``beta/c`` axis ``x``."""
    a1 = np.full_like(x, 1.0 / p_off_ratio)
    a2 = 1.0 + x / ell
    h2 = (1.0 + x / ell) / p_off_ratio
    return {"A1: $p^{\\mathrm{off}}\\mu > c$": a1,
            "A2: $\\kappa\\,\\ell > \\beta$": a2,
            "H2: $\\kappa''\\ell \\geq \\beta$": h2}


def _panel(ax: plt.Axes, cal: Calibration, sigma2: float, cfg: FigureConfig) -> None:
    """One synchrony level: the three premise curves, the region, and the calibration point."""
    ell = ell_of(sigma2)
    x = np.logspace(np.log10(BETA_OVER_C_RANGE[0]), np.log10(BETA_OVER_C_RANGE[1]), GRID_POINTS)
    curves = _curves(x, ell, cal.p_off_ratio)
    binding = np.maximum.reduce(list(curves.values()))
    ax.fill_between(x, np.clip(binding, *R_RANGE), R_RANGE[1], color=cfg.palette[3], alpha=0.30,
                    linewidth=0.0, label="all three hold")
    for i, (label, curve) in enumerate(curves.items()):
        ax.plot(x, curve, color=cfg.palette[i], linestyle=cfg.linestyles[i % 4], linewidth=1.2,
                label=label)
    ax.axvline(cal.beta_over_c, color=cfg.palette[4], linestyle=":", linewidth=1.0,
               label=f"$\\beta/c = {cal.beta_over_c:g}$")
    # the binding premise at the calibration's beta/c, evaluated in closed form rather than read
    # off the plotted grid: a scan point is filled when every premise holds there and open when
    # one of them does not
    at_cal = max(1.0 / cal.p_off_ratio, (1.0 + cal.beta_over_c / ell) / cal.p_off_ratio,
                 1.0 + cal.beta_over_c / ell)
    for supported, marker in ((True, "o"), (False, "x")):
        values = [r for r in cal.r if (r >= at_cal) is supported]
        if not values:
            continue
        ax.plot([cal.beta_over_c] * len(values), values, marker=marker, linestyle="none",
                markersize=4.5, markerfacecolor=cfg.palette[5] if supported else "none",
                markeredgecolor=cfg.palette[5],
                label="$r$ scan" if supported else "$r$ scan, unsupported")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*BETA_OVER_C_RANGE)
    ax.set_ylim(*R_RANGE)
    ax.set_xlabel("$\\beta/c$ (replica-minutes per start)")
    ax.annotate(f"$\\sigma_2 = {sigma2:.2f}$, $\\ell = {ell:.2f}$ min", xy=(0.04, 0.93),
                xycoords="axes fraction", fontsize=cfg.small_font_pt, va="top")


def feasibility(cal: Calibration, out_dir: Path, cfg: FigureConfig) -> Path:
    """Draw the three-panel feasibility figure and return the directory it was written to."""
    n = len(cal.sigma2)
    fig, axes = plt.subplots(1, n, figsize=(PANEL_WIDTH_IN * n, PANEL_HEIGHT_IN), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sigma2 in zip(axes, cal.sigma2):
        _panel(ax, cal, float(sigma2), cfg)
    axes[0].set_ylabel("$r = p^{\\mathrm{in}}\\mu / c$")
    # the premise curves run through every corner of a panel, so the legend goes underneath them
    # all rather than inside one; the entries come from the first panel, where every label exists
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3),
               fontsize=cfg.small_font_pt, handlelength=1.8, columnspacing=1.2,
               borderaxespad=0.1)
    fig.tight_layout(pad=0.6, w_pad=1.2, rect=(0.0, LEGEND_ROOM, 1.0, 1.0))
    return save(fig, out_dir, NAME)


def save(fig: plt.Figure, out_dir: Path, name: str, formats: Sequence[str] = FORMATS) -> Path:
    """Write one figure in every format; SVG keeps its text editable (``svg.fonttype = none``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    logger.info("wrote %s.{%s}", out_dir / name, ",".join(formats))
    return out_dir


def draw_all(cal: Calibration, out_dir: Path) -> Path:
    """Install the house style and draw every figure section 7 asks for."""
    cfg = apply_style(FigureConfig())
    return feasibility(cal, out_dir, cfg)


__all__ = ["BETA_OVER_C_RANGE", "FORMATS", "NAME", "R_RANGE", "draw_all", "feasibility", "save"]
