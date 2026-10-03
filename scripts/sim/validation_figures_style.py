"""Printed-size style for the validation figures: the size each PDF is placed at in the paper.

``validation_figures.py`` draws a figure at the size it is printed (``\\includegraphics`` with no
scale factor), so the text in the PDF is the text on the page: one serif face at ``FONT_PT`` for
every label, ``savefig.bbox = "standard"`` so the PDF page is exactly the figure, and a
constrained layout that fits the axes and the legend inside it (``grid``, ``legend_below``), or,
for two files whose panels must match, axes placed in cm (``axes_cm``, ``legend_band``).
``printed`` puts the caller's house style back when it exits.

At 8 pt the house policy names do not fit the legends, so the printed figures get their own
shorter names here.  Colours, markers, dashes and the order of every legend come from
``pilot.policy_style``: the timetable plan with known timing keeps its prefix and the variants
that add to it continue with ``+ ...``; matplotlib fills a legend column by column, and
``ORDER`` lists them straight after it, so they follow it down its column (and on at the head
of the next one when that column is full).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox
import numpy as np

from pilot.config import FigureConfig
from pilot.figures import apply_style

logger = logging.getLogger("validation_figures.style")

CM_PER_IN = 2.54
COLUMN_CM = 8.8    # \columnwidth of the two-column body
FONT_PT = 8.0      # every rendered text, ticks included
DPI = 600
PAD_IN = 0.012     # outer padding of the constrained layout, in inches

# Tight legends: at 8 pt the widest of them take nearly the whole width of their figure.
LEGEND_KW: Dict[str, object] = {"frameon": False, "handlelength": 1.0, "handletextpad": 0.35,
                                "columnspacing": 0.6, "labelspacing": 0.30, "borderpad": 0.05,
                                "borderaxespad": 0.05}

SHORT_LABELS: Dict[str, str] = {
    "tap": "TAP", "offline": "Offline optimum", "informed": "Informed optimum",
    "calendar": "Timetable + timing", "calendar_adapt": "+ size correction",
    "calendar_learn_prior": "Timetable + learned, prior", "calendar_learn": "+ learned timing",
    "stale_calendar": "Stale timetable", "forecast_ilp": "History + ILP",
    "guard": "Bell-aware guard", "lcp": "LCP", "velocity": "Token velocity",
    "velocity (SLO>=0.95, disp)": "Velocity, SLO-tuned", "reactive": "Reactive (HPA)",
    "reactive (SLO>=0.95, disp)": "Reactive, SLO-tuned", "fixed": "Base tier only",
}
SHORT_CASE: Dict[str, str] = {"c2_32b_shared4k": "32B", "c3_72b_shared4k": "72B"}
SHORT_TIER: Dict[str, str] = {"d_rep": "replica tier", "d_inst": "instance tier"}


@contextmanager
def printed(cfg: FigureConfig, size_cm: Sequence[float], font_pt: float = FONT_PT,
            dpi: int = DPI) -> Iterator[FigureConfig]:
    """Style for a figure drawn at ``size_cm``; ``cfg``'s house style is restored on exit."""
    printed_cfg = apply_style(FigureConfig(width_in=size_cm[0] / CM_PER_IN,
                                           height_in=size_cm[1] / CM_PER_IN,
                                           base_font_pt=font_pt, small_font_pt=font_pt, dpi=dpi))
    plt.rcParams["savefig.bbox"] = "standard"  # the PDF page is then exactly size_cm
    try:
        yield printed_cfg
    finally:
        apply_style(cfg)


def grid(pcfg: FigureConfig, nrows: int = 1, ncols: int = 1, wspace: float = 0.02,
         hspace: float = 0.02, **kwargs: object) -> Tuple[plt.Figure, object]:
    """``plt.subplots`` at the printed size, with the constrained layout and its pads set."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(pcfg.width_in, pcfg.height_in),
                             layout="constrained", **kwargs)
    fig.get_layout_engine().set(w_pad=PAD_IN, h_pad=PAD_IN, wspace=wspace, hspace=hspace)
    return fig, axes


def legend_below(fig: plt.Figure, handles: Sequence[object], labels: Sequence[str], ncol: int,
                 font_pt: float = FONT_PT, **kwargs: object) -> Legend:
    """Legend under the axes, inside the figure: ``outside`` is what reserves room for it.

    ``kwargs`` override ``LEGEND_KW``.
    """
    leg = fig.legend(handles, labels, loc="outside lower center", ncol=ncol, fontsize=font_pt,
                     **{**LEGEND_KW, **kwargs})
    fig.draw_without_rendering()
    width_cm = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width * CM_PER_IN
    if width_cm > fig.get_figwidth() * CM_PER_IN:
        logger.warning("legend is %.2f cm wide, the figure %.2f cm: shorten it or drop a column",
                       width_cm, fig.get_figwidth() * CM_PER_IN)
    return leg


def thin_axes(ax: plt.Axes, font_pt: float = FONT_PT, spine_pt: float = 0.7) -> None:
    """Hairline spines and short ticks on one axes (0.6 pt spines for an inset)."""
    ax.tick_params(labelsize=font_pt, length=2.0, width=0.6, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_pt)


def thin(fig: plt.Figure, font_pt: float = FONT_PT) -> None:
    """Hairline spines and short ticks: at 8 pt the default weights swamp a 3 cm panel.

    ``fig.axes`` does not hold an inset, which is a child of the panel it sits in, so the insets
    are collected from the panels themselves.
    """
    insets = [child for ax in fig.axes for child in ax.child_axes]
    for ax in fig.axes + insets:
        thin_axes(ax, font_pt, 0.7 if ax not in insets else 0.6)


def axes_cm(fig: plt.Figure, box_cm: Sequence[float], **kwargs: object) -> plt.Axes:
    """An axes box ``(left, bottom, width, height)`` in cm from the figure's lower-left corner.

    For a figure whose panels must match another file's, where a layout engine would size each
    file's panels to its own labels.
    """
    width_cm, height_cm = fig.get_figwidth() * CM_PER_IN, fig.get_figheight() * CM_PER_IN
    left, bottom, width, height = box_cm
    return fig.add_axes((left / width_cm, bottom / height_cm, width / width_cm, height / height_cm),
                        **kwargs)


def legend_band(fig: plt.Figure, handles: Sequence[object], labels: Sequence[str], ncol: int,
                top_cm: float, font_pt: float = FONT_PT, **kwargs: object) -> Legend:
    """Legend centred on the figure with its top ``top_cm`` above the bottom edge.

    The fixed-geometry counterpart of ``legend_below``: two files that put their legends at the
    same ``top_cm`` share the band.  ``kwargs`` override ``LEGEND_KW``.
    """
    leg = fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=font_pt,
                     bbox_to_anchor=(0.5, top_cm / (fig.get_figheight() * CM_PER_IN)),
                     **{**LEGEND_KW, **kwargs})
    fig.draw_without_rendering()
    box = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if box.x0 < 0.0 or box.x1 > fig.get_figwidth() or box.y0 < 0.0:
        logger.warning("legend runs off the figure: %.2f-%.2f cm wide of %.2f, bottom at %.2f cm",
                       box.x0 * CM_PER_IN, box.x1 * CM_PER_IN, fig.get_figwidth() * CM_PER_IN,
                       box.y0 * CM_PER_IN)
    return leg


def phase_dashes(leg: Legend, dashes: Sequence[object], gap_at_pt: float = 1.5) -> None:
    """Start the first gap of each dashed legend line ``gap_at_pt`` from the handle's left end.

    A handle short enough for the legend to fit carries its marker in the middle, where a dash
    pattern that starts at the left end puts its first gap: the line then reads as solid.  Only
    plain dashes are shifted; a pattern with a dot in it reads as one either way.  ``dashes``
    are the linestyles of the handles in order, ``(offset, pattern)`` or a name.
    """
    for handle, dash in zip(leg.legend_handles, dashes):
        if isinstance(dash, tuple) and len(dash[1]) == 2:
            # matplotlib scales the offset and the pattern by the line width alike
            handle.set_linestyle((dash[1][0] - gap_at_pt / handle.get_linewidth(), dash[1]))


# The order of ``InsetIndicator.connectors``: the inset's lower-left, upper-left, lower-right
# and upper-right corners.
CORNERS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


def marker_discs(ax: plt.Axes) -> List[Tuple[float, float, float]]:
    """Centre and radius, in display units, of every marker drawn on ``ax``."""
    px = ax.get_figure().dpi / 72.0
    discs: List[Tuple[float, float, float]] = []
    for line in ax.get_lines():
        if line.get_marker() in (None, "", "None", "none"):
            continue
        radius = 0.5 * (line.get_markersize() + line.get_markeredgewidth()) * px
        discs.extend((float(x), float(y), radius)
                     for x, y in ax.transData.transform(line.get_xydata()))
    return discs


def box_gap(box: Bbox, x: float, y: float) -> float:
    """Distance from the point (x, y) to ``box``, zero inside it."""
    return float(np.hypot(max(box.x0 - x, 0.0, x - box.x1), max(box.y0 - y, 0.0, y - box.y1)))


def segment_gap(x: float, y: float, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from the point (x, y) to the segment from ``a`` to ``b``."""
    ab, ap = b - a, np.array([x, y]) - a
    t = float(np.clip(np.dot(ap, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(ap - t * ab)))


def connectors_shown(zoom: Bbox, inset: Bbox) -> Tuple[bool, ...]:
    """Which of the four corner-to-corner lines to draw: the two that miss both boxes.

    ``indicate_inset_zoom`` makes this choice itself, but on a zoomed box it has transformed
    twice (matplotlib 3.11): for a box in data coordinates it then shows the pair that runs
    across the inset.  Both boxes are in display units.
    """
    x0, x1 = zoom.x0 < inset.x0, zoom.x1 < inset.x1
    y0, y1 = zoom.y0 < inset.y0, zoom.y1 < inset.y1
    return (x0 ^ y0, x0 == y1, x1 == y0, x1 ^ y1)


def connector_segments(zoom: Bbox, inset: Bbox) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The two connector lines ``connectors_shown`` keeps, from the inset's corner."""
    return [(np.array([inset.x0 + ex * inset.width, inset.y0 + ey * inset.height]),
             np.array([zoom.x0 + ex * zoom.width, zoom.y0 + ey * zoom.height]))
            for (ex, ey), keep in zip(CORNERS, connectors_shown(zoom, inset)) if keep]
