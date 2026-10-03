"""Shared page geometry of the four Fig. 3 panels, so that their frames line up in the paper.

Fig. 3 prints (a) f3a, (b) f3b, (c) f3c and (d) F_session_starts side by side at 0.24\\textwidth
each.  Two scripts draw them (``scripts/q1/make_assistant_figures.py`` and ``scripts/t2/plots.py``);
both place their axes with :func:`add_axes` from the constants below instead of letting a layout
engine size each page on its own, so every frame has the same top and bottom.  Below the x-axis
labels every page keeps a band two legend rows high: (a) and (d) put their legends there, (b) and
(c) leave it empty.

Lengths are in points and measured from the page bottom unless the name says otherwise.  The
vertical stack, from the bottom: ``BAND_BOTTOM_PT``, the band, 1 pt, the x-axis label (7.4 pt),
``X_LABELPAD_PT``, the 45-degree bell labels of (b) and (d) with their ticks (21.8 pt), the frame.
The frame of (c) is shorter than its 66.5 pt y label, so (c) lowers the label just enough to stay
on the page (:func:`fit_ylabel`); the page height is what keeps the heading after Fig. 3's page
in its column.
"""

from __future__ import annotations

from typing import Dict, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure

PT_PER_IN = 72.0
CM_PER_IN = 2.54
PAGE_CM: Tuple[float, float] = (4.3, 3.888)  # 121.89 x 110.21 pt: 0.24\textwidth wide
TOP_PAD_PT = 0.8          # page top to frame: keeps the frame line and (c)'s y label on the page
BAND_BOTTOM_PT = 0.5      # page bottom to the legend band
BAND_H_PT = 15.9          # two rows of 8 pt legend entries at labelspacing 0.15
X_LABELPAD_PT = 2.5       # tick labels to the x-axis label
AXES_BOTTOM_PT = 49.1     # page bottom to frame: the stack above, for the 45-degree labels
# (left, right) margins: the y label and y tick labels, and what the last x tick label overhangs
MARGINS_PT: Dict[str, Tuple[float, float]] = {
    "a": (30.0, 0.9),
    "b": (23.9, 0.9),
    "c": (28.0, 2.0),
    "d": (33.7, 0.9),
}


def page_in() -> Tuple[float, float]:
    """Page size in inches, for ``plt.figure(figsize=...)``."""
    return PAGE_CM[0] / CM_PER_IN, PAGE_CM[1] / CM_PER_IN


def add_axes(fig: Figure, panel: str) -> Axes:
    """The frame of ``panel`` ('a'-'d') on ``fig``: shared top and bottom, its own left/right."""
    width_pt, height_pt = (s * PT_PER_IN for s in fig.get_size_inches())
    left, right = MARGINS_PT[panel]
    return fig.add_axes((left / width_pt, AXES_BOTTOM_PT / height_pt,
                         (width_pt - left - right) / width_pt,
                         (height_pt - TOP_PAD_PT - AXES_BOTTOM_PT) / height_pt))


def fit_ylabel(ax: Axes, clearance_pt: float = 0.2) -> None:
    """Lower a y label longer than its frame until its top clears the page top, keeping it centred
    otherwise; a label that fits is left where matplotlib put it."""
    fig = ax.figure
    fig.canvas.draw()
    label = ax.yaxis.label
    height_pt = label.get_window_extent().height * PT_PER_IN / fig.dpi
    frame_pt = ax.get_position().height * fig.get_size_inches()[1] * PT_PER_IN
    room_top = frame_pt + TOP_PAD_PT - clearance_pt          # from the frame bottom to the page top
    if height_pt / 2 + frame_pt / 2 > room_top:
        label.set_y((room_top - height_pt / 2) / frame_pt)


def band_top(fig: Figure) -> float:
    """Top of the legend band as a figure fraction: anchor a legend's upper centre at (0.5, it)."""
    return (BAND_BOTTOM_PT + BAND_H_PT) / (fig.get_size_inches()[1] * PT_PER_IN)


__all__ = ["AXES_BOTTOM_PT", "BAND_BOTTOM_PT", "BAND_H_PT", "MARGINS_PT", "PAGE_CM", "TOP_PAD_PT",
           "X_LABELPAD_PT", "add_axes", "band_top", "fit_ylabel", "page_in"]
