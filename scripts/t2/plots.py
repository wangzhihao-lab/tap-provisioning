"""Redraw the T2 session-start panel (F3d) at the size the manuscript prints it.

    uv run python scripts/t2/plots.py            # results/t2/session_starts.csv -> F_session_starts

The figure is panel (d) of F3, included with ``width=0.24\\textwidth``, i.e. about 4.3 cm wide.
``pilot.timetable.plots.session_start_bars`` draws the same bars across the full text width for
the T2 note; this script redraws them from the table the pipeline already wrote, so nothing is
recomputed and no T2 table is touched.

Differences from the wide version, all of them forced by the printed width:
* only the periods that appear in ``session_starts.csv`` are on the x axis.  They are the block
  starts (1, 3, 5, 7, 9); no session starts in any other period, so no bar is dropped.
* the ticks are the bell times themselves; "period number + bell" needs two lines and 12 slots.
* seats are drawn in thousands, so that the y tick labels stay two digits wide.
* the weekday legend is one row in the legend band under the x-axis label, and the frame sits on
  the page grid all four Fig. 3 panels share (``pilot.fig3_layout``), so the four frames align.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from pilot import fig3_layout  # noqa: E402  (the page grid F3d shares with F3a-F3c)
from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style, save  # noqa: E402
from pilot.timetable.parse import BellSchedule  # noqa: E402
from pilot.timetable.stats import StatsConfig  # noqa: E402

logger = logging.getLogger("t2_plots")

CM_PER_IN = 2.54
FONT_PT = 8.0     # every glyph at the printed size
PRINT_DPI = 600   # the PNG is a proof copy of the PDF LaTeX includes
PANEL_CM: Tuple[float, float] = fig3_layout.PAGE_CM   # 0.24\textwidth, the Fig. 3 page grid
WEEKDAY_LABELS: Tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
SEAT_UNIT = 1000.0


def printed_style(size_cm: Sequence[float], font_pt: float = FONT_PT) -> FigureConfig:
    """House style at the printed size; ``savefig.bbox`` standard keeps the page at ``size_cm``."""
    cfg = apply_style(FigureConfig(width_in=size_cm[0] / CM_PER_IN,
                                   height_in=size_cm[1] / CM_PER_IN,
                                   base_font_pt=font_pt, small_font_pt=font_pt, dpi=PRINT_DPI))
    plt.rcParams["savefig.bbox"] = "standard"
    return cfg


def session_start_bars(starts: pd.DataFrame, out_dir: Path, cfg: FigureConfig,
                       bells: Optional[BellSchedule] = None,
                       stats_cfg: Optional[StatsConfig] = None,
                       name: str = "F_session_starts") -> Path:
    """Seats whose class starts at each bell, grouped by weekday (Mon-Fri)."""
    bells = bells or BellSchedule()
    stats_cfg = stats_cfg or StatsConfig()
    weekdays = list(stats_cfg.teaching_weekdays)
    periods = sorted(int(p) for p in starts["period"].unique())
    table = starts.pivot_table(index="period", columns="weekday", values="seats_typ",
                               aggfunc="sum").reindex(index=periods, columns=weekdays).fillna(0.0)
    width = 0.84 / len(weekdays)
    fig = plt.figure(figsize=(cfg.width_in, cfg.height_in))
    ax = fig3_layout.add_axes(fig, "d")   # same frame top and bottom as F3a-F3c
    ax.xaxis.labelpad = fig3_layout.X_LABELPAD_PT
    base = np.arange(len(periods), dtype=float)
    for i, weekday in enumerate(weekdays):
        ax.bar(base + (i - (len(weekdays) - 1) / 2) * width,
               table[weekday].to_numpy(dtype=float) / SEAT_UNIT, width,
               color=cfg.palette[i], label=WEEKDAY_LABELS[weekday - 1], linewidth=0)
    ax.set_xticks(base, [bells.start(p) for p in periods], rotation=45, ha="right",
                  rotation_mode="anchor")
    ax.set_xlabel("Bell of the block start")
    ax.set_ylabel("Students starting\n(thousands)")  # two lines: one would be taller than the axes
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(length=2.0, width=0.6, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    # one row in the legend band under the x-axis label, level with the first row of F3a's; a
    # swatch sits three times nearer its own day than the day before it (1.6 pt against 4.8 pt)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, fig3_layout.band_top(fig)), ncol=len(weekdays),
               frameon=False, fontsize=cfg.small_font_pt, handlelength=0.6, handletextpad=0.2,
               columnspacing=0.6, borderpad=0.0, borderaxespad=0.0)
    save(fig, out_dir, name, exts=("pdf", "png"))
    return out_dir / f"{name}.pdf"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--starts", type=Path, default=REPO / "results/t2/session_starts.csv")
    ap.add_argument("--out", type=Path, default=REPO / "results/t2")
    ap.add_argument("--width-cm", type=float, default=PANEL_CM[0])
    ap.add_argument("--height-cm", type=float, default=PANEL_CM[1])
    ap.add_argument("--font-pt", type=float, default=FONT_PT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    starts = pd.read_csv(args.starts)
    cfg = printed_style((args.width_cm, args.height_cm), args.font_pt)
    path = session_start_bars(starts, args.out, cfg)
    logger.info("redrew %s from %s", path, args.starts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
