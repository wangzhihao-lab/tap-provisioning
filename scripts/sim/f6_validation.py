"""F6 of the validation round: cost against the adoption-estimate error, and the switch point.

    uv run python scripts/sim/f6_validation.py            # the four r*_d_rep validation sweeps
    uv run python scripts/sim/f6_validation.py --dev      # the batch-C epsilon sweep instead
    uv run python scripts/sim/f6_validation.py --panels 10 30 --layout column --stem f6_forecast_error_col --figure-only

The sweep scales the timetable plan by ``forecast_scale`` and runs the plan-following policies
against the robust fallback at four price ratios.  The main text's remark on the switch point says
BDINTERP-d leaves the plan once the plan's cumulative cost exceeds ``4/eps`` times the robust
sequence's cost, so a growing over-forecast should push ``bdinterp@eps`` off ``calendar`` towards
``lcp`` at a scale that grows with ``1/eps``, and an under-forecast should never switch.

That prediction is evaluated on the *net* cost, the decision-dependent one the phase test compares
(``pilot.sim.policies.bdinterp._slot_cost``): the always-on base tier is a sunk cost both branches
pay whatever they decide, so its running cost comes out of both sides.  The total-cost reading is
kept beside it as ``plan_over_robust_total`` so the two stay auditable.  Written:
``tables/f6_switch.csv`` / ``.md`` (the observed and predicted switch scale per price ratio and
eps), ``tables/f6_forecast_scale.csv`` (the aggregate behind both) and
``figures/f6_forecast_error.pdf`` / ``.png`` (one panel per price ratio, the vertical lines at the
net prediction).  ``f6_validation_util.py`` holds the loading and aggregation helpers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from f6_validation_util import (  # noqa: E402
    C_RUN_DEFAULT, NONE, REPO, aggregate, curve, first_or_none, load, md_table, rel,
    resolve_inputs)

logger = logging.getLogger("f6_validation")

DEFAULT_INPUTS: Tuple[str, ...] = tuple(
    f"outputs/validation_local/f6_32b/r{r}_d_rep/epsilon/epsilon.csv" for r in (30, 6, 100, 10))
DEV_INPUTS: Tuple[str, ...] = ("outputs/batchC/eps/epsilon/epsilon.csv",)
OUT_DIR = REPO / "results" / "sim" / "validation"

# house style, standalone copy of pilot.figures.apply_style
PALETTE: Tuple[str, ...] = ("#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677",
                            "#AA4499", "#882255")
INDIGO, GREEN, TEAL, LIGHTBLUE, SAND, _ROSE, PURPLE, WINE = PALETTE
SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times", "DejaVu Serif"]
BASE_PT, SMALL_PT, DPI = 9.0, 8.0, 300
FIG_W_IN, FIG_H_IN = 17.8 / 2.54, 5.4 / 2.54  # two-column strip

# policies in palette order; the labels are validation_figures.py's wording
POLICIES: Tuple[str, ...] = ("calendar", "calendar_adapt", "bdinterp@0.1", "bdinterp@0.5",
                             "bdinterp@1.0", "lcp", "offline")  # informed = offline at these leads
LABELS: Dict[str, str] = {"calendar": "Plan (the advice)",
                          "calendar_adapt": "Plan + online size correction",
                          "bdinterp@0.1": r"BDINTERP-$d$, $\varepsilon$ = 0.1",
                          "bdinterp@0.5": r"BDINTERP-$d$, $\varepsilon$ = 0.5",
                          "bdinterp@1.0": r"BDINTERP-$d$, $\varepsilon$ = 1.0",
                          "lcp": "LCP (the robust sequence)", "informed": "Informed optimum",
                          "offline": "Offline optimum"}
# (colour, marker, dash) per series: the palette in the order of POLICIES, rose skipped (the
# reactive rule's colour elsewhere) so that the size correction, LCP and the optimum keep their
# pilot.policy_style colours and markers.  No curve is dotted: dotted lines are the predicted
# switches, which the caption relies on (policy_style draws the size correction and LCP dotted).
SERIES: Dict[str, Tuple[str, str, object]] = {
    "calendar": (INDIGO, "o", "-"), "calendar_adapt": (GREEN, "h", "--"),
    "bdinterp@0.1": (TEAL, "^", "-."), "bdinterp@0.5": (LIGHTBLUE, "D", "-."),
    "bdinterp@1.0": (SAND, "v", "-"), "lcp": (PURPLE, "d", "--"), "offline": (WINE, "*", "-")}
LIGHT: Tuple[str, ...] = (LIGHTBLUE, SAND)  # drawn no thinner than LIGHT_LW
LINE_LW, LIGHT_LW = 1.1, 1.2
MARKER_PT, STAR_PT = 3.2, 4.5  # a star this small reads as a blob on its line, so it is larger
RING_PT = 6.0  # the plan's open ring must enclose the bdinterp marker on it; at 4.5 corners poke out
SWITCH_DASH: Tuple[float, Tuple[float, float]] = (0, (1, 2))  # the predicted-switch lines
EPS_VALUES: Tuple[str, ...] = ("0.1", "0.5", "1.0")  # the trust settings of bdinterp@<eps>
SWITCH_FACTOR = 4.0  # the remark's constant: the plan is left once cost(plan) >= factor/eps * LCP
DEPART_TOL = 0.01  # "has left the plan" = more than 1% under calendar's mean cost
NET_COL = "net_cost_mean"
PRED_COL = "predicted_switch_scale_net"
XLABEL = "Forecast scale $s$ (multiplier on the booked demand)"
YLABEL = "Cost ratio to the\noffline optimum"  # two lines: one is taller than the panels
XTICKS: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
YTICKS_LOG: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0)  # log y axis: the switch and the underestimate side together
LEGEND_NCOL = 3  # nine entries in three columns of three, filled down: the palette in order
LAYOUTS: Dict[str, Tuple[float, float, int, float]] = {  # width, height (in), legend columns, gap
    "strip": (FIG_W_IN, FIG_H_IN, LEGEND_NCOL, 1.8), "column": (8.8 / 2.54, 6.6 / 2.54, 2, 1.0)}
PAD_PT = 1.5  # page margin
GAP_PT = 3.0  # gap between the legend, the x label and the tick labels
WSPACE = 0.12  # gap between the panels, as a fraction of a panel's width


# --------------------------------------------------------------------------- switch point
def _switch_table(agg: pd.DataFrame, factor: float = SWITCH_FACTOR) -> pd.DataFrame:
    """Observed and predicted switch scale per price ratio and per eps.

    Observed: the smallest ``forecast_scale > 1`` at which ``bdinterp@eps`` sits more than
    ``DEPART_TOL`` of calendar's mean cost below it, i.e. the interpolation has begun to leave the
    plan; reported on the total cost ratio and on the net cost.  Predicted: the smallest
    ``forecast_scale > 1`` at which calendar's *net* cost reaches ``factor/eps`` times LCP's, the
    remark's condition on the quantity the phase test actually compares.
    """
    rows: List[Dict[str, object]] = []
    for ratio in sorted(agg["price_ratio"].dropna().unique()):
        cal, lcp = curve(agg, ratio, "calendar"), curve(agg, ratio, "lcp")
        cal_net = curve(agg, ratio, "calendar", NET_COL)
        lcp_net = curve(agg, ratio, "lcp", NET_COL)
        for eps in EPS_VALUES:
            bd = curve(agg, ratio, f"bdinterp@{eps}")
            bd_net = curve(agg, ratio, f"bdinterp@{eps}", NET_COL)
            if cal.empty or bd.empty or lcp.empty:
                logger.warning("r = %g: no calendar, lcp or bdinterp@%s rows", ratio, eps)
                continue
            over = cal.index > 1.0
            gap, gap_net = (cal - bd) / cal, (cal_net - bd_net) / cal_net
            por_net, por_tot = cal_net / lcp_net, cal / lcp
            need = factor / float(eps)
            below = [f"{s:g}" for s in gap.index[(gap > DEPART_TOL) & (gap.index < 1.0)]]
            rows.append({
                "price_ratio": ratio, "eps": float(eps),
                "observed_switch_scale": first_or_none(gap.index[(gap > DEPART_TOL) & over]),
                "observed_switch_scale_net":
                    first_or_none(gap_net.index[(gap_net > DEPART_TOL) & over]),
                PRED_COL: first_or_none(por_net.index[(por_net >= need) & over]),
                "required_factor": need,
                "max_plan_over_robust_net": float(por_net[over].max()),
                "plan_over_robust_total": float(por_tot[over].max()),
                "switch_below_1": ";".join(below) if below else NONE,
            })
    return pd.DataFrame(rows)


def _header(inputs: Sequence[Path], sunk: float, factor: float, min_replicas: int) -> List[str]:
    """The generated block above the Markdown table: provenance and the three definitions."""
    return [
        "# F6: where BDINTERP-d leaves the timetable plan",
        "",
        f"Generated {date.today().isoformat()} by `scripts/sim/f6_validation.py`.",
        "",
        "Sources:",
        *[f"- `{rel(p)}`" for p in inputs],
        "",
        f"Net cost: the always-on base tier is a sunk cost both branches pay whatever they decide, "
        f"so `bdinterp`'s phase test compares costs with its running cost removed "
        f"(`c_run * min_replicas * minutes`, {sunk:.0f} USD per run at --min-replicas "
        f"{min_replicas}); the predicted switch is evaluated on that net cost and "
        f"`plan_over_robust_total` keeps the total-cost reading beside it.",
        f"Observed switch: the smallest forecast scale above 1 at which `bdinterp@eps` is more "
        f"than {DEPART_TOL:.0%} under `calendar`'s mean cost, on the total cost ratio "
        f"(`observed_switch_scale`) and on the net cost (`observed_switch_scale_net`).",
        f"Predicted switch: the smallest forecast scale above 1 at which `calendar`'s mean net "
        f"cost reaches `{factor:g}/eps` times `lcp`'s (the remark on the switch point).",
        "`switch_below_1` lists any scale under 1 that meets the observed criterion.  Means are "
        "over the seeds; the per-cell aggregate with its seed sd and in-class attainment is in "
        "`f6_forecast_scale.csv`.",
        "",
        "",
    ]


def _write_tables(switch: pd.DataFrame, agg: pd.DataFrame, inputs: Sequence[Path], out_dir: Path,
                  factor: float = SWITCH_FACTOR, min_replicas: int = 1) -> None:
    """The switch table as CSV and Markdown, and the aggregate the figure is drawn from."""
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    switch.to_csv(tables / "f6_switch.csv", index=False)
    agg.to_csv(tables / "f6_forecast_scale.csv", index=False)
    sunk = float(agg["sunk_usd"].mean()) if "sunk_usd" in agg.columns else float("nan")
    head = _header(inputs, sunk, factor, min_replicas)
    (tables / "f6_switch.md").write_text("\n".join(head) + md_table(switch), encoding="utf-8")
    logger.info("wrote %s and .csv", rel(tables / "f6_switch.md"))
    logger.info("wrote %s", rel(tables / "f6_forecast_scale.csv"))


# --------------------------------------------------------------------------- figure
def _apply_style() -> None:
    """The house style: serif, >= 8 pt, no title inside the axes, exact figure size on save."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": BASE_PT, "axes.labelsize": BASE_PT, "axes.titlesize": SMALL_PT,
        "xtick.labelsize": SMALL_PT, "ytick.labelsize": SMALL_PT, "legend.fontsize": SMALL_PT,
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": None,
        "axes.grid": False, "legend.frameon": False,
    })


def _thin(fig: plt.Figure) -> None:
    """Hairline spines and short ticks, as ``validation_figures_style.thin`` gives Fig. 5-8."""
    for ax in fig.axes:
        ax.tick_params(labelsize=SMALL_PT, length=2.0, width=0.6, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)


def _style(policy: str) -> Dict[str, object]:
    """Colour, marker and dash of one series, from ``SERIES``; the plan is drawn as open rings."""
    color, marker, dash = SERIES[policy]
    style: Dict[str, object] = {"color": color, "marker": marker, "linestyle": dash,
                                "markersize": STAR_PT if marker == "*" else MARKER_PT,
                                "linewidth": LIGHT_LW if color in LIGHT else LINE_LW}
    if policy == "calendar":  # bdinterp lies exactly on the plan until it switches
        style.update({"markersize": RING_PT, "markerfacecolor": "none", "markeredgewidth": 0.8,
                      "zorder": 4})
    return style


def _ylim(agg: pd.DataFrame, cap: float) -> float:
    """Upper limit of the shared y axis: the data, or ``cap`` when a curve runs far above it."""
    top = float(agg["cost_ratio_mean"].max())
    return min(cap, top * 1.04) if cap > 0 else top * 1.04


def _panel(ax, agg: pd.DataFrame, ratio: float, switch: pd.DataFrame, ymax: float,
           ylog: bool = False) -> None:
    """One price-ratio panel: the policy curves (clipped at ``ymax`` on a linear axis) and the predicted switches."""
    off_scale: List[float] = []
    for policy in POLICIES:
        series = curve(agg, ratio, policy)
        if series.empty:
            continue
        x, y = series.index.to_numpy(dtype=float), series.to_numpy(dtype=float)
        over = (y > ymax) & (not ylog)
        off_scale.extend(y[over].tolist())
        style = _style(policy)
        ax.plot(x, y if ylog else y.clip(max=ymax), **style)
        if over.any():  # carets where the curve was clipped, so the reader sees which one left
            ax.plot(x[over], [ymax] * int(over.sum()), linestyle="none", marker="^",
                    markersize=3.6, color=style["color"], clip_on=False, zorder=5)
    hits = switch[(switch["price_ratio"] == ratio) & (switch[PRED_COL] != NONE)] \
        if not switch.empty else switch
    for _, row in hits.iterrows():
        name = f"bdinterp@{row['eps']:.1f}"
        ax.axvline(float(row[PRED_COL]), color=SERIES.get(name, SERIES["calendar"])[0],
                   linestyle=SWITCH_DASH, linewidth=LIGHT_LW, zorder=1)
    if off_scale:
        lo, hi = min(off_scale), max(off_scale)
        text = f"off scale: {lo:.2f}" if hi - lo < 5e-3 else f"off scale: {lo:.2f}-{hi:.2f}"
        ax.text(0.03, 0.96, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=SMALL_PT, color="0.30")
    ax.set_xscale("log")
    scales = sorted(agg.loc[agg["price_ratio"] == ratio, "forecast_scale"].unique())
    ax.set_xticks([t for t in XTICKS if min(scales) <= t <= max(scales)])
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    if ylog:
        ax.set_yscale("log")
        ax.set_yticks([t for t in YTICKS_LOG if t <= ymax])
        ax.set_yticks([], minor=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_ylim(0.95, ymax)
    else:
        ax.set_ylim(0.985, ymax)
    ax.margins(x=0.10)
    ax.set_title(f"$r$ = {ratio:.0f}")  # italic variable only: no mathtext sub/superscripts (0.7x)


def _fit(fig: plt.Figure, axes: Sequence[plt.Axes], legend: Legend, xlabel: Text) -> None:
    """Stack the legend, the x label and the panels from the bottom edge up, measured, not guessed.

    The y label and the titles set the left and top margins.  The legend and the x label are
    centred on the panels, not on the page: the y label puts the panels right of the page centre.
    """
    fig.subplots_adjust(wspace=WSPACE)
    inv = fig.transFigure.inverted()

    def box(artist):
        return artist.get_window_extent().transformed(inv)

    first = axes[0]
    pad_x, pad_y = PAD_PT / 72.0 / fig.get_figwidth(), PAD_PT / 72.0 / fig.get_figheight()
    gap = GAP_PT / 72.0 / fig.get_figheight()
    left = right = 0.0
    legend_top = 0.0
    for _ in range(2):  # twice: on a linear axis the locator picks other ticks once the panels resize
        fig.draw_without_rendering()
        pos = first.get_position()
        labels = [first.yaxis.label, *first.get_yticklabels()]
        left_most = min(box(t).x0 for t in labels if t.get_text())
        lowest_tick = min(box(t).y0 for t in first.get_xticklabels() if t.get_text())
        title_rise = max(box(ax.title).y1 for ax in axes) - pos.y1
        legend_top = pad_y + box(legend).height
        xlabel_top = legend_top + gap + box(xlabel).height
        left, right = pad_x + pos.x0 - left_most, 1.0 - pad_x
        fig.subplots_adjust(left=left, right=right, top=1.0 - pad_y - title_rise,
                            bottom=xlabel_top + gap + pos.y0 - lowest_tick)
    centre = 0.5 * (left + right)
    half = 0.5 * box(legend).width  # a legend wider than the panels is kept on the page
    centre = min(max(centre, pad_x + half), 1.0 - pad_x - half)
    legend.set_bbox_to_anchor((centre, pad_y), transform=fig.transFigure)
    xlabel.set_position((centre, legend_top + gap))


def _figure(agg: pd.DataFrame, switch: pd.DataFrame, out_dir: Path, cap: float,
            factor: float = SWITCH_FACTOR, ylog: bool = True, panels: Sequence[float] = (),
            layout: str = "strip", stem: str = "f6_forecast_error") -> None:
    """The strip of price-ratio panels, one shared legend below, written as PDF and PNG;
    ``panels`` keeps only those ratios and ``layout`` picks the page geometry, same data."""
    ratios = sorted(agg["price_ratio"].dropna().unique())
    if panels:
        ratios = [r for r in ratios if any(abs(r - float(p)) < 1e-9 for p in panels)]
        agg = agg[agg["price_ratio"].isin(ratios)]
        switch = switch[switch["price_ratio"].isin(ratios)] if not switch.empty else switch
    ymax = float(agg["cost_ratio_mean"].max()) * 1.10 if ylog else _ylim(agg, cap)
    logger.info("figure: %d panel(s), y axis %s, top %.3f", len(ratios), "log" if ylog else "linear (clipped)", ymax)
    width, height, ncol, spacing = LAYOUTS[layout]
    fig, axes = plt.subplots(1, len(ratios), sharey=True, squeeze=False, figsize=(width, height))
    for ax, ratio in zip(axes[0], ratios):
        _panel(ax, agg, ratio, switch, ymax, ylog)
    _thin(fig)
    axes[0][0].set_ylabel(YLABEL, fontsize=SMALL_PT)
    drawn = [p for p in POLICIES if not curve(agg, ratios[0], p).empty]
    handles = [plt.Line2D([], [], **_style(p)) for p in drawn]
    labels = [LABELS.get(p, p) for p in drawn]
    hits = switch[switch[PRED_COL] != NONE].drop_duplicates("eps") if not switch.empty else switch
    for _, row in hits.iterrows():  # one dotted entry per trust value that predicts a switch in the grid
        color = SERIES[f"bdinterp@{row['eps']:.1f}"][0]
        handles.append(plt.Line2D([], [], color=color, linestyle=SWITCH_DASH, linewidth=LIGHT_LW))
        labels.append(f"Predicted switch, $\\varepsilon$ = {row['eps']:.1f}")
    legend = fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=SMALL_PT,
                        handlelength=2.2, columnspacing=spacing, handletextpad=0.4, borderpad=0.0,
                        borderaxespad=0.0, frameon=False)
    xlabel = fig.text(0.5, 0.0, XLABEL, ha="center", va="bottom", fontsize=SMALL_PT)
    _fit(fig, list(axes[0]), legend, xlabel)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"{stem}.{ext}")
    plt.close(fig)
    logger.info("wrote %s.{pdf,png}", rel(figures / stem))


# --------------------------------------------------------------------------- entry point
def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", nargs="+", metavar="CSV",
                   help="sweep CSVs or glob patterns; missing files are skipped")
    p.add_argument("--dev", action="store_true",
                   help="read the batch-C epsilon sweep instead of the validation sweeps")
    p.add_argument("--out", type=Path, default=OUT_DIR, help=f"output root (default {OUT_DIR})")
    p.add_argument("--yscale", choices=("log", "linear"), default="log",
                   help="y axis of the figure: log (default) shows the switch and the underestimate "
                        "side together; linear clips at --ymax")
    p.add_argument("--ymax", type=float, default=2.0,
                   help="cap of the shared y axis; curves above it are clipped (0 = no cap)")
    p.add_argument("--switch-factor", type=float, default=SWITCH_FACTOR,
                   help="constant of the predicted switch rule (default 4, i.e. 4/eps)")
    p.add_argument("--min-replicas", type=int, default=1,
                   help="always-on base tier whose running cost is sunk (default 1)")
    p.add_argument("--c-run", type=float, default=None,
                   help=f"USD per replica-minute when the CSVs carry no column "
                        f"(default {C_RUN_DEFAULT})")
    p.add_argument("--panels", nargs="+", type=float, metavar="R", help="price ratios to draw")
    p.add_argument("--layout", choices=tuple(LAYOUTS), default="strip",
                   help="strip: two columns wide (default); column: one column wide")
    p.add_argument("--stem", default="f6_forecast_error", help="figure file name, no extension")
    p.add_argument("--figure-only", action="store_true", help="draw the figure, write no tables")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("fontTools", "matplotlib"):  # the PDF font subsetter talks at INFO
        logging.getLogger(noisy).setLevel(logging.WARNING)
    patterns = args.inputs or (DEV_INPUTS if args.dev else DEFAULT_INPUTS)
    paths = resolve_inputs(patterns)
    if not paths:
        logger.error("no input CSV found; pass --inputs or --dev")
        return 1
    df = load(paths)
    if df.empty:
        logger.error("the inputs carry no forecast_scale rows")
        return 1
    agg = aggregate(df, args.c_run, args.min_replicas)
    switch = _switch_table(agg, args.switch_factor)
    if not args.figure_only:
        _write_tables(switch, agg, paths, args.out, args.switch_factor, args.min_replicas)
    _apply_style()
    _figure(agg, switch, args.out, args.ymax, args.switch_factor, ylog=args.yscale == "log",
            panels=args.panels or (), layout=args.layout, stem=args.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
