"""Draw F4, F4b, F5, F6, F6b and F7 from the sweep CSVs written by run_sweep.py.

    uv run python scripts/sim/make_figures.py --sweeps outputs/sim --out outputs/sim/figures
    uv run python scripts/sim/make_figures.py --sweeps outputs/batchC/r30 --min-replicas 1 \
        --c-run 0.0333

F4  cost-violation Pareto: GPU-hours against in-class SLO attainment, one marker per policy.
F4b weekly cost over the offline optimum at the reference cell, with the R1 weeks shaded.
F5  gain against the *measured* synchrony sigma_2: cost saved relative to the reactive baseline,
    with the two data anchors (prompted sign-ins, self-paced ASSISTments) as dashed verticals.
F6  cost ratio against the adoption-estimate error (forecast scale), one facet per price ratio.
F6b the same ratio against the *measured* calendar error eta_norm.
F7  provisioning tiers x policies: in-class violation rate at d_rep / d_inst / d_res.

The sweep CSVs carry whole-run rows (``week = 0``) and one row per teaching week; F4b is the one
figure drawn from the weekly rows and every other figure filters on ``week = 0``.

``--min-replicas M`` with ``--c-run C`` reports the sweeps of a deployment that pays for an
always-on base tier (``run_sweep.py --min-replicas``).  The constant ``M * C * minutes`` sits in
the numerator and the denominator of every cost ratio and compresses the differences the figures
are about, so F5 takes its gain on the cost *above* the tier (``pilot.batchc.derive``).  F4 keeps
the total GPU-hours, per teaching week, because the point of an institution-wide deployment is how
large that floor is next to what the scheduler decides; F6 and F6b keep the total cost ratio,
because that is the quantity lemma B bounds.  At the default ``--min-replicas 0`` there is no base
tier and every figure is exactly the one drawn before the option existed.

House style comes from ``pilot.figures.apply_style`` (serif, >= 8 pt, the fixed colour cycle, no
titles inside the axes, ``svg.fonttype = "none"``); every figure is written as PDF and PNG.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

PILOT = Path(__file__).resolve().parents[2]
if str(PILOT / "src") not in sys.path:
    sys.path.insert(0, str(PILOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from pilot.batchc import FloorSpec, reference_cell  # noqa: E402
from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style, save  # noqa: E402

logger = logging.getLogger("make_figures")

LABELS: Dict[str, str] = {
    "static": "Static peak",
    "reactive": "Reactive (HPA)",
    "velocity": "Token velocity",
    "forecast_ilp": "History + ILP",
    "lcp": "LCP",
    "calendar": "Timetable FtP",
    "bdinterp": "BDINTERP-$d$",
    "offline": "Offline optimum",
    "informed": "Informed optimum",
    "stale_calendar": "Stale timetable",
    "calendar_adapt": "Timetable + online correction",
    "fixed": "Base tier only",
    "guard": "Bell-aware guard",
}
ORDER = list(LABELS)
ALIASES: Dict[str, str] = {"trust": "bdinterp"}  # CSVs written before policy 7 was renamed
# policy 7 at three trust values shares one colour and is told apart by dash and marker
VARIANTS: Dict[str, int] = {"0.1": 0, "0.5": 1, "1.0": 2}
F6_POLICIES = ("calendar", "calendar_adapt", "bdinterp@0.1", "bdinterp@0.5", "bdinterp@1.0", "lcp",
               "informed")
F6_LABELS: Dict[str, str] = {"calendar": "FtP-cal"}  # the name the BDINTERP-d analysis and the probe note use
F6_THEORY_EPS = 0.5
F6_XTICKS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)  # a readable subset of the log-spaced scale grid
F6_MAX_WIDTH_IN = 7.16  # two-column width: the facets are drawn side by side
F6_XLABEL = "Adoption estimate error (forecast scale)"
F5_XCOL = "sigma2_measured"  # the measured synchrony, not the mixing weight ``sigma``
F5_SIGNIN_ANCHOR = 0.70  # prompted sign-ins (pilot Q1); the self-paced anchor is measured
TIER_LABELS: Dict[str, str] = {"d_rep": "$d_{\\mathrm{rep}}$ (0.5 min)",
                               "d_inst": "$d_{\\mathrm{inst}}$ (8 min)",
                               "d_res": "$d_{\\mathrm{res}}$ (8 min, 60 min block)"}
R1_WEEKS: Tuple[int, ...] = (1, 5)  # novel timetable: week 1 has no history, week 5 the step
# the flat line, the zero line, and the control that never scales at all
F4B_EXCLUDED: Tuple[str, ...] = ("static", "offline", "fixed")
F4B_SHADE = "0.90"  # the grey the R1 weeks are shaded with
F4_MATCHED_SUFFIX: str = ", SLO $\\geq$ 0.95"  # the hollow markers of the SLO-matched pair
F4_MATCHED_GROUP: str = "r30_slo95"  # its sweep directory, beside the one being plotted
F4_MATCHED_SWEEP: str = "lead"
# The base tier lives in ``pilot.batchc`` rather than here: this module is loaded by path in the
# tests (``importlib`` without a sys.modules entry), which a dataclass defined under
# ``from __future__ import annotations`` cannot survive.


def _canonical(name: str) -> str:
    """Policy key under its current name, with any ``@eps`` trust suffix stripped."""
    base = str(name).partition("@")[0]
    return ALIASES.get(base, base)


def _label(name: str) -> str:
    """Legend entry; a trust-pinned policy 7 carries its epsilon."""
    base, _, eps = str(name).partition("@")
    text = LABELS.get(_canonical(base), base)
    return f"{text}, $\\varepsilon={eps}$" if eps else text


def _whole_run(df: pd.DataFrame) -> pd.DataFrame:
    """The whole-run rows of a sweep CSV (``week = 0``); older CSVs have no week column."""
    return df[df["week"] == 0] if "week" in df.columns else df


def _style(cfg: FigureConfig, name: str) -> Dict[str, object]:
    """Colour, marker and dash pattern of one policy, identical in every figure.

    The index is the policy's position in ``ORDER``, not in the plotted subset, so a policy keeps
    its appearance across F4, F5, F6b and F7 (F6 takes the palette in order, ``_f6_style``).  Ten
    policies against eight colours: the ninth and tenth reuse the first two, and the marker index
    is shifted by five on every wrap so a reused colour never repeats a marker and dash.
    """
    key = _canonical(name)
    i = ORDER.index(key) if key in ORDER else len(ORDER)
    n_c, n_m = len(cfg.palette), len(cfg.markers)
    style = {"color": cfg.palette[i % n_c],
             "marker": cfg.markers[(i + 5 * (i // n_c)) % n_m],
             "linestyle": cfg.linestyles[(i + i // n_c) % len(cfg.linestyles)],
             "markersize": 3.5, "linewidth": 1.1}
    eps = str(name).partition("@")[2]
    if eps in VARIANTS:  # same colour, different dash and marker
        v = VARIANTS[eps]
        style["linestyle"] = cfg.linestyles[v % len(cfg.linestyles)]
        style["marker"] = cfg.markers[(i + v) % n_m]
    return style


def _legend(ax, cfg: FigureConfig, ncol: int = 3) -> None:
    """Legend below the axes so that it never covers a series."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=ncol, frameon=False,
              fontsize=cfg.small_font_pt - 1.0, handlelength=2.4, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.0)


def _read(path: Path) -> Optional[pd.DataFrame]:
    """Read one sweep CSV, or return ``None`` when it has not been produced."""
    if not path.exists():
        logger.warning("missing sweep file %s", path)
        return None
    return pd.read_csv(path)


def _ordered(names: Sequence[str]) -> list:
    """Policies in the paper's order, unknown ones appended."""
    rank = {n: ORDER.index(_canonical(n)) if _canonical(n) in ORDER else len(ORDER) for n in names}
    return sorted(dict.fromkeys(names), key=lambda n: (rank[n], n))


def _slo_matched(sweeps: Path) -> Optional[pd.DataFrame]:
    """The SLO-matched lead sweep beside ``sweeps``, when the batch wrote one.

    ``batch_c.sh`` puts it in a sibling directory of the group being plotted
    (``<sweeps>/../r30_slo95/lead/lead.csv``); every other group has none, so its absence is a
    debug line and not the warning ``_read`` would log.
    """
    path = Path(sweeps).parent / F4_MATCHED_GROUP / F4_MATCHED_SWEEP / f"{F4_MATCHED_SWEEP}.csv"
    if not path.exists():
        logger.debug("no SLO-matched sweep at %s", path)
        return None
    logger.info("F4 also draws the SLO-matched baselines of %s", path)
    return pd.read_csv(path)


def _f4_points(sel: pd.DataFrame, floor: FloorSpec) -> Tuple[pd.DataFrame, str, str]:
    """Mean attainment and GPU-hours per policy, with the y column and its axis label.

    The y axis stays the *total* GPU-hours a policy spent, base tier included; with a base tier it
    is divided by the number of teaching weeks so that runs of different length are readable on one
    axis.  Reporting only the attributable component here would hide how large the always-on floor
    is next to what the scheduler decides, which is exactly what the reader of an institution-wide
    deployment needs to see (the cost tables carry the attributable component instead).
    """
    ycol, ylabel = "gpu_hours", "GPU-hours"
    if not floor.on:
        return sel.groupby("policy")[["slo_in", ycol]].mean().reset_index(), ycol, ylabel
    if "weeks" in sel.columns:
        weeks = pd.to_numeric(sel["weeks"], errors="coerce")
    else:
        logger.warning("F4: no 'weeks' column; GPU-hours are shown for the whole run")
        weeks = pd.Series(1.0, index=sel.index)
    sel = sel.assign(gpu_hours_per_week=pd.to_numeric(sel["gpu_hours"], errors="coerce") / weeks)
    ycol, ylabel = "gpu_hours_per_week", "GPU-hours per week"
    return sel.groupby("policy")[["slo_in", ycol]].mean().reset_index(), ycol, ylabel


def _f4_matched(ax, matched: Optional[pd.DataFrame], cfg: FigureConfig, floor: FloorSpec) -> None:
    """The SLO-matched baselines as hollow markers of the same colour, marker and size.

    ``outputs/batchC/r30_slo95`` re-tunes the two feedback baselines to the cheapest configuration
    that reaches ``slo_in >= 0.95``, so the pair says what the same baseline costs once it is held
    to the attainment the planner reaches; hollow markers keep that pair visibly the same policy.
    """
    if matched is None or matched.empty:
        return
    frame = _whole_run(matched)
    cell = reference_cell(frame)
    agg, ycol, _ = _f4_points(cell.rows(frame), floor)
    for name in _ordered(agg["policy"].tolist()):
        row = agg[agg["policy"] == name].iloc[0]
        st = _style(cfg, name)
        ax.plot(row["slo_in"], row[ycol], marker=st["marker"], color=st["color"],
                linestyle="none", markersize=5, markerfacecolor="none", markeredgewidth=1.0,
                label=f"{LABELS.get(_canonical(name), name)}{F4_MATCHED_SUFFIX}")
    logger.info("F4 SLO-matched cell: %s", cell.description)


def figure_f4(df: pd.DataFrame, out: Path, cfg: FigureConfig, floor: Optional[FloorSpec] = None,
              matched: Optional[pd.DataFrame] = None) -> None:
    """GPU-hours against in-class SLO attainment, one marker per policy.

    With a base tier the y axis is the total GPU-hours per teaching week rather than per run; it
    stays the total either way (``_f4_points``).  ``matched`` is the optional SLO-matched retuning
    of the two feedback baselines, drawn as hollow markers (``_f4_matched``).
    """
    floor = floor or FloorSpec()
    df = _whole_run(df)
    sel = df[df["sigma"] == df["sigma"].max()] if "sigma" in df else df
    agg, ycol, ylabel = _f4_points(sel, floor)
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    for name in _ordered(agg["policy"].tolist()):
        row = agg[agg["policy"] == name].iloc[0]
        st = _style(cfg, name)
        ax.plot(row["slo_in"], row[ycol], marker=st["marker"], color=st["color"],
                linestyle="none", markersize=5, label=LABELS.get(_canonical(name), name))
    _f4_matched(ax, matched, cfg, floor)
    ax.set_xlabel("In-class SLO attainment (TTFT p95 $\\leq$ 2 s)")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.10, y=0.12)
    _legend(ax, cfg)
    save(fig, out, "F4_cost_violation_pareto")


def figure_f4b(df: pd.DataFrame, out: Path, cfg: FigureConfig,
               floor: Optional[FloorSpec] = None) -> None:
    """Weekly cost over the offline optimum at the reference cell, with the R1 weeks shaded.

    The companion of F4: instead of one whole-run point per policy it shows *when* a policy pays,
    week by week, against the optimum of its own run.  Weeks 1 and 5 are the novel-timetable weeks
    (no history at all, and the freshman step of this institution's timetable); a policy that only
    works once it has seen the week before is separated from one that does not exactly there.

    ``excess_usd`` is a difference of two costs, so the base tier cancels out of it and the figure
    is drawn whether or not there is a floor.  The cell is the replica tier of a lead sweep or the
    most synchronous cell of a sigma sweep (``pilot.batchc.reference_cell``).
    """
    floor = floor or FloorSpec()
    frame = floor.derived(df)
    cell = reference_cell(frame)
    rows = cell.rows(frame)
    week = pd.to_numeric(rows.get("week", pd.Series(dtype=float)), errors="coerce")
    weekly = rows[week > 0]
    if weekly.empty:
        logger.warning("F4b needs per-week rows (week >= 1); skipping")
        return
    piv = weekly.groupby(["policy", "week"])["excess_usd"].mean().unstack("week")
    names = [n for n in _ordered(piv.index.tolist()) if _canonical(n) not in F4B_EXCLUDED]
    if not names:
        logger.warning("F4b has no policy left after dropping %s; skipping", list(F4B_EXCLUDED))
        return
    weeks = [int(w) for w in piv.columns]
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    for i, w in enumerate(sorted(set(R1_WEEKS) & set(weeks))):
        ax.axvspan(w - 0.5, w + 0.5, color=F4B_SHADE, linewidth=0.0, zorder=0,
                   label="R1" if i == 0 else None)
    for name in names:
        ax.plot(weeks, piv.loc[name].to_numpy(dtype=float), label=_label(name),
                **_style(cfg, name))
    ax.set_xticks(weeks)
    ax.set_xlabel("Teaching week")
    ax.set_ylabel("Cost over the offline optimum (USD per week)")
    ax.margins(x=0.04, y=0.12)
    logger.info("F4b cell: %s", cell.description)
    _legend(ax, cfg)
    save(fig, out, "F4b_weekly_excess")


def figure_f5(df: pd.DataFrame, out: Path, cfg: FigureConfig,
              floor: Optional[FloorSpec] = None) -> None:
    """Cost saved relative to the reactive baseline, against the *measured* synchrony sigma_2.

    The x axis is ``sigma2_measured`` (the median over sessions of the share of per-student first
    arrivals within 2 min of the session's first), not the knob ``sigma``, which is only the weight
    the two offset pools are mixed with.  The dashed verticals are the data anchors.

    With a base tier the gain is taken on the attributable cost, ``total_cost - base_usd``: the
    saving a scheduler can actually make is a saving on the capacity it decides, not on the replica
    the deployment keeps up whatever the scheduler does.
    """
    floor = floor or FloorSpec()
    whole = _whole_run(df)
    value, ylabel = "total_cost", "Cost saved vs reactive"
    if floor.on:
        whole = floor.derived(whole)
        value, ylabel = "attr_cost", "Cost above the base tier saved vs reactive"
    axis = F5_XCOL if F5_XCOL in whole.columns else "sigma"
    if axis == "sigma":
        logger.warning("F5: no %s column, falling back to the sigma knob; re-run the sweep",
                       F5_XCOL)
    piv = whole.groupby(["policy", "sigma"])[value].mean().unstack("sigma")
    if "reactive" not in piv.index:
        logger.warning("F5 needs the reactive baseline; skipping")
        return
    xs = whole.groupby("sigma")[axis].mean().reindex(piv.columns).to_numpy(dtype=float)
    gain = 1.0 - piv.div(piv.loc["reactive"], axis=1)
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    for name in _ordered([n for n in gain.index if n != "reactive"]):
        ax.plot(xs, gain.loc[name].to_numpy(), label=_label(name), **_style(cfg, name))
    ax.axhline(0.0, color="0.4", linewidth=0.6, linestyle=":")
    anchors = _f5_anchors(whole, axis)
    mid = float(np.mean(xs)) if xs.size else 0.5
    for value, text in anchors:
        ax.axvline(value, color="0.35", linestyle=(0, (4, 2)), linewidth=0.8, zorder=0)
        right = value > mid  # keep the label inside the axes at either end of the range
        ax.annotate(text, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-2.0 if right else 2.0, -2.0), textcoords="offset points",
                    ha="right" if right else "left", va="top",
                    fontsize=cfg.small_font_pt - 1.0, color="0.35")
    ax.set_xlabel("Measured synchrony $\\sigma_2$ (first arrivals within 2 min)")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.08, y=0.12)
    _legend(ax, cfg)
    save(fig, out, "F5_gain_vs_sigma")


def _f5_anchors(whole: pd.DataFrame, axis: str) -> list:
    """F5's two data anchors: the prompted sign-ins at 0.70, and the sweep's own ``w = 0`` cell.

    The ASSISTments anchor is measured, not hard-coded, so the offset pools and it move together.
    """
    anchors = [(F5_SIGNIN_ANCHOR, "sign-ins")]
    if axis == "sigma":
        return anchors
    w_min = float(whole["sigma"].min())
    self_paced = float(whole.loc[whole["sigma"] == w_min, axis].mean())
    if w_min > 0.0:
        logger.warning("F5: the sweep has no w = 0 cell; the ASSISTments anchor uses w = %g", w_min)
    if np.isfinite(self_paced):
        anchors.append((self_paced, "ASSISTments"))
    return anchors


def _f6_style(cfg: FigureConfig, i: int) -> Dict[str, object]:
    """Colour, marker and dash of F6's ``i``-th series: the house palette from its first entry.

    F6 is the only figure whose series are not the ten policies - it draws one policy three times,
    once per trust setting - so it takes the palette in order instead of at each policy's ``ORDER``
    position (``_style``).
    """
    return {"color": cfg.palette[i % len(cfg.palette)],
            "marker": cfg.markers[i % len(cfg.markers)],
            "linestyle": cfg.linestyles[i % len(cfg.linestyles)],
            "markersize": 3.5, "linewidth": 1.1}


def _f6_panel(ax, part: pd.DataFrame, names: Sequence[str], cfg: FigureConfig,
              ratio_col: str = "cost_ratio", opt_col: str = "offline_cost") -> None:
    """One price-ratio facet: mean cost ratio over the seeds, min/max whiskers.

    ``ratio_col`` and ``opt_col`` are the whole-run ratio and the optimum the theory bound divides
    by; above a base tier they are ``attr_ratio`` and ``offline_cost - base_usd``.
    """
    for i, name in enumerate(names):
        grp = part[part["policy"] == name].groupby("forecast_scale")[ratio_col]
        mean, lo, hi = grp.mean(), grp.min(), grp.max()
        style = _f6_style(cfg, i)
        if _canonical(name) == "calendar":  # policy 7 lies exactly on it in the ADV branch
            style.update({"markersize": 7.0, "markerfacecolor": "none", "markeredgewidth": 0.9,
                          "zorder": 3})
        ax.errorbar(mean.index.to_numpy(), mean.to_numpy(),
                    yerr=[(mean - lo).to_numpy(), (hi - mean).to_numpy()], capsize=1.6,
                    elinewidth=0.7, label=F6_LABELS.get(name, _label(name)), **style)
    ref = (part[part["policy"] == names[0]]
           .groupby("forecast_scale")[["eta", opt_col, "p_in"]].mean())
    bound = (1.0 + F6_THEORY_EPS) * (1.0 + 2.0 * ref["p_in"] * ref["eta"] / ref[opt_col])
    ax.plot(ref.index.to_numpy(), bound.to_numpy(), color="0.35", linestyle=(0, (4, 2)),
            linewidth=0.9, label=f"Lemma B, $\\varepsilon={F6_THEORY_EPS}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = part["forecast_scale"]
    ax.set_xticks([t for t in F6_XTICKS if xs.min() <= t <= xs.max()])
    ax.set_xticks([], minor=True)  # the log minor ticks would be unlabelled clutter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.margins(x=0.10, y=0.15)


def figure_f6(df: pd.DataFrame, out: Path, cfg: FigureConfig,
              floor: Optional[FloorSpec] = None) -> None:
    """Cost ratio against the adoption-estimate error, one facet per dimensionless price ratio.

    The x axis is ``forecast_scale``, the factor by which the plan misjudges the size of every
    booked session; it is logarithmic and only the ``forecast_scale`` cells are drawn, so one curve
    is one policy along one error source (F6b puts all three on the *measured* error axis).  The
    facets are the dimensionless price ratio ``p_in mu / c_run``: policy 7 leaves the advice only
    when the calendar plan costs about ``4/eps`` times LCP, which needs both a catastrophic
    over-forecast and capacity that is not free, so that axis is what makes the trust knob visible
    at all (results/sim/f6_regime_probe.md).  Points are
    means over the seeds with min/max whiskers; the dashed line is lemma B's consistency bound
    ``(1 + eps) (1 + 2 p_in eta / OPT)`` at ``eps = 0.5``, with ``OPT`` that run's own optimum.

    ``floor`` is accepted for a uniform call signature and deliberately not used: lemma B bounds
    the *total* cost against the *total* optimum, so the base tier belongs in both the ratio and
    the bound.  Taking the ratio above the floor would separate the curves more, but it would no
    longer be the quantity the theorem is about (the tables report that component instead).
    """
    whole = _whole_run(df).copy()
    missing = {"forecast_scale", "cost_ratio", "eta", "offline_cost", "p_in"} - set(whole.columns)
    if missing:
        logger.warning("F6 needs %s; re-run the epsilon sweep", sorted(missing))
        return
    ratio_col, opt_col, ylabel = "cost_ratio", "offline_cost", "Cost ratio to the offline optimum"
    sel = whole[whole["error_knob"] == "forecast_scale"] if "error_knob" in whole else whole
    names = [p for p in F6_POLICIES if p in set(sel["policy"])]
    if sel.empty or not names:
        logger.warning("F6 found no forecast_scale rows for %s; skipping", list(F6_POLICIES))
        return
    ratios = (sorted(sel["price_ratio"].dropna().unique())
              if "price_ratio" in sel.columns else []) or [None]  # no price axis: one facet
    fig, axes = plt.subplots(1, len(ratios), sharey=True, squeeze=False,
                             figsize=(min(F6_MAX_WIDTH_IN, cfg.width_in * len(ratios)),
                                      cfg.height_in))
    for ax, ratio in zip(axes[0], ratios):
        _f6_panel(ax, sel if ratio is None else sel[sel["price_ratio"] == ratio], names, cfg,
                  ratio_col, opt_col)
        if ratio is not None:  # facet header, not a title inside the axes
            ax.set_title(f"$p_{{\\mathrm{{in}}}}\\mu/c_{{\\mathrm{{run}}}} = {ratio:.0f}$",
                         fontsize=cfg.small_font_pt)
    axes[0][0].set_ylabel(ylabel)
    # one x label and one legend for the whole strip, both under the figure box so that neither
    # collides with the tick labels of a facet
    fig.text(0.5, -0.005, F6_XLABEL, ha="center", va="top", fontsize=cfg.base_font_pt)
    handles, labels = axes[0][0].get_legend_handles_labels()
    # matplotlib hands back the plain lines before the error bars, which would open the legend with
    # the theory line; sort it back so that the series come first, in palette order.
    order = sorted(range(len(labels)), key=lambda i: labels[i].startswith("Lemma B"))
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False,
               ncol=min(4, len(labels)) if len(ratios) > 1 else 2,
               fontsize=cfg.small_font_pt - 1.0, handlelength=2.4, columnspacing=1.0,
               handletextpad=0.4)
    save(fig, out, "F6_cost_ratio_vs_forecast_scale")


def figure_f6b(df: pd.DataFrame, out: Path, cfg: FigureConfig,
               floor: Optional[FloorSpec] = None) -> None:
    """Cost ratio against the measured calendar error ``eta_norm``, all three error knobs at once.

    The companion of F6: the cells F6 leaves out (``bell_shift_frac``, ``p_cancel``) are reported
    here, on the error the run was measured to have rather than the one injected.  The ratio is the
    total one and ``floor`` is ignored, for the reason given in ``figure_f6``.
    """
    whole = _whole_run(df).copy()
    if "eta_norm" not in whole.columns:
        logger.warning("F6b needs the eta columns; re-run the epsilon sweep")
        return
    ratio_col, opt_col, ylabel = "cost_ratio", "offline_cost", "Cost ratio to the offline optimum"
    names = [p for p in F6_POLICIES if p in set(whole["policy"])]
    if not names:
        logger.warning("F6b found none of %s in the sweep; skipping", list(F6_POLICIES))
        return
    keys = [k for k in ("error_knob", "error_value") if k in whole.columns] or ["eta_norm"]
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    for name in names:
        agg = (whole[whole["policy"] == name].groupby(keys)[["eta_norm", ratio_col]]
               .mean().sort_values("eta_norm"))
        style = _style(cfg, name)
        # markers only: the cost ratio is not a function of eta alone.  An under-forecast and an
        # over-forecast of the same eta cost very different amounts, so joining the cells of the
        # three error knobs into one curve would invent a relation that is not there.
        style.update({"linestyle": "none", "markersize": 4.5})
        if _canonical(name) == "calendar":  # policy 7 sits exactly on it in the ADV branch
            style.update({"markersize": 8.0, "markerfacecolor": "none", "markeredgewidth": 0.9,
                          "zorder": 3})
        ax.plot(agg["eta_norm"].to_numpy(), agg[ratio_col].to_numpy(), label=_label(name),
                **style)
    ref = whole[whole["policy"] == names[0]].groupby(keys)[["eta_norm", "eta", opt_col,
                                                            "p_in"]].mean().sort_values("eta_norm")
    bound = ((1.0 + F6_THEORY_EPS)
             * (1.0 + 2.0 * ref["p_in"] * ref["eta"] / ref[opt_col]))
    ax.plot(ref["eta_norm"].to_numpy(), bound.to_numpy(), color="0.35", linestyle=(0, (4, 2)),
            linewidth=0.9, label=f"Lemma B, $\\varepsilon={F6_THEORY_EPS}$")
    ax.set_xlabel("Calendar error $\\eta/\\sum_t\\lambda_t$")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.margins(x=0.08, y=0.12)
    _legend(ax, cfg, ncol=2)
    save(fig, out, "F6b_cost_ratio_vs_calendar_error")


def figure_f7(df: pd.DataFrame, out: Path, cfg: FigureConfig) -> None:
    """In-class violation rate across the three provisioning tiers, one series per policy.

    A tier is a lead time *and* a commitment / billing block (d_res shares the instance lead time
    and commits by the hour), so the axis is not ordered by lead time alone.
    """
    df = _whole_run(df)
    key = "lead_tier" if "lead_tier" in df else "lead_min"
    piv = df.groupby(["policy", key])["viol_in"].mean().unstack(key)
    tiers = [c for c in ("d_rep", "d_inst", "d_res") if c in piv.columns] or list(piv.columns)
    piv = piv[tiers]
    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in))
    xs = range(len(tiers))
    for name in _ordered(piv.index.tolist()):
        ax.plot(list(xs), piv.loc[name].to_numpy(), label=_label(name), **_style(cfg, name))
    ax.set_xticks(list(xs))
    ax.set_xticklabels([TIER_LABELS.get(str(t), str(t)) for t in tiers])
    ax.set_xlabel("Provisioning tier")
    ax.set_ylabel("In-class SLO violation rate")
    ax.margins(x=0.12, y=0.12)
    _legend(ax, cfg)
    save(fig, out, "F7_lead_time_tiers")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    ap = argparse.ArgumentParser(
        description="Figures F4, F4b, F5, F6, F6b and F7 from the sweep CSVs")
    ap.add_argument("--sweeps", type=Path, default=PILOT / "outputs" / "sim")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-replicas", type=int, default=0,
                    help="the always-on base tier the sweep ran with (run_sweep.py "
                         "--min-replicas); above 0 the cost figures report above that tier and "
                         "--c-run is required")
    ap.add_argument("--c-run", type=float, default=None,
                    help="USD per replica-minute, used to price the base tier")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def floor_from(args: argparse.Namespace, parser: argparse.ArgumentParser) -> FloorSpec:
    """The base tier the figures report above; pricing it needs both numbers, not one."""
    if args.min_replicas < 0:
        parser.error("--min-replicas cannot be negative")
    if args.min_replicas > 0 and args.c_run is None:
        parser.error("--min-replicas > 0 needs --c-run (USD per replica-minute) to price the "
                     "base tier")
    return FloorSpec(min_replicas=args.min_replicas, c_run=args.c_run or 0.0)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    out = args.out or Path(args.sweeps) / "figures"
    floor = floor_from(args, parser)
    if floor.on:
        logger.info("reporting above a base tier of %d replica(s) at %g USD per replica-minute",
                    floor.min_replicas, floor.c_run)
    cfg = apply_style()
    sigma = _read(Path(args.sweeps) / "sigma" / "sigma.csv")
    lead = _read(Path(args.sweeps) / "lead" / "lead.csv")
    error = _read(Path(args.sweeps) / "epsilon" / "epsilon.csv")
    matched = _slo_matched(Path(args.sweeps))
    if sigma is not None:
        figure_f4(sigma, out, cfg, floor, matched)
        if sigma["sigma"].nunique() > 1:
            figure_f5(sigma, out, cfg, floor)
        else:
            logger.warning("F5 needs more than one sigma; skipping")
    if error is not None:
        figure_f6(error, out, cfg, floor)
        figure_f6b(error, out, cfg, floor)
    if lead is not None:
        figure_f7(lead, out, cfg)
    elif sigma is not None:
        logger.info("no lead sweep yet; F7 skipped")
    weekly = lead if lead is not None else sigma  # the lead sweep's replica tier, else sigma
    if weekly is not None:
        figure_f4b(weekly, out, cfg, floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
