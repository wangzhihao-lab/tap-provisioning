"""Fig. 1: the busiest bell of week 3 on one minute axis.

Run from the repository root::

    uv run python scripts/figs/make_fig1.py

Rows, top to bottom, share the mapping "minutes from the bell": a header with the timetable facts
of the bell; (A) campus in-class requests per minute with one recorded class as an inset and the
lead time d; (B) replicas held by TAP (``tap``) and by the reactive feedback rule (SLO-floor tuned,
selected by resource cost) against the demand in replica units; (C) one cell per minute coloured by
each policy's in-class TTFT p95. Every number printed in the figure is computed from the data files
and compared with section 26 of ``results/paper_numbers.md``; any difference stops the
script before a file is written.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from svgkit import (AXIS, DEMAND, PALETTE, PT_PER_CM, SECONDARY, TEXT, Axes, Canvas, icon_bell,
                    icon_calendar, icon_stopwatch, label_width, step_points, tint)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "figures"
STEM = "f1_bell_timeline"
F1_CSV = REPO / "results/runs/fig1/f1_right_bell.csv"
F1_JSON = REPO / "results/runs/fig1/f1_right_bell.json"
BELL_META = REPO / "scripts/q2/pool_inputs/bell_c3_w3.json"
BURSTS = REPO / "results/q1/assistant/bursts.csv"
ARRIVALS = REPO / "results/q1/assistant/arrivals_per_min.csv"
CANONICAL = REPO / "results/paper_numbers.md"

WINDOW = (-20, 24)          # minutes from the bell drawn, inclusive
INSET_WINDOW = (-10, 2)     # minutes of the recorded class drawn in the inset, inclusive
RECORDED = ("ds_prompted", "2026-03-24", "08:30")
SLO_S, UNSERVED_S = 2.0, 60.0
POLICIES = (("tap", "TAP"), ("reactive", "Reactive"))
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

FS = 8.0                    # every label is 8 pt
CAP = 0.662 * FS            # cap height of Times New Roman at 8 pt
DESC = 0.217 * FS
TICK = 2.0
EDGE = "#A6A6A6"            # outline of the demand area
CLASS_BAR = "#8C8C8C"       # recorded class in the inset: demand, so grey
SHADE = {"ok": tint(PALETTE["green"], 0.55), "slow": PALETTE["sand"], "lost": PALETTE["wine"]}
COLOUR = {"TAP": PALETTE["indigo"], "Reactive": PALETTE["rose"]}


@dataclass(frozen=True)
class Layout:
    """Page geometry in pt (y grows downwards)."""

    width_cm: float = 8.8
    left: float = 34.0
    right: float = 6.0
    xlim: Tuple[float, float] = (-20.0, 25.0)
    ylabel_x: float = 13.4          # baseline of the rotated y labels
    header_top: float = 1.2
    row_a_top: float = 14.0
    row_a_h: float = 64.0
    row_a_ylim: Tuple[float, float] = (0.0, 1250.0)
    gap_ab: float = 6.5
    row_b_h: float = 60.0
    row_b_ylim: Tuple[float, float] = (0.0, 20.0)
    gap_bc: float = 6.5
    strip_h: float = 6.5
    strip_text: float = 7.0         # strip bottom to the baseline of its totals
    strip_gap: float = 11.0         # strip bottom to the next strip top
    legend_gap: float = 9.6         # totals baseline to the legend baseline
    axis_gap: float = 7.0           # legend baseline to the x axis line
    bottom_pad: float = 2.0

    @property
    def width(self) -> float:
        return self.width_cm * PT_PER_CM

    @property
    def row_b_top(self) -> float:
        return self.row_a_top + self.row_a_h + self.gap_ab

    @property
    def row_c_top(self) -> float:
        return self.row_b_top + self.row_b_h + self.gap_bc

    def strip_top(self, k: int) -> float:
        return self.row_c_top + k * (self.strip_h + self.strip_gap)

    @property
    def legend_base(self) -> float:
        return self.strip_top(1) + self.strip_h + self.strip_text + self.legend_gap

    @property
    def axis_y(self) -> float:
        return self.legend_base + self.axis_gap

    @property
    def height(self) -> float:
        return self.axis_y + TICK + 1.2 + CAP + 9.8 + self.bottom_pad


@dataclass(frozen=True)
class Bell:
    """The 45 minutes around the bell and the recorded class, plus the printed facts."""

    minutes: np.ndarray
    demand: np.ndarray
    mu: float
    replicas: Dict[str, np.ndarray]
    p95: Dict[str, np.ndarray]
    class_minutes: np.ndarray
    class_counts: np.ndarray
    facts: Dict[str, str]
    printed: Dict[str, str]


# ---------------------------------------------------------------------- data and checks
def load_bell() -> Bell:
    """Read the replay, the bell metadata and the recorded class; derive every printed number."""
    df = pd.read_csv(F1_CSV)
    meta = json.loads(F1_JSON.read_text(encoding="utf-8"))
    bell_meta = json.loads(BELL_META.read_text(encoding="utf-8"))
    w = df[(df["minute_from_bell"] >= WINDOW[0]) & (df["minute_from_bell"] <= WINDOW[1])]
    minutes = w["minute_from_bell"].to_numpy(dtype=float)
    if len(minutes) != WINDOW[1] - WINDOW[0] + 1 or np.any(np.diff(minutes) != 1):
        raise SystemExit(f"{F1_CSV}: minutes {WINDOW} are not complete")
    mu = float(meta["mu_req_min"])
    demand = w["arrivals_in"].to_numpy(dtype=float)
    replicas = {n: w[f"replicas_{p}"].to_numpy(dtype=float) for p, n in POLICIES}
    p95 = {n: w[f"ttft_p95_{p}"].to_numpy(dtype=float) for p, n in POLICIES}

    arr = pd.read_csv(ARRIVALS)
    one = arr[(arr["dataset"] == RECORDED[0]) & (arr["date"] == RECORDED[1])
              & (arr["minute_from_bell"] >= INSET_WINDOW[0]) & (arr["minute_from_bell"] <= INSET_WINDOW[1])]
    bursts = pd.read_csv(BURSTS)
    row = bursts[(bursts["dataset"] == RECORDED[0]) & (bursts["date"] == RECORDED[1])
                 & (bursts["bell"] == RECORDED[2])].iloc[0]
    students = int(one["first_arrivals"].sum())
    if students != int(row["n_students"]):
        raise SystemExit(f"inset: {students} first arrivals in {INSET_WINDOW}, bursts.csv says {row['n_students']}")

    week, rem = divmod(int(meta["bell_min"]), 7 * 1440)
    day, clock = divmod(rem, 1440)
    when = f"week {week + 1}, {DAYS[day]} {clock // 60:02d}:{clock % 60:02d}"
    first_over = int(minutes[demand > mu].min())
    facts = {
        "bell": when,
        "sections starting at the bell": f"{int(bell_meta['n_sessions_at_bell'])}",
        "students (roster seats) at the bell": f"{int(meta['seats_at_bell']):,}",
        "peak campus in-class arrivals per minute": f"{int(demand.max())}",
        "first minute the demand exceeds one replica": f"{first_over}",
        "lead time drawn (instance tier, min)": f"{float(meta['lead_min']):.0f}",
        "recorded class: students, 10-90% ramp (min)": f"{students}, {float(row['ramp_10_90_min']):.1f}",
    }
    printed = {"header": f"{when[0].upper()}{when[1:]}: {facts['sections starting at the bell']} sections, "
                         f"{facts['students (roster seats) at the bell']} students",
               "class_1": f"one class: {students} students,",
               "class_2": f"10–90% in {float(row['ramp_10_90_min']):.1f} min",
               "lead": f"lead time $d$ = {facts['lead time drawn (instance tier, min)']} min"}
    for _, name in POLICIES:
        t, x = p95[name], replicas[name]
        late, lost = int((t > SLO_S).sum()), int((t > UNSERVED_S).sum())
        facts[f"{name}: minutes with p95 > 2 s / not served, replica-minutes, peak"] = \
            f"{late} / {lost}, {x.sum():.0f}, {x.max():.0f}"
        printed[f"totals_{name}"] = f"{late} min late · {x.sum():.0f} replica-min"
    r = replicas["Reactive"]
    at_peak = minutes[r == r.max()]
    facts["Reactive at its peak: minutes, first, last"] = f"{len(at_peak)}, {int(at_peak.min())}, {int(at_peak.max())}"
    return Bell(minutes, demand, mu, replicas, p95, one["minute_from_bell"].to_numpy(dtype=float),
                one["first_arrivals"].to_numpy(dtype=float), facts, printed)


def check_canonical(facts: Dict[str, str]) -> None:
    """Stop unless the computed facts equal the table of section 26 of the canonical file."""
    text = CANONICAL.read_text(encoding="utf-8")
    sec = re.search(r"^## 26\.[^\n]*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    if sec is None:
        raise SystemExit(f"{CANONICAL}: section 26 not found")
    table: Dict[str, str] = {}
    for line in sec.group(1).splitlines():
        cells = [s.strip() for s in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] != "fact" and set(cells[0]) != {"-"}:
            table[cells[0]] = cells[1]
    diff = [f"  {k}: computed {facts.get(k)!r}, canonical {table.get(k)!r}"
            for k in sorted(set(facts) | set(table)) if facts.get(k) != table.get(k)]
    if diff:
        raise SystemExit("Fig. 1 numbers differ from canonical section 26:\n" + "\n".join(diff))
    for k, v in facts.items():
        logger.info("asserted  %-70s %s", k, v)


def span(minutes: np.ndarray, mask: np.ndarray, contiguous: bool = True) -> Tuple[int, int]:
    """First and last minute of ``mask``; with ``contiguous``, stops unless they form one run."""
    sel = minutes[mask]
    if len(sel) == 0 or (contiguous and sel.max() - sel.min() + 1 != len(sel)):
        raise SystemExit(f"expected one contiguous run of minutes, got {sel.tolist()}")
    return int(sel.min()), int(sel.max())


# ---------------------------------------------------------------------- primitives
def vtext(c: Canvas, x: float, y: float, label: str, gid: str) -> None:
    """``label`` reading bottom to top, centred on ``y``, baseline on the vertical line ``x``."""
    with c.group(gid, transform=f"rotate(-90 {x:.2f} {y:.2f})"):
        c.text(x, y, label, size=FS, anchor="middle")


def step_area(ax: Axes, xs: Sequence[float], ys: Sequence[float]) -> List[Tuple[float, float]]:
    pts = step_points(ax, xs, ys)
    return [(pts[0][0], ax.Y(0))] + pts + [(pts[-1][0], ax.Y(0))]


def between(ax: Axes, xs: Sequence[float], top: Sequence[float], bottom: Sequence[float]) -> List[Tuple[float, float]]:
    """Polygon between two per-minute step curves over ``xs``."""
    upper = step_points(ax, xs, top)
    lower = step_points(ax, xs, bottom)
    return upper + lower[::-1]


def left_axis(c: Canvas, ax: Axes, ticks: Sequence[float], label: str, gid: str, lay: Layout) -> None:
    """Left and bottom spines, outward y ticks with labels, rotated y label."""
    with c.group(gid):
        c.polyline([(ax.x0, ax.y0), (ax.x0, ax.y0 + ax.h), (ax.x0 + ax.w, ax.y0 + ax.h)],
                   stroke=AXIS, sw=0.6, join="miter")
        for t in ticks:
            y = ax.Y(t)
            c.line(ax.x0 - TICK, y, ax.x0, y, stroke=AXIS, sw=0.6)
            c.text(ax.x0 - TICK - 1.5, y + CAP / 2, f"{t:g}", size=FS, anchor="end")
        vtext(c, lay.ylabel_x, ax.y0 + ax.h / 2, label, gid=f"{gid}_label")


def demand_area(c: Canvas, ax: Axes, xs: np.ndarray, ys: np.ndarray, gid: str) -> None:
    with c.group(gid):
        c.polygon(step_area(ax, xs, ys), fill=DEMAND)
        c.polyline(step_points(ax, xs, ys), stroke=EDGE, sw=0.5, join="miter")


# ---------------------------------------------------------------------- rows
def draw_header(c: Canvas, lay: Layout, bell: Bell) -> None:
    s = 9.0
    with c.group("header"):
        icon_calendar(c, lay.left, lay.header_top, s, PALETTE["indigo"])
        c.text(lay.left + s + 3.0, lay.header_top + 0.58 * s + CAP / 2, bell.printed["header"], size=FS)


def draw_row_a(c: Canvas, ax: Axes, lay: Layout, bell: Bell) -> None:
    xs = bell.minutes
    left_axis(c, ax, (0, 400, 800), "Requests/min", "row_a_axes", lay)
    demand_area(c, ax, xs, bell.demand, "row_a_demand")
    draw_lead_time(c, ax, bell)
    draw_inset(c, ax, bell)


def draw_lead_time(c: Canvas, ax: Axes, bell: Bell) -> None:
    """Double arrow from the first minute above one replica to d minutes later, with a leader."""
    t0 = float(bell.facts["first minute the demand exceeds one replica"])
    d = float(bell.facts["lead time drawn (instance tier, min)"])
    x0, x1 = ax.X(t0), ax.X(t0 + d)
    inside = (bell.minutes >= t0) & (bell.minutes < t0 + d)
    beyond = (bell.minutes >= t0) & (bell.minutes < t0 + d + 4)     # under the label's tail
    rise, s = 6.2, 7.5                             # label baseline to arrow; stopwatch size
    base = min(ax.Y(bell.demand[beyond].max()) - 2.5, ax.Y(bell.demand[inside].max()) - 3.4 - rise)
    y_arrow = base + rise
    with c.group("row_a_lead_time"):
        c.line(x0, y_arrow, x0, ax.Y(bell.mu), stroke=TEXT, sw=0.5, dash=(0.5, 1.1), cap="round")
        c.circle(x0, ax.Y(bell.mu), 0.9, fill=TEXT)
        c.arrow([(x0, y_arrow), (x1, y_arrow)], stroke=TEXT, sw=0.8, head=3.0, head_w=2.6, start_head=True)
        icon_stopwatch(c, x0 - 1.2 - s, y_arrow - 0.58 * s, s, TEXT)
        c.text(x0 - 1.2 - s, base, bell.printed["lead"], size=FS)


def draw_inset(c: Canvas, ax: Axes, bell: Bell) -> None:
    """Framed mini-axis in the empty upper right: one recorded class, first arrivals per minute."""
    pad, lead, bar_h = 3.0, 1.18 * FS, 11.5
    w_note = max(label_width(bell.printed["class_1"], FS), label_width(bell.printed["class_2"], FS))
    scale = ax.w / (ax.xlim[1] - ax.xlim[0])          # same pt per minute as the main axis
    lo, hi = INSET_WINDOW[0] - 0.6, INSET_WINDOW[1] + 1.6
    frame_w = max(w_note + 2 * pad, (hi - lo) * scale + 2 * pad)
    fx1 = ax.x0 + ax.w - 1.0
    fx0, fy0 = fx1 - frame_w, ax.y0 + 1.0
    base1 = fy0 + pad + CAP
    bar_top = base1 + lead + DESC + 2.0
    axis_y = bar_top + bar_h
    frame_h = axis_y + TICK + 1.0 + CAP + pad - fy0
    mini = Axes(fx0 + (frame_w - (hi - lo) * scale) / 2, bar_top, (hi - lo) * scale, bar_h,
                (lo, hi), (0.0, float(bell.class_counts.max())))
    with c.group("row_a_inset"):
        c.rect(fx0, fy0, frame_w, frame_h, rx=1.5, fill="white", stroke=AXIS, sw=0.5)
        c.text(fx0 + pad, base1, bell.printed["class_1"], size=FS)
        c.text(fx0 + pad, base1 + lead, bell.printed["class_2"], size=FS)
        for m, n in zip(bell.class_minutes, bell.class_counts):
            if n > 0:
                c.rect(mini.X(m + 0.1), mini.Y(n), mini.X(m + 0.9) - mini.X(m + 0.1), mini.Y(0) - mini.Y(n),
                       fill=CLASS_BAR)
        c.line(mini.X(INSET_WINDOW[0]), axis_y, mini.X(INSET_WINDOW[1] + 1), axis_y, stroke=AXIS, sw=0.5)
        for t in (-8, -4, 0):
            c.line(mini.X(t), axis_y, mini.X(t), axis_y + TICK, stroke=AXIS, sw=0.5)
            c.text(mini.X(t), axis_y + TICK + 1.0 + CAP, f"{t:g}".replace("-", "−"), size=FS, anchor="middle")


def draw_row_b(c: Canvas, ax: Axes, lay: Layout, bell: Bell) -> None:
    xs = bell.minutes
    need = bell.demand / bell.mu
    rea, tap = bell.replicas["Reactive"], bell.replicas["TAP"]
    left_axis(c, ax, (0, 5, 10, 15), "Replicas", "row_b_axes", lay)
    demand_area(c, ax, xs, need, "row_b_demand")
    i0, i1 = span(xs, rea == rea.max(), contiguous=False)   # +2..+12, three minutes at 16 inside
    idle = (xs >= i0) & (xs <= i1)
    with c.group("row_b_idle"):
        c.polygon(between(ax, xs[idle], rea[idle], need[idle]), fill=PALETTE["rose"], opacity=0.15)
        c.text(ax.X((i0 + i1 + 1) / 2), ax.Y(4.6), "idle", size=FS, anchor="middle", fill=PALETTE["rose"])
    with c.group("row_b_tap"):
        c.polyline(step_points(ax, xs, tap), stroke=COLOUR["TAP"], sw=1.2, join="miter")
    with c.group("row_b_reactive"):
        c.polyline(step_points(ax, xs, rea), stroke=COLOUR["Reactive"], sw=1.2, dash=(3, 1.6), join="miter")
    l0, l1 = span(xs, (bell.p95["Reactive"] > SLO_S) | (bell.p95["TAP"] > SLO_S))   # both late over -15..-8
    y_late = ax.Y(7.4)
    with c.group("row_b_late"):   # neutral: the span is shared, the strips below tell the two apart
        xa, xb = ax.X(l0), ax.X(l1 + 1)
        c.polyline([(xa, y_late + 1.8), (xa, y_late), (xb, y_late), (xb, y_late + 1.8)],
                   stroke=TEXT, sw=0.7, join="miter")
        c.text((xa + xb) / 2, y_late - 1.8, "late", size=FS, anchor="middle", fill=TEXT)
    steps = xs[np.flatnonzero(np.diff(rea)) + 1]              # minutes at which Reactive changes
    after = steps[steps > i1 + 1]
    next_step = after.min() if after.size else xs[-1] + 1     # Reactive's first change after idle (+18)
    with c.group("row_b_labels"):   # Reactive above its peak plateau; TAP under Reactive's next plateau
        c.text(ax.X(i1 + 1) - 1.5, ax.Y(rea.max()) - 1.9, "Reactive", size=FS, anchor="end",
               fill=COLOUR["Reactive"])
        c.text((ax.X(i1 + 1) + ax.X(next_step)) / 2, ax.Y(tap[-1]) - 1.9, "TAP", size=FS, anchor="middle",
               fill=COLOUR["TAP"])


def draw_row_c(c: Canvas, ax: Axes, lay: Layout, bell: Bell) -> None:
    xs = bell.minutes
    for k, (_, name) in enumerate(POLICIES):
        top = lay.strip_top(k)
        with c.group(f"row_c_{name.lower()}"):
            for m, t in zip(xs, bell.p95[name]):
                key = "ok" if t <= SLO_S else ("slow" if t <= UNSERVED_S else "lost")
                c.rect(ax.X(m) + 0.18, top, ax.X(m + 1) - ax.X(m) - 0.36, lay.strip_h, fill=SHADE[key])
            c.text(ax.x0 - 3.0, top + lay.strip_h / 2 + CAP / 2, name, size=FS, anchor="end", fill=COLOUR[name])
            c.text(ax.x0 + ax.w, top + lay.strip_h + lay.strip_text, bell.printed[f"totals_{name}"], size=FS,
                   anchor="end", fill=SECONDARY)
    sq, x = 5.6, ax.x0
    with c.group("row_c_legend"):
        x += c.text(x, lay.legend_base, "In-class TTFT p95", size=FS) + 6.0
        for key, label in (("ok", "≤ 2 s"), ("slow", "2–60 s"), ("lost", "> 60 s")):
            c.rect(x, lay.legend_base - CAP + 0.1, sq, sq, fill=SHADE[key])
            x += sq + 2.0
            x += c.text(x, lay.legend_base, label, size=FS) + 7.0


def draw_x_axis(c: Canvas, ax: Axes, lay: Layout) -> None:
    y = lay.axis_y
    with c.group("x_axis"):
        c.line(ax.x0, y, ax.x0 + ax.w, y, stroke=AXIS, sw=0.6)
        for t in (-20, -10, 0, 10, 20):
            c.line(ax.X(t), y, ax.X(t), y + TICK, stroke=AXIS, sw=0.6)
            c.text(ax.X(t), y + TICK + 1.2 + CAP, f"{t:g}".replace("-", "−"), size=FS, anchor="middle")
        c.text(ax.x0 + ax.w / 2, y + TICK + 1.2 + CAP + 9.8, "Minutes from the bell", size=FS, anchor="middle")


def draw_bell(c: Canvas, ax: Axes, lay: Layout) -> None:
    s = 8.5
    x = ax.X(0)
    with c.group("bell"):
        icon_bell(c, x - s / 2, lay.row_a_top + 0.6, s, SECONDARY)
    with c.group("bell_line"):
        c.line(x, lay.row_a_top + 0.6 + s + 1.2, x, lay.strip_top(1) + lay.strip_h, stroke=AXIS, sw=0.6,
               dash=(2.0, 1.5))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bell = load_bell()
    check_canonical(bell.facts)
    lay = Layout()
    height_cm = round(lay.height / PT_PER_CM, 2)
    if not 7.4 <= height_cm <= 7.8:
        raise SystemExit(f"height {height_cm} cm outside 7.4-7.8 cm")
    c = Canvas(lay.width_cm, height_cm)
    w = lay.width - lay.left - lay.right
    ax_a = Axes(lay.left, lay.row_a_top, w, lay.row_a_h, lay.xlim, lay.row_a_ylim)
    ax_b = Axes(lay.left, lay.row_b_top, w, lay.row_b_h, lay.xlim, lay.row_b_ylim)
    draw_header(c, lay, bell)
    draw_row_a(c, ax_a, lay, bell)
    draw_row_b(c, ax_b, lay, bell)
    draw_bell(c, ax_a, lay)
    draw_row_c(c, ax_a, lay, bell)
    draw_x_axis(c, ax_a, lay)
    svg, pdf, png = c.save(OUT_DIR, STEM)
    logger.info("wrote %s, %s, %s (%.2f x %.2f cm)", svg, pdf, png, lay.width_cm, height_cm)


if __name__ == "__main__":
    main()
