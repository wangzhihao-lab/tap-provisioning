"""Fig. 2: TAP as the controller of the campus teaching-assistant service.

Run from the repository root::

    uv run python scripts/figs/make_fig2.py

Bottom row, the managed system: the campus, whose sections start at the bells of a published
timetable; the LLM service with its request log, two admission queues and replica pool; the GPU
provider, which delivers a replica d minutes after it is ordered. Top, the TAP controller: each
module's header is coloured by when it acts (before the semester, weekly, every minute). Indigo is
the control path, dashed rose the observations read from the request log, grey the environment.
Every label is measured before it is drawn and the script stops if one leaves its box.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

FIGS = Path(__file__).resolve().parent
if str(FIGS) not in sys.path:
    sys.path.insert(0, str(FIGS))

from svgkit import (AXIS, DEMAND, PALETTE, PT_PER_CM, SECONDARY, TEXT, Canvas, hatch, icon_bell,  # noqa: E402
                    icon_calendar, icon_histogram, icon_stopwatch, label_width, tint)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "figures"
STEM = "f2_tap_framework"

WIDTH_CM, HEIGHT_PT = 18.1, 214.0
FS, FS_MATH, FS_TITLE = 8.0, 10.0, 9.0   # text; symbols with subscripts (subscript = 8 pt); titles
PAD, HEAD = 5.0, 12.5                    # card padding; height of a card's header band
FLOW, BACK, ENV = PALETTE["indigo"], PALETTE["rose"], "#7A7A7A"
ENV_FILL = "#F1F1F1"
EDGE = "#A6A6A6"
WHEN = {"semester": PALETTE["lightblue"], "weekly": PALETTE["teal"], "minute": PALETTE["sand"]}
WHEN_FILL = {"semester": tint(PALETTE["lightblue"], 0.45), "weekly": tint(PALETTE["teal"], 0.38),
             "minute": tint(PALETTE["sand"], 0.55)}


@dataclass(frozen=True)
class Box:
    """A rectangle in pt, y downwards."""

    x: float
    y: float
    w: float
    h: float

    @property
    def r(self) -> float:
        return self.x + self.w

    @property
    def b(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


# controller (top) and managed system (bottom); the channel between them carries the log arrows
CONTROLLER = Box(2.0, 2.0, 509.0, 132.0)
TIMETABLE, PROJECTION = Box(10.0, 18.0, 90.0, 54.0), Box(114.0, 18.0, 100.0, 54.0)
PLAN, GUARD = Box(228.0, 18.0, 100.0, 54.0), Box(342.0, 18.0, 152.0, 110.0)
TRIGGER, CORRECTION = Box(114.0, 82.0, 100.0, 46.0), Box(228.0, 82.0, 100.0, 46.0)
ROB = Box(350.0, 92.0, 66.0, 28.0)
CAMPUS, SERVICE, PROVIDER = Box(2.0, 150.0, 148.0, 62.0), Box(164.0, 150.0, 240.0, 62.0), Box(434.0, 150.0, 77.0, 62.0)
LOG_H = 10.5
ROUTE_X = 502.0      # the committed count runs down the right margin to the provider
NODE = (440.0, 57.0)


# ---------------------------------------------------------------------- text that must fit
def fit(label: str, limit: float, size: float = FS, weight: str = "normal", where: str = "") -> float:
    w = label_width(label, size, weight)
    if w > limit + 1e-6:
        raise SystemExit(f"'{label}' is {w:.1f} pt wide, {limit:.1f} pt available ({where})")
    return w


def text_in(c: Canvas, box: Box, x: float, y: float, label: str, size: float = FS, anchor: str = "start",
            **kw) -> None:
    """Text that must stay inside ``box`` horizontally."""
    w = label_width(label, size, kw.get("weight", "normal"))
    x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
    if x0 < box.x + 1.5 or x0 + w > box.r - 1.5:
        raise SystemExit(f"'{label}' spans {x0:.1f}-{x0 + w:.1f}, box {box.x:.1f}-{box.r:.1f}")
    c.text(x, y, label, size=size, anchor=anchor, **kw)


def card(c: Canvas, box: Box, title: str, when: str, gid: str, note: str = "") -> None:
    """White card whose header band carries the title in the colour of when the module acts."""
    r = 3.0
    hx, hy, hw = box.x, box.y, box.w
    band = (f"M {hx:.2f} {hy + HEAD:.2f} L {hx:.2f} {hy + r:.2f} Q {hx:.2f} {hy:.2f} {hx + r:.2f} {hy:.2f} "
            f"L {hx + hw - r:.2f} {hy:.2f} Q {hx + hw:.2f} {hy:.2f} {hx + hw:.2f} {hy + r:.2f} "
            f"L {hx + hw:.2f} {hy + HEAD:.2f} Z")
    c.rect(box.x, box.y, box.w, box.h, rx=r, fill="white")
    c.path(band, stroke="none", fill=WHEN_FILL[when])
    c.rect(box.x, box.y, box.w, box.h, rx=r, stroke=WHEN[when], sw=0.8)
    tw = fit(title, box.w - 2 * PAD, weight="bold", where=gid)
    c.text(box.x + PAD, box.y + 8.9, title, size=FS, weight="bold")
    if note:
        fit(note, box.w - 2 * PAD - tw - 5.0, where=f"{gid} note")
        c.text(box.r - PAD, box.y + 8.9, note, size=FS, anchor="end", fill=SECONDARY)


def env_box(c: Canvas, box: Box, title: str, gid: str, title_y: float) -> None:
    c.rect(box.x, box.y, box.w, box.h, rx=3.0, fill=ENV_FILL)
    text_in(c, box, box.x + PAD, title_y, title, weight="bold")


def varrow(c: Canvas, x: float, y0: float, y1: float, color: str, dashed: bool = False, sw: float = 0.9) -> None:
    c.arrow([(x, y0), (x, y1)], stroke=color, sw=sw, head=3.6, head_w=3.0, dash=(3.0, 2.0) if dashed else None)


# ---------------------------------------------------------------------- glyphs
def _bump(xs: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return np.exp(-0.5 * ((xs - mu) / sig) ** 2)


def _pts(g: Box, xs: np.ndarray, ys: np.ndarray) -> List[Tuple[float, float]]:
    return [(g.x + x * g.w, g.b - y * g.h) for x, y in zip(xs, ys)]


def glyph_projection(c: Canvas, g: Box) -> None:
    xs = np.linspace(0.0, 1.0, 81)
    parts = [a * _bump(xs, m, 0.06) for m, a in zip((0.2, 0.33, 0.45, 0.58, 0.7), (0.7, 1.0, 0.85, 0.9, 0.55))]
    total = np.sum(parts, axis=0)
    scale = 0.9 / total.max()
    for p in parts:
        c.polygon(_pts(g, xs, p * scale) + [(g.r, g.b), (g.x, g.b)], fill=DEMAND, stroke=EDGE, sw=0.4)
    c.polyline(_pts(g, xs, total * scale), stroke=FLOW, sw=1.0)
    c.line(g.x, g.b, g.r, g.b, stroke=AXIS, sw=0.5)


def glyph_plan(c: Canvas, g: Box) -> None:
    xs = np.linspace(0.0, 1.0, 81)
    f = _bump(xs, 0.42, 0.13) + 0.45 * _bump(xs, 0.7, 0.09)
    f = f / f.max() * 0.72
    c.polygon(_pts(g, xs, f) + [(g.r, g.b), (g.x, g.b)], fill=tint(FLOW, 0.1))
    c.polyline(_pts(g, xs, f), stroke=FLOW, sw=0.5, dash=(1.6, 1.0))
    edges, q = np.linspace(0.0, 1.0, 11), 0.24
    steps: List[Tuple[float, float]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        level = max(q * np.ceil(f[(xs >= a) & (xs <= b)].max() / q), q)
        steps += [(g.x + a * g.w, g.b - level * g.h), (g.x + b * g.w, g.b - level * g.h)]
    c.polyline(steps, stroke=FLOW, sw=1.1, join="miter")
    c.line(g.x, g.b, g.r, g.b, stroke=AXIS, sw=0.5)


def glyph_correction(c: Canvas, g: Box) -> None:
    xs = np.linspace(0.0, 1.0, 81)
    c.polyline(_pts(g, xs, 0.95 * _bump(xs, 0.5, 0.13)), stroke=BACK, sw=0.9)
    c.polyline(_pts(g, xs, 0.52 * _bump(xs, 0.5, 0.13)), stroke=FLOW, sw=0.7, dash=(1.6, 1.0))
    xa = g.cx
    c.arrow([(xa, g.b - 0.52 * g.h - 0.6), (xa, g.b - 0.95 * g.h + 1.0)], stroke=TEXT, sw=0.6, head=2.4, head_w=2.2)
    c.line(g.x, g.b, g.r, g.b, stroke=AXIS, sw=0.5)


def icon_classroom(c: Canvas, x: float, y: float, w: float, h: float, color: str) -> None:
    c.rect(x, y, w, h, rx=1.5, fill="white", stroke=color, sw=0.7)
    c.rect(x + 0.3 * w, y + 0.12 * h, 0.4 * w, 0.14 * h, fill=color)
    for row in range(2):
        for k in range(4):
            c.circle(x + (0.2 + 0.2 * k) * w, y + (0.55 + 0.25 * row) * h, 0.055 * w, fill=tint(color, 0.7))


def replica(c: Canvas, x: float, y: float, s: float, warm: bool) -> None:
    w = 0.78 * s
    colour = PALETTE["green"]
    if warm:
        c.rect(x, y, w, s, rx=0.1 * s, fill=colour, stroke=colour, sw=0.7)
    else:
        c.rect(x, y, w, s, rx=0.1 * s, fill=hatch(c, tint(colour, 0.8)), stroke=colour, sw=0.7)
    slot = "white" if warm else colour
    for k in range(3):
        yy = y + (0.22 + 0.26 * k) * s
        c.line(x + 0.18 * w, yy, x + 0.62 * w, yy, stroke=slot, sw=0.7, cap="round")
        c.circle(x + 0.8 * w, yy, 0.035 * s, fill=slot)


# ---------------------------------------------------------------------- controller
def draw_controller(c: Canvas) -> None:
    p = CONTROLLER
    with c.group("controller_panel"):
        c.rect(p.x, p.y, p.w, p.h, rx=4.0, fill=tint(FLOW, 0.05), stroke=tint(FLOW, 0.35), sw=0.6)
        x = p.x + 8.0
        x += c.text(x, 13.2, "TAP", size=FS_TITLE, weight="bold", fill=FLOW) + 4.0
        c.text(x, 13.2, "timetable-aware provisioning", size=FS, fill=SECONDARY)
        xr = p.r - 8.0
        for when, label in (("minute", "every minute $t$"), ("weekly", "weekly"), ("semester", "before the semester")):
            xr -= label_width(label, FS)
            c.text(xr, 13.2, label, size=FS)
            xr -= 9.0
            c.rect(xr, 13.2 - 6.2, 6.6, 6.6, rx=1.0, fill=WHEN_FILL[when], stroke=WHEN[when], sw=0.7)
            xr -= 8.0
    with c.group("timetable"):
        card(c, TIMETABLE, "Timetable", "semester", "timetable")
        icon_calendar(c, TIMETABLE.x + PAD, TIMETABLE.y + 21.0, 16.0, SECONDARY)
        text_in(c, TIMETABLE, TIMETABLE.x + 27.0, TIMETABLE.y + 30.5, "sections, bells")
        text_in(c, TIMETABLE, TIMETABLE.x + 27.0, TIMETABLE.y + 40.0, "and rosters")
    with c.group("projection"):
        card(c, PROJECTION, "Demand projection", "semester", "projection")
        glyph_projection(c, Box(PROJECTION.x + PAD, PROJECTION.y + 16.5, PROJECTION.w - 2 * PAD, 24.0))
        text_in(c, PROJECTION, PROJECTION.r - PAD, PROJECTION.y + 26.0, "$\\hat{\\lambda}_t$", size=FS_MATH,
                anchor="end", fill=FLOW)
        text_in(c, PROJECTION, PROJECTION.x + PAD, PROJECTION.y + 50.5, "sum of session profiles")
    with c.group("plan"):
        card(c, PLAN, "Plan", "semester", "plan", note="LP, integer DP")
        glyph_plan(c, Box(PLAN.x + PAD, PLAN.y + 17.0, PLAN.w - 2 * PAD, 22.0))
        text_in(c, PLAN, PLAN.x + PAD, PLAN.y + 50.5, "$\\tilde{x} = \\arg \\min C(X;\\hat{\\lambda})$")
    with c.group("trigger"):
        card(c, TRIGGER, "Trigger times", "weekly", "trigger")
        icon_histogram(c, TRIGGER.x + PAD + 0.5, TRIGGER.y + 19.0, 14.0, SECONDARY, heights=(0.3, 0.75, 1.0, 0.55, 0.25))
        text_in(c, TRIGGER, TRIGGER.x + 25.0, TRIGGER.y + 25.5, "offset of the burst")
        text_in(c, TRIGGER, TRIGGER.x + 25.0, TRIGGER.y + 35.0, "prior: last semester")
    with c.group("correction"):
        card(c, CORRECTION, "Scale correction", "minute", "correction")
        glyph_correction(c, Box(CORRECTION.x + PAD, CORRECTION.y + 15.5, CORRECTION.w - 2 * PAD, 17.0))
        x = CORRECTION.x + PAD
        for piece, colour in (("realised", BACK), (" / ", TEXT), ("forecast", FLOW)):
            x += c.text(x, CORRECTION.y + 42.0, piece, size=FS, fill=colour)
    draw_guard(c)
    draw_controller_flow(c)


def draw_guard(c: Canvas) -> None:
    g = GUARD
    with c.group("guard"):
        card(c, g, "Robust guard", "minute", "guard", note="BDINTERP-$d$")
        y_plan = PLAN.cy
        c.polyline([(g.x, y_plan), (NODE[0] - 30.0, y_plan), (NODE[0] - 4.6, NODE[1] - 3.6)], stroke=FLOW, sw=1.0)
        c.rect(ROB.x, ROB.y, ROB.w, ROB.h, rx=2.0, fill="white", stroke=AXIS, sw=0.6)
        text_in(c, ROB, ROB.x + 4.0, ROB.y + 11.0, "ROB($d$)")
        text_in(c, ROB, ROB.x + 4.0, ROB.y + 21.5, "lazy rule", fill=SECONDARY)
        c.polyline([(ROB.r - 12.0, ROB.y), (ROB.r - 12.0, NODE[1] + 8.0), (NODE[0] - 4.6, NODE[1] + 3.6)],
                   stroke=AXIS, sw=1.0)
        c.circle(NODE[0], NODE[1], 6.0, fill="white", stroke=TEXT, sw=0.8)
        c.text(NODE[0], NODE[1] + 2.8, "$\\nu$", size=FS, anchor="middle")
        text_in(c, g, NODE[0] + 16.0, NODE[1] + 26.0, "follow the plan while", anchor="middle")
        text_in(c, g, NODE[0] + 16.0, NODE[1] + 39.5, "$C_{\\mathrm{ROB}} \\ge \\delta$$C_{\\mathrm{ADV}}$",
                size=FS_MATH, anchor="middle")
        text_in(c, g, (NODE[0] + 6.0 + g.r) / 2, NODE[1] - 4.5, "$x_{t+d}$", size=FS_MATH, anchor="middle",
                fill=FLOW)


def draw_controller_flow(c: Canvas) -> None:
    y = PLAN.cy
    with c.group("controller_flow"):
        c.arrow([(TIMETABLE.r + 0.8, y), (PROJECTION.x - 0.8, y)], stroke=FLOW, sw=1.0, head=4.0, head_w=3.4)
        c.arrow([(PROJECTION.r + 0.8, y), (PLAN.x - 0.8, y)], stroke=FLOW, sw=1.0, head=4.0, head_w=3.4)
        c.text((PROJECTION.r + PLAN.x) / 2, y - 4.5, "$\\hat{\\lambda}$", size=FS, anchor="middle", fill=FLOW)
        c.arrow([(PLAN.r + 0.8, y), (GUARD.x - 0.8, y)], stroke=FLOW, sw=1.0, head=4.0, head_w=3.4)
        c.text((PLAN.r + GUARD.x) / 2, y - 4.5, "$\\tilde{x}$", size=FS, anchor="middle", fill=FLOW)
        x = PROJECTION.cx
        varrow(c, x, TRIGGER.y - 0.6, PROJECTION.b + 0.6, FLOW)
        c.text(x + 4.0, (PROJECTION.b + TRIGGER.y) / 2 + 1.6, "$o_k$", size=FS_MATH, fill=FLOW)
        x = PLAN.cx
        varrow(c, x, CORRECTION.y - 0.6, PLAN.b + 0.6, FLOW)
        c.text(x + 4.0, (PLAN.b + CORRECTION.y) / 2 + 2.8, "re-solve", size=FS, fill=FLOW)
        c.polyline([(NODE[0] + 6.0, NODE[1]), (ROUTE_X, NODE[1]), (ROUTE_X, CONTROLLER.b)], stroke=FLOW, sw=1.2,
                   join="miter")
        c.arrow([(ROUTE_X, CONTROLLER.b), (ROUTE_X, PROVIDER.y - 0.6)], stroke=FLOW, sw=1.2, head=4.2, head_w=3.6)


# ---------------------------------------------------------------------- managed system
def draw_system(c: Canvas) -> None:
    with c.group("campus"):
        env_box(c, CAMPUS, "Campus", "campus", CAMPUS.y + 10.5)
        icon_bell(c, CAMPUS.x + PAD, CAMPUS.y + 16.5, 13.0, SECONDARY)
        for k in range(3):
            icon_classroom(c, CAMPUS.x + 26.0 + k * 31.0, CAMPUS.y + 15.5, 27.0, 19.0, SECONDARY)
        text_in(c, CAMPUS, CAMPUS.x + PAD, CAMPUS.y + 46.0, "prompts near each bell")
        text_in(c, CAMPUS, CAMPUS.x + PAD, CAMPUS.y + 55.5, "trigger bursts of questions")
    with c.group("service"):
        s = SERVICE
        c.rect(s.x, s.y, s.w, s.h, rx=3.0, fill=ENV_FILL)
        c.path(f"M {s.x:.2f} {s.y + LOG_H:.2f} L {s.x:.2f} {s.y + 3:.2f} Q {s.x:.2f} {s.y:.2f} {s.x + 3:.2f} {s.y:.2f} "
               f"L {s.r - 3:.2f} {s.y:.2f} Q {s.r:.2f} {s.y:.2f} {s.r:.2f} {s.y + 3:.2f} L {s.r:.2f} {s.y + LOG_H:.2f} Z",
               stroke="none", fill=tint(BACK, 0.22))
        text_in(c, s, s.x + PAD, s.y + 8.0, "request log", fill=BACK)
        text_in(c, s, s.x + PAD, s.b - 5.0, "LLM teaching-assistant service", weight="bold")
        sq, gap, x0 = 4.4, 1.4, s.x + PAD + 2.0
        for row, (filled, label) in enumerate(((True, "in-class first"), (False, "off-class waits"))):
            base = s.y + 24.0 + row * 10.0
            for k in range(4):
                c.rect(x0 + k * (sq + gap), base - 5.0, sq, sq, fill=ENV if filled else "white",
                       stroke="none" if filled else AXIS, sw=0.6)
            text_in(c, s, x0 + 4 * (sq + gap) + 3.0, base, label)
        c.arrow([(s.x + 92.0, s.y + 26.5), (s.x + 112.0, s.y + 26.5)], stroke=ENV, sw=0.9, head=3.4, head_w=3.0)
        xs = [s.x + 118.0 + k * 13.5 for k in range(4)]
        for k, x in enumerate(xs):
            replica(c, x, s.y + 19.5, 14.0, warm=k < 3)
        c.text((xs[0] + xs[2] + 10.9) / 2, s.y + 43.0, "warm", size=FS, anchor="middle", fill=SECONDARY)
        c.text(xs[3] + 5.5, s.y + 43.0, "starting", size=FS, anchor="middle", fill=SECONDARY)
        text_in(c, s, xs[0], s.y + 53.0, "$\\mu$ requests/min per replica", fill=SECONDARY)
    with c.group("provider"):
        pv = PROVIDER
        env_box(c, pv, "GPU provider", "provider", pv.y + 10.5)
        icon_stopwatch(c, pv.x + PAD, pv.y + 15.0, 10.0, SECONDARY)
        text_in(c, pv, pv.x + PAD + 13.0, pv.y + 23.0, "lead time $d$")
        text_in(c, pv, pv.x + PAD, pv.y + 35.0, "replica ≈ 1 min")
        text_in(c, pv, pv.x + PAD, pv.y + 45.0, "instance 8 min")
        text_in(c, pv, pv.x + PAD, pv.y + 55.0, "reserved: 1-h blocks")
    with c.group("system_flow"):
        y = SERVICE.y + 26.5
        c.arrow([(CAMPUS.r + 0.8, y), (SERVICE.x + 5.5, y)], stroke=ENV, sw=1.4, head=4.2, head_w=3.8)
        x_start = xs[3] + 0.78 * 14.0 + 1.2
        c.arrow([(PROVIDER.x - 0.6, SERVICE.y + 26.5), (x_start, SERVICE.y + 26.5)], stroke=FLOW, sw=1.0,
                head=3.8, head_w=3.2)
        c.text((x_start + SERVICE.r) / 2 + 4.0, SERVICE.y + 22.0, "warm at $t+d$", size=FS, anchor="middle", fill=FLOW)


def draw_log_links(c: Canvas) -> None:
    """Observations read from the request log, dashed rose; the published timetable, grey."""
    with c.group("log_links"):
        base = CONTROLLER.b + 11.0
        for x, top, label in ((TRIGGER.cx + 26.0, TRIGGER.b, "first arrivals, weekly"),
                              (CORRECTION.cx + 12.0, CORRECTION.b, "in-class counts"),
                              (ROB.x + 22.0, ROB.b, "observed demand")):
            varrow(c, x, SERVICE.y, top + 0.6, BACK, dashed=True)
            c.text(x + 4.0, base, label, size=FS, fill=BACK)
    with c.group("timetable_link"):
        x = TIMETABLE.x + 40.0
        varrow(c, x, CAMPUS.y, TIMETABLE.b + 0.6, ENV, sw=1.0)
        c.text(x + 4.0, TRIGGER.y + 18.0, "published", size=FS, fill=ENV)
        c.text(x + 4.0, TRIGGER.y + 27.5, "months ahead", size=FS, fill=ENV)


def main(out_dir: Path = OUT_DIR) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = Canvas(WIDTH_CM, round(HEIGHT_PT / PT_PER_CM, 2))
    draw_controller(c)
    draw_system(c)
    draw_log_links(c)
    svg, pdf, png = c.save(out_dir, STEM)
    logger.info("wrote %s, %s, %s (%.2f x %.2f cm)", svg, pdf, png, WIDTH_CM, HEIGHT_PT / PT_PER_CM)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR)
