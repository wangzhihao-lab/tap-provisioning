"""Minimal SVG toolkit for the paper's schematic figures.

Units are points (1 pt = 1/72 in): the <svg> element carries its size in cm and a viewBox in pt,
so ``size=8`` is 8 pt on paper. Text stays <text> (editable in Illustrator/Inkscape), every group
can carry an id, and ``Canvas.save`` writes the SVG and converts it with ``rsvg-convert`` to a
vector PDF (fonts embedded) for LaTeX and a 300-dpi PNG for checking.

Math in labels: put it between ``$...$``. Inside, letters are italic, digits and punctuation
upright, ``_{...}`` or ``_x`` is a subscript, ``\\hat{x}`` / ``\\tilde{x}`` add an accent, and
``\\lambda \\mu \\nu \\delta ...`` are Greek, ``\\mathrm{..}`` is upright and
``\\ge \\le \\cdot \\times \\to`` are symbols. The house floor is 8 pt for
every glyph, subscripts included (``SUB_SCALE`` of a 10-pt symbol line is 8 pt).
"""

from __future__ import annotations

import math
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

from PIL import ImageFont

PT_PER_CM = 72.0 / 2.54
FONT = "Times New Roman, Times, serif"
_FONT_DIR = "/System/Library/Fonts/Supplemental"
_FONT_FILES = {("normal", "normal"): "Times New Roman.ttf", ("bold", "normal"): "Times New Roman Bold.ttf",
               ("normal", "italic"): "Times New Roman Italic.ttf",
               ("bold", "italic"): "Times New Roman Bold Italic.ttf"}
MIN_PT = 8.0
SUB_SCALE = 0.8
SUB_SHIFT = 0.25  # of the base size

PALETTE = {"indigo": "#332288", "green": "#117733", "teal": "#44AA99", "lightblue": "#88CCEE",
           "sand": "#DDCC77", "rose": "#CC6677", "purple": "#AA4499", "wine": "#882255"}
TEXT = "#1A1A1A"
SECONDARY = "#555555"
AXIS = "#888888"
DEMAND = "#D9D9D9"

GREEK = {"lambda": "λ", "mu": "μ", "sigma": "σ", "varepsilon": "ε", "epsilon": "ε", "rho": "ρ",
         "tau": "τ", "eta": "η", "delta": "δ", "beta": "β", "gamma": "γ", "nu": "ν", "kappa": "κ"}
SYMBOLS = {"ge": "≥", "le": "≤", "cdot": "·", "times": "×", "to": "→", "approx": "≈", "infty": "∞"}
HAT, TILDE = "̂", "̃"
_UPRIGHT = set("0123456789()[]+-=,./:;·×−→≈ ")


def tint(color: str, amount: float) -> str:
    """``color`` blended with white: ``amount`` 0 is white, 1 is the colour itself."""
    c = color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(255 - amount * (255 - v)) for v in rgb]
    return "#" + "".join(f"{v:02X}" for v in mixed)


_FONT_CACHE: dict = {}


def _font(weight: str, style: str) -> ImageFont.FreeTypeFont:
    key = (weight, style)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(f"{_FONT_DIR}/{_FONT_FILES[key]}", 100)
    return _FONT_CACHE[key]


def plain_width(s: str, size: float, weight: str = "normal", style: str = "normal") -> float:
    """Advance width of ``s`` in pt (combining accents add nothing)."""
    s = s.replace(HAT, "").replace(TILDE, "")
    return _font(weight, style).getlength(s) * size / 100.0


@dataclass(frozen=True)
class Run:
    text: str
    italic: bool
    sub: bool


def parse(label: str) -> List[Run]:
    """``label`` with ``$...$`` math into runs of (text, italic, subscript)."""
    runs: List[Run] = []
    parts = re.split(r"(\$[^$]*\$)", label)
    for part in parts:
        if not part:
            continue
        if not (part.startswith("$") and part.endswith("$")):
            runs.append(Run(part, False, False))
            continue
        runs.extend(_parse_math(part[1:-1]))
    return _merge(runs)


def _close(m: str, j: int) -> int:
    """Index of the brace that closes the one at ``m[j]``."""
    depth = 0
    for k in range(j, len(m)):
        if m[k] == "{":
            depth += 1
        elif m[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"unbalanced braces in {m!r}")


def _parse_math(m: str) -> List[Run]:
    out: List[Run] = []
    i = 0
    sub = False
    depth_close = False
    while i < len(m):
        ch = m[i]
        if ch == "\\":
            j = i + 1
            while j < len(m) and m[j].isalpha():
                j += 1
            name = m[i + 1:j]
            if name in ("hat", "tilde") and j < len(m) and m[j] == "{":
                k = _close(m, j)
                inner = _parse_math(m[j + 1:k])
                acc = HAT if name == "hat" else TILDE
                if inner:
                    last = inner[-1]
                    inner[-1] = Run(last.text + acc, last.italic, sub or last.sub)
                out.extend(Run(r.text, r.italic, sub or r.sub) for r in inner)
                i = k + 1
                continue
            if name == "mathrm" and j < len(m) and m[j] == "{":
                k = _close(m, j)
                out.append(Run(m[j + 1:k], False, sub))
                i = k + 1
                continue
            if name in SYMBOLS:
                out.append(Run(SYMBOLS[name], False, sub))
                i = j
                continue
            if name in GREEK:
                out.append(Run(GREEK[name], True, sub))
                i = j
                continue
            out.append(Run(name, False, sub))
            i = j
            continue
        if ch == "_":
            if i + 1 < len(m) and m[i + 1] == "{":
                k = _close(m, i + 1)
                for r in _parse_math(m[i + 2:k]):
                    out.append(Run(r.text, r.italic, True))
                i = k + 1
            else:
                out.append(Run(m[i + 1], m[i + 1] not in _UPRIGHT, True))
                i += 2
            continue
        out.append(Run(ch, ch not in _UPRIGHT and not ch.isspace(), sub))
        i += 1
    del depth_close
    return out


def _merge(runs: List[Run]) -> List[Run]:
    merged: List[Run] = []
    for r in runs:
        if merged and merged[-1].italic == r.italic and merged[-1].sub == r.sub:
            merged[-1] = Run(merged[-1].text + r.text, r.italic, r.sub)
        else:
            merged.append(r)
    return merged


def label_width(label: str, size: float, weight: str = "normal") -> float:
    """Advance width in pt of a label with math markup."""
    total = 0.0
    for r in parse(label):
        total += plain_width(r.text, size * (SUB_SCALE if r.sub else 1.0), weight,
                             "italic" if r.italic else "normal")
    return total


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class TextItem:
    """One line of text: pieces are (text, italic, subscript, size, x offset from x0)."""
    x0: float
    y: float
    pieces: Tuple[Tuple[str, bool, bool, float, float], ...]
    weight: str
    fill: str
    gid: Optional[str]

    def editable(self) -> str:
        """One <text> with a tspan per piece and ordinary spaces (for editing the SVG)."""
        spans, state_sub = [], False
        base = max((p[3] for p in self.pieces if not p[2]), default=self.pieces[0][3])
        shift = SUB_SHIFT * base
        for text, italic, sub, size, _ in self.pieces:
            attrs = ['font-style="italic"'] if italic else []
            if sub != state_sub:
                attrs.append(f'dy="{_fmt(shift if sub else -shift)}"')
            state_sub = sub
            attrs.append(f'font-size="{_fmt(size)}"')
            spans.append(f'<tspan {" ".join(attrs)}>{escape(text)}</tspan>')
        idattr = f' id="{self.gid}"' if self.gid else ""
        return (f'<text{idattr} x="{_fmt(self.x0)}" y="{_fmt(self.y)}" font-family="{FONT}" '
                f'font-weight="{self.weight}" fill="{self.fill}" xml:space="preserve">{"".join(spans)}</text>')

    def pdf_safe(self) -> str:
        """Every word placed on its own: cairo emits a Type 3 font for any space glyph."""
        spans = []
        base = max((p[3] for p in self.pieces if not p[2]), default=self.pieces[0][3])
        shift = SUB_SHIFT * base
        for text, italic, sub, size, dx in self.pieces:
            style = "italic" if italic else "normal"
            y = self.y + (shift if sub else 0.0)
            offset = 0.0
            for word in text.split(" "):
                if word:
                    st = ' font-style="italic"' if italic else ""
                    spans.append(f'<tspan x="{_fmt(self.x0 + dx + offset)}" y="{_fmt(y)}" font-size="{_fmt(size)}"{st}>'
                                 f'{escape(word)}</tspan>')
                offset += plain_width(word + " ", size, self.weight, style)
        idattr = f' id="{self.gid}"' if self.gid else ""
        return (f'<text{idattr} font-family="{FONT}" font-weight="{self.weight}" fill="{self.fill}">'
                f'{"".join(spans)}</text>')


@dataclass
class Canvas:
    width_cm: float
    height_cm: float
    body: List = field(default_factory=list)
    defs: List[str] = field(default_factory=list)

    @property
    def w(self) -> float:
        return self.width_cm * PT_PER_CM

    @property
    def h(self) -> float:
        return self.height_cm * PT_PER_CM

    # ------------------------------------------------------------------ structure
    @contextmanager
    def group(self, gid: str, **attrs: str) -> Iterator[None]:
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.body.append(f'<g id="{gid}"{extra}>')
        try:
            yield
        finally:
            self.body.append("</g>")

    def raw(self, s: str) -> None:
        self.body.append(s)

    # ------------------------------------------------------------------ shapes
    @staticmethod
    def _stroke(stroke: str, sw: float, dash: Optional[Sequence[float]], cap: str = "butt",
                join: str = "miter") -> str:
        if stroke == "none":
            return ' stroke="none"'
        s = f' stroke="{stroke}" stroke-width="{_fmt(sw)}" stroke-linecap="{cap}" stroke-linejoin="{join}"'
        if dash:
            s += f' stroke-dasharray="{",".join(_fmt(d) for d in dash)}"'
        return s

    def rect(self, x: float, y: float, w: float, h: float, rx: float = 0.0, fill: str = "none",
             stroke: str = "none", sw: float = 0.6, dash: Optional[Sequence[float]] = None,
             opacity: Optional[float] = None, gid: Optional[str] = None) -> None:
        op = f' fill-opacity="{_fmt(opacity)}"' if opacity is not None else ""
        idattr = f' id="{gid}"' if gid else ""
        self.body.append(f'<rect{idattr} x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" '
                         f'rx="{_fmt(rx)}" fill="{fill}"{op}{self._stroke(stroke, sw, dash)}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = TEXT, sw: float = 0.6,
             dash: Optional[Sequence[float]] = None, cap: str = "butt") -> None:
        self.body.append(f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}"'
                         f'{self._stroke(stroke, sw, dash, cap)}/>')

    def polyline(self, pts: Sequence[Tuple[float, float]], stroke: str = TEXT, sw: float = 0.6,
                 fill: str = "none", dash: Optional[Sequence[float]] = None, join: str = "round",
                 opacity: Optional[float] = None) -> None:
        p = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)
        op = f' fill-opacity="{_fmt(opacity)}"' if opacity is not None else ""
        self.body.append(f'<polyline points="{p}" fill="{fill}"{op}{self._stroke(stroke, sw, dash, "butt", join)}/>')

    def polygon(self, pts: Sequence[Tuple[float, float]], fill: str, stroke: str = "none", sw: float = 0.6,
                opacity: Optional[float] = None) -> None:
        p = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)
        op = f' fill-opacity="{_fmt(opacity)}"' if opacity is not None else ""
        self.body.append(f'<polygon points="{p}" fill="{fill}"{op}{self._stroke(stroke, sw, None)}/>')

    def path(self, d: str, stroke: str = TEXT, sw: float = 0.6, fill: str = "none",
             dash: Optional[Sequence[float]] = None, cap: str = "round", join: str = "round") -> None:
        self.body.append(f'<path d="{d}" fill="{fill}"{self._stroke(stroke, sw, dash, cap, join)}/>')

    def circle(self, cx: float, cy: float, r: float, fill: str = "none", stroke: str = "none",
               sw: float = 0.6) -> None:
        self.body.append(f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}" fill="{fill}"'
                         f'{self._stroke(stroke, sw, None)}/>')

    def arrow(self, pts: Sequence[Tuple[float, float]], stroke: str = TEXT, sw: float = 1.0,
              head: float = 4.2, head_w: float = 3.4, dash: Optional[Sequence[float]] = None,
              start_head: bool = False) -> None:
        """Polyline with a filled triangular head at the end (and optionally at the start)."""
        pts = list(pts)

        def trimmed(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[float, float], List]:
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = math.hypot(dx, dy) or 1.0
            ux, uy = dx / n, dy / n
            base = (b[0] - ux * head, b[1] - uy * head)
            px, py = -uy, ux
            tri = [b, (base[0] + px * head_w / 2, base[1] + py * head_w / 2),
                   (base[0] - px * head_w / 2, base[1] - py * head_w / 2)]
            return base, tri

        end_base, end_tri = trimmed(pts[-2], pts[-1])
        body = pts[:-1] + [end_base]
        tris = [end_tri]
        if start_head:
            st_base, st_tri = trimmed(pts[1], pts[0])
            body = [st_base] + body[1:]
            tris.append(st_tri)
        self.polyline(body, stroke=stroke, sw=sw, dash=dash, join="miter")
        for tri in tris:
            self.polygon(tri, fill=stroke)

    # ------------------------------------------------------------------ text
    def text(self, x: float, y: float, label: str, size: float = 8.0, weight: str = "normal",
             anchor: str = "start", fill: str = TEXT, italic: bool = False,
             gid: Optional[str] = None) -> float:
        """One line at baseline ``y``; math between ``$...$``; returns the advance width.

        Accents (hat, tilde) are drawn as separate spacing glyphs (U+02C6, U+02DC) centred over
        their letter: combining marks sit badly in Times and make cairo emit Type 3 fonts.
        """
        runs = parse(label)
        if italic:
            runs = [Run(r.text, True, r.sub) for r in runs]
        width = sum(plain_width(r.text, size * (SUB_SCALE if r.sub else 1.0), weight,
                                "italic" if r.italic else "normal") for r in runs)
        x0 = x - width / 2 if anchor == "middle" else (x - width if anchor == "end" else x)
        pieces: List[Tuple[str, bool, bool, float, float]] = []
        accents: List[Tuple[float, float, str, float, bool]] = []
        shift = SUB_SHIFT * size
        cursor = x0
        for r in runs:
            sz = size * (SUB_SCALE if r.sub else 1.0)
            style = "italic" if r.italic else "normal"
            stripped = ""
            for ch in r.text:
                if ch in (HAT, TILDE) and stripped:
                    base = stripped[-1]
                    left = cursor + plain_width(stripped[:-1], sz, weight, style)
                    wch = plain_width(base, sz, weight, style)
                    skew = 0.07 * sz if r.italic else 0.0
                    tall = base in "λbdfhklt" or base.isupper() or base.isdigit()
                    yb = y + (shift if r.sub else 0.0) - (0.22 * sz if tall else 0.0)
                    accents.append((left + wch / 2 + skew, yb, "\u02c6" if ch == HAT else "\u02dc", sz, r.italic))
                else:
                    stripped += ch
            pieces.append((stripped, r.italic, r.sub, sz, cursor - x0))
            cursor += plain_width(stripped, sz, weight, style)
        self.body.append(TextItem(x0, y, tuple(pieces), weight, fill, gid))
        for cx, yb, glyph, sz, it in accents:
            self.body.append(TextItem(cx - plain_width(glyph, sz) / 2, yb, ((glyph, False, False, sz, 0.0),),
                                      "normal", fill, None))
        return width

    def lines(self, x: float, y: float, rows: Sequence[str], size: float = 8.0, leading: float = 1.18,
              **kw) -> float:
        """Several lines from baseline ``y``; returns the baseline of the last line."""
        for i, row in enumerate(rows):
            self.text(x, y + i * size * leading, row, size=size, **kw)
        return y + (len(rows) - 1) * size * leading

    # ------------------------------------------------------------------ output
    def svg(self, pdf_safe: bool = False) -> str:
        defs = f"<defs>{''.join(self.defs)}</defs>" if self.defs else ""
        parts = [(b.pdf_safe() if pdf_safe else b.editable()) if isinstance(b, TextItem) else b for b in self.body]
        return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(self.width_cm)}cm" '
                f'height="{_fmt(self.height_cm)}cm" viewBox="0 0 {_fmt(self.w)} {_fmt(self.h)}">\n'
                f'{defs}\n' + "\n".join(parts) + "\n</svg>\n")

    def save(self, out_dir: Path, stem: str) -> Tuple[Path, Path, Path]:
        """``stem.svg`` (editable), ``stem.pdf`` (vector, TrueType only) and ``stem.png`` (300 dpi)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        svg = out_dir / f"{stem}.svg"
        svg.write_text(self.svg(), encoding="utf-8")
        check_min_font(svg.read_text(encoding="utf-8"))
        safe = out_dir / f".{stem}.pdfsafe.svg"
        safe.write_text(self.svg(pdf_safe=True), encoding="utf-8")
        pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf), str(safe)], check=True)
        subprocess.run(["rsvg-convert", "-f", "png", "-d", "300", "-p", "300", "-b", "white",
                        "-o", str(png), str(safe)], check=True)
        safe.unlink()
        fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
        if "Type 3" in fonts:
            raise RuntimeError(f"{pdf} contains Type 3 fonts:\n{fonts}")
        return svg, pdf, png


def check_min_font(svg_text: str, floor: float = MIN_PT) -> float:
    """Smallest font-size in the SVG; raises if any glyph is below the house floor."""
    sizes = [float(v) for v in re.findall(r'font-size="([0-9.]+)"', svg_text)]
    smallest = min(sizes) if sizes else floor
    if smallest < floor - 1e-6:
        raise ValueError(f"font-size {smallest} pt below the {floor} pt floor")
    return smallest


# ---------------------------------------------------------------------- axes for data rows
@dataclass(frozen=True)
class Axes:
    """A data window mapped onto a rectangle in pt (y grows downwards on the page)."""
    x0: float
    y0: float
    w: float
    h: float
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]

    def X(self, v: float) -> float:
        return self.x0 + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, v: float) -> float:
        return self.y0 + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h


def step_points(ax: Axes, xs: Sequence[float], ys: Sequence[float], width: float = 1.0) -> List[Tuple[float, float]]:
    """A per-minute step curve: value ``ys[i]`` held over ``[xs[i], xs[i] + width)``."""
    pts: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        pts.append((ax.X(x), ax.Y(y)))
        pts.append((ax.X(x + width), ax.Y(y)))
    return pts


# ---------------------------------------------------------------------- icons (line icons in a size box)
def icon_calendar(c: Canvas, x: float, y: float, s: float, color: str) -> None:
    c.rect(x, y + 0.16 * s, s, 0.84 * s, rx=0.12 * s, fill="white", stroke=color, sw=0.8)
    c.rect(x, y + 0.16 * s, s, 0.24 * s, rx=0.12 * s, fill=color, stroke=color, sw=0.8)
    for fx in (0.28, 0.72):
        c.line(x + fx * s, y, x + fx * s, y + 0.26 * s, stroke=color, sw=1.0, cap="round")
    for r in range(3):
        for k in range(3):
            c.rect(x + (0.14 + 0.26 * k) * s, y + (0.48 + 0.16 * r) * s, 0.18 * s, 0.1 * s, fill=color)


def icon_bell(c: Canvas, x: float, y: float, s: float, color: str) -> None:
    cx = x + s / 2
    d = (f"M {_fmt(cx - 0.36 * s)} {_fmt(y + 0.78 * s)} "
         f"C {_fmt(cx - 0.3 * s)} {_fmt(y + 0.6 * s)}, {_fmt(cx - 0.34 * s)} {_fmt(y + 0.14 * s)}, {_fmt(cx)} {_fmt(y + 0.12 * s)} "
         f"C {_fmt(cx + 0.34 * s)} {_fmt(y + 0.14 * s)}, {_fmt(cx + 0.3 * s)} {_fmt(y + 0.6 * s)}, {_fmt(cx + 0.36 * s)} {_fmt(y + 0.78 * s)} Z")
    c.path(d, stroke=color, sw=0.8, fill="white")
    c.line(cx - 0.44 * s, y + 0.78 * s, cx + 0.44 * s, y + 0.78 * s, stroke=color, sw=0.9, cap="round")
    c.circle(cx, y + 0.9 * s, 0.08 * s, fill=color)
    c.circle(cx, y + 0.07 * s, 0.05 * s, fill=color)


def icon_stopwatch(c: Canvas, x: float, y: float, s: float, color: str) -> None:
    cx, cy, r = x + 0.5 * s, y + 0.58 * s, 0.38 * s
    c.circle(cx, cy, r, fill="white", stroke=color, sw=0.8)
    c.rect(cx - 0.08 * s, y + 0.02 * s, 0.16 * s, 0.1 * s, fill=color)
    c.line(cx, y + 0.12 * s, cx, cy - r, stroke=color, sw=0.8)
    c.line(cx, cy, cx, cy - 0.26 * s, stroke=color, sw=0.9, cap="round")
    c.line(cx, cy, cx + 0.18 * s, cy + 0.08 * s, stroke=color, sw=0.9, cap="round")
    c.line(cx + 0.28 * s, y + 0.2 * s, cx + 0.36 * s, y + 0.12 * s, stroke=color, sw=0.8, cap="round")


def icon_histogram(c: Canvas, x: float, y: float, s: float, color: str,
                   heights: Sequence[float] = (0.35, 0.9, 0.6, 0.25)) -> None:
    n = len(heights)
    bw = s / (n * 1.25)
    for i, hgt in enumerate(heights):
        c.rect(x + i * bw * 1.25, y + s * (1 - hgt), bw, s * hgt, fill=color)
    c.line(x - 0.05 * s, y + s, x + s, y + s, stroke=color, sw=0.6)


def icon_gpu(c: Canvas, x: float, y: float, s: float, color: str) -> None:
    b, e = 0.2 * s, 0.6 * s
    c.rect(x + b, y + b, e, e, rx=0.06 * s, fill="white", stroke=color, sw=0.8)
    c.rect(x + b + 0.14 * s, y + b + 0.14 * s, e - 0.28 * s, e - 0.28 * s, fill=tint(color, 0.35), stroke=color, sw=0.6)
    for k in range(4):
        t = b + 0.09 * s + k * (e - 0.18 * s) / 3
        c.line(x + t, y + 0.04 * s, x + t, y + b, stroke=color, sw=0.7)
        c.line(x + t, y + b + e, x + t, y + 0.96 * s, stroke=color, sw=0.7)
        c.line(x + 0.04 * s, y + t, x + b, y + t, stroke=color, sw=0.7)
        c.line(x + b + e, y + t, x + 0.96 * s, y + t, stroke=color, sw=0.7)


def hatch(c: Canvas, color: str, gap: float = 1.6, sw: float = 0.5) -> str:
    """A diagonal-hatch fill pattern registered in the canvas defs; returns ``url(#id)``."""
    pid = f"hatch{len(c.defs)}"
    c.defs.append(f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="{_fmt(gap)}" height="{_fmt(gap)}" '
                  f'patternTransform="rotate(45)"><rect width="{_fmt(gap)}" height="{_fmt(gap)}" fill="white"/>'
                  f'<line x1="0" y1="0" x2="0" y2="{_fmt(gap)}" stroke="{color}" stroke-width="{_fmt(sw)}"/></pattern>')
    return f"url(#{pid})"


def icon_replica(c: Canvas, x: float, y: float, s: float, color: str, state: str = "warm") -> None:
    """A server box; ``state`` warm (filled), starting (hatched) or off (dashed outline)."""
    w, h = 0.78 * s, s
    x0 = x + (s - w) / 2
    if state == "warm":
        c.rect(x0, y, w, h, rx=0.1 * s, fill=color, stroke=color, sw=0.7)
        slot = "white"
    elif state == "starting":
        c.rect(x0, y, w, h, rx=0.1 * s, fill=hatch(c, tint(color, 0.75)), stroke=color, sw=0.7)
        slot = color
    else:
        c.rect(x0, y, w, h, rx=0.1 * s, fill="white", stroke=color, sw=0.7, dash=(1.6, 1.2))
        slot = tint(color, 0.6)
    for k in range(3):
        yy = y + (0.22 + 0.26 * k) * h
        c.line(x0 + 0.18 * w, yy, x0 + 0.62 * w, yy, stroke=slot, sw=0.7, cap="round")
        c.circle(x0 + 0.8 * w, yy, 0.035 * s, fill=slot)


def icon_person(c: Canvas, x: float, y: float, s: float, color: str) -> None:
    cx = x + s / 2
    c.circle(cx, y + 0.28 * s, 0.2 * s, fill=color)
    d = (f"M {_fmt(x + 0.12 * s)} {_fmt(y + s)} C {_fmt(x + 0.12 * s)} {_fmt(y + 0.55 * s)}, "
         f"{_fmt(x + 0.88 * s)} {_fmt(y + 0.55 * s)}, {_fmt(x + 0.88 * s)} {_fmt(y + s)} Z")
    c.path(d, stroke="none", fill=color)


def icon_students(c: Canvas, x: float, y: float, s: float, color: str) -> None:
    icon_person(c, x, y + 0.12 * s, 0.62 * s, tint(color, 0.55))
    icon_person(c, x + 0.38 * s, y + 0.12 * s, 0.62 * s, tint(color, 0.55))
    icon_person(c, x + 0.19 * s, y, 0.66 * s, color)


def badge(c: Canvas, cx: float, cy: float, r: float, label: str, color: str, size: float = 8.0) -> None:
    """A filled circle with a white number (stage marker)."""
    c.circle(cx, cy, r, fill=color)
    c.text(cx, cy + 0.34 * size, label, size=size, weight="bold", anchor="middle", fill="white")


__all__ = ["AXIS", "Axes", "Canvas", "TextItem", "DEMAND", "FONT", "MIN_PT", "PALETTE", "PT_PER_CM", "SECONDARY",
           "TEXT", "badge", "check_min_font", "hatch", "icon_bell", "icon_calendar", "icon_gpu", "icon_histogram",
           "icon_person", "icon_replica", "icon_stopwatch", "icon_students", "label_width", "parse",
           "plain_width", "step_points", "tint"]
