"""Mechanical text-vs-data audit: every number of paper/main.tex against the canonical tables.

    cd pilot && uv run python scripts/audit/audit_numbers.py   # -> results/audit_numbers.md

The manuscript may quote a number only if that number exists in results/paper_numbers.md.
This script lists every numeric token of the manuscript that has no counterpart there, with the
sentence or table row it sits in, so that a human can check it.  Only the document body is read
(the preamble carries float, spacing and counter constants that are not claims); equations,
algorithms, the notation table and pure markup (labels, refs, cites, graphics, column specs) are
skipped; inline math is kept but tagged separately.  A manuscript number counts as matched when
it equals a canonical value or one of its declared roundings.  Exits 1 while anything is unmatched.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger("audit_numbers")

Span = Tuple[int, int]

DROP_ENVS = ("equation", "equation*", "align", "align*", "algorithm", "algorithmic", "tikzpicture")
ONE_ARG = ("label", "ref", "eqref", "cite", "graphicspath")
TWO_ARG = ("setlength", "figtodo", "multicolumn")
NUMBER = re.compile(r"\d{1,3}(?:(?:,|\{,\})\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
CLOCK = re.compile(r"\d{1,2}:\d{2}")
CLOCK_LITERALS: Set[str] = set()  # HH:MM tokens present in the canonical file (bell times)
YEAR = re.compile(r"20\d{2}(?:--\d{2})?")
COMMENT = re.compile(r"(?<!\\)%[^\n]*")
SECTION = re.compile(r"\\(?:sub)*section\*?\{([^{}]*)\}")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(])")
ROW_BREAK = re.compile(r"\\\\|\\midrule|\\toprule|\\bottomrule|\\cmidrule")
# A single digit right after one of these words is a structural marker, not a quantity.
MARKER_BEFORE = re.compile(r"(?:(?:^|[^A-Za-z])c|\b(?:weeks?|cells?|cases?|seeds?))[\s~]*$", re.I)
# A number glued to one of these letter runs names a model or a context size (32B, 4k, 95th).
NAME_SUFFIX = frozenset({"b", "k", "d", "th", "st", "nd", "rd"})
MAX_CONTEXT = 200
MAX_UNUSED_LINES = 60


@dataclass(frozen=True)
class Found:
    """One numeric token of the manuscript, with where it came from."""

    text: str
    value: Optional[float]
    tag: str
    section: str
    context: str
    offset: int


def _brace_end(text: str, i: int) -> int:
    """Index just past the balanced group that starts at ``text[i] == '{'``."""
    depth = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _skip_optional(text: str, i: int, opener: str = "[", closer: str = "]") -> int:
    """Index just past an optional ``[...]`` (or ``(...)``) argument at ``i``, if present."""
    while i < len(text) and text[i] in " \t\n":
        i += 1
    if i < len(text) and text[i] == opener:
        closed = text.find(closer, i)
        return i if closed < 0 else closed + 1
    return i


def _blank(buf: List[str], start: int, end: int) -> None:
    """Overwrite ``buf[start:end]`` with spaces, keeping newlines so offsets stay valid."""
    for i in range(max(start, 0), min(end, len(buf))):
        if buf[i] != "\n":
            buf[i] = " "


def env_spans(text: str, name: str) -> List[Span]:
    """Spans of every ``\\begin{name} ... \\end{name}`` block (these do not nest here)."""
    starts = [m.start() for m in re.finditer(r"\\begin\{" + re.escape(name) + r"\}", text)]
    ends = [m.end() for m in re.finditer(r"\\end\{" + re.escape(name) + r"\}", text)]
    spans: List[Span] = []
    for start in starts:
        after = [end for end in ends if end > start]
        if after:
            spans.append((start, after[0]))
    return spans


def _blank_commands(buf: List[str], text: str) -> None:
    """Blank the markup commands whose arguments carry no prose number."""
    for name in ONE_ARG:
        for match in re.finditer(r"\\" + name + r"\b", text):
            i = _skip_optional(text, match.end())
            _blank(buf, match.start(), _brace_end(text, i) if i < len(text) and text[i] == "{" else i)
    for name in TWO_ARG:
        for match in re.finditer(r"\\" + name + r"\b", text):
            i = _brace_end(text, _skip_optional(text, match.end()))
            _blank(buf, match.start(), _brace_end(text, i))
    for match in re.finditer(r"\\includegraphics\b", text):
        i = _skip_optional(text, match.end())
        _blank(buf, match.start(), _brace_end(text, i))
    for match in re.finditer(r"\\cmidrule\b", text):
        i = _skip_optional(text, match.end(), "(", ")")
        _blank(buf, match.start(), _brace_end(text, i))
    for match in re.finditer(r"\\subfloat\b", text):
        _blank(buf, match.start(), _skip_optional(text, match.end()))
    for match in re.finditer(r"\\begin\{tabular\}", text):
        _blank(buf, match.start(), _brace_end(text, match.end()))
    for match in re.finditer(r"\\tabcolsep\b", text):
        _blank(buf, match.start(), match.end())


def strip_latex(text: str) -> Tuple[str, List[Span], List[Tuple[int, str]]]:
    """Blank everything that is not manuscript prose or table text.

    Returns the blanked copy (same length as ``text``, so offsets still point into the original),
    the spans of the table environments, and the ``(offset, title)`` of every sectioning command.
    """
    buf = list(text)
    body_start = text.find(r"\begin{document}")
    body_end = text.find(r"\end{document}")
    if body_start < 0 or body_end < 0:
        raise ValueError("main.tex has no \\begin{document} / \\end{document} pair")
    _blank(buf, 0, body_start)
    _blank(buf, body_end, len(text))
    for match in COMMENT.finditer(text):
        _blank(buf, match.start(), match.end())
    for name in DROP_ENVS:
        for span in env_spans(text, name):
            _blank(buf, *span)
    tables: List[Span] = env_spans(text, "table") + env_spans(text, "table*")
    for span in tables:
        if "tab:notation" in text[span[0]:span[1]]:
            _blank(buf, *span)
    sections: List[Tuple[int, str]] = []
    for match in SECTION.finditer(text):
        sections.append((match.start(), match.group(1)))
        _blank(buf, match.start(), match.end())
    _blank_commands(buf, text)
    masked = "".join(buf)
    for match in YEAR.finditer(masked):  # years and the --27 / --26 fragment after them
        _blank(buf, match.start(), match.end())
    logger.info("stripped %d table environments, %d sectioning commands", len(tables), len(sections))
    return "".join(buf), tables, sections


def math_spans(masked: str) -> List[Span]:
    """Spans of inline ``$...$`` math left in the blanked text."""
    marks = [m.start() for m in re.finditer(r"(?<!\\)\$", masked)]
    return [(marks[i], marks[i + 1] + 1) for i in range(0, len(marks) - 1, 2)]


def _inside(spans: Sequence[Span], offset: int) -> bool:
    return any(start <= offset < end for start, end in spans)


def _context(text: str, offset: int, in_table: bool) -> str:
    """The table row or the sentence around ``offset``, whitespace-collapsed and truncated."""
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    line_end = len(text) if line_end < 0 else line_end
    unit = text[line_start:line_end]
    position = offset - line_start
    if in_table:
        breaks = [(m.start(), m.end()) for m in ROW_BREAK.finditer(unit)]
        start = max((e for _, e in breaks if e <= position), default=0)
        end = min((s for s, _ in breaks if s > position), default=len(unit))
        unit, position = unit[start:end], position - start
    else:
        cursor = 0
        for piece in SENTENCE.split(unit):
            found = unit.find(piece, cursor)
            cursor = found + len(piece)
            if found <= position < cursor:
                unit, position = piece, position - found
                break
    head = " ".join(unit[:position].split())
    tail = " ".join(unit[position:].split())
    separator = " " if position and unit[position - 1].isspace() and head else ""
    lead = MAX_CONTEXT // 3  # window the long units so that the flagged number stays visible
    if len(head) + len(separator) > lead:
        head = "\u2026" + head[len(head) - lead:]
    collapsed = head + separator + tail
    return collapsed if len(collapsed) <= MAX_CONTEXT else collapsed[: MAX_CONTEXT - 1] + "\u2026"


def _skip_number(text: str, start: int, end: int, token: str) -> bool:
    """True when the token is a symbol index, a structural marker or part of a name."""
    before = text[start - 1] if start else " "
    if before.isalpha() or before in "_^":
        return True
    if before == "{" and start >= 2 and text[start - 2] in "_^":
        return True
    if len(token) == 1 and MARKER_BEFORE.search(text[max(0, start - 12):start]):
        return True
    suffix = re.match(r"[A-Za-z]+", text[end:])
    return bool(suffix) and suffix.group(0).lower() in NAME_SUFFIX


def find_numbers(text: str, masked: str, tables: Sequence[Span],
                 sections: Sequence[Tuple[int, str]]) -> List[Found]:
    """Every numeric token of the manuscript body, tagged prose / table / math."""
    buf = list(masked)
    hits: List[Tuple[int, int, str]] = []
    for match in CLOCK.finditer(masked):  # a clock time is one token, never a quantity
        hits.append((match.start(), match.end(), match.group(0)))
        _blank(buf, match.start(), match.end())
    masked = "".join(buf)
    maths = math_spans(masked)
    for match in NUMBER.finditer(masked):
        if not _skip_number(text, match.start(), match.end(), match.group(0)):
            hits.append((match.start(), match.end(), match.group(0)))
    found: List[Found] = []
    for start, end, token in sorted(hits):
        in_table = _inside(tables, start)
        tag = "table" if in_table else ("math" if _inside(maths, start) else "prose")
        title = "(front matter)"
        for offset, name in sections:
            if offset < start:
                title = name
        written = token + "%" if text[end:end + 2] == r"\%" else token
        value = None if ":" in token else float(token.replace("{,}", "").replace(",", ""))
        found.append(Found(written, value, tag, title, _context(text, start, in_table), start))
    return found


def _roundings(literal: str) -> Set[float]:
    """A canonical value plus the roundings the manuscript is allowed to quote it with."""
    try:
        value = Decimal(literal)
    except InvalidOperation:
        return set()
    places = max(0, -value.as_tuple().exponent)
    out = {float(value)}
    for target in {places - 1, 2, 3} & set(range(places)):
        out.add(float(value.quantize(Decimal(1).scaleb(-target), rounding=ROUND_HALF_UP)))
    return out


def _percent_forms(literal: str) -> Set[float]:
    """The percent form of a ratio, which only a token written with a ``%`` may quote.

    Restricting it to such tokens matters: without it any ratio, say an SLO of 0.33, would
    silently validate an unrelated "33\\,s" elsewhere in the text.
    """
    value = Decimal(literal)
    return {float(value * 100)} if 0 <= value <= 1 else set()


def read_canonical(path: str) -> Tuple[Set[float], Set[float], Dict[int, Set[str]]]:
    """Every numeric token of the canonical file: plain and percent forms, and literals per section."""
    CLOCK_LITERALS.update(CLOCK.findall(Path(path).read_text(encoding="utf-8")) if isinstance(path, (str, Path)) else set())
    text = Path(path).read_text(encoding="utf-8")
    plain: Set[float] = set()
    percent: Set[float] = set()
    per_section: Dict[int, Set[str]] = {}
    current = 0
    for line in text.splitlines():
        heading = re.match(r"##\s*(\d+)\.", line)
        if heading:
            current = int(heading.group(1))
            continue
        if line.startswith("#") or not line.strip():
            continue
        cells = line.split("|") if line.lstrip().startswith("|") else [line]
        for cell in cells:
            if set(cell.strip()) <= {"-", ":"}:
                continue
            for match in NUMBER.finditer(cell):
                literal = match.group(0).replace("{,}", "").replace(",", "")
                plain.update(_roundings(literal))
                percent.update(_percent_forms(literal))
                per_section.setdefault(current, set()).add(literal)
    logger.info("canonical file: %d accepted values over %d sections", len(plain), len(per_section))
    return plain, percent, per_section


def _matches(item: Found, plain: Set[float], percent: Set[float]) -> bool:
    if CLOCK.fullmatch(item.text):
        return item.text in CLOCK_LITERALS
    if item.value is None:
        return False
    allowed = plain | percent if item.text.endswith("%") else plain
    return any(abs(item.value - ok) <= 1e-9 for ok in allowed)


def unused_canonical(per_section: Dict[int, Set[str]], quoted: Sequence[float]) -> List[str]:
    """Literals of canonical sections 1-2 that no manuscript number reproduces."""
    lines: List[str] = []
    for section in (1, 2):
        for literal in sorted(per_section.get(section, set()), key=float):
            allowed = _roundings(literal) | _percent_forms(literal)
            if not any(abs(q - ok) <= 1e-9 for q in quoted for ok in allowed):
                lines.append(f"- {literal} (canonical section {section})")
    return lines


def render(found: Sequence[Found], unmatched: Sequence[Found], unused: Sequence[str]) -> str:
    """The audit report, identical on stdout and in the output file."""
    tags = {tag: sum(1 for f in found if f.tag == tag) for tag in ("prose", "table", "math")}
    lines = [
        "# Text-vs-data audit of paper/main.tex",
        "",
        f"Numbers found in the manuscript: {len(found)} "
        f"(prose {tags['prose']}, table {tags['table']}, math {tags['math']})",
        f"Matched in results/paper_numbers.md: {len(found) - len(unmatched)}",
        f"Unmatched: {len(unmatched)}",
        "",
        "## Unmatched numbers",
        "",
    ]
    if not unmatched:
        lines.append("None: every number in the manuscript prose has a canonical counterpart.")
    for item in unmatched:
        lines.append(f"- `{item.text}` [{item.tag}] {item.section} -- {item.context}")
    lines += ["", "## Canonical values of sections 1-2 never quoted in the manuscript", ""]
    if len(unused) > MAX_UNUSED_LINES:
        lines.append(f"{len(unused)} unquoted values; block skipped (over {MAX_UNUSED_LINES} lines).")
    elif not unused:
        lines.append("None: every value of canonical sections 1-2 appears in the manuscript.")
    else:
        lines += list(unused)
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit manuscript numbers against the canonical file.")
    parser.add_argument("--tex", required=True, help="the manuscript source (not part of this repository)")
    parser.add_argument("--canonical", default=str(REPO / "results" / "paper_numbers.md"))
    parser.add_argument("--out", default=str(REPO / "results" / "audit_numbers.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        text = Path(args.tex).read_text(encoding="utf-8")
        plain, percent, per_section = read_canonical(args.canonical)
    except (FileNotFoundError, OSError):
        logger.error("cannot read %s or %s", args.tex, args.canonical)
        raise
    masked, tables, sections = strip_latex(text)
    found = find_numbers(text, masked, tables, sections)
    unmatched = [f for f in found if not _matches(f, plain, percent)]
    quoted = [f.value for f in found if f.value is not None]
    report = render(found, unmatched, unused_canonical(per_section, quoted))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    logger.info("%d of %d manuscript numbers unmatched; wrote %s", len(unmatched), len(found), out)
    print(report)
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
