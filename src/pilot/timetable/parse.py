"""Parse the institution-wide timetable export into section, slot and week frames.

The export has one row per course section with comma-separated, position-aligned lists of
meeting times and rooms.  Three frames come out of this module:

``load_timetable``  one row per section (raw list strings kept, personal columns dropped),
``explode_slots``   one row per section x meeting slot (weekday + periods + weeks + room),
``expand_weeks``    one row per section x slot x teaching week.

Privacy: the teacher name (``授课教师``) and the staff id (``教工号``) are personal data.  They are
dropped as soon as the file is read and never reach any frame, file, log or figure produced here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# --- source columns -------------------------------------------------------------------------
HEAD_COLUMNS: Tuple[str, ...] = (
    "校区", "上课学院", "上课班级", "课程编号", "课程名称", "上课周次", "上课时间",
)
ROOM_COLUMN = "上课地点"
TAIL_COLUMNS: Tuple[str, ...] = ("授课教师", "教工号", "课程性质", "学分", "上课人数")
PERSONAL_COLUMNS: Tuple[str, ...] = ("授课教师", "教工号")
RENAME: Dict[str, str] = {
    "校区": "campus", "上课学院": "college", "上课班级": "class_label", "课程编号": "course_code",
    "课程名称": "course_name", "上课周次": "weeks_raw", "上课时间": "time_raw",
    "上课地点": "room_raw", "课程性质": "course_type", "学分": "credits", "上课人数": "enrollment",
}
SECTION_COLUMNS: Tuple[str, ...] = (
    "section_id", "campus", "college", "class_label", "n_classes", "course_code", "course_name",
    "weeks_raw", "time_raw", "room_raw", "course_type", "credits", "enrollment",
    "general_elective", "ambiguous",
)

_FIELD_SEP = re.compile(r"\t+|\s{2,}")
_BRACKET = re.compile(r"\[(\d+)\s*-\s*(\d+)\]")
_WEEK_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_LINBAN = re.compile(r"^临班\d*")


@dataclass(frozen=True)
class TimetableConfig:
    """Semester geometry of the timetable export."""

    weeks: int = 18
    periods: int = 12
    # Fall semester 2026-2027-1: the summer break ended 2026-08-21, so week 1 is Aug 24-30.
    # Freshman sections that start in week 5 ("5-18") begin on Sept 21, after military training.
    week1_monday: str = "2026-08-24"
    typical_week_lo: int = 5
    typical_week_hi: int = 16


@dataclass(frozen=True)
class BellSchedule:
    """Period -> (start, end) wall-clock strings, plus the six teaching blocks.

    Defaults are the bell times of the studied university, checked for periods 1-11.  Period 12 is not on the published
    sheet; its window (21:05-21:50) is a placeholder.
    """

    windows: Tuple[Tuple[str, str], ...] = (
        ("08:30", "09:15"), ("09:15", "10:00"),      # block 1, bell 08:30
        ("10:20", "11:05"), ("11:05", "11:50"),      # block 2, bell 10:20
        ("14:00", "14:45"), ("14:45", "15:30"),      # block 3, bell 14:00
        ("15:50", "16:35"), ("16:35", "17:20"),      # block 4, bell 15:50
        ("18:40", "19:25"), ("19:25", "20:10"),      # block 5, bell 18:40
        ("20:20", "21:05"), ("21:05", "21:50"),      # block 6, bell 20:20; period 12 placeholder
    )
    block_size: int = 2

    def window(self, period: int) -> Tuple[str, str]:
        """Wall-clock window of ``period`` (1-based)."""
        if not 1 <= period <= len(self.windows):
            raise ValueError(f"period out of range: {period}")
        return self.windows[period - 1]

    def start(self, period: int) -> str:
        """Start time of ``period``."""
        return self.window(period)[0]

    def block(self, period: int) -> int:
        """1-based index of the teaching block that contains ``period``."""
        return (period - 1) // self.block_size + 1

    def block_start(self, period: int) -> str:
        """Start time of the block that contains ``period`` (the bell students hear)."""
        return self.start((self.block(period) - 1) * self.block_size + 1)

    def bells(self) -> Tuple[str, ...]:
        """The block start times, in order (six with the default schedule)."""
        n_blocks = len(self.windows) // self.block_size
        return tuple(self.start(i * self.block_size + 1) for i in range(n_blocks))


# --- small parsers --------------------------------------------------------------------------
def parse_time_code(code: str, periods: int = 12) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """``"3010203"`` -> ``(3, (1, 2, 3))``: first digit weekday 1-7, then two-digit periods."""
    code = code.strip()
    if not code.isdigit() or len(code) < 3 or len(code) % 2 == 0:
        logger.warning("unparsable 上课时间 code %r", code)
        return None
    weekday = int(code[0])
    slots = tuple(int(code[i:i + 2]) for i in range(1, len(code), 2))
    if not 1 <= weekday <= 7 or any(not 1 <= p <= periods for p in slots):
        logger.warning("out-of-range 上课时间 code %r", code)
        return None
    return weekday, slots


def parse_week_part(part: str) -> Tuple[int, ...]:
    """``"4-18"`` -> ``(4, ..., 18)``; ``"10"`` -> ``(10,)``; unparsable -> ``()``."""
    part = part.strip()
    m = _WEEK_RANGE.match(part)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return tuple(range(lo, hi + 1)) if lo <= hi else (lo,)
    if part.isdigit():
        return (int(part),)
    if part:
        logger.warning("unparsable 上课周次 part %r", part)
    return ()


def group_week_parts(parts: Sequence[str]) -> List[List[str]]:
    """Group a concatenated week list monotonically.

    A new group starts whenever a part's first week is <= the previous part's last week, which is
    where the export has wrapped around to the next slot's own week list.
    """
    groups: List[List[str]] = []
    prev_last: Optional[int] = None
    for part in parts:
        weeks = parse_week_part(part)
        if not weeks:
            continue
        if prev_last is None or weeks[0] > prev_last:
            if not groups:
                groups.append([part])
            else:
                groups[-1].append(part)
        else:
            groups.append([part])
        prev_last = weeks[-1]
    return groups


def align_weeks(weeks_raw: str, n_slots: int, total_weeks: int = 18) -> Tuple[List[Tuple[int, ...]], bool]:
    """Align the concatenated 上课周次 list to ``n_slots`` slots.

    (a) as many parts as slots -> 1:1; (b) otherwise monotone grouping (``group_week_parts``) when
    that yields exactly ``n_slots`` groups; (c) otherwise every slot gets the union of all parts and
    the row is flagged ambiguous.
    """
    parts = [p for p in _split_list(weeks_raw) if p]
    if n_slots <= 0:
        return [], False
    if not parts:
        logger.warning("empty 上课周次 with %d slot(s); assuming weeks 1-%d", n_slots, total_weeks)
        return [tuple(range(1, total_weeks + 1))] * n_slots, True
    if len(parts) == n_slots:
        return [parse_week_part(p) for p in parts], False
    groups = group_week_parts(parts)
    if len(groups) == n_slots:
        return [_merge_weeks(g) for g in groups], False
    union = _merge_weeks(parts)
    return [union] * n_slots, True


def count_classes(label: str) -> int:
    """Number of administrative classes in a 上课班级 label (``[1-4]`` ranges expanded)."""
    total = 0
    for item in _split_list(label):
        if not item:
            continue
        n = 1
        for lo, hi in _BRACKET.findall(item):
            n *= max(int(hi) - int(lo) + 1, 1)
        total += n
    return total


def is_general_elective(label: str) -> bool:
    """Temporary-class labels (``临班NNN``) mark university-wide elective sections."""
    return bool(_LINBAN.match(label.strip()))


def _split_list(value: str) -> List[str]:
    return [p.strip() for p in str(value).replace("，", ",").split(",")]


def _merge_weeks(parts: Sequence[str]) -> Tuple[int, ...]:
    weeks: set = set()
    for part in parts:
        weeks.update(parse_week_part(part))
    return tuple(sorted(weeks))


# --- reading --------------------------------------------------------------------------------
def split_text_row(line: str, lineno: int = 0) -> Optional[Dict[str, str]]:
    """Split one whitespace-separated text row, anchoring on the first 7 and last 5 fields."""
    return _from_fields([f for f in _FIELD_SEP.split(line.strip()) if f != ""], lineno)


def _from_fields(fields: List[str], lineno: int = 0) -> Optional[Dict[str, str]]:
    """Build a row from already split, non-empty fields (campus and room may have collapsed)."""
    if len(fields) < len(HEAD_COLUMNS) + len(TAIL_COLUMNS) - 1:
        logger.warning("line %d: skipping row with %d fields", lineno, len(fields))
        return None
    campus = ""
    if fields[0].endswith("校区"):
        campus, fields = fields[0], fields[1:]
    head, tail = fields[:len(HEAD_COLUMNS) - 1], fields[-len(TAIL_COLUMNS):]
    middle = fields[len(HEAD_COLUMNS) - 1:-len(TAIL_COLUMNS)]
    if len(middle) > 1:
        logger.warning("line %d: %d middle fields; joining them as 上课地点", lineno, len(middle))
    row = dict(zip(HEAD_COLUMNS[1:], head))
    row[HEAD_COLUMNS[0]] = campus
    row[ROOM_COLUMN] = " ".join(middle)
    row.update(zip(TAIL_COLUMNS, tail))
    return row


def read_text(path: Path) -> pd.DataFrame:
    """Read a .txt copy of the web table, tab separated or separated by runs of >= 2 spaces.

    Leading blank lines (the export starts with a line of U+3000) and the header are skipped.  A
    tab-separated row keeps its empty cells, so it is mapped positionally; a space-separated row
    has empty cells collapsed into the separator and is anchored from both ends instead.
    """
    columns = list(HEAD_COLUMNS) + [ROOM_COLUMN] + list(TAIL_COLUMNS)
    rows: List[Dict[str, str]] = []
    skipped: List[int] = []
    with path.open(encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if "课程编号" in line and "上课时间" in line:  # header
                continue
            if "\t" in line:
                cells = [c.strip() for c in line.rstrip("\r\n").split("\t")]
                if len(cells) == len(columns):
                    rows.append(dict(zip(columns, cells)))
                    continue
                row = _from_fields([c for c in cells if c], lineno)
            else:
                row = split_text_row(line, lineno)
            if row is None:
                skipped.append(lineno)
            else:
                rows.append(row)
    if skipped:
        logger.warning("%d line(s) could not be parsed: %s", len(skipped), skipped)
    logger.info("read %d data rows from %s", len(rows), path.name)
    return pd.DataFrame(rows, columns=columns)


def read_table(path: Path) -> pd.DataFrame:
    """Read an .xlsx/.xls/.csv export by header name."""
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        frame = pd.read_excel(path, dtype=str)
    else:
        frame = pd.read_csv(path, dtype=str)
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in HEAD_COLUMNS + (ROOM_COLUMN,) if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    keep = [c for c in list(HEAD_COLUMNS) + [ROOM_COLUMN] + list(TAIL_COLUMNS) if c in frame.columns]
    return frame[keep]


def load_timetable(path: str | Path, cfg: TimetableConfig | None = None) -> pd.DataFrame:
    """Read a timetable export and return one row per course section.

    Args:
        path: .txt (whitespace separated), .csv or .xlsx export.
        cfg: semester geometry; defaults to :class:`TimetableConfig`.

    Returns:
        DataFrame with :data:`SECTION_COLUMNS`; teacher columns are dropped on read.

    Raises:
        FileNotFoundError: the export does not exist.
        ValueError: a tabular export is missing required columns.
    """
    cfg = cfg or TimetableConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"timetable export not found: {path}")
    raw = read_text(path) if path.suffix.lower() == ".txt" else read_table(path)
    raw = raw.drop(columns=[c for c in PERSONAL_COLUMNS if c in raw.columns])
    frame = raw.rename(columns=RENAME)
    for column in RENAME.values():
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame = frame[frame["course_code"] != ""].reset_index(drop=True)
    frame.insert(0, "section_id", range(len(frame)))
    frame["n_classes"] = frame["class_label"].map(count_classes)
    frame["general_elective"] = frame["class_label"].map(is_general_elective)
    frame["credits"] = pd.to_numeric(frame["credits"], errors="coerce")
    frame["enrollment"] = pd.to_numeric(frame["enrollment"], errors="coerce").fillna(0).astype(int)
    frame["ambiguous"] = [
        _resolve_slots(r.time_raw, r.weeks_raw, r.room_raw, cfg)[1]
        for r in frame.itertuples()
    ]
    logger.info("parsed %d sections, %d ambiguous week lists", len(frame), int(frame["ambiguous"].sum()))
    return frame[list(SECTION_COLUMNS)]


# --- slots ----------------------------------------------------------------------------------
def _resolve_slots(time_raw: str, weeks_raw: str, room_raw: str,
                   cfg: TimetableConfig) -> Tuple[List[Dict[str, object]], bool]:
    """Resolve one section's time/week/room lists into per-slot records."""
    codes = [c for c in _split_list(time_raw) if c]
    if not codes:
        return [], False
    parsed = [(c, parse_time_code(c, cfg.periods)) for c in codes]
    rooms = _split_list(room_raw)
    if len(rooms) == 1 and len(codes) > 1:
        rooms = rooms * len(codes)
    rooms = (rooms + [""] * len(codes))[:len(codes)]
    weeks_per_slot, ambiguous = align_weeks(weeks_raw, len(codes), cfg.weeks)
    slots: List[Dict[str, object]] = []
    for index, ((code, decoded), room, weeks) in enumerate(zip(parsed, rooms, weeks_per_slot)):
        if decoded is None:
            continue
        weekday, periods = decoded
        slots.append({
            "slot_index": index, "weekday": weekday, "periods": periods,
            "period_start": min(periods), "period_end": max(periods), "n_periods": len(periods),
            "weeks": weeks, "n_weeks": len(weeks), "room": room, "time_code": code,
        })
    return slots, ambiguous


def explode_slots(sections: pd.DataFrame, cfg: TimetableConfig | None = None) -> pd.DataFrame:
    """One row per section x meeting slot.

    Args:
        sections: output of :func:`load_timetable`.
        cfg: semester geometry.

    Returns:
        DataFrame with section_id, slot_index, campus, weekday, period_start/_end, n_periods,
        periods, weeks (tuple), n_weeks, room, enrollment and the section descriptors.
    """
    cfg = cfg or TimetableConfig()
    carry = ["section_id", "campus", "college", "course_code", "course_name", "course_type",
             "class_label", "n_classes", "enrollment", "general_elective"]
    rows: List[Dict[str, object]] = []
    for row in sections.itertuples():
        slots, ambiguous = _resolve_slots(row.time_raw, row.weeks_raw, row.room_raw, cfg)
        for slot in slots:
            rows.append({**{c: getattr(row, c) for c in carry}, **slot, "ambiguous": ambiguous})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    logger.info("exploded %d sections into %d slots", len(sections), len(frame))
    return frame


def expand_weeks(slots: pd.DataFrame) -> pd.DataFrame:
    """One row per section x slot x teaching week (the ``weeks`` tuple exploded)."""
    if slots.empty:
        return slots.copy()
    frame = slots.explode("weeks", ignore_index=True).rename(columns={"weeks": "week"})
    frame = frame[frame["week"].notna()].copy()
    frame["week"] = frame["week"].astype(int)
    logger.info("expanded %d slots into %d slot-weeks", len(slots), len(frame))
    return frame


def format_weeks(weeks: Sequence[int]) -> str:
    """``(1, 2, 3, 5)`` -> ``"1-3,5"`` for compact CSV output."""
    if len(weeks) == 0:
        return ""
    ordered = sorted(int(w) for w in weeks)
    out, lo, prev = [], ordered[0], ordered[0]
    for week in ordered[1:] + [None]:  # type: ignore[list-item]
        if week is not None and week == prev + 1:
            prev = week
            continue
        out.append(str(lo) if lo == prev else f"{lo}-{prev}")
        if week is not None:
            lo = prev = week
    return ",".join(out)
