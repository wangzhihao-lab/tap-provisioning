"""Writing a table twice: the CSV a script reads and the Markdown a reader reads.

Every Markdown file opens with the same provenance block - which CSVs it came from, the base-tier
formula with the values it was computed with, when it was generated and the sentence that the
numbers are means over seeds - so a table lifted out of the directory still says where it is from.

Numbers follow one rule throughout: three significant digits for USD and counts, three decimals for
ratios and rates, integers for shapes and ranks.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED_NOTE: str = ("Numbers are means over seeds, with the standard deviation in parentheses where "
                  "shown.")
NA: str = "n/a"
RATE_COLUMNS: Tuple[str, ...] = (
    "attr_ratio", "cost_ratio", "slo_in", "slo_off", "slo_first_in", "viol_in", "eta_norm",
    "sigma2_measured", "sigma2_pooled", "x_mean_min", "slo_floor", "tuning_ratio_chosen",
    "tuning_ratio_default", "heldout_ratio_chosen", "heldout_ratio_default",
    "heldout_slo_in_chosen", "heldout_slo_in_default")
INT_COLUMNS: Tuple[str, ...] = (
    "n_rows", "n_cells", "week", "weeks", "seed", "grid_size", "rank_of_default",
    "n_distinct_scores", "nan_total_cost", "nan_gpu_hours", "nan_slo_in",
    "dominance_violations", "floor_violations", "price_ratio")
SUFFIXES: Tuple[str, ...] = ("_mean", "_sd")


def base_tier_formula(min_replicas: int, c_run: float) -> str:
    """The base-tier definition with the values this run used, for the provenance block."""
    return (f"Base tier: `base_usd = M * c_run * minutes` and `base_hours = M * minutes / 60` "
            f"with `M = {min_replicas}` replica(s), `c_run = {c_run:g}` USD per replica-minute and "
            f"`minutes = 10080 * weeks` for whole-run rows (`week = 0`), `10080` for weekly rows; "
            f"`excess_usd = total_cost - offline_total`, "
            f"`attr_ratio = (total_cost - base_usd) / (offline_total - base_usd)`.")


@dataclass(frozen=True)
class Provenance:
    """The header every Markdown table opens with."""

    title: str
    sources: Tuple[str, ...]
    base_formula: str
    generated: str
    notes: Tuple[str, ...] = ()

    def lines(self) -> List[str]:
        """The header as Markdown lines, sources first."""
        out = [f"# {self.title}", ""]
        if self.sources:
            out.append("Source" + ("s" if len(self.sources) > 1 else "") + ":")
            out.extend(f"- `{src}`" for src in self.sources)
        else:
            out.append("Sources: none found.")
        out.extend(["", self.base_formula, "", f"Generated: {self.generated}", "", SEED_NOTE])
        out.extend(["", *self.notes] if self.notes else [])
        return out


def now_stamp() -> str:
    """Local run timestamp, to the second, with the UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _base_name(column: str) -> str:
    """``excess_usd_mean`` -> ``excess_usd``: the metric a mean/sd column belongs to."""
    for suffix in SUFFIXES:
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def format_value(column: str, value: object) -> str:
    """One cell of a Markdown table: 3 decimals for rates, 3 significant digits otherwise."""
    if isinstance(value, str):
        return value
    if value is None:
        return NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return NA
    base = _base_name(column)
    if base in INT_COLUMNS and column == base:
        return f"{int(round(number))}"
    if base in RATE_COLUMNS:
        return f"{number:.3f}"
    return f"{number:.3g}"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A GitHub-flavoured Markdown table; an empty body becomes a one-line note."""
    if not header:
        return "_(empty table)_"
    body = [f"| {' | '.join(header)} |", f"| {' | '.join(['---'] * len(header))} |"]
    body.extend(f"| {' | '.join(row)} |" for row in rows)
    if not rows:
        body.append(f"| {' | '.join([NA] * len(header))} |")
    return "\n".join(body)


def _merge_mean_sd(frame: pd.DataFrame, format_as: Optional[str] = None
                   ) -> Tuple[List[str], List[List[str]]]:
    """Collapse ``<m>_mean`` / ``<m>_sd`` pairs into one ``mean (sd)`` column."""
    columns: List[Tuple[str, str, Optional[str]]] = []
    seen: set = set()
    for column in frame.columns:
        base = _base_name(column)
        if column.endswith("_mean") and f"{base}_sd" in frame.columns:
            columns.append((base, column, f"{base}_sd"))
            seen.add(f"{base}_sd")
        elif column not in seen:
            columns.append((column, column, None))
    header = [name for name, _, _ in columns]
    rows: List[List[str]] = []
    for _, row in frame.iterrows():
        cells = []
        for name, mean_col, sd_col in columns:
            key = format_as or name
            text = format_value(key, row[mean_col])
            if sd_col is not None:
                sd = format_value(key, row[sd_col])
                text = f"{text} ({sd})" if sd != NA else text
            cells.append(text)
        rows.append(cells)
    return header, rows


def frame_to_markdown(frame: pd.DataFrame, merge_sd: bool = True,
                      format_as: Optional[str] = None) -> str:
    """A tidy frame as one Markdown table, mean/sd pairs merged unless asked otherwise.

    ``format_as`` names the metric every numeric cell holds, for a table whose column headers are
    not metric names (the r-sweep's columns are price ratios); text cells are unaffected.
    """
    if frame is None or frame.empty:
        return "_(no rows)_"
    if merge_sd:
        header, rows = _merge_mean_sd(frame, format_as)
    else:
        header = [str(c) for c in frame.columns]
        rows = [[format_value(format_as or str(c), row[c]) for c in frame.columns]
                for _, row in frame.iterrows()]
    return markdown_table(header, rows)


def wide_by_week(frame: pd.DataFrame, metric: str, r1_weeks: Sequence[int],
                 index: str = "policy") -> str:
    """One metric as policies x weeks, the R1 weeks marked with a star in the header."""
    if frame is None or frame.empty or metric not in frame.columns:
        return "_(no rows)_"
    pivot = frame.pivot_table(index=index, columns="week", values=metric, sort=True)
    novel = {int(w) for w in r1_weeks}
    header = [index] + [f"w{int(w)}*" if int(w) in novel else f"w{int(w)}" for w in pivot.columns]
    rows = [[str(name)] + [format_value(metric, value) for value in series]
            for name, series in pivot.iterrows()]
    return markdown_table(header, rows)


def blocks_by(frame: pd.DataFrame, column: str, titles: Optional[Dict[str, str]] = None) -> str:
    """One Markdown table per distinct value of ``column`` (the r-sweep's three metric blocks)."""
    if frame is None or frame.empty or column not in frame.columns:
        return "_(no rows)_"
    parts: List[str] = []
    for value in frame[column].drop_duplicates():
        part = frame[frame[column] == value].drop(columns=[column])
        name = (titles or {}).get(str(value), str(value))
        parts.extend([f"## {name}", "", frame_to_markdown(part, format_as=str(value)), ""])
    return "\n".join(parts).rstrip() + "\n"


def write_table(out_dir: Path, name: str, frame: pd.DataFrame, provenance: Provenance,
                markdown: Optional[str] = None) -> Tuple[Path, Path]:
    """Write ``<name>.csv`` and ``<name>.md`` into ``out_dir``; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"{name}.csv", out_dir / f"{name}.md"
    body = frame if frame is not None else pd.DataFrame()
    body.to_csv(csv_path, index=False)
    text = markdown if markdown is not None else frame_to_markdown(body)
    md_path.write_text("\n".join([*provenance.lines(), "", text, ""]), encoding="utf-8")
    logger.info("wrote %s and %s (%d rows)", csv_path.name, md_path.name, len(body))
    return csv_path, md_path


__all__ = ["INT_COLUMNS", "NA", "RATE_COLUMNS", "SEED_NOTE", "Provenance", "base_tier_formula",
           "blocks_by", "format_value", "frame_to_markdown", "markdown_table", "now_stamp",
           "wide_by_week", "write_table"]
