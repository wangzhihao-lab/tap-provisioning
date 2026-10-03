"""Common interface for arrival sources."""

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import pandas as pd

COLUMNS = ("ts", "group", "kind")


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse wall-clock timestamps that may be ISO strings, mixed formats or unix epochs.

    Unparsable entries become NaT; ``ArrivalSource.validate`` drops them.
    """
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        big = numeric.dropna()
        unit = "ms" if not big.empty and (big > 1e11).mean() > 0.5 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    text = values.astype("string").str.strip()
    out = pd.to_datetime(text, errors="coerce", format="ISO8601")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(text, errors="coerce", format="mixed")
    return out


class ArrivalSource(ABC):
    """Produces arrival events; subclasses wrap synthetic generators or real exports.

    ``workload_class`` separates education workloads from generic LLM traces in the Q1 tables.
    ``n_paths`` tells the CLI how many files the constructor takes: 0 (none), 1 (``path`` first),
    -1 (a list of paths first).
    """

    workload_class: ClassVar[str] = "education"
    n_paths: ClassVar[int] = 0
    # where sessions come from: "timetable" (known in advance), "inferred" (clustered from the
    # log) or "detected" (threshold onsets, no schedule exists)
    timetable_kind: ClassVar[str] = "inferred"

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a DataFrame with columns ts (datetime64), group (str), kind (str)."""

    def sessions(self) -> Optional[pd.DataFrame]:
        """Known timetable (group, start, duration_min, n_students), or None when there is none."""
        return None

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Check the schema, sort by time and drop rows without a timestamp."""
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"arrival frame missing columns {missing}")
        out = df.dropna(subset=["ts"]).copy()
        out["ts"] = pd.to_datetime(out["ts"])
        return out.sort_values("ts").reset_index(drop=True)
