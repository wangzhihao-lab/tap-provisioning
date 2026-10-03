"""Finding and reading the batch C artefacts, tolerating whatever the running batch has not written.

The report is meant to be run *while* ``scripts/sim/batch_c.sh`` is still going, so every read here
returns ``None`` and logs a warning rather than raising: a lane that has not reached its sweep yet
simply drops out of the tables.

Layout (``scripts/sim/batch_c.sh``): ``<root>/<group>/<sweep>/<sweep>.csv`` with ``group`` one of
``r6 r10 r30 r100 r30_slo95 r30_32b eps``, plus
``<root>/tuning_r<R>[_slo95]/{best.json, <policy>_grid.csv}``.  The sweep files are discovered by
the shape of the path (the CSV's stem repeats its directory) so that a group nobody asked about
still shows up in the verification table.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

TUNED_POLICIES: Tuple[str, ...] = ("reactive", "velocity")  # the two grids tune_baselines.py runs
GRID_SCORE: str = "cost_ratio_tune_mean"  # the tuning objective, one column of <policy>_grid.csv
SCORE_ROUNDING: int = 9  # distinct scores are counted after rounding to 1e-9
CONFIG_KEY: str = "config"
POLICIES_KEY: str = "policies"
SLICE_KEY: str = "slice"  # the provenance block of best.json (``TuningSpec.to_dict``)
SLO_FLOOR_KEY: str = "slo_floor"  # ``null`` unless tune_baselines.py ran with --slo-floor
SLO95_SUFFIX: str = "_slo95"  # the directory suffix of the SLO-matched selection


@dataclass(frozen=True)
class SweepFile:
    """One sweep CSV: its group directory (``r30``, ``eps``, ...), its sweep name and its path."""

    group: str
    sweep: str
    path: Path

    @property
    def label(self) -> str:
        """``r30/lead``: how the verification table names the file."""
        return f"{self.group}/{self.sweep}"


def read_frame(path: Path) -> Optional[pd.DataFrame]:
    """Read one CSV, or log a warning and return ``None`` when it is missing or unreadable."""
    if not path.exists():
        logger.warning("missing %s (skipped)", path)
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning("could not read %s: %s (skipped)", path, exc)
        return None
    if frame.empty:
        logger.warning("%s has no rows (skipped)", path)
        return None
    return frame


def sweep_path(root: Path, group: str, sweep: str) -> Path:
    """``<root>/<group>/<sweep>/<sweep>.csv``, the path ``run_sweep.py --out <root>/<group>`` writes."""
    return Path(root) / group / sweep / f"{sweep}.csv"


def find_sweep_files(root: Path) -> List[SweepFile]:
    """Every ``<group>/<sweep>/<sweep>.csv`` under ``root``, sorted by group then sweep."""
    root = Path(root)
    if not root.exists():
        logger.warning("missing batch root %s: nothing to report", root)
        return []
    found = [SweepFile(path.parent.parent.name, path.parent.name, path)
             for path in sorted(root.glob("*/*/*.csv")) if path.stem == path.parent.name]
    if not found:
        logger.warning("no <group>/<sweep>/<sweep>.csv under %s", root)
    return found


@dataclass(frozen=True)
class TuningRun:
    """One ``tuning_r<R>`` directory: the chosen configurations and the grids behind them.

    ``suffix`` distinguishes the variants of the same price ratio that batch C tunes twice:
    ``tuning_r30`` selects the cheapest configuration outright and ``tuning_r30_slo95`` the
    cheapest one that reaches ``slo_in >= 0.95`` on the tuning seeds, which is a different
    baseline and has to be reported as such (``tune_baselines.chosen``).
    """

    ratio: int
    path: Path
    best: Mapping[str, object]
    grids: Mapping[str, Path]
    suffix: str = ""

    @property
    def label(self) -> str:
        """``r30`` / ``r30_slo95``: how the tuning table names this selection."""
        return f"r{self.ratio}{self.suffix}"

    @property
    def slo_floor(self) -> float:
        """``slice.slo_floor``: the attainment the selection was constrained to, NaN when free."""
        block = self.best.get(SLICE_KEY)
        if not isinstance(block, Mapping):
            logger.warning("%s: no '%s' block; the SLO floor is unknown", self.path, SLICE_KEY)
            return float("nan")
        try:
            return float(block.get(SLO_FLOOR_KEY))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")  # an unconstrained selection writes ``null``

    def sections(self) -> Dict[str, Mapping[str, object]]:
        """The ``policies.<name>`` blocks of ``best.json``, in the order of ``TUNED_POLICIES``."""
        raw = self.best.get(POLICIES_KEY) or {}
        if not isinstance(raw, Mapping):
            logger.warning("%s: '%s' is not an object; ignored", self.path, POLICIES_KEY)
            return {}
        known = [p for p in TUNED_POLICIES if p in raw]
        extra = [p for p in raw if p not in TUNED_POLICIES]
        return {name: raw[name] for name in known + sorted(extra) if isinstance(raw[name], Mapping)}


def read_tuning(root: Path, ratio: int, suffix: str = "") -> Optional[TuningRun]:
    """Read ``<root>/tuning_r<ratio><suffix>/best.json`` and note which grid CSVs sit beside it."""
    folder = Path(root) / f"tuning_r{ratio}{suffix}"
    best = folder / "best.json"
    if not best.exists():
        logger.warning("missing %s (r=%d%s tuning skipped)", best, ratio, suffix)
        return None
    try:
        payload = json.loads(best.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s (skipped)", best, exc)
        return None
    if not isinstance(payload, Mapping):
        logger.warning("%s is not a JSON object (skipped)", best)
        return None
    grids = {policy: folder / f"{policy}_grid.csv" for policy in TUNED_POLICIES
             if (folder / f"{policy}_grid.csv").exists()}
    return TuningRun(ratio=ratio, path=folder, best=payload, grids=grids, suffix=suffix)


def distinct_scores(path: Optional[Path]) -> Optional[int]:
    """How many distinct tuning scores a grid CSV holds - whether the grid discriminated at all.

    The count is taken on ``cost_ratio_tune_mean`` rounded to 1e-9: a knob that never binds on the
    slice leaves every configuration with the same cost, and a grid of 54 rows and one score is a
    tuning result that says nothing, which the table has to make visible.
    """
    if path is None:
        return None
    frame = read_frame(path)
    if frame is None or GRID_SCORE not in frame.columns:
        if frame is not None:
            logger.warning("%s has no '%s' column", path, GRID_SCORE)
        return None
    scores = pd.to_numeric(frame[GRID_SCORE], errors="coerce").dropna()
    return int(scores.round(SCORE_ROUNDING).nunique())


__all__ = ["CONFIG_KEY", "GRID_SCORE", "POLICIES_KEY", "SCORE_ROUNDING", "SLICE_KEY",
           "SLO95_SUFFIX", "SLO_FLOOR_KEY", "TUNED_POLICIES", "SweepFile", "TuningRun",
           "distinct_scores", "find_sweep_files", "read_frame", "read_tuning", "sweep_path"]
