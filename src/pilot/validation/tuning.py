"""What the baseline tuning of the validation round selected, and whether its SLO floor was reachable.

``tune_baselines.py`` writes ``best.json`` (the chosen configurations) and one ``<policy>_grid.csv``
per tuned policy under ``<case>/tuning[8]_r<ratio>[_slo95]``.  The SLO-matched selection is only
meaningful when at least one grid point reaches the floor on the tuning seeds; when none does the
tuner falls back to the cheapest point and the "matched" baseline is the freely tuned one under
another name.  ``tuning_rows`` reports that per directory so the results note can say it.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .runs import CASES, CASE_ORDER

logger = logging.getLogger(__name__)

TUNING_RE = re.compile(r"^tuning(?P<t8>8)?_r(?P<ratio>\d+)(?P<slo>_slo95disp|_slo95)?$")
SCORE: str = "cost_ratio_tune_mean"
SLO_SCORE: str = "slo_in_tune_mean"


def discover_tuning(root: Path) -> List[Path]:
    """Every ``<case>/tuning*/best.json`` under ``root`` (any depth), first copy per (case, dir)."""
    seen: Dict[tuple, Path] = {}
    for path in sorted(Path(root).glob("**/best.json")):
        case, name = path.parent.parent.name, path.parent.name
        if case in CASES and TUNING_RE.match(name) and (case, name) not in seen:
            seen[(case, name)] = path
    return [seen[k] for k in sorted(seen, key=lambda k: (CASE_ORDER.index(k[0]), k[1]))]


def _config_text(config: Dict[str, object]) -> str:
    return ", ".join(f"{k}={v}" for k, v in config.items())


def _grid_summary(grid_path: Path, floor: Optional[float]) -> Dict[str, object]:
    """Grid size, best attainable SLO on the tuning seeds and how many points reach the floor."""
    if not grid_path.exists():
        return {"grid_size": None, "max_slo_tune": None, "n_reach_floor": None}
    grid = pd.read_csv(grid_path)
    out: Dict[str, object] = {"grid_size": len(grid)}
    if SLO_SCORE in grid.columns:
        slo = pd.to_numeric(grid[SLO_SCORE], errors="coerce")
        out["max_slo_tune"] = float(slo.max())
        out["n_reach_floor"] = int((slo >= floor).sum()) if floor is not None else None
    else:
        out["max_slo_tune"] = None
        out["n_reach_floor"] = None
    return out


def tuning_rows(best_paths: List[Path]) -> pd.DataFrame:
    """One row per (case, tuning dir, policy): the selection and the floor's reachability."""
    rows: List[Dict[str, object]] = []
    for path in best_paths:
        best = json.loads(path.read_text())
        name = path.parent.name
        m = TUNING_RE.match(name)
        slice_ = best.get("slice", {})
        floor = slice_.get("slo_floor")
        for policy, block in best.get("policies", {}).items():
            summary = _grid_summary(path.parent / f"{policy}_grid.csv", floor)
            rows.append({
                "case": path.parent.parent.name, "tuning": name,
                "tree": path.parents[2].name,
                "trigger_offset": " ".join(str(v) for v in slice_.get("trigger_offset", [0, 0, 0, 0])),
                "tuned_at": slice_.get("lead_tier", "d_inst" if m and m.group("t8") else "d_rep"),
                "ratio": int(m.group("ratio")) if m else None,
                "slo_floor": floor, "policy": policy,
                "config": _config_text(block.get("config", {})),
                **summary,
                "floor_reachable": (None if floor is None or summary["n_reach_floor"] is None
                                    else bool(summary["n_reach_floor"] > 0)),
            })
    return pd.DataFrame(rows)


__all__ = ["discover_tuning", "tuning_rows"]
