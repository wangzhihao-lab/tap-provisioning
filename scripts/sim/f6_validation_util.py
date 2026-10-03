"""Input resolution, aggregation and Markdown helpers for ``f6_validation.py``.

Split out of that script so it stays inside the project's file-size rule; nothing here is
specific to the figure, and nothing here imports matplotlib.

The aggregation is the one ``make_figures.py::figure_f6`` used for batch C, reproduced rather than
imported so that the pair stands alone: a seed's whole-run cost ratio is its ``week = 0`` row, the
summed weekly cost over that run's offline optimum with the base tier in both.  The weekly rows
repeat the *whole-run* ``offline_cost``, so those must not be summed.

Beside the total cost every cell also carries the *net* cost, the decision-dependent one the
interpolation's phase test actually compares (``pilot.sim.policies.bdinterp._slot_cost``): the
always-on base tier is a sunk cost both branches pay whatever they decide, so its running cost is
taken out of both sides.  Over a whole run that constant is
``c_run * min_replicas * weeks * 7 * 24 * 60``.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("f6_validation")

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT

RATIO_COL, ATTAIN_COL, COST_COL = "cost_ratio", "slo_in", "total_cost"
C_RUN_COL, C_RUN_DEFAULT = "c_run", 0.0333  # USD per replica-minute, batch C's --c-run
MIN_PER_WEEK = 7 * 24 * 60
KEYS = ["price_ratio", "forecast_scale", "policy"]
NONE = "none"


def rel(path: Path) -> str:
    """``path`` under the repository root when it lives there, for the generated headers."""
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def resolve_inputs(patterns: Sequence[str]) -> List[Path]:
    """Existing CSVs behind the given paths or glob patterns; relative ones are taken from PILOT."""
    found: List[Path] = []
    for raw in patterns:
        base = Path(raw)
        candidate = base if base.is_absolute() else PILOT / base
        if any(ch in raw for ch in "*?["):
            hits = sorted(Path(p) for p in glob.glob(str(candidate)))
            if not hits:
                logger.debug("no match for pattern %s", raw)
            found.extend(hits)
            continue
        if candidate.exists():
            found.append(candidate)
        else:
            logger.debug("skipping missing input %s", candidate)
    unique = list(dict.fromkeys(found))
    logger.info("found %d of %d inputs", len(unique), len(patterns))
    for path in unique:
        logger.info("  input %s", rel(path))
    return unique


def load(paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate the sweep CSVs and keep the ``forecast_scale`` rows of the error axis."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "error_knob" in df.columns:
        df = df[df["error_knob"] == "forecast_scale"]
    logger.info("%d forecast_scale rows over %d file(s)", len(df), len(paths))
    return df


def whole_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (price_ratio, forecast_scale, seed, policy): the whole run of that seed.

    ``week = 0`` already carries the summed cost over that run's own offline optimum; the weekly
    rows repeat the whole-run ``offline_cost``, so the fallback sums only the numerator.
    """
    if "week" in df.columns and (df["week"] == 0).any():
        return df[df["week"] == 0].copy()
    logger.warning("no week = 0 rows; rebuilding the whole-run ratio from the weekly rows")
    keys = ["price_ratio", "forecast_scale", "seed", "policy"]
    how: Dict[str, str] = {COST_COL: "sum", "offline_cost": "mean"}
    for extra in (ATTAIN_COL, "weeks", C_RUN_COL):
        if extra in df.columns:
            how[extra] = "mean"
    out = df.groupby(keys, as_index=False).agg(how)
    out[RATIO_COL] = out[COST_COL] / out["offline_cost"]
    return out


def _sunk(whole: pd.DataFrame, c_run: Optional[float], min_replicas: int) -> pd.Series:
    """Running cost of the always-on base tier over each row's whole run, in USD."""
    if C_RUN_COL in whole.columns:
        rate = pd.to_numeric(whole[C_RUN_COL], errors="coerce")
        logger.info("c_run read from the sweep CSVs (mean %.4f USD per replica-minute)",
                    float(rate.mean()))
    else:
        rate = pd.Series(C_RUN_DEFAULT if c_run is None else c_run, index=whole.index)
        logger.info("no '%s' column; c_run = %.4f USD per replica-minute", C_RUN_COL,
                    float(rate.iloc[0]) if len(rate) else float("nan"))
    if "weeks" not in whole.columns:
        logger.warning("no 'weeks' column; the base tier is priced over a single week")
        weeks = pd.Series(1.0, index=whole.index)
    else:
        weeks = pd.to_numeric(whole["weeks"], errors="coerce")
    return rate * float(min_replicas) * weeks * MIN_PER_WEEK


def aggregate(df: pd.DataFrame, c_run: Optional[float] = None,
              min_replicas: int = 1) -> pd.DataFrame:
    """Mean and sd over seeds of the whole-run cost ratio, the total cost and the net cost.

    ``net_cost_mean`` is the decision-dependent cost the phase test compares: the whole-run cost
    with the always-on base tier's running cost removed from it.
    """
    rows = whole_run(df)
    missing = (set(KEYS) | {RATIO_COL}) - set(rows.columns)
    if missing:
        raise KeyError(f"the sweep CSVs lack {sorted(missing)}")
    if COST_COL not in rows.columns:
        logger.info("no '%s' column; the cost is rebuilt as cost_ratio * offline_cost", COST_COL)
        rows[COST_COL] = rows[RATIO_COL] * rows["offline_cost"]
    rows["sunk_usd"] = _sunk(rows, c_run, min_replicas)
    rows["net_cost"] = rows[COST_COL] - rows["sunk_usd"]
    if (rows["net_cost"] <= 0).any():
        logger.warning("%d row(s) have a non-positive net cost; --min-replicas may be too large",
                       int((rows["net_cost"] <= 0).sum()))
    grp = rows.groupby(KEYS)
    out = pd.DataFrame({"cost_ratio_mean": grp[RATIO_COL].mean(),
                        "cost_ratio_sd": grp[RATIO_COL].std(ddof=1),
                        "n_seeds": grp[RATIO_COL].size(),
                        "total_cost_mean": grp[COST_COL].mean(),
                        "net_cost_mean": grp["net_cost"].mean(),
                        "sunk_usd": grp["sunk_usd"].mean()})
    if ATTAIN_COL in rows.columns:
        out["slo_in_mean"] = grp[ATTAIN_COL].mean()
    else:
        logger.info("no '%s' column in the sweeps; in-class attainment is skipped", ATTAIN_COL)
    return out.reset_index().sort_values(KEYS).reset_index(drop=True)


def curve(agg: pd.DataFrame, ratio: float, policy: str,
          col: str = "cost_ratio_mean") -> pd.Series:
    """One policy's ``col`` in one price-ratio panel, indexed by forecast scale."""
    part = agg[(agg["price_ratio"] == ratio) & (agg["policy"] == policy)]
    return part.set_index("forecast_scale")[col].sort_index()


def first_or_none(scales: Sequence[float]) -> object:
    """The smallest scale of a hit list, or the ``none`` marker when it is empty."""
    return float(min(scales)) if len(scales) else NONE


def cell(value: object) -> str:
    """One Markdown cell: floats to three significant digits, ``n/a`` for a missing number."""
    if isinstance(value, float):
        return f"{value:g}" if np.isfinite(value) else "n/a"
    return str(value)


def md_table(frame: pd.DataFrame) -> str:
    """A frame as a Markdown pipe table (tabulate is not a dependency of this project)."""
    if frame.empty:
        return "_(no rows)_\n"
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines) + "\n"
