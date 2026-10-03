"""Selection of the tuned configuration from a scored grid, and the ``best.json`` it goes into.

Split out of ``tune_baselines.py`` (which scores the grid) so that the selection rules sit in one
place.  Two rules pick the row of a policy's grid that ``best.json`` carries:

``cost`` (the default, the rule every earlier ``best.json`` was written with)
    the first row of the ranked grid - lowest mean cost ratio (penalties included) over the tuning
    seeds - or, with ``--slo-floor``, the first ranked row whose mean in-class attainment over the
    tuning seeds reaches the floor.
``resource`` (``--slo-select resource``, needs ``--slo-floor``)
    among the rows whose mean in-class attainment over the tuning seeds is at least the floor (no
    rounding), the one with the lowest mean resource cost ``run_cost + switch_cost`` over the
    tuning seeds; ties keep the ranked order.  This is the SLO-floor baseline's own objective: the
    cheapest capacity that meets the floor, with the penalty weights playing no part.

Both fall back to the unconstrained best (the first ranked row) when no row reaches the floor, and
say so; the resource rule also records that in ``best.json`` (``selection.floor_met``) together
with the row the cost rule would have picked on the same grid.  Only the tuning seeds enter either
rule; the held-out seed is reported, never selected on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from tuned_config import CONFIG_KEY, POLICIES_KEY

logger = logging.getLogger("tune_baselines")

SLO_SELECT: Tuple[str, ...] = ("cost", "resource")
# metrics kept per tuning seed so that a selection can be made on the resource cost alone
RESOURCE_KEYS: Tuple[str, ...] = ("gpu_hours", "starts", "run_cost", "switch_cost", "slo_in")
RESOURCE_COST = "resource_cost_tune_mean"  # mean of run_cost + switch_cost over the tuning seeds


def resource_columns(per_seed: Sequence[Tuple[int, Mapping[str, float]]]) -> Dict[str, float]:
    """Per-seed ``RESOURCE_KEYS`` columns, their tuning means and the mean resource cost."""
    out: Dict[str, float] = {}
    for seed, metrics in per_seed:
        out.update({f"{k}_seed{seed}": float(metrics[k]) for k in RESOURCE_KEYS})
    n = max(1, len(per_seed))
    for k in RESOURCE_KEYS:
        if k != "slo_in":  # the tuning mean of slo_in is the grid's own slo_in_tune_mean
            out[f"{k}_tune_mean"] = sum(float(m[k]) for _, m in per_seed) / n
    out[RESOURCE_COST] = sum(float(m["run_cost"]) + float(m["switch_cost"])
                             for _, m in per_seed) / n
    return out


def plain(value: object) -> object:
    """numpy scalar -> plain Python, so the JSON stays readable and re-loadable."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def pick_cost(frame: pd.DataFrame, policy: str, slo_floor: Optional[float]) -> pd.Series:
    """The ``cost`` rule: first ranked row, or first ranked row that reaches the floor."""
    best = frame.iloc[0]
    if slo_floor is not None:
        eligible = frame[frame["slo_in_tune_mean"] >= float(slo_floor)]
        if len(eligible):
            best = eligible.iloc[0]
        else:
            logger.warning("%s: no cell reaches slo_in >= %.2f on the tuning seeds; unconstrained "
                           "best kept", policy, slo_floor)
    return best


def pick_resource(frame: pd.DataFrame, policy: str, slo_floor: float) -> Tuple[pd.Series, int]:
    """The ``resource`` rule: ``(row, number of rows reaching the floor)``."""
    eligible = frame[frame["slo_in_tune_mean"] >= float(slo_floor)]
    if not len(eligible):
        logger.warning("%s: no cell reaches slo_in >= %.2f on the tuning seeds; unconstrained "
                       "best kept", policy, slo_floor)
        return frame.iloc[0], 0
    return eligible.sort_values(RESOURCE_COST, kind="mergesort").iloc[0], int(len(eligible))


def chosen(frame: pd.DataFrame, policy: str, knobs: Sequence[str],
           slo_floor: Optional[float] = None, slo_select: str = "cost") -> Dict[str, object]:
    """Best and default rows of one policy's grid, as the ``best.json`` section.

    With ``slo_floor`` and the ``cost`` rule the best row is the first ranked one whose in-class
    attainment on the tuning seeds reaches the floor (the SLO-matched baseline of F4); the
    ``resource`` rule takes the cheapest in resource cost among those rows instead (module
    docstring).  When no cell reaches the floor the unconstrained best is taken and logged.
    """
    if slo_select not in SLO_SELECT:
        raise ValueError(f"unknown selection rule '{slo_select}'; known: {list(SLO_SELECT)}")
    selection: Optional[Dict[str, object]] = None
    if slo_select == "resource":
        if slo_floor is None:
            raise ValueError("the resource selection rule needs an SLO floor (--slo-floor)")
        best, n_ok = pick_resource(frame, policy, slo_floor)
        by_cost = pick_cost(frame, policy, slo_floor)
        selection = {"rule": "resource", "slo_floor": float(slo_floor), "floor_met": n_ok > 0,
                     "eligible": n_ok,
                     "tuning_resource": {f"{k}_mean": float(best[f"{k}_tune_mean"])
                                         for k in RESOURCE_KEYS if k != "slo_in"},
                     "resource_cost_mean": float(best[RESOURCE_COST]),
                     "cost_rule_config": {k: plain(by_cost[k]) for k in knobs},
                     "same_as_cost_rule": bool(int(best["config_id"]) == int(by_cost["config_id"]))}
    else:
        best = pick_cost(frame, policy, slo_floor)
    default_rows = frame[frame["is_default"]]
    default = default_rows.iloc[0] if len(default_rows) else None
    section: Dict[str, object] = {
        CONFIG_KEY: {k: plain(best[k]) for k in knobs},
        "rank_of_default": int(default["rank"]) if default is not None else None,
        "grid_size": int(len(frame)),
        "tuning": {"cost_ratio_mean": float(best["cost_ratio_tune_mean"]),
                   "cost_ratio_max": float(best["cost_ratio_tune_max"]),
                   "slo_in_mean": float(best["slo_in_tune_mean"])},
        "heldout": {"cost_ratio": float(best["cost_ratio_heldout"]),
                    "slo_in": float(best["slo_in_heldout"]),
                    "slo_first_in": float(best["slo_first_in_heldout"]),
                    "viol_in": float(best["viol_in_heldout"]),
                    "gpu_hours": float(best["gpu_hours_heldout"]),
                    "starts": float(best["starts_heldout"])},
    }
    if selection is not None:
        section["selection"] = selection
    if default is not None:
        section["default"] = {k: plain(default[k]) for k in knobs}
        section["default_tuning"] = {"cost_ratio_mean": float(default["cost_ratio_tune_mean"])}
        section["default_heldout"] = {"cost_ratio": float(default["cost_ratio_heldout"]),
                                      "slo_in": float(default["slo_in_heldout"]),
                                      "slo_first_in": float(default["slo_first_in_heldout"])}
    return section


def payload(spec, sections: Dict[str, Dict[str, object]], score: str) -> Dict[str, object]:
    """The whole ``best.json``: the chosen values plus the provenance of the selection."""
    body: Dict[str, object] = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/sim/tune_baselines.py",
        "objective": f"mean {score} over the tuning seeds (lower is better)",
        "slice": spec.to_dict(), "tuning_seeds": list(spec.tuning_seeds),
        "heldout_seed": spec.heldout_seed, POLICIES_KEY: sections}
    if getattr(spec, "slo_select", "cost") == "resource":
        body["objective"] = ("mean run_cost + switch_cost over the tuning seeds among the cells "
                             "whose mean slo_in reaches slo_floor (lower is better)")
        body["slo_select"] = "resource"
    return body


def configs_of(best: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """``{policy: config}`` of a ``best.json`` payload."""
    sections = best.get(POLICIES_KEY, {})
    return {name: dict((section or {}).get(CONFIG_KEY, {}))
            for name, section in sections.items()}  # type: ignore[union-attr]


def changed_policies(old: Mapping[str, object], new: Mapping[str, object]) -> List[str]:
    """Policies whose chosen configuration differs between two ``best.json`` payloads."""
    a, b = configs_of(old), configs_of(new)
    return sorted(name for name in set(a) | set(b) if a.get(name) != b.get(name))


__all__ = ["RESOURCE_COST", "RESOURCE_KEYS", "SLO_SELECT", "changed_policies", "chosen",
           "configs_of", "payload", "pick_cost", "pick_resource", "plain", "resource_columns"]
