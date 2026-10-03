"""Printable tables of a tuning run (moved out of ``tune_baselines.py`` unchanged)."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

TOP_ROWS = 6  # how many configurations the printed table shows per policy


def table(frame: pd.DataFrame, policy: str, spec, knobs: Sequence[str], rows: int = TOP_ROWS) -> str:
    """Printable table: the best configurations plus the default, wherever it ranked."""
    cols = ["rank", *knobs, "cost_ratio_tune_mean", "cost_ratio_heldout",
            "slo_in_heldout", "slo_first_in_heldout", "is_default"]
    shown = frame.head(rows)
    default_rows = frame[frame["is_default"]]
    if len(default_rows) and not bool(shown["is_default"].any()):
        shown = pd.concat([shown, default_rows])
    body = shown[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    seeds = "+".join(str(s) for s in spec.tuning_seeds)
    return (f"\n[{policy}] top {rows} of {len(frame)} configurations "
            f"(selected on seeds {seeds}, reported on held-out seed {spec.heldout_seed})\n{body}")


def before_after(sections: Dict[str, Dict[str, object]]) -> str:
    """The one table the report needs: default against tuned on the held-out seed."""
    lines = ["\nbefore / after on the held-out seed",
             f"{'policy':<10}{'ratio before':>14}{'ratio after':>13}{'change':>9}"
             f"{'slo_in before':>15}{'slo_in after':>14}"]
    for name, section in sections.items():
        before = section.get("default_heldout", {})
        after = section["heldout"]
        if not before:
            continue
        gain = 1.0 - float(after["cost_ratio"]) / float(before["cost_ratio"])
        lines.append(f"{name:<10}{float(before['cost_ratio']):>14.3f}"
                     f"{float(after['cost_ratio']):>13.3f}{gain:>8.1%}"
                     f"{float(before['slo_in']):>15.4f}{float(after['slo_in']):>14.4f}")
    return "\n".join(lines)


__all__ = ["TOP_ROWS", "before_after", "table"]
