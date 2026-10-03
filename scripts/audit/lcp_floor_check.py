#!/usr/bin/env python3
"""Check that the lazy rule clamped at the always-on replica is the lazy rule on {1..X}.

The implementation computes the lazy capacity provisioning bounds on {0..X} and clamps its output
at one replica (``Policy.clip``); Theorem 1 is stated on {1..X}.  This script computes the exact
LCP trajectory (the recurrences of ``pilot.sim.policies.lcp``) on the realised demand of a central
run both ways and prints how many minutes differ.  Run from the repository root::

    uv run python scripts/audit/lcp_floor_check.py c2 10 6    # case, seed, weeks
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import prefix_invariance as pi  # noqa: E402
import run_sweep  # noqa: E402
from pilot.sim.engine import build_stream, make_context  # noqa: E402
from pilot.sim.config import CostConfig  # noqa: E402
from pilot.sim.policies.base import hit_cost  # noqa: E402


def lcp(lam_in: np.ndarray, lam_off: np.ndarray, mu: float, cost: CostConfig, lo_state: int,
        n_x: int) -> np.ndarray:
    """Exact LCP(0) on states {lo_state..n_x-1}, x_0 = 0."""
    st = np.arange(lo_state, n_x, dtype=float)
    beta = cost.beta_start
    g = h = None
    x = 0.0
    out = np.zeros(lam_in.size)
    for t in range(lam_in.size):
        f = hit_cost(st, float(lam_in[t]), float(lam_off[t]), mu, cost)
        if g is None:
            g = f + beta * st
            h = f.copy()
        else:
            up = np.minimum.accumulate(g - beta * st) + beta * st
            down = np.minimum.accumulate(g[::-1])[::-1]
            g = f + np.minimum(up, down)
            keep = np.minimum.accumulate(h)
            shed = np.minimum.accumulate((h + beta * st)[::-1])[::-1] - beta * st
            h = f + np.minimum(keep, shed)
        xl = st[int(np.argmin(g))]
        xu = st[h.size - 1 - int(np.argmin(h[::-1]))]
        x = float(np.clip(x, xl, max(xl, xu)))
        out[t] = x
    return out


def main() -> None:
    """Parse ``case seed weeks`` from argv and print the comparison."""
    case = sys.argv[1] if len(sys.argv) > 1 else "c2"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    weeks = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    args = pi.sweep_args(case)
    args[args.index("--weeks") + 1] = str(weeks)
    cfg = run_sweep.base_config(run_sweep.build_parser().parse_args(args))
    cfg = replace(cfg, workload=replace(cfg.workload, **pi.CELL_C), seed=seed)
    stream = build_stream(cfg)
    ctx = make_context(cfg, stream)
    top = int(np.ceil((stream.lam_in + stream.lam_off).max() / ctx.mu)) + 2
    a = np.maximum(lcp(stream.lam_in, stream.lam_off, ctx.mu, ctx.cost, 0, top), 1.0)
    b = lcp(stream.lam_in, stream.lam_off, ctx.mu, ctx.cost, 1, top)
    diff = np.flatnonzero(a != b)
    print(f"{case} seed {seed} weeks {weeks}: {diff.size} of {a.size} minutes differ; "
          f"replica-minutes clamp {a.sum():.0f} vs floor-domain {b.sum():.0f}")
    if diff.size:
        print("  first differing minutes:", diff[:10].tolist(), "values", a[diff[:5]].tolist(), b[diff[:5]].tolist())


if __name__ == "__main__":
    main()
