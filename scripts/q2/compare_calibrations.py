#!/usr/bin/env python3
"""Batch B3: the short-prompt calibration against the long-prompt T3, model by model.

Reads two T3 CSVs (batch B's long tutoring prompts, ~300 tokens median, and batch B3's classroom
prompts, ~45 tokens), plus batch B's long-prompt probe on the B3 card when present (which separates
the card's effect from the prompt length's), and prints, per model, the knee concurrency ``K`` at a 2 s TTFT p95, the
aggregate decode rate, the single-stream rate and the fluid capacity ``mu`` at the workload's mean
output length, plus the ratios.  Writes ``<out>/calibration_comparison.csv``.  Run from the repository root:

    uv run python scripts/q2/compare_calibrations.py \\
        --long scripts/q2/results/b1_remote/b1/T3_calibration.csv \\
        --short scripts/q2/results/b3_remote/b3/shortprompt/T3_calibration.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pilot.sim import ServiceConfig
from pilot.sim.config import with_lead_tier
from pilot.sim.service import load_service_config

logger = logging.getLogger("compare_calibrations")
PILOT = Path(__file__).resolve().parents[2]
MEAN_OUTPUT_TOKENS = 211.0  # Azure conversation trace, the workload's output length


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--long", type=Path, default=PILOT / "scripts/q2/results/b1_remote/b1/T3_calibration.csv")
    ap.add_argument("--short", type=Path, default=PILOT / "scripts/q2/results/b3_remote/b3/shortprompt/T3_calibration.csv")
    ap.add_argument("--probe", type=Path, default=PILOT / "scripts/q2/results/b2_remote/b2/T3_calibration.csv",
                    help="batch B' long-prompt probe on the same card as --short (separates card from prompt length)")
    ap.add_argument("--out", type=Path, default=PILOT / "scripts/q2/results/b3_remote/b3")
    ap.add_argument("--mean-output-tokens", type=float, default=MEAN_OUTPUT_TOKENS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    rows: List[Dict[str, object]] = []
    models = sorted(set(pd.read_csv(args.short)["model"]) & set(pd.read_csv(args.long)["model"]))
    base = with_lead_tier(ServiceConfig(), "d_rep")
    for model in models:
        rec: Dict[str, object] = {"model": model.split("/")[-1]}
        sources = [("long", args.long), ("short", args.short)]
        if args.probe.exists() and model in set(pd.read_csv(args.probe)["model"]):
            sources.insert(1, ("probe", args.probe))
        for tag, path in sources:
            svc = load_service_config(path, model, base=base, lead_stat="median")
            rec[f"K_{tag}"] = svc.max_concurrency
            rec[f"decode_tok_s_{tag}"] = svc.decode_tokens_per_s
            rec[f"per_stream_tok_s_{tag}"] = svc.per_stream_tokens_per_s
            rec[f"ttft_floor_s_{tag}"] = svc.ttft_floor_s
            rec[f"mu_req_min_{tag}"] = svc.requests_per_min(args.mean_output_tokens)
        for key in ("K", "decode_tok_s", "per_stream_tok_s", "mu_req_min"):
            rec[f"{key}_ratio_short_over_long"] = float(rec[f"{key}_short"]) / max(1e-9, float(rec[f"{key}_long"]))
            if f"{key}_probe" in rec:  # card effect (probe / long) and prompt-length effect (short / probe)
                rec[f"{key}_ratio_probe_over_long"] = float(rec[f"{key}_probe"]) / max(1e-9, float(rec[f"{key}_long"]))
                rec[f"{key}_ratio_short_over_probe"] = float(rec[f"{key}_short"]) / max(1e-9, float(rec[f"{key}_probe"]))
        rows.append(rec)
    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "calibration_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    logger.info("wrote %s\n%s", args.out / "calibration_comparison.csv", frame.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
