"""Q1 orchestrator: are education arrivals sharper and more calendar-predictable?

For every configured source this computes, on the same definitions:

* per-burst shape -- 10-90% ramp time, synchronization fraction sigma_k, peak arrivals per minute;
* whole-series peak-to-mean at a fine and a coarse bin;
* forecasting errors of a history-only predictor against a calendar-informed one.

Outputs land in ``out_dir``: ``<source>_bursts.csv`` per source, ``summary.csv``, ``summary.md``
and the three figures. Nothing that identifies a student is written; group labels are class or
trace labels only.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import figures
from .config import AnalysisConfig, FigureConfig, PredictConfig, SessionDetectConfig
from .metrics.burst import minute_counts, peak_to_mean, per_session_bursts
from .metrics.predict import evaluate_forecasts, to_series
from .metrics.sessions import gap_sessions, onset_times
from .sources.base import ArrivalSource

logger = logging.getLogger(__name__)

_MD_COLUMNS = (
    ("source", "Source", "{}"),
    ("workload_class", "Class", "{}"),
    ("timetable", "Sessions from", "{}"),
    ("n_arrivals", "Arrivals", "{:,.0f}"),
    ("span_days", "Span (d)", "{:.1f}"),
    ("n_bursts", "Bursts", "{:.0f}"),
    ("ramp_min_median", "Ramp 10-90% (min)", "{:.2f}"),
    ("sync_1min_median", "sigma_1", "{:.2f}"),
    ("sync_2min_median", "sigma_2", "{:.2f}"),
    ("sync_5min_median", "sigma_5", "{:.2f}"),
    ("p2m_1min", "P2M 1 min", "{:.1f}"),
    ("p2m_5min", "P2M 5 min", "{:.1f}"),
    ("mase_ewma", "MASE EWMA", "{:.3f}"),
    ("mase_calendar", "MASE cal.", "{:.3f}"),
    ("mase_calendar_first", "MASE cal. wk1", "{:.3f}"),
    ("window_gain", "Window MAE gain", "{:.1%}"),
)


def _detected_sessions(df: pd.DataFrame, detect: SessionDetectConfig, window_min: int,
                       bin_minutes: int) -> pd.DataFrame:
    """Threshold onsets as pseudo-sessions, for traces that have no timetable at all."""
    rows: List[Dict[str, object]] = []
    for group, sub in df.groupby("group", sort=True):
        for t0 in onset_times(sub["ts"], detect, bin_minutes, window_min):
            rows.append({"group": group, "start": t0, "end": t0 + pd.Timedelta(minutes=window_min),
                         "duration_min": float(window_min), "n_students": np.nan,
                         "n_events": np.nan, "triggers": [t0]})
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True) if rows else pd.DataFrame()


def sessions_for(src: ArrivalSource, df: pd.DataFrame, detect: SessionDetectConfig,
                 analysis: AnalysisConfig) -> Tuple[pd.DataFrame, str]:
    """Sessions of a source plus a label saying where they came from."""
    known = src.sessions()
    if known is not None and not known.empty:
        out = known.copy()
    elif src.workload_class == "generic":
        out = _detected_sessions(df, detect, analysis.burst_window_min, analysis.bin_minutes)
    else:
        out = gap_sessions(df, detect)
    if out.empty:
        logger.warning("%s: no sessions found", type(src).__name__)
        return out, src.timetable_kind
    if "triggers" not in out.columns:
        out["triggers"] = [[t] for t in pd.to_datetime(out["start"])]
    return out.reset_index(drop=True), src.timetable_kind


def _empty_bursts(ks: Sequence[int]) -> pd.DataFrame:
    cols = ["group", "start", "trigger", "t0", "n", "ramp_min", "peak_per_min"]
    return pd.DataFrame(columns=cols + [f"sync_{k}min" for k in ks])


def analyse_source(name: str, src: ArrivalSource, analysis: AnalysisConfig,
                   predict: PredictConfig, detect: SessionDetectConfig
                   ) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Burst, peak-to-mean and forecasting metrics for one source."""
    df = src.load()
    sessions, timetable = sessions_for(src, df, detect, analysis)
    if sessions.empty:
        bursts = _empty_bursts(analysis.sync_k_minutes)
    else:
        bursts = per_session_bursts(df, sessions, analysis.triggers_per_session,
                                    analysis.burst_window_min, analysis.sync_k_minutes)
    counts_fine = minute_counts(df["ts"], bin_minutes=analysis.bin_minutes)
    counts_coarse = minute_counts(df["ts"], bin_minutes=analysis.bin_minutes_coarse)
    windows = list(pd.to_datetime(bursts["t0"])) if not bursts.empty else []
    scores, _ = evaluate_forecasts(to_series(df, predict.bin_minutes), sessions, windows, predict,
                                   analysis.burst_window_min, arrivals=df)

    span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
    row: Dict[str, object] = {
        "source": name,
        "workload_class": src.workload_class,
        "timetable": timetable,
        "n_arrivals": int(len(df)),
        "span_days": float(span),
        "n_sessions": int(len(sessions)),
        "n_bursts": int(len(bursts)),
        "arrivals_per_burst_median": float(bursts["n"].median()) if not bursts.empty else np.nan,
        "ramp_min_median": float(bursts["ramp_min"].median()) if not bursts.empty else np.nan,
        "ramp_min_p90": float(bursts["ramp_min"].quantile(0.9)) if not bursts.empty else np.nan,
        "peak_per_min_median": float(bursts["peak_per_min"].median()) if not bursts.empty else np.nan,
        "p2m_1min": peak_to_mean(counts_fine),
        "p2m_5min": peak_to_mean(counts_coarse),
    }
    for k in analysis.sync_k_minutes:
        col = f"sync_{k}min"
        row[f"{col}_median"] = float(bursts[col].median()) if not bursts.empty else np.nan
    row.update(scores.as_dict())
    row["gate_ramp_pass"] = bool(row["ramp_min_median"] <= analysis.gate_ramp_min) \
        if np.isfinite(row["ramp_min_median"]) else False
    row["gate_predict_pass"] = bool(row["mase_calendar"] <= 1.0 - analysis.gate_error_reduction) \
        if np.isfinite(row["mase_calendar"]) else False
    logger.info("%s: %d arrivals, %d bursts, ramp median %.2f min, MASE calendar %.3f",
                name, row["n_arrivals"], row["n_bursts"], row["ramp_min_median"],
                row["mase_calendar"])
    return row, bursts


def _format(value: object, spec: str) -> str:
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def summary_markdown(summary: pd.DataFrame, analysis: AnalysisConfig) -> str:
    """Render the comparison table and the Q1 gate verdict as markdown."""
    head = "| " + " | ".join(label for _, label, _ in _MD_COLUMNS) + " |"
    rule = "|" + "|".join(["---"] * len(_MD_COLUMNS)) + "|"
    lines = [head, rule]
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(_format(r.get(col), spec) for col, _, spec in _MD_COLUMNS)
                     + " |")
    table = "\n".join(lines)

    verdicts = []
    for _, r in summary.iterrows():
        ramp = "pass" if r.get("gate_ramp_pass") else "fail"
        pred = "pass" if r.get("gate_predict_pass") else "fail"
        verdicts.append(f"- {r['source']} ({r['workload_class']}): ramp gate {ramp}, "
                        f"predictability gate {pred}")
    return (
        "# Q1 summary\n\n"
        f"Written by `pilot q1` on {datetime.now():%Y-%m-%d %H:%M}. One row per source; the full\n"
        "set of columns is in `summary.csv`.\n\n"
        f"{table}\n\n"
        "## Pre-registered Q1 gate\n\n"
        f"Ramp: median 10-90% ramp <= {analysis.gate_ramp_min:g} min. Predictability: the\n"
        f"calendar-informed predictor cuts the per-bin MAE by >= {analysis.gate_error_reduction:.0%}\n"
        "against the seasonal-naive benchmark (MASE <= "
        f"{1 - analysis.gate_error_reduction:.2f}).\n\n" + "\n".join(verdicts) + "\n\n"
        "## Reading the columns\n\n"
        f"- `sigma_k`: share of a burst's arrivals landing within k minutes of the trigger; the\n"
        f"  burst window is {analysis.burst_window_min} min.\n"
        "- `P2M`: peak-to-mean of the whole series at "
        f"{analysis.bin_minutes} min and {analysis.bin_minutes_coarse} min bins.\n"
        "- `MASE`: per-bin MAE at horizon "
        f"{analysis.forecast_horizon_min} min divided by the seasonal-naive MAE on the same bins,\n"
        "  so 1.0 means no better than history. `wk1` restricts scoring to the first week, where\n"
        "  no same-slot history exists yet.\n"
        "- `Window MAE gain`: error reduction of the calendar predictor on session-start windows.\n"
        "- `Sessions from`: `timetable` = known in advance; `inferred` = gap-clustered from the\n"
        "  log; `detected` = threshold onsets found in the same series. For `detected` sources no\n"
        "  schedule exists, so the calendar predictor is handed its own detected onsets and its\n"
        "  numbers are an optimistic upper bound, not an exogenous-prior result.\n"
    )


def run_q1(sources: Mapping[str, ArrivalSource], out_dir: Path,
           analysis: Optional[AnalysisConfig] = None, predict: Optional[PredictConfig] = None,
           detect: Optional[SessionDetectConfig] = None,
           fig_cfg: Optional[FigureConfig] = None) -> pd.DataFrame:
    """Run the Q1 comparison over ``sources`` and write every output into ``out_dir``."""
    analysis = analysis or AnalysisConfig()
    predict = predict or PredictConfig(bin_minutes=analysis.bin_minutes,
                                       horizon_min=analysis.forecast_horizon_min)
    detect = detect or SessionDetectConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    bursts_by_source: Dict[str, pd.DataFrame] = {}
    for name, src in sources.items():
        row, bursts = analyse_source(name, src, analysis, predict, detect)
        rows.append(row)
        bursts_by_source[name] = bursts
        bursts.to_csv(out_dir / f"{name}_bursts.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(summary_markdown(summary, analysis), encoding="utf-8")

    cfg = figures.apply_style(fig_cfg)
    figures.ramp_cdf(bursts_by_source, out_dir, cfg, analysis.gate_ramp_min)
    figures.sigma_hist(bursts_by_source, out_dir, cfg, k_minutes=analysis.sync_k_minutes[1]
                       if len(analysis.sync_k_minutes) > 1 else analysis.sync_k_minutes[0])
    figures.predictability(summary, out_dir, cfg, analysis.gate_error_reduction)
    logger.info("Q1 outputs written to %s", out_dir)
    return summary
