#!/usr/bin/env python3
"""Score the replica pool's real-machine rounds against the simulator (B' step 2).

    uv run python scripts/q2/pool_compare.py --results scripts/q2/results/b5_remote/pool

Three views of every round under ``<results>/runs/<label>/``: measured (the client's
``<label>.jsonl``, TTFT from the scheduled arrival, NaN = gave up; the pool's ``pool_minutes.csv``
and ``replica_<id>_server.json``), sim A (``pool_inputs/sim_predictions.json``: the same policy
code on the same window inside the simulator) and sim B (the pool's realised admitted trajectory
through ``pilot.sim.service.replay`` on the window rebuilt from the client's records).  Measured
TTFTs are also scored under the simulator's abandonment rule, as ``replay_compare.py`` does.  The
feedback policies are re-run with ``pilot.sim.policies`` on the pool's own per-minute observations
and compared with the targets it recorded; the vendored MANIFEST sha256 is checked against the
sources.  Writes ``pool_compare.csv``, ``pool_minutes_compare.csv`` and
``f10b_pool_validation.{pdf,png}`` under ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

PILOT = Path(__file__).resolve().parents[2]
for _extra in (PILOT / "src", PILOT / "scripts" / "sim", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from bell_window import (SLO_S, context_of, measured_minutes, minute_table,  # noqa: E402
                         stream_from_records)
from pilot.config import FigureConfig  # noqa: E402
from pilot.figures import apply_style, save  # noqa: E402
from pilot.sim import PolicyFactory  # noqa: E402
from pilot.sim.policies.base import SimState, as_int_replicas  # noqa: E402
from pilot.sim.service import replay  # noqa: E402

logger = logging.getLogger("pool_compare")

SIM_SRC = PILOT / "src" / "pilot" / "sim"
VENDORED = ("config.py", "base.py", "reactive.py", "velocity.py")  # what vendor_policies.py copies
FEEDBACK = ("velocity", "reactive")  # the policies the pool runs live (the rest execute a plan)
FAMILIES = ("plan_learnprior", "plan_offline", "velocity", "reactive")  # one colour each
FAMILY_LABELS = {"plan_learnprior": "Timetable plan", "plan_offline": "Offline plan",
                 "velocity": "Velocity", "reactive": "Reactive"}
TIER_LABELS = {"d_rep": "replica tier", "d_inst": "instance tier"}
# pool_minutes.csv (pool_control.MINUTE_COLUMNS): what is read, and the design 5.3 spellings
MINUTE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "t": ("t", "minute"), "arrivals": ("arrivals",), "tokens": ("tokens",),
    "admitted": ("admitted",), "target": ("target_s", "target"), "launched": ("launched_ids",)}
CM_PER_IN, FONT_PT, PRINT_DPI, PAD_IN = 2.54, 8.0, 600, 0.012  # printed at 8 pt, 600 dpi
F10B_SIZE_CM: Tuple[float, float] = (8.8, 7.6)  # one column wide


@dataclass(frozen=True)
class RunFiles:
    """One round's ``<label>.jsonl``, ``pool_minutes.csv``, ``pool_meta.json``, replica records."""

    label: str
    records: Path
    minutes: Path
    meta: Path
    servers: Tuple[Path, ...]


def discover_runs(results: Path) -> List[RunFiles]:
    """Every ``runs/<label>/`` that holds the client's records and the pool's minute log."""
    found: List[RunFiles] = []
    for d in sorted((results / "runs").glob("*/")):
        files = RunFiles(d.name, d / f"{d.name}.jsonl", d / "pool_minutes.csv", d / "pool_meta.json",
                         tuple(sorted(d.glob("replica_*_server.json"))))
        if files.records.exists() and files.minutes.exists():
            found.append(files)
        else:
            logger.warning("%s: no %s or pool_minutes.csv; skipped", d, files.records.name)
    return found


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_records(path: Path) -> pd.DataFrame:
    """Client records sorted by ``idx``; ``ttft_s`` NaN where the client gave up."""
    rec = pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                        if line]).sort_values("idx").reset_index(drop=True)
    return rec.assign(ttft_s=pd.to_numeric(rec["ttft_s"], errors="coerce"))


def _count_ids(text: str) -> int:
    """Ids in a cell written as JSON, ``1;2`` or ``1,2``."""
    parts = text.strip().strip("[]").replace(";", ",").split(",")
    return sum(1 for part in parts if part.strip() not in ("", "nan"))


def load_minutes(path: Path, duration: int) -> pd.DataFrame:
    """``pool_minutes.csv`` reduced to the columns read here, for ``t < duration``; raises with the
    missing names when the box wrote other columns (``target`` and ``launched`` are optional)."""
    raw = pd.read_csv(path)
    found = {key: next((n for n in names if n in raw.columns), None)
             for key, names in MINUTE_COLUMNS.items()}
    missing = [f"{k} (one of {MINUTE_COLUMNS[k]})" for k, c in found.items()
               if c is None and k not in ("target", "launched")]
    if missing:
        raise ValueError(f"{path}: pool_minutes.csv lacks {missing}; columns are {list(raw.columns)}")
    out = pd.DataFrame({k: pd.to_numeric(raw[c], errors="coerce")
                        for k, c in found.items() if c and k != "launched"})
    out["target"] = out["target"] if "target" in out else np.nan
    launched = found["launched"]
    out["launched"] = raw[launched].fillna("").astype(str).map(_count_ids) if launched else 0
    out = out[(out["t"] >= 0) & (out["t"] < duration)].sort_values("t")
    return out.drop_duplicates("t", keep="last").reset_index(drop=True)


def _seconds(server: Dict[str, object], key: str, start: str, end: str) -> Optional[float]:
    """``server[key]`` (pool_replicas.Replica.record) or ``end - start`` from the wall clocks."""
    if isinstance(server.get(key), (int, float)):
        return float(server[key])  # type: ignore[arg-type]
    clocks = [next((server[k] for k in (f"{m}_wallclock", f"{m}_ts", m)
                    if isinstance(server.get(k), (int, float))), None) for m in (start, end)]
    return None if None in clocks else float(clocks[1]) - float(clocks[0])  # type: ignore[arg-type]


def cold_starts(files: RunFiles) -> Dict[str, object]:
    """Cold-start seconds (launch to ready) and gate waits (ready to admit) over the replicas."""
    starts, waits = [], []
    for path in files.servers:
        server = _read_json(path)
        cold = _seconds(server, "ready_after_s", "launch", "ready")
        wait = _seconds(server, "gate_wait_s", "ready", "admit")
        if cold is not None:
            starts.append(cold)
        if wait is not None:
            waits.append(max(0.0, wait))
    nan = float("nan")
    return {"cold_start_n": len(starts), "gate_wait_median_s": float(np.median(waits)) if waits else nan,
            "cold_start_median_s": float(np.median(starts)) if starts else nan,
            "cold_start_max_s": float(np.max(starts)) if starts else nan}


def recompute_targets(policy_name: str, ctx: Dict[str, object], minutes: pd.DataFrame) -> np.ndarray:
    """The targets the policy code gives on the pool's own observations (design 5.3 steps 2-3)."""
    pctx = context_of(ctx)
    t_idx = minutes["t"].to_numpy(dtype=int)
    horizon = max(pctx.horizon_min, int(t_idx.max()) + 1 if t_idx.size else 0)
    arrivals, tokens = np.zeros(horizon), np.zeros(horizon)
    arrivals[t_idx] = minutes["arrivals"].to_numpy(dtype=float)
    tokens[t_idx] = minutes["tokens"].to_numpy(dtype=float)
    policy = PolicyFactory(policy_name)(pctx)  # __init__ resets; start_run as engine.trajectory
    policy.start_run()
    state = SimState(arrivals, np.zeros(horizon), tokens)
    out = np.zeros(t_idx.size)
    for i, (t, admitted) in enumerate(zip(t_idx, minutes["admitted"].to_numpy(dtype=float))):
        state.t, state.x_now = int(t), float(admitted)
        out[i] = as_int_replicas(policy.decide(int(t), state), pctx.cost,
                                 floor=pctx.policy.min_replicas)
    return out


def manifest_check(meta: Dict[str, object]) -> Tuple[Optional[bool], List[str]]:
    """Vendored MANIFEST sha256 against the current ``src/pilot/sim`` sources: whether every
    vendored source still matches and the names that do not; ``(None, [])`` without a manifest."""
    manifest = next((v for k, v in meta.items() if "manifest" in k.lower()), None)
    if not isinstance(manifest, dict):
        return None, []
    table = manifest.get("sources") or manifest.get("generated") or manifest.get("files") or manifest
    entries: Dict[str, str] = {}
    for key, value in (table.items() if isinstance(table, dict) else []):
        if isinstance(value, str) and len(value) == 64:
            entries[str(key)] = value
        elif isinstance(value, dict):
            sha = value.get("source_sha256") or value.get("sha256")
            if isinstance(sha, str):
                entries[str(value.get("path") or value.get("source") or key)] = sha
    bad: List[str] = []
    checked = 0
    for name, sha in entries.items():
        base = Path(name).name
        if base in VENDORED:
            source = SIM_SRC / base if base == "config.py" else SIM_SRC / "policies" / base
            checked += 1
            if hashlib.sha256(source.read_bytes()).hexdigest() != sha.lower():
                bad.append(name)
    return (not bad) if checked else None, bad


def compare_run(files: RunFiles, spec: Dict[str, object], prediction: Optional[Dict[str, object]],
                ctx: Dict[str, object], duration: int, horizon: int, minute_s: float = 60.0
                ) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Measured, sim A and sim B for one round (summary row, per-minute table); the records are
    binned by the pool's ``minute_s`` (6 s in a dry run) and stretched onto 60 s minutes for sim B."""
    rec = load_records(files.records)
    minutes = load_minutes(files.minutes, duration)
    if minutes.empty or rec.empty:
        raise ValueError(f"{files.label}: no minute rows below {duration} or no client records")
    service = context_of(ctx).service
    table = measured_minutes(rec, duration, service.abandon_in_s, minute_s=minute_s).merge(
        minutes, on="t", how="left")
    # sim B: the realised admitted trajectory through the simulator's service model
    admitted = minutes.set_index("t")["admitted"].reindex(range(duration)).ffill().bfill()
    x_real = np.pad(admitted.to_numpy(dtype=float), (0, horizon - duration), mode="edge")
    stream = stream_from_records(rec.assign(offset_s=rec["offset_s"] * (60.0 / minute_s)), horizon)
    result = replay(stream, x_real, service)
    sim_b = minute_table(stream, result, duration)
    table["admitted_meas"] = admitted.to_numpy()
    table["share_simB"], table["p95_simB"] = sim_b["share_slo"], sim_b["ttft_p95_s"]
    table["timeouts_simB"] = sim_b["timeouts"]
    # sim A: the policy's own run in the simulator
    keys = {"x_simA": "x", "arrivals_sim": "arrivals", "tokens_sim": "tokens",
            "share_simA": "share_slo", "p95_simA": "ttft_p95_s"}
    for col, key in keys.items():
        values = np.asarray((prediction or {}).get(key, []), dtype=float)[:duration]
        table[col] = np.pad(values, (0, duration - values.size), mode="edge") if values.size else np.nan
    if prediction is None:
        logger.warning("%s: no entry in sim_predictions.json; sim A left empty", files.label)
    policy = str(spec.get("policy", ""))
    table["target_recomputed"] = np.nan
    checked = mismatched = 0
    if policy in FEEDBACK:
        recomputed = recompute_targets(policy, ctx, minutes)
        table.loc[table["t"].isin(minutes["t"]), "target_recomputed"] = recomputed
        have = minutes["target"].notna().to_numpy()
        if have.any():
            checked = int(have.sum())
            mismatched = int((minutes["target"].to_numpy(dtype=float)[have] != recomputed[have]).sum())
            if mismatched:
                logger.warning("%s: %d/%d recorded targets differ from the policy code's",
                               files.label, mismatched, checked)
        else:
            logger.warning("%s: pool_minutes.csv has no target column; targets not checked",
                           files.label)
    ok, bad = manifest_check(_read_json(files.meta))
    if ok is False:
        logger.warning("%s: vendored sources changed since MANIFEST: %s", files.label, bad)
    ttft = rec["ttft_s"].to_numpy(dtype=float)
    finite, aligned = ttft[np.isfinite(ttft)], ttft[ttft <= service.abandon_in_s]
    b_ok, pred = np.asarray(result.served) & (result.ttft_s <= SLO_S), prediction or {}
    b_finite = result.ttft_s[np.asarray(result.served)]
    row: Dict[str, object] = {
        "label": files.label, "policy": policy, "tier": str(spec.get("tier", "")),
        "lead_gate_min": spec.get("lead_gate_min"), "n_requests": int(len(rec)),
        "dry_run": bool(rec["dry_run"].any()) if "dry_run" in rec else False,
        "meas_share_slo": float(np.mean(np.nan_to_num(ttft, nan=np.inf) <= SLO_S)),
        "meas_p95_s": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "meas_p95_aligned_s": float(np.percentile(aligned, 95)) if aligned.size else float("nan"),
        "meas_gave_up": int((~np.isfinite(ttft)).sum()),
        "meas_over_abandon": int((finite > service.abandon_in_s).sum()),
        "meas_replica_minutes": float(admitted.sum()), "meas_peak_admitted": float(admitted.max()),
        "meas_starts_after_t0": float(minutes["launched"].sum()), **cold_starts(files),
        "simA_share_slo": pred.get("slo_in"), "simA_p95_s": pred.get("ttft_p95_in_s"),
        "simA_replica_minutes": pred.get("replica_minutes"),
        "simA_starts_after_t0": pred.get("starts_after_t0"),
        "simA_peak_x": float(table["x_simA"].max()) if prediction else None,
        "simB_share_slo": float(b_ok.mean()) if len(rec) else float("nan"),
        "simB_p95_s": float(np.percentile(b_finite, 95)) if b_finite.size else float("nan"),
        "simB_timeouts": int(np.asarray(result.timed_out).sum()),
        "x_abs_diff_minutes": float(np.abs(table["admitted_meas"] - table["x_simA"]).sum()),
        "x_equal_minutes": int((table["admitted_meas"] == table["x_simA"]).sum()),
        "targets_checked": checked, "targets_mismatched": mismatched,
        "manifest_ok": ok, "manifest_mismatched": ";".join(bad)}
    table.insert(0, "label", files.label)
    return row, table


def printed_style(size_cm: Sequence[float], font_pt: float = FONT_PT) -> FigureConfig:
    """House style at the printed size; ``savefig.bbox`` standard keeps the page at ``size_cm``."""
    cfg = apply_style(FigureConfig(width_in=size_cm[0] / CM_PER_IN, height_in=size_cm[1] / CM_PER_IN,
                                   base_font_pt=font_pt, small_font_pt=font_pt, dpi=PRINT_DPI))
    plt.rcParams["savefig.bbox"] = "standard"
    return cfg


def figure(table: pd.DataFrame, minutes: pd.DataFrame, out: Path, bell_offset_min: int = 18,
           size_cm: Sequence[float] = F10B_SIZE_CM, font_pt: float = FONT_PT) -> None:
    """F10b: a column per tier, replicas above (admitted vs sim A) and share within 2 s below
    (measured, sim A, sim B); colour is the policy, line style the source; no titles."""
    cfg = printed_style(size_cm, font_pt)
    tiers = [t for t in TIER_LABELS if (table["tier"] == t).any()] or [""]
    fig, axes = plt.subplots(2, len(tiers), figsize=(cfg.width_in, cfg.height_in), sharex=True,
                             sharey="row", squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=PAD_IN, h_pad=PAD_IN, wspace=0.02, hspace=0.02)
    seen: Dict[int, str] = {}  # palette slot -> legend text; other labels get slots after the four
    extra: Dict[str, int] = {}
    for j, tier in enumerate(tiers):
        ax_x, ax_s = axes[0, j], axes[1, j]
        rows = table[(table["tier"] == tier) | (tier == "")]
        for label, policy in zip(rows["label"], rows["policy"].astype(str)):
            part = minutes[minutes["label"] == label].sort_values("t")
            family = next((f for f in FAMILIES if f in label), policy if policy in FAMILIES else label)
            i = (FAMILIES.index(family) if family in FAMILIES
                 else extra.setdefault(family, len(FAMILIES) + len(extra)))
            seen[i] = FAMILY_LABELS.get(family, family)
            t = part["t"].to_numpy(dtype=float) - bell_offset_min
            color = cfg.palette[i % len(cfg.palette)]
            ax_x.step(t, part["admitted_meas"], where="post", color=color, linewidth=1.0)
            ax_x.step(t, part["x_simA"], where="post", color=color, linewidth=0.9, linestyle="--")
            ax_s.plot(t, part["share_meas"], color=color, linewidth=1.0, markersize=1.8,
                      marker=cfg.markers[i % len(cfg.markers)])
            ax_s.plot(t, part["share_simA"], color=color, linewidth=0.9, linestyle="--")
            ax_s.plot(t, part["share_simB"], color=color, linewidth=0.9, linestyle=":")
        ax_s.set_xlabel("Minutes from the bell")
        if tier:  # a note in the clear space before the bell, not a title
            ax_x.text(0.03, 0.96, TIER_LABELS[tier], transform=ax_x.transAxes, va="top",
                      ha="left", fontsize=font_pt, color="0.35")
        for ax in (ax_x, ax_s):
            ax.axvline(0, color="0.6", linewidth=0.7)
            ax.tick_params(length=2.0, width=0.6, pad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
    axes[0, 0].set_ylabel("Replicas")
    axes[0, 0].set_ylim(0, max(1.0, float(np.nanmax(minutes[["admitted_meas", "x_simA"]].to_numpy(
        dtype=float)))) * 1.15)
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(1))
    axes[1, 0].set_ylabel("Share within 2 s")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].yaxis.set_major_locator(MultipleLocator(0.5))
    axes[1, 0].xaxis.set_major_locator(MultipleLocator(10))
    policies = [Line2D([], [], color=cfg.palette[i % len(cfg.palette)], linewidth=1.0, label=name)
                for i, name in sorted(seen.items())]
    sources = [Line2D([], [], color="0.3", linewidth=1.0, label="measured"),
               Line2D([], [], color="0.3", linewidth=0.9, linestyle="--", label="sim. A (policy)"),
               Line2D([], [], color="0.3", linewidth=0.9, linestyle=":", label="sim. B (replicas)")]
    # as F10: the policies (colour) above the panels, the sources (line style) below them
    for handles, loc in ((policies, "outside upper center"), (sources, "outside lower center")):
        fig.legend(handles=handles, loc=loc, ncol=len(handles), frameon=False, fontsize=font_pt,
                   handlelength=1.3, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0)
    save(fig, out, "f10b_pool_validation")


def compare_all(results: Path, inputs: Path, out: Path, runs_file: str = "pool_runs.json"
                ) -> pd.DataFrame:
    """Every round under ``results`` against ``inputs``; writes the two CSVs and the figure."""
    runs_json = _read_json(inputs / runs_file)
    predictions = _read_json(inputs / "sim_predictions.json")
    trace_meta = _read_json(inputs / Path(str(runs_json.get("trace", ""))).with_suffix(".json").name)
    duration, minute_s = int(runs_json.get("duration_min", 45)), float(runs_json.get("minute_s", 60))
    horizon = int(trace_meta.get("horizon_min", duration + 10))  # type: ignore[arg-type]
    bell_offset = -int(trace_meta.get("window_from_bell_min", [0])[0])  # type: ignore[index]
    specs = {str(r["label"]): r for r in runs_json.get("runs", [])}  # type: ignore[union-attr]
    rows, tables = [], []
    for files in discover_runs(results):
        pred = predictions.get(files.label) or {}
        meta = _read_json(files.meta)  # a round missing from pool_runs.json (dry run) is read from it
        spec = {"policy": pred.get("policy") or str(meta.get("policy", "")).split(":")[0],
                "tier": pred.get("tier", ""), "lead_gate_min": meta.get("lead_gate_min"),
                **specs.get(files.label, {})}  # type: ignore[dict-item]
        ctx = (_read_json(inputs.parent / str(spec["context"])) if spec.get("context")
               else meta.get("context") or {})
        if not ctx:
            raise FileNotFoundError(f"{files.label}: no context in pool_runs.json or pool_meta.json")
        row, table = compare_run(files, spec, pred or None, ctx, duration, horizon, minute_s)
        rows.append(row)
        tables.append(table)
        logger.info("%-24s share within 2 s: measured %.3f | sim A %s | sim B %.3f; replica-minutes "
                    "measured %.0f | sim A %s; x equal on %d/%d minutes; targets %d/%d mismatched",
                    row["label"], row["meas_share_slo"], row["simA_share_slo"],
                    row["simB_share_slo"], row["meas_replica_minutes"], row["simA_replica_minutes"],
                    row["x_equal_minutes"], duration, row["targets_mismatched"], row["targets_checked"])
    if not rows:
        raise SystemExit(f"no rounds with records under {results / 'runs'}")
    out.mkdir(parents=True, exist_ok=True)
    summary, per_minute = pd.DataFrame(rows), pd.concat(tables, ignore_index=True)
    summary.to_csv(out / "pool_compare.csv", index=False)
    per_minute.to_csv(out / "pool_minutes_compare.csv", index=False)
    figure(summary, per_minute, out, bell_offset)
    logger.info("wrote %s", out)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=PILOT / "scripts/q2/results/b5_remote/pool")
    ap.add_argument("--inputs", type=Path, default=PILOT / "scripts/q2/pool_inputs")
    ap.add_argument("--out", type=Path, default=PILOT / "results/q2/pool")
    ap.add_argument("--runs-file", default="pool_runs.json", help="pool_runs_dry.json for a dry run")
    ap.add_argument("--draw-only", action="store_true", help="redraw the figure from the CSVs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if args.draw_only:
        figure(pd.read_csv(args.out / "pool_compare.csv"),
               pd.read_csv(args.out / "pool_minutes_compare.csv"), args.out)
    else:
        compare_all(args.results, args.inputs, args.out, args.runs_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
