#!/usr/bin/env python3
"""Q2 tier (ii): how long from "create instance" to "the replica can serve".

Two modes.

``on-instance`` is run **on the rented box** after each step is reached.  It
appends one JSON line with an ISO-8601 local timestamp to a log file and echoes
the timestamp, so the operator can paste it into their notes::

    python instance_acquire.py on-instance --step ssh-ready --trial t1 \\
        --log ~/q2_acquire_t1.jsonl --provider <provider> --gpu A100-80G

``report`` parses one or more such logs plus the **manually recorded** moment the
operator clicked "create" and computes the acquisition lead times::

    python instance_acquire.py report --logs ~/q2_acquire_*.jsonl \\
        --submitted t1=2026-09-20T09:03:12+08:00 --out results/b4

The manual part is unavoidable: no timestamp inside the instance can tell you
when the order was placed.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("q2.instance")

STEPS = ("boot", "ssh-ready", "deps-ready", "weights-present", "server-ready")


@dataclass(frozen=True)
class AcquireConfig:
    steps: tuple[str, ...] = STEPS
    out_dir: Path = Path("results")
    trials: int = 3
    seed: int = 20260917


# Dry-run only: plausible seconds between consecutive milestones on a rented GPU
# box whose image already carries CUDA 12.x, weights pulled to local NVMe.
_DRY_GAPS: dict[str, tuple[float, float]] = {
    "boot": (35.0, 95.0),
    "ssh-ready": (15.0, 45.0),
    "deps-ready": (70.0, 240.0),
    "weights-present": (150.0, 620.0),
    "server-ready": (30.0, 75.0),
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_event(
    log_path: Path, step: str, trial: str, iso: str | None = None, **meta: Any
) -> dict[str, Any]:
    """Append one milestone record to ``log_path`` and return it."""
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r}; expected one of {STEPS}")
    record = {
        "schema": "q2.instance.event.v1",
        "trial": trial,
        "step": step,
        "iso": iso or _now_iso(),
        **{k: v for k, v in meta.items() if v not in (None, "")},
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _parse_iso(text: str) -> datetime:
    """Parse an ISO-8601 timestamp; a naive one (e.g. from `uptime -s`) gets the local zone."""
    try:
        stamp = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise ValueError(f"not an ISO-8601 timestamp: {text!r}") from exc
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    return stamp


def load_events(patterns: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            logger.warning("no log file matched %s", pattern)
        for path in matches:
            with open(path, encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("skipping %s:%d (not JSON)", path, line_no)
                        continue
                    obj["source_log"] = path
                    events.append(obj)
    return events


def load_submitted(pairs: list[str], csv_path: Path | None) -> dict[str, dict[str, Any]]:
    """Manually recorded 'create clicked' times, as TRIAL=ISO pairs and/or a CSV."""
    out: dict[str, dict[str, Any]] = {}
    if csv_path is not None:
        frame = pd.read_csv(csv_path)
        for _, row in frame.iterrows():
            trial = str(row["trial"])
            out[trial] = {
                "submitted": _parse_iso(str(row["submitted_iso"])),
                "meta": {
                    k: row[k] for k in ("provider", "gpu", "region", "note")
                    if k in frame.columns and pd.notna(row[k])
                },
            }
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--submitted expects TRIAL=ISO, got {pair!r}")
        trial, iso = pair.split("=", 1)
        out.setdefault(trial, {"meta": {}})["submitted"] = _parse_iso(iso)
    return out


def build_trials(
    events: list[dict[str, Any]], submitted: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One record per trial: absolute milestones, lead times from submit, stage gaps."""
    trials: list[dict[str, Any]] = []
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_trial.setdefault(str(ev.get("trial", "unknown")), []).append(ev)
    for trial, evs in sorted(by_trial.items()):
        info = submitted.get(trial)
        if info is None or "submitted" not in info:
            logger.warning("trial %s has no manually recorded submit time; skipped", trial)
            continue
        t_submit: datetime = info["submitted"]
        rec: dict[str, Any] = {
            "schema": "q2.instance.trial.v1",
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trial": trial,
            "submitted_iso": t_submit.isoformat(timespec="seconds"),
            "dry_run": any(bool(e.get("dry_run")) for e in evs),
        }
        for key in ("provider", "gpu", "region", "note"):
            value = next((e[key] for e in evs if e.get(key)), info.get("meta", {}).get(key))
            if value is not None:
                rec[key] = value
        marks: dict[str, datetime] = {}
        for step in STEPS:
            hits = [e for e in evs if e.get("step") == step]
            if not hits:
                rec[f"{_key(step)}_iso"] = None
                rec[f"request_to_{_key(step)}_s"] = None
                continue
            stamp = min(_parse_iso(str(e["iso"])) for e in hits)
            marks[step] = stamp
            rec[f"{_key(step)}_iso"] = stamp.isoformat(timespec="seconds")
            rec[f"request_to_{_key(step)}_s"] = (stamp - t_submit).total_seconds()
        prev_step, prev_time = "submitted", t_submit
        for step in STEPS:
            if step in marks:
                gap = (marks[step] - prev_time).total_seconds()
                rec[f"gap_{_key(prev_step)}_to_{_key(step)}_s"] = gap
                prev_step, prev_time = step, marks[step]
        rec["missing_steps"] = [s for s in STEPS if s not in marks]
        trials.append(rec)
    return trials


def _key(step: str) -> str:
    return step.replace("-", "_")


def synth_trials(cfg: AcquireConfig, out_dir: Path) -> tuple[list[str], Path]:
    """Dry run: write plausible per-trial logs plus the submitted-times CSV."""
    rng = np.random.default_rng(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    rows: list[dict[str, Any]] = []
    base = datetime.now().astimezone().replace(microsecond=0)
    for idx in range(cfg.trials):
        trial = f"dry{idx + 1}"
        log_path = out_dir / f"acquire_{trial}.jsonl"
        log_path.unlink(missing_ok=True)
        submitted = base + timedelta(hours=idx)
        stamp = submitted
        for step in STEPS:
            lo, hi = _DRY_GAPS[step]
            stamp = stamp + timedelta(seconds=float(rng.uniform(lo, hi)))
            append_event(
                log_path, step, trial, iso=stamp.isoformat(timespec="seconds"),
                provider="dry-run", gpu="A100-80G", region="dry-run", dry_run=True,
            )
        rows.append({
            "trial": trial, "submitted_iso": submitted.isoformat(timespec="seconds"),
            "provider": "dry-run", "gpu": "A100-80G", "region": "dry-run",
            "note": "synthetic (--dry-run)",
        })
        logs.append(str(log_path))
    csv_path = out_dir / "submitted_times.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("dry run: wrote %d synthetic trial logs and %s", len(logs), csv_path)
    return logs, csv_path


def write_report(trials: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "instance_acquire_summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in trials:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    csv_path = out_dir / "instance_acquire.csv"
    pd.DataFrame(trials).to_csv(csv_path, index=False)
    return jsonl_path, csv_path


def _cmd_on_instance(args: argparse.Namespace) -> int:
    iso = _parse_iso(args.iso).isoformat(timespec="seconds") if args.iso else None
    rec = append_event(
        Path(args.log).expanduser(), args.step, args.trial, iso=iso, provider=args.provider,
        gpu=args.gpu, region=args.region, note=args.note,
        dry_run=True if args.dry_run else None,
    )
    logger.info("%s  %s  trial=%s  -> %s", rec["step"], rec["iso"], rec["trial"], args.log)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out) / "instance"
    logs, csv_path = list(args.logs), args.submitted_csv
    if args.dry_run and not logs:
        logs, csv_path = synth_trials(AcquireConfig(seed=args.seed, trials=args.trials), out_dir)
    if not logs:
        logger.error("no --logs given (and no --dry-run); nothing to report")
        return 2
    events = load_events(logs)
    submitted = load_submitted(list(args.submitted), Path(csv_path) if csv_path else None)
    trials = build_trials(events, submitted)
    if not trials:
        logger.error("no trial had both a log and a recorded submit time")
        return 2
    jsonl_path, csv_out = write_report(trials, out_dir)
    logger.info("wrote %d trials to %s and %s", len(trials), jsonl_path, csv_out)
    for rec in trials:
        total = rec.get("request_to_server_ready_s")
        logger.info(
            "trial %s: submit -> ssh %.0fs, -> deps %.0fs, -> weights %.0fs, -> serving %s",
            rec["trial"], rec.get("request_to_ssh_ready_s") or float("nan"),
            rec.get("request_to_deps_ready_s") or float("nan"),
            rec.get("request_to_weights_present_s") or float("nan"),
            f"{total:.0f}s ({total / 60:.1f} min)" if total else "n/a",
        )
    totals = [r["request_to_server_ready_s"] for r in trials if r.get("request_to_server_ready_s")]
    if totals:
        logger.info(
            "acquisition lead time (submit -> serving): median %.0f s, min %.0f s, max %.0f s",
            float(np.median(totals)), float(np.min(totals)), float(np.max(totals)),
        )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    on = sub.add_parser("on-instance", aliases=["on_instance"], help="record one milestone")
    on.add_argument("--step", required=True, choices=list(STEPS))
    on.add_argument("--trial", required=True, help="trial id, e.g. t1")
    on.add_argument("--log", required=True, help="append-only JSONL log path")
    on.add_argument("--provider", default=None)
    on.add_argument("--gpu", default=None)
    on.add_argument("--region", default=None)
    on.add_argument("--note", default=None)
    on.add_argument("--iso", default=None,
                    help="back-fill a past moment, e.g. --iso \"$(uptime -s)\" for boot")
    on.add_argument("--dry-run", action="store_true", help="tag the record as synthetic")
    on.set_defaults(func=_cmd_on_instance)

    rp = sub.add_parser("report", help="turn logs + manual submit times into lead times")
    rp.add_argument("--logs", nargs="*", default=[], help="paths or globs of on-instance logs")
    rp.add_argument("--submitted", action="append", default=[], help="TRIAL=ISO8601, repeatable")
    rp.add_argument("--submitted-csv", default=None, help="csv: trial,submitted_iso[,provider,...]")
    rp.add_argument("--out", default="results")
    rp.add_argument("--trials", type=int, default=3, help="--dry-run only: synthetic trial count")
    rp.add_argument("--seed", type=int, default=20260917)
    rp.add_argument("--dry-run", action="store_true")
    rp.set_defaults(func=_cmd_report)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        return int(args.func(args))
    except (OSError, ValueError, KeyError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
