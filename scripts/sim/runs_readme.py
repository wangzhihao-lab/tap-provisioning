"""``results/runs/JOBS_README.md``: the job list of the rerun round, explained."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from runs_groups import Original, flag_value

RUNNER = ("nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- caffeinate -i nice -n 10 "
          "uv run python scripts/sim/runs_runner.py --workers 6 --mem-budget-gb 11 "
          "> outputs/validation_local/runs/logs/runner.out 2>&1 < /dev/null &")
PLATFORM = """\
**Platform.** The cloud boxes (x86) and this laptop (arm64, same numpy 2.5.3 / scipy 1.18.1) do not
generate the same arrival stream from the same arguments.  Before this round: the laptop's own
sensitivity-round run `sensitivity_sweeps/c2_32b_shared4k_lead_d8` and box1's `r30_d_inst_t8_learnprior` are
the same stream group (same arguments, cell C, seeds 10-14) but their offline rows differ (in-class
requests 4,343,098 vs 4,339,254 for seed 10; offline GPU-hours up to 0.13 % apart), while runs made
on one platform agree exactly (box1 = box2; `lead_d8` = `priority_ablation/c2_32b_shared4k_on`).
So the plan's invariant "offline rows identical to the original" can only hold against laptop
originals.  Every job's check (`runs_check.py offline`) therefore demands exact equality with
every laptop mirror (and halts the queue if one differs) and records, without halting, how far the
cloud mirrors are ("platform" entries of `invariant.json`).  Groups whose mirrors are all cloud runs
(R1 cells A, B, D-F and all of c1, R2 replica tier, R7) produce TAP rows on the laptop's streams:
they cannot be joined row for row with the cloud baselines of the same group.
"""


def _tier(argv: Sequence[str]) -> str:
    tier, lead = flag_value(argv, "--lead-tier"), flag_value(argv, "--lead-min")
    return f"{tier} {lead or '(tier)'}"


def _extra(argv: Sequence[str]) -> str:
    bits = []
    if "--no-in-class-priority" in argv:
        bits.append("priority off")
    if flag_value(argv, "--p-off"):
        bits.append("p_off = p_in")
    if flag_value(argv, "--adoption") not in (None, "1.0"):
        bits.append(f"adoption {flag_value(argv, '--adoption')}")
    return ", ".join(bits) or "-"


def _short(path: str) -> str:
    return path.replace("outputs/validation_remote/", "").replace("outputs/validation_local/", "")


def write_readme(path: Path, jobs: List, origs: Dict[str, Original], mirrors: Dict[str, List[str]],
                 order: str) -> None:
    """Write the README next to the round's other notes."""
    lines = [f"# Rerun job list ({datetime.now():%Y-%m-%d %H:%M})", "",
             "Written by `scripts/sim/runs_jobs.py --order " + order + "`; the jobs are in "
             "`outputs/validation_local/runs/JOBS.txt` (launch order), results under "
             "`outputs/validation_local/runs/`, logs and DONE markers in its `logs/`.", "",
             "Start or resume (from the repository root; jobs with a DONE marker are skipped):", "",
             f"    {RUNNER}", "",
             "Progress: `cat outputs/validation_local/runs/logs/status.txt`; stop everything: "
             "`kill -TERM -$(cat outputs/validation_local/runs/logs/runner.pgid)`.", "",
             "Launch order (`--order priority`): the order the rerun round gives for a queue longer than 14 h "
             "(the one-seed probe of the first central job put it at 13-19 h under the load of "
             "2026-09-23; the first jobs then ran at about half their estimate, see `logs/status.txt` "
             "for the running ETA), except that the groups whose originals all ran on the cloud boxes go last "
             "(R2 replica tier, R7, R1 for c1; see Platform).  `touch logs/<id>.SKIP` drops a queued "
             "job without restarting the runner.", "",
             PLATFORM, "## TAP stream groups", "",
             "Each job runs `--policies offline tap` with the base original's arguments; `--tuned` "
             "is dropped (it moves only the feedback baselines).  Mirrors = every original with the "
             "same stream key and an overlapping cell.", "",
             "| job | case | tier, lead | p_in | timetable | prior | cells | other | est. min | base "
             "original | laptop mirrors | cloud mirrors |", "|" + "---|" * 12]
    for job in jobs:
        if "spec" not in job.info:
            continue
        spec, argv = job.info["spec"], job.info["argv"]
        mine = mirrors[job.job_id]
        lap = "<br>".join(_short(m) for m in mine if origs[m].laptop) or "none"
        cloud = "<br>".join(_short(m) for m in mine if not origs[m].laptop) or "none"
        slots = flag_value(argv, "--slots") or "2026-27 s1 (default)"
        note = f" ({spec.note})" if spec.note else ""
        lines.append(f"| `{job.job_id}` ({spec.family}) | {spec.case} | {_tier(argv)} | "
                     f"{flag_value(argv, '--p-in')} | "
                     f"{_short(slots)} | {_short(spec.prior)}{note} | {''.join(spec.cells)} | "
                     f"{_extra(argv)} | {job.est:.0f} | {_short(spec.base)} | {lap} | {cloud} |")
    lines += ["", "Covered by R1 without a job of their own (same stream key, cell C): the lead-time "
              "points d = 1 / 1.5 (replica tier) and d = 8 (instance tier) of R3, and the priority-on "
              "arm of R8; they appear above as laptop mirrors of the R1 jobs.  Not run: c4, c5 "
              "(the paper does not rank policies on them) and F6.", "",
              "## Central cell C on the laptop's streams (paired with TAP)", "",
              "Every policy variant of the central originals (same stream key, cell C) that no laptop "
              "run has produced, derived by `runs_paired.py` from the originals' `--policies` "
              "and `--tuned` (a reactive / velocity variant is its knob values, so two tuned files "
              "with the same values count once).  Each run keeps its base original's arguments; the "
              "check demands offline rows equal to the R1 job and every laptop mirror.  The SLO-floor "
              "variants of the dispersed tuning are already on the laptop (`sensitivity_sweeps/*_lead_d*`), "
              "and `s3_eval_*` reruns them there if section 3 selects another configuration.", "",
              "| job | base original | tuned | missing policies run |", "|---|---|---|---|"]
    for job in jobs:
        if job.family == "P1" and "specs" in job.info:
            info = job.info
            lines.append(f"| `{job.job_id}` | {_short(info['base'].out)} | "
                         f"{_short(info['tuned'] or '-')} | {' '.join(info['specs'])} |")
    lines += ["", "`p1_table` (after every `p1_*` and `s3_eval_*` job) writes "
              "`results/runs/paired_cell_c.{md,csv}`: per case x tier, every policy variant "
              "of cell C from laptop runs only, offline rows re-checked against the R1 job.", "",
              "## Floor tuning (section 3), billing sensitivity (section 4), Fig. 1 data, audits", "",
              "| job | after | est. min | what |", "|---|---|---|---|"]
    for job in jobs:
        if job.family in ("S3", "S4", "A1", "F1"):
            what = {"S3": "resource-rule floor tuning / conditional six-week rerun",
                    "S4": "billing from the order (beta = c d)", "A1": "TAP prefix invariance",
                    "F1": "Fig. 1 single-bell table with TAP (checked against f1_right_bell.csv)"}
            lines.append(f"| `{job.job_id}` | {', '.join(job.after) or '-'} | {job.est:.0f} | "
                         f"{what[job.family]} |")
    lines += ["", "`s3_eval_*` compares the new resource-rule `best.json` with the one the cloud run "
              "selected (cost rule); an unchanged configuration is not rerun (`compare.json` says "
              "which).  `s4_eval_cost_*` runs offline, tap, calendar_learn_prior, calendar, reactive, "
              "velocity and fixed at beta = c d with the baselines retuned at that price, "
              "`s4_eval_floor_*` the floor-tuned baselines, `s4_ref_main_*` the cost-tuned baselines "
              "at beta = c on the same laptop streams (the reference block); `s4_summary` writes "
              "`results/runs/billing_sensitivity.md`.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["RUNNER", "write_readme"]
