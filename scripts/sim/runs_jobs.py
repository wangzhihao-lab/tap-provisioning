"""Job list of the rerun round: TAP reruns, floor tuning (s3), billing (s4).

    uv run python scripts/sim/runs_jobs.py [--order priority|longest]

Writes ``outputs/validation_local/runs/JOBS.txt`` (one job per line, run by
``runs_runner.py``) and ``results/runs/JOBS_README.md`` (the stream groups and the
originals each one mirrors).  A TAP job reuses the argument list of one original verbatim except
for ``--policies offline tap``, ``--tuned`` (dropped: it only moves the feedback baselines),
``--learn-prior-from`` (the prior TAP's forecast is warm-started from), ``--offset-cells`` (always
explicit: the timing grid has twelve cells now, the cloud runs had A-F) and ``--case/--out``.
Its mirrors are all originals with the same stream key (``runs_groups.stream_key``) and an
overlapping timing cell; the job's check compares its offline rows with theirs.
"""

from __future__ import annotations

import argparse
import math
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from runs_groups import (CASES, CELLS_AF, LOCAL, PILOT, PRIOR_FWD, PRIOR_SEM2, PRIOR_SYM,
                              REMOTE, Original, cells_of, check_laptop_reconstruction, drop_flags,
                              flag_value, originals, set_flag, stream_key)
from runs_paired import missing_runs, paired_argv
from runs_readme import write_readme

OUT = f"{LOCAL}/runs"
DOCS = PILOT / "results" / "runs"
C_RUN = 0.0333
SEEDS = 5
# wall minutes per (cell x seed) and GB of peak RSS, from the one-seed probe of the first central
# job under the 2026-09-23 load (6.3 min wall, 1.8 min CPU, 1.52 GB); the sweep keeps one stream at
# a time, so a six-cell job peaks where a one-cell job does
MIN_PER_CELL_SEED = {"tap": 6.3, "eval7": 16.0, "eval4": 9.0, "eval3": 7.5}
MEM_GB = {"tap6": 1.8, "tap1": 1.8, "tune": 1.2, "eval1": 1.8, "eval6": 1.8}
TUNE_MIN = 120.0
AUDIT_MIN = 90.0


@dataclass
class Job:
    """One line of JOBS.txt."""

    job_id: str
    family: str
    command: str
    after: Tuple[str, ...] = ()
    mem: float = 2.0
    est: float = 30.0
    info: Dict[str, object] = field(default_factory=dict)

    def line(self) -> str:
        """``id after=.. mem=.. est=.. :: command``."""
        after = ",".join(self.after) or "-"
        return f"{self.job_id} after={after} mem={self.mem:g} est={self.est:.0f} :: {self.command}"


def py(script: str, argv: Sequence[str]) -> str:
    """``uv run python <script> <argv>``, shell-quoted."""
    return shlex.join(["uv", "run", "python", script, *argv])


def check(new_csv: str, mirrors: Sequence[str], report: str, extra: Sequence[str] = ()) -> str:
    """The offline-row check of one new run against its mirrors."""
    argv = ["offline", "--new", new_csv, "--report", report, *extra]
    for m in mirrors:
        argv += ["--orig", m]
    return py("scripts/sim/runs_check.py", argv)


# --------------------------------------------------------------------------- TAP stream groups
@dataclass(frozen=True)
class TapSpec:
    """One TAP job: the original whose arguments it reuses, its cells and its prior."""

    job_id: str
    family: str
    case: str
    base: str
    cells: Tuple[str, ...]
    prior: str
    note: str = ""


def tap_specs() -> List[TapSpec]:
    """The stream groups of the rerun round (R1-R8), one TAP job each."""
    out: List[TapSpec] = []
    v1, v2 = "validation", "validation2"
    for key in ("c2", "c3", "c1"):
        c = CASES[key]
        tree = v1 if key == "c1" else v2
        inst = "r30_d_inst_t8" + ("" if key == "c1" else "_learnprior")
        rep = "r30_d_rep" + ("" if key == "c1" else "_learnprior")
        out += [TapSpec(f"r1_{key}_inst", "R1", key, f"{REMOTE}/{c.box}/{tree}/{c.name}/{inst}", CELLS_AF,
                        PRIOR_SEM2),
                TapSpec(f"r1_{key}_rep", "R1", key, f"{REMOTE}/{c.box}/{tree}/{c.name}/{rep}", CELLS_AF,
                        PRIOR_SEM2)]
    for key in ("c2", "c3"):
        c, root = CASES[key], f"{REMOTE}/{CASES[key].box}/{v2}/{CASES[key].name}"
        sens, abl = f"{LOCAL}/sensitivity_sweeps/{c.name}", f"{LOCAL}/priority_ablation/{c.name}"
        out += [TapSpec(f"r2_{key}_inst_sym", "R2", key, f"{root}/r30_d_inst_t8_heldout_sem2_learnprior",
                        CELLS_AF, PRIOR_SYM),
                TapSpec(f"r2_{key}_inst_fwd", "R2", key, f"{sens}_heldout_fwd", ("C",), PRIOR_FWD),
                TapSpec(f"r2_{key}_rep_sym", "R2", key, f"{root}/r30_d_rep_heldout_sem2", CELLS_AF,
                        PRIOR_SYM, "originals carry no prior; the symmetric one, as at the instance tier"),
                TapSpec(f"r2_{key}_rep_fwd", "R2", key, f"{root}/r30_d_rep_heldout_sem2", ("C",),
                        PRIOR_FWD, "no original with this prior; cell C, as at the instance tier")]
        for d in ("2", "3", "4", "6", "12", "15"):
            out.append(TapSpec(f"r3_{key}_d{d}", "R3", key, f"{sens}_lead_d{d}", ("C",), PRIOR_SEM2))
        out.append(TapSpec(f"r4_{key}_cellsGL", "R4", key, f"{sens}_cells_GL", tuple("GHIJKL"),
                           PRIOR_SEM2))
        out.append(TapSpec(f"r5_{key}_res", "R5", key, f"{sens}_res", ("C",), PRIOR_SEM2))
        for a in ("0.25", "0.5", "0.75"):
            out.append(TapSpec(f"r6_{key}_adopt{a}", "R6", key, f"{sens}_adopt{a}", ("C",), PRIOR_SEM2))
        r100_rep = f"{root}/r100_d_rep"
        r100_inst = f"{REMOTE}/{c.box}/{v1}/{c.name}/r100_d_inst_t8"
        out += [TapSpec(f"r7_{key}_rep", "R7", key, r100_rep, CELLS_AF, PRIOR_SEM2),
                TapSpec(f"r7_{key}_inst", "R7", key, r100_inst, CELLS_AF, PRIOR_SEM2)]
        out += [TapSpec(f"r8_{key}_off", "R8", key, f"{abl}_off", ("C",), PRIOR_SEM2),
                TapSpec(f"r8_{key}_ungraded", "R8", key, f"{abl}_ungraded", ("C",), PRIOR_SEM2)]
    return out


def tap_argv(spec: TapSpec, base: Original, out_dir: str) -> List[str]:
    """The base original's arguments with only the TAP job's own flags changed."""
    argv = drop_flags(base.argv, ["--policies", "--tuned", "--out", "--learn-prior-from",
                                  "--offset-cells", "--case"])
    return [*argv, "--offset-cells", *spec.cells, "--policies", "offline", "tap",
            "--learn-prior-from", spec.prior, "--case", CASES[spec.case].name, "--out", out_dir]


def build_tap_jobs(origs: Dict[str, Original]) -> Tuple[List[Job], Dict[str, List[str]]]:
    """TAP jobs with their mirrors; raises when a group has none or two jobs share a group."""
    sweeps = [o for o in origs.values() if o.script == "run_sweep"]
    keys = {o.out: stream_key(o.argv) for o in sweeps}
    cells = {o.out: cells_of(o.argv, PILOT / o.csv) for o in sweeps}
    jobs, mirrors, seen = [], {}, {}
    for spec in tap_specs():
        base = origs[spec.base]
        out_dir = f"{OUT}/tap/{spec.job_id}"
        argv = tap_argv(spec, base, out_dir)
        key = stream_key(argv)
        mine = sorted(o for o in keys if keys[o] == key and set(cells[o]) & set(spec.cells))
        if spec.base not in mine or not mine:
            raise RuntimeError(f"{spec.job_id}: its base {spec.base} is not in its own stream group")
        for cell in spec.cells:
            prev = seen.setdefault((key, spec.prior, cell), spec.job_id)
            if prev != spec.job_id:
                raise RuntimeError(f"{spec.job_id} and {prev} replay the same stream group (cell {cell})")
        mirrors[spec.job_id] = mine
        n = len(spec.cells) * SEEDS
        csv = f"{out_dir}/offset/offset.csv"
        command = py("scripts/sim/run_sweep.py", argv) + " && " + check(
            csv, [origs[m].csv for m in mine], f"{out_dir}/invariant.json")
        jobs.append(Job(spec.job_id, spec.family, command, (), MEM_GB["tap6" if n > 10 else "tap1"],
                        n * MIN_PER_CELL_SEED["tap"],
                        {"spec": spec, "argv": argv, "base": base, "mirrors": mine,
                         "lead": flag_value(argv, "--lead-min")}))
    return jobs, mirrors


# --------------------------------------------------------------------------- sections 3 and 4
def tier_dirs(case: str, tier: str) -> Dict[str, str]:
    """Original tuning and evaluation directories of one case x tier."""
    c = CASES[case]
    t8 = "8" if tier == "inst" else ""
    run = "r30_d_inst_t8" if tier == "inst" else "r30_d_rep"
    return {"floor_tuning": f"{REMOTE}/box2/validation2/{c.name}/tuning{t8}_r30_slo95disp",
            "floor_eval": f"{REMOTE}/box2/validation2/{c.name}/{run}_slo95disp",
            "cost_tuning": f"{REMOTE}/{c.box}/validation/{c.name}/tuning{t8}_r30"}


def section3_jobs(origs: Dict[str, Original], taps: Dict[str, Job]) -> List[Job]:
    """Resource-cost floor tuning of c2/c3 x both tiers, then the conditional re-evaluation."""
    jobs: List[Job] = []
    for case in ("c2", "c3"):
        for tier in ("inst", "rep"):
            dirs, name = tier_dirs(case, tier), f"{CASES[case].name}_{tier}"
            tune_out = f"{OUT}/tuning_resource/{name}"
            tune = origs[dirs["floor_tuning"]]
            argv = [*drop_flags(tune.argv, ["--out"]), "--slo-select", "resource", "--out", tune_out]
            jobs.append(Job(f"s3_tune_{case}_{tier}", "S3", py("scripts/sim/tune_baselines.py", argv),
                            (), MEM_GB["tune"], TUNE_MIN))
            ev = origs[dirs["floor_eval"]]
            eval_out = f"{tune_out}/eval"
            eargv = [*drop_flags(ev.argv, ["--out", "--tuned", "--offset-cells"]), "--offset-cells",
                     *CELLS_AF, "--tuned", f"{tune_out}/best.json", "--out", eval_out]
            r1 = taps[f"r1_{case}_{tier}"]
            argv = ["floor-eval", "--old", f"{dirs['floor_tuning']}/best.json", "--new",
                    f"{tune_out}/best.json", "--note", f"{tune_out}/compare.json", "--pair",
                    f"{OUT}/tap/{r1.job_id}/offset/offset.csv", "--orig", ev.csv, "--", *eargv]
            jobs.append(Job(f"s3_eval_{case}_{tier}", "S3", py("scripts/sim/runs_check.py", argv),
                            (f"s3_tune_{case}_{tier}", r1.job_id), MEM_GB["eval6"],
                            len(CELLS_AF) * SEEDS * MIN_PER_CELL_SEED["eval3"]))
    return jobs


def beta_for(lead_min: str) -> str:
    """``c_run * d_slots`` for the whole-lead billing (d_slots = the engine's ceil of the lead)."""
    return repr(C_RUN * math.ceil(float(lead_min) - 1e-9))


def section4_jobs(origs: Dict[str, Original], taps: Dict[str, Job]) -> List[Job]:
    """Billing from the order: retune at beta = c d, evaluate cell C, and the beta = c reference."""
    jobs: List[Job] = []
    for case, tier in (("c2", "inst"), ("c3", "inst"), ("c3", "rep")):
        dirs, name = tier_dirs(case, tier), f"{CASES[case].name}_{tier}"
        r1 = taps[f"r1_{case}_{tier}"]
        beta = beta_for(str(r1.info["lead"]))
        root = f"{OUT}/billing/{name}"
        ids = {k: f"s4_{k}_{case}_{tier}" for k in ("tune_cost", "tune_floor", "eval_cost",
                                                     "eval_floor", "ref_main")}
        for kind, src, extra in (("tune_cost", dirs["cost_tuning"], []),
                                 ("tune_floor", dirs["floor_tuning"], ["--slo-select", "resource"])):
            argv = [*set_flag(drop_flags(origs[src].argv, ["--out"]), "--beta-start", beta), *extra,
                    "--out", f"{root}/{kind}"]
            jobs.append(Job(ids[kind], "S4", py("scripts/sim/tune_baselines.py", argv), (),
                            MEM_GB["tune"], TUNE_MIN))
        base = drop_flags(r1.info["argv"], ["--policies", "--out", "--offset-cells", "--beta-start"])
        laptop_mirrors = [origs[m].csv for m in r1.info["mirrors"] if origs[m].laptop]
        runs = (("eval_cost", beta, ["offline", "tap", "calendar_learn_prior", "calendar", "reactive",
                                     "velocity", "fixed"], f"{root}/tune_cost/best.json", "eval7"),
                ("eval_floor", beta, ["offline", "reactive", "velocity"],
                 f"{root}/tune_floor/best.json", "eval3"),
                ("ref_main", repr(C_RUN), ["offline", "reactive", "velocity", "fixed"],
                 f"{dirs['cost_tuning']}/best.json", "eval4"))
        for kind, b, pols, tuned, weight in runs:
            out_dir = f"{root}/{kind}"
            argv = [*base, "--beta-start", b, "--offset-cells", "C", "--policies", *pols, "--tuned",
                    tuned, "--out", out_dir]
            csv = f"{out_dir}/offset/offset.csv"
            if kind == "eval_cost":  # new beta: only the stream columns can match the originals
                after, chk = (ids["tune_cost"],), check(csv, laptop_mirrors, f"{out_dir}/invariant.json",
                                                        ["--stream-only"])
            elif kind == "eval_floor":  # same beta and stream as eval_cost: offline rows identical
                after, chk = (ids["tune_floor"], ids["eval_cost"]), check(
                    csv, [f"{root}/eval_cost/offset/offset.csv"], f"{out_dir}/invariant.json")
            else:
                after, chk = (), check(csv, laptop_mirrors, f"{out_dir}/invariant.json")
            jobs.append(Job(ids[kind], "S4", py("scripts/sim/run_sweep.py", argv) + " && " + chk, after,
                            MEM_GB["eval1"], SEEDS * MIN_PER_CELL_SEED[weight]))
    deps = tuple(j.job_id for j in jobs if "_eval_" in j.job_id or "_ref_" in j.job_id)
    deps += ("r1_c2_inst", "r1_c3_inst", "r1_c3_rep", "s3_eval_c2_inst", "s3_eval_c3_inst",
             "s3_eval_c3_rep")
    jobs.append(Job("s4_summary", "S4", py("scripts/sim/runs_billing.py", []), deps, 1.0, 2.0))
    return jobs


def paired_jobs(origs: Dict[str, Original], taps: Dict[str, Job]) -> List[Job]:
    """Every central cell-C policy variant not yet on the laptop (``runs_paired``)."""
    jobs: List[Job] = []
    for case in ("c2", "c3"):
        for tier in ("inst", "rep"):
            r1 = taps[f"r1_{case}_{tier}"]
            mirrors = [origs[m] for m in r1.info["mirrors"]]
            refs = ([f"{OUT}/tap/{r1.job_id}/offset/offset.csv"]
                    + [m.csv for m in mirrors if m.laptop] + [m.csv for m in mirrors if not m.laptop])
            for base, specs in missing_runs(mirrors):
                tuned = flag_value(base.argv, "--tuned")
                tag = tuned.split("/")[-2] if tuned else "untuned"
                out_dir = f"{OUT}/paired/{CASES[case].name}_{tier}/{tag}"
                csv = f"{out_dir}/offset/offset.csv"
                command = (py("scripts/sim/run_sweep.py", paired_argv(base, specs, out_dir)) + " && "
                           + check(csv, refs, f"{out_dir}/invariant.json"))
                jobs.append(Job(f"p1_{case}_{tier}_{tag}", "P1", command, (r1.job_id,),
                                MEM_GB["eval1"], SEEDS * (0.8 + 1.2 * len(specs)),
                                {"base": base, "specs": specs, "tuned": tuned, "out": out_dir}))
    after = tuple(j.job_id for j in jobs) + tuple(f"s3_eval_{c}_{t}" for c in ("c2", "c3")
                                                 for t in ("inst", "rep"))
    jobs.append(Job("p1_table", "P1", py("scripts/sim/runs_paired_table.py", []), after, 1.0, 2.0))
    return jobs


def fig1_job() -> Job:
    """Fig. 1's single-bell table with TAP added."""
    out = f"{OUT}/fig1"
    orig = "../results/sim/validation/f1_right_bell.csv"
    command = (py("scripts/sim/single_bell_replay.py", ["--with-tap", "--out", out]) + " && "
               + py("scripts/sim/runs_check.py", ["columns", "--new", f"{out}/f1_right_bell.csv",
                                                       "--orig", orig, "--report", f"{out}/invariant.json"]))
    return Job("f1_bell_tap", "F1", command, (), MEM_GB["tap1"], 3 * 7 * 2.0)


def audit_jobs() -> List[Job]:
    """Prefix invariance of TAP and its advice at the central cell."""
    return [Job(f"audit_tap_prefix_{case}", "A1",
                py("scripts/audit/tap_prefix_invariance.py",
                   ["--case", case, "--seeds", "10", "11", "12", "13", "14", "--out",
                    f"{OUT}/audit/tap_prefix_{case}.json"]), (), MEM_GB["tap1"], AUDIT_MIN)
            for case in ("c2", "c3")]


# the caller's order for a queue longer than 14 h, except that the groups whose originals all ran on
# the cloud boxes (their TAP rows cannot be paired with the cloud baselines, JOBS_README "Platform")
# go last: R2 at the replica tier, R7, and c1 (last in the given order anyway)
CLOUD_ONLY: Tuple[str, ...] = ("r2_c2_rep", "r2_c3_rep", "R7", "r1_c1")
PRIORITY: Tuple[str, ...] = ("r1_c2", "r1_c3", "P1", "F1", "R2", "S4", "S3", "R3", "R6", "R5", "R4",
                             "R8", "A1", *CLOUD_ONLY)


def ordered(jobs: List[Job], order: str) -> List[Job]:
    """Launch order: the given family priority, or longest first (ties by priority)."""
    def rank(job: Job) -> int:
        tags = [(i, t) for i, t in enumerate(PRIORITY) if job.job_id.startswith(t) or job.family == t]
        return max(tags)[0] if tags else len(PRIORITY)  # the most specific (latest) tag wins
    first = sorted(jobs, key=lambda j: (rank(j), j.job_id.startswith("s4_summary")))
    if order == "priority":
        return first
    return sorted(first, key=lambda j: -j.est)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write the job list and its README."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", choices=("priority", "longest"), default="priority")
    ap.add_argument("--readme-only", action="store_true", help="rewrite JOBS_README.md, not JOBS.txt")
    args = ap.parse_args(argv)
    origs = originals()
    bad = {o.out: check_laptop_reconstruction(o) for o in origs.values() if o.laptop}
    if any(bad.values()):
        raise RuntimeError(f"laptop originals rebuilt wrongly: { {k: v for k, v in bad.items() if v} }")
    taps, mirrors = build_tap_jobs(origs)
    by_id = {j.job_id: j for j in taps}
    jobs = ordered(taps + paired_jobs(origs, by_id) + [fig1_job()] + section3_jobs(origs, by_id)
                   + section4_jobs(origs, by_id) + audit_jobs(), args.order)
    target = PILOT / OUT / "JOBS.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    head = ["# rerun jobs: <id> after=<deps> mem=<GB> est=<min> :: <command>   (run from the repository root)",
            f"# written by scripts/sim/runs_jobs.py --order {args.order}"]
    text = "\n".join(head + [j.line() for j in jobs]) + "\n"
    if args.readme_only:
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"{target} differs from what this code would write; rerun without --readme-only")
    else:
        target.write_text(text, encoding="utf-8")
    write_readme(DOCS / "JOBS_README.md", jobs, origs, mirrors, args.order)
    print(f"{'kept' if args.readme_only else 'wrote'} {target} ({len(jobs)} jobs); wrote "
          f"{DOCS / 'JOBS_README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
