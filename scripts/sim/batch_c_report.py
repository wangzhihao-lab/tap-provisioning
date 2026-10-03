"""Results tables of batch C, reported above the always-on base tier.

    uv run python scripts/sim/batch_c_report.py                       # outputs/batchC -> docs
    uv run python scripts/sim/batch_c_report.py --root outputs/batchC --out /tmp/bc --ratios 30

Batch C deploys the service institution-wide with one always-on base replica (``--min-replicas 1``),
which every policy pays for and the offline optimum pays for too.  That replica is a constant, so
``cost_ratio = total / OPT`` is compressed towards 1 for every policy and stops separating them.
Every table here is therefore reported on the component *above* the base tier: ``excess_usd``, the
USD a policy paid over its own optimum, and ``attr_ratio``, the ratio of the two attributable costs
(``pilot.batchc.derive``).

The script is safe to run while ``scripts/sim/batch_c.sh`` is still going: a sweep that has not
landed yet is logged and skipped rather than raising, so the report can be re-run as the lanes
finish.  It exits 1 when a run violates one of the two invariants the scenario guarantees - no
policy can beat its own offline optimum, and no trajectory can spend fewer GPU-hours than the base
tier - because either one means the numbers are not measuring what the paper says they measure.

Every table is written as ``<name>.csv`` next to a ``<name>.md`` that opens with the sources it was
built from, the base-tier formula with the values used and the run timestamp.
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PILOT = Path(__file__).resolve().parents[2]
REPO = PILOT
if str(PILOT / "src") not in sys.path:  # allow running without installing the package
    sys.path.insert(0, str(PILOT / "src"))

from pilot.batchc import (  # noqa: E402
    CELL_METRICS,
    DEFAULT_C_RUN,
    DEFAULT_MIN_REPLICAS,
    SLO95_SUFFIX,
    SLO_MATCHED_SUFFIX,
    SWEEP_CELL_METRICS,
    WEEKLY_METRICS,
    CellSelection,
    Provenance,
    TuningRun,
    appended,
    base_tier_formula,
    blocks_by,
    cell_table,
    derive,
    find_sweep_files,
    frame_to_markdown,
    main_table,
    now_stamp,
    ratio_table,
    read_frame,
    read_tuning,
    reference_cell,
    relabelled,
    sweep_path,
    tuning_table,
    verification_row,
    violation_count,
    weekly_table,
    wide_by_week,
    write_table,
)
from pilot.batchc import R1_WEEKS as NOVEL_TIMETABLE_WEEKS  # noqa: E402

logger = logging.getLogger("batch_c_report")

R1_WEEKS: Tuple[int, ...] = NOVEL_TIMETABLE_WEEKS  # (1, 5): no history, and the freshman step
R2_NOTE: str = ("R1 weeks are starred: week 1 has no history and week 5 is the freshman step of "
                "the timetable; every other week is regime R2.")
MAIN_SWEEP: str = "lead"  # the main cell is the replica tier of the lead sweep
FALLBACK_SWEEP: str = "sigma"  # ... and its stand-in until that sweep lands
SENSITIVITY_GROUP: str = "r30_32b"
EPSILON_GROUP: str = "eps"
SLO_MATCHED_RATIO: int = 30  # the price ratio batch C also tunes under an SLO floor
SLO_MATCHED_GROUP: str = f"r{SLO_MATCHED_RATIO}{SLO95_SUFFIX}"  # r30_slo95
CELL_TABLES: Tuple[Tuple[str, str, str], ...] = (  # (group, sweep, output name)
    ("r30", "adoption", "adoption_r30"),
    ("r30", "margin", "margin_r30"),
    ("r30", "trust", "trust_r30"),
    ("r30", "pool", "pool_r30"),
    (EPSILON_GROUP, "epsilon", "epsilon"),
)
METRIC_TITLES: Dict[str, str] = {
    "excess_usd": "Mean cost over the offline optimum (USD)",
    "attr_ratio": "Mean cost ratio to OPT above the base tier",
    "slo_in": "Mean in-class SLO attainment",
    "gpu_hours": "Mean GPU-hours (total, base tier included)",
    "attr_gpu_hours": "Mean GPU-hours above the base tier",
}
SLO_MATCHED_NOTE: str = (
    f"The rows marked `{SLO_MATCHED_SUFFIX.strip()}` come from `{SLO_MATCHED_GROUP}/lead/lead.csv`:"
    " the same two feedback baselines retuned to the cheapest configuration that reaches "
    "`slo_in >= 0.95` on the tuning seeds, so that they are read at a matched attainment rather "
    "than at their own cheapest operating point.")


@dataclass(frozen=True)
class ReportConfig:
    """Everything the report needs that the command line decides."""

    root: Path
    out: Path
    min_replicas: int = DEFAULT_MIN_REPLICAS
    c_run: float = DEFAULT_C_RUN
    ratios: Tuple[int, ...] = (6, 10, 30, 100)
    generated: str = field(default_factory=now_stamp)

    @property
    def base_formula(self) -> str:
        """The base-tier formula with this run's ``M`` and ``c_run`` substituted in."""
        return base_tier_formula(self.min_replicas, self.c_run)

    def provenance(self, title: str, sources: Sequence[Path], notes: Sequence[str] = ()
                   ) -> Provenance:
        """The Markdown header of one table."""
        return Provenance(title=title, sources=tuple(_rel(p) for p in sources),
                          base_formula=self.base_formula, generated=self.generated,
                          notes=tuple(n for n in notes if n))


def _rel(path: Path) -> str:
    """A source path relative to the pilot directory when it sits inside it."""
    try:
        return str(Path(path).resolve().relative_to(PILOT))
    except ValueError:
        return str(path)


def load_derived(cfg: ReportConfig, group: str, sweep: str) -> Optional[pd.DataFrame]:
    """Read one sweep CSV and add the base-tier columns; ``None`` when it is not there yet."""
    frame = read_frame(sweep_path(cfg.root, group, sweep))
    return None if frame is None else derive(frame, cfg.min_replicas, cfg.c_run)


def whole_run(frame: pd.DataFrame) -> pd.DataFrame:
    """The whole-run rows of a sweep frame (``week = 0``)."""
    return frame[pd.to_numeric(frame["week"], errors="coerce") == 0] if "week" in frame else frame


@dataclass(frozen=True)
class Reference:
    """The cell every main table reports: its rows, which cell it is and where it came from."""

    frame: pd.DataFrame
    cell: CellSelection
    path: Path
    note: str = ""

    @property
    def cell_note(self) -> str:
        """The header line naming the cell."""
        return f"Cell: {self.cell.description}."


def reference(cfg: ReportConfig, group: str) -> Optional[Reference]:
    """The replica tier of ``<group>/lead``, or the most synchronous cell of its sigma sweep."""
    for sweep in (MAIN_SWEEP, FALLBACK_SWEEP):
        frame = load_derived(cfg, group, sweep)
        if frame is None:
            continue
        cell = reference_cell(frame)
        note = ("" if sweep == MAIN_SWEEP else
                f"`{MAIN_SWEEP}/{MAIN_SWEEP}.csv` was not found under `{group}`, so this table "
                f"falls back to the `{FALLBACK_SWEEP}` sweep at its largest sigma.")
        return Reference(cell.rows(frame), cell, sweep_path(cfg.root, group, sweep), note)
    logger.warning("group %s has neither a %s nor a %s sweep; skipped", group, MAIN_SWEEP,
                   FALLBACK_SWEEP)
    return None


def write_verification(cfg: ReportConfig) -> int:
    """One row per sweep CSV found: shape, coverage and the two invariants.  Returns the count."""
    sources: List[Path] = []
    rows: List[Dict[str, object]] = []
    for found in find_sweep_files(cfg.root):
        frame = read_frame(found.path)
        if frame is None:
            continue
        rows.append(verification_row(found, derive(frame, cfg.min_replicas, cfg.c_run)))
        sources.append(found.path)
    table = pd.DataFrame(rows)
    notes = ["`dominance_violations` counts whole-run rows cheaper than their own offline optimum "
             "and `floor_violations` rows below the base tier; both must be 0."]
    _write(cfg, "verification", table, sources, notes, frame_to_markdown(table, merge_sd=False))
    violations = violation_count(table)
    if violations:
        for _, row in table.iterrows():
            if float(row.get("dominance_violations", 0)) or float(row.get("floor_violations", 0)):
                logger.error("%s: %s dominance, %s floor violations", row["file"],
                             row["dominance_violations"], row["floor_violations"])
    return violations


def write_main(cfg: ReportConfig, ref: Reference, name: str, title: str,
               matched: Optional[Reference] = None) -> pd.DataFrame:
    """Per policy, mean and sd over seeds of the whole-run metrics of the reference cell.

    ``matched`` is the SLO-matched retuning of the same price ratio: its two baselines are
    appended under names that carry the floor, and the frame returned for the r-sweep stays the
    freely tuned one so that the ratios column keeps comparing like with like.
    """
    rows = whole_run(ref.frame)
    table, sources = main_table(rows), [ref.path]
    notes = [ref.cell_note, ref.note]
    if matched is not None:
        table = appended(table, relabelled(main_table(whole_run(matched.frame))))
        sources.append(matched.path)
        notes.append(SLO_MATCHED_NOTE)
    _write(cfg, name, table, sources, notes, title=title)
    return rows


def write_weekly(cfg: ReportConfig, ref: Reference, name: str, title: str) -> None:
    """Per policy and teaching week, the mean over seeds, one wide table per metric."""
    table = weekly_table(ref.frame, WEEKLY_METRICS, R1_WEEKS)
    parts: List[str] = []
    for metric in WEEKLY_METRICS:
        if metric in table.columns:
            parts.extend([f"## {METRIC_TITLES.get(metric, metric)}", "",
                          wide_by_week(table, metric, R1_WEEKS), ""])
    parts.append(R2_NOTE)
    _write(cfg, name, table, [ref.path], [ref.cell_note, ref.note], "\n".join(parts), title)


def write_r_sweep(cfg: ReportConfig, frames: Dict[int, pd.DataFrame],
                  sources: Dict[int, Path]) -> None:
    """Policies against price ratios: one block per metric, one column per ratio."""
    table = ratio_table(frames, CELL_METRICS, cfg.ratios)
    notes = ["Each column is the reference cell of that price ratio, over its whole-run rows."]
    _write(cfg, "r_sweep", table, [sources[r] for r in sorted(sources)], notes,
           blocks_by(table, "metric", METRIC_TITLES))


def tuning_runs(cfg: ReportConfig) -> List[TuningRun]:
    """Every tuning directory the report reads: one per price ratio, plus the SLO-matched one."""
    wanted = [(ratio, "") for ratio in cfg.ratios]
    if SLO_MATCHED_RATIO in cfg.ratios:
        wanted.append((SLO_MATCHED_RATIO, SLO95_SUFFIX))
    return [run for run in (read_tuning(cfg.root, ratio, suffix) for ratio, suffix in wanted)
            if run is not None]


def write_tuning(cfg: ReportConfig) -> None:
    """What the baseline tuning chose per selection, against the default it replaced."""
    runs = tuning_runs(cfg)
    table = tuning_table(runs)
    notes = ["`n_distinct_scores` is the number of distinct `cost_ratio_tune_mean` values in that "
             "policy's grid CSV (rounded to 1e-9): a grid whose score never moved selected "
             "nothing, whatever `rank_of_default` says.",
             "`slice` names the tuning directory and `slo_floor` the attainment the selection was "
             "constrained to (`n/a` when it was free), so the SLO-matched pair of `r30_slo95` can "
             "be told from the freely tuned one.",
             "Ratios are means over the tuning seeds; held-out numbers are the single held-out "
             "seed, so no standard deviation is shown."]
    sources: List[Path] = []
    for run in runs:
        sources.extend([run.path / "best.json", *sorted(run.grids.values())])
    _write(cfg, "tuning", table, sources, notes, frame_to_markdown(table, merge_sd=False))


def write_cells(cfg: ReportConfig, group: str, sweep: str, name: str) -> None:
    """Per cell and policy of one secondary sweep: the headline means and the total GPU-hours."""
    frame = load_derived(cfg, group, sweep)
    if frame is None:
        return
    table = cell_table(whole_run(frame), SWEEP_CELL_METRICS)
    notes = ["Every cell-key column of the sweep is kept, so a row is one grid cell of one policy."]
    _write(cfg, name, table, [sweep_path(cfg.root, group, sweep)], notes)


def _write(cfg: ReportConfig, name: str, table: pd.DataFrame, sources: Sequence[Path],
           notes: Sequence[str] = (), markdown: Optional[str] = None,
           title: Optional[str] = None) -> None:
    """Write one table as CSV and Markdown under ``cfg.out``."""
    heading = title or _TITLES.get(name, name)
    write_table(cfg.out, name, table, cfg.provenance(heading, sources, notes), markdown)


_TITLES: Dict[str, str] = {
    "verification": "Batch C verification: shape, coverage and invariants",
    "r_sweep": "Batch C across the price ratios",
    "tuning": "Batch C baseline tuning",
    "sensitivity_32b": "Batch C sensitivity: the 32B-AWQ service model at r = 30",
    "epsilon": "Batch C calendar error x price ratio",
    "adoption_r30": "Batch C adoption sweep at r = 30",
    "margin_r30": "Batch C planning-margin sweep at r = 30",
    "trust_r30": "Batch C BDINTERP trust sweep at r = 30",
    "pool_r30": "Batch C offset-pool sweep at r = 30",
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    ap = argparse.ArgumentParser(description="Batch C results tables, above the base tier")
    ap.add_argument("--root", type=Path, default=PILOT / "outputs" / "batchC",
                    help="batch C output root written by scripts/sim/batch_c.sh")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "sim" / "batch_c")
    ap.add_argument("--min-replicas", type=int, default=DEFAULT_MIN_REPLICAS,
                    help="the always-on base tier every policy pays for")
    ap.add_argument("--c-run", type=float, default=DEFAULT_C_RUN,
                    help="USD per replica-minute, used to price the base tier")
    ap.add_argument("--ratios", type=int, nargs="+", default=[6, 10, 30, 100],
                    help="dimensionless price ratios p_in*mu/c_run the batch ran")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def config_from(args: argparse.Namespace) -> ReportConfig:
    """Assemble the report config from the parsed arguments."""
    return ReportConfig(root=Path(args.root), out=Path(args.out), min_replicas=args.min_replicas,
                        c_run=args.c_run, ratios=tuple(args.ratios))


def run(cfg: ReportConfig) -> int:
    """Write every table; return 1 when an invariant was violated, 0 otherwise."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    violations = write_verification(cfg)
    frames: Dict[int, pd.DataFrame] = {}
    sources: Dict[int, Path] = {}
    for ratio in cfg.ratios:
        ref = reference(cfg, f"r{ratio}")
        if ref is None:
            continue
        matched = reference(cfg, SLO_MATCHED_GROUP) if ratio == SLO_MATCHED_RATIO else None
        frames[ratio] = write_main(cfg, ref, f"main_r{ratio}",
                                   f"Batch C main table, r = {ratio}", matched)
        sources[ratio] = ref.path
        write_weekly(cfg, ref, f"weekly_r{ratio}", f"Batch C weekly table, r = {ratio}")
    write_r_sweep(cfg, frames, sources)
    write_tuning(cfg)
    sensitivity = reference(cfg, SENSITIVITY_GROUP)
    if sensitivity is not None:
        write_main(cfg, sensitivity, "sensitivity_32b", _TITLES["sensitivity_32b"])
    for group, sweep, name in CELL_TABLES:
        write_cells(cfg, group, sweep, name)
    if violations:
        logger.error("%d invariant violation(s); see %s", violations, cfg.out / "verification.md")
        return 1
    logger.info("tables written to %s (no invariant violations)", cfg.out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("pilot.batchc").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    return run(config_from(args))


if __name__ == "__main__":
    raise SystemExit(main())
