# TAP: Timetable-Aware Provisioning for LLM Teaching-Assistant Services

Code and data for the paper "Timetable-Aware Provisioning for LLM Teaching-Assistant Services"
(TAP). The repository holds the provisioning simulator and its policies, the scripts that
produce every table and figure of the paper and its supplementary material, the derived input
data the simulator reads, and the result tables the paper reports.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```bash
uv sync
uv run pytest -q
```

`uv sync` creates `.venv/` from `uv.lock`. The GPU measurement scripts under `scripts/q2`
additionally need vLLM on the GPU machine (see `scripts/q2/README.md`); nothing else does.

## Repository layout

```
src/pilot/            Python package
  sim/                discrete-event provisioning simulator; policies/ holds TAP and every baseline
  timetable/          parser and statistics of the timetable export
  assistantlogs/      parser and metrics of the AI-assistant session exports
  sources/, metrics/  arrival sources and burst metrics of the arrival-structure analysis
  batchc/, validation/  readers of the sweep outputs
scripts/
  sim/                sweeps, baseline tuning, result tables (canonical_*.py), figures, run queues
  theory/             numerical instantiation of the bounds and the theory-vs-simulation table
  figs/               Figs. 1 and 2
  t2/                 timetable statistics
  q1/                 arrival-structure analysis and offset pools
  q2/                 GPU measurements, burst replay and the real-machine replica pool
  audit/              consistency checks (prefix invariance, number audit)
  data/               builds the released derived tables from the raw exports
tests/                unit and integration tests (pytest)
data/
  timetable/          de-identified timetable slots, three semesters
  offsets/            arrival-offset pools (minutes after a session's first arrival)
  tokens/             prompt/output token lengths of the Azure 2023 conversation trace
results/
  runs/               per-run summaries behind Table II and Figs. 3-4 (paired_cell_c.csv, cells_laptop.csv, ...)
  sim/                validation-round tables (validation/tables) and earlier sweep summaries (batch_c)
  theory/             instantiated bounds and the theory-vs-simulation comparison
  q1/, t2/, q2/       arrival-structure, timetable and GPU-replay summaries
  tables/             the LaTeX tables as printed in the paper
  paper_numbers.md    every number quoted in the paper, with its source
outputs/              run outputs (created by the scripts); ships only the tuned baseline
                      configurations (outputs/**/best.json) that the runs read
```

All paths are relative to the repository root, and every script is run from there.

## Data

### Derived data in this repository

| file | content | de-identification |
|---|---|---|
| `data/timetable/slots_<semester>.csv` | one row per section x weekly slot: section number, campus code, college code, roster size, weekday, periods, teaching weeks (semesters 2025-2026-1, 2025-2026-2, 2026-2027-1) | section numbers are a running index; campus and college names replaced by `campus_<k>` and `college_<kk>` (numbered in the sort order of the names); course names and codes, class labels, rooms and teachers removed |
| `data/offsets/signin_offsets_min.csv` | 3,054 check-in offsets from 82 classroom sign-in sessions | minutes after the session's first check-in, rounded to 0.001 min and sorted; no student, class or activity identifier |
| `data/offsets/prompted_ai_offsets_min.csv`, `selfpaced_ai_offsets_min.csv` | per-student first-arrival offsets of the AI-assistant sessions (prompted theory class, self-paced lab class) | offsets only; no identifier and no message text |
| `data/offsets/assistments_inclass_offsets_min.csv` | 200,000 first-arrival offsets of in-class ASSISTments sessions | derived from public data (below) |
| `data/tokens/azure_conv_tokens.csv` | `ContextTokens`, `GeneratedTokens` of the Azure LLM inference trace 2023 (conversation) | public data; the timestamp column is dropped |
| `scripts/q2/traces/burst_20260324_0830.*` | the classroom burst replayed on the GPU: request offsets and prompt/reply lengths | students renumbered 0..93; no text |
| `scripts/q2/results/**.csv` | capacity calibrations (`T3_calibration.csv`) and per-minute records of the replica-pool rounds | measurements only |

The result tables under `results/` hold aggregates only (per policy, case, tier, cell, week
and seed). The AI-assistant summaries in `results/q1/assistant/` are per day, per burst or
per bin; the sign-in summaries in `results/q1/` name sessions as `course_<k>_session_<nnn>`.

### Data that are not distributed

* The university timetable exports (course names, class labels, rooms and teachers) and the
  classroom sign-in exports (student names and ids) are internal records of the university.
* The AI-assistant session exports hold student ids and the text of every question.

Neither can be published. `scripts/data/make_public_tables.py` (timetable slots, token
lengths), `scripts/q1/offset_pools.py` (sign-in and ASSISTments pools) and
`scripts/q1/assistant_workload.py` (AI-assistant pools and summaries) show exactly how each
released table is derived from them; `scripts/t2/run_t2.py`, `scripts/q1/signin_sessions.py`
and `scripts/q1/xxt_assistant_sessions.py` are the parsers of the raw formats.

### Public data

* ASSISTments 2012-2013 school data (`2012-2013-data-with-predictions-4-final`), available from
  the ASSISTments data page, https://sites.google.com/site/assistmentsdata/ (dataset
  "2012-13 School Data with Affect"). With the file on disk,
  `uv run python scripts/q1/offset_pools.py assistments --path <file>` rebuilds the released
  pool: a uniform subsample (seed 0) of 200,000 of the 211,068 first arrivals in the 10,362
  in-class sessions.
* Azure LLM inference trace 2023, https://github.com/Azure/AzurePublicDataset
  (`AzureLLMInferenceTrace_conv.csv`).

## Reproducing the paper

### From the result tables (no simulation)

| item | command | output |
|---|---|---|
| Table II and the supplementary timing-cell, weekly and boundary-case tables | `uv run python scripts/sim/canonical_tables.py --from-results` | `results/tables/` |
| Fig. 1 (bell timeline) | `uv run python scripts/figs/make_fig1.py` | `results/figures/` |
| Fig. 2 (framework) | `uv run python scripts/figs/make_fig2.py` | `results/figures/` |
| Fig. 3 (a) and (b) | `uv run python scripts/sim/validation_figures.py --only f4` and `--only f4b` | `results/sim/validation/figures/` |
| Fig. 4 (timing cells) | `uv run python scripts/sim/validation_figures.py --only f5` | same |
| Supplementary information-ladder figure | `uv run python scripts/sim/validation_figures.py --only f8` | same |
| Supplementary arrival-structure panels (a)-(c) | `uv run python scripts/q1/make_assistant_figures.py` | `results/q1/assistant/figures/` |
| Supplementary arrival-structure panel (d) | `uv run python scripts/t2/plots.py` | `results/t2/` |
| Supplementary replay figure | `uv run python scripts/q2/replay_compare.py --draw-only` | `results/q2/replay/` |

Tables I and III of the main text are descriptive and are typeset in the manuscript. Fig. 5
(lead time), the supplementary forecast-error figure and the remaining supplementary tables
(tuning, ablation, sensitivity, billing, pool) read the per-run outputs under `outputs/`,
which are regenerated by the runs below; with those in place, `canonical_tables.py` without
`--from-results`, `validation_figures.py --only f7` and `scripts/sim/canonical_numbers.py`
(which writes `results/paper_numbers.md`) rebuild them.

### Simulation runs

One simulation run covers one capacity case, one provisioning tier, a set of timing cells, a
set of seeds and the policies named. For example, the central case c2 (32B, shared context) at
the instance tier, cell C, one week and one seed:

```bash
uv run python scripts/sim/run_sweep.py --sweep offset --offset-cells C --weeks 1 --seeds 10 \
    --adoption 1.0 --min-replicas 1 --c-run 0.0333 --beta-start 0.0333 --lead-stat median \
    --policies offline calendar calendar_learn calendar_learn_prior stale_calendar tap reactive velocity \
    --calibration scripts/q2/results/b4_remote/b4/ladder/rag_shared_4000/T3_calibration.csv \
    --model qwen2.5-32b-awq --slo-in-s 2 --lead-tier d_inst --lead-min 8 --p-in 0.002500756 \
    --learn-prior-from data/timetable/slots_2025-2026-2.csv --out outputs/smoke
```

This takes about 15 s on one core of a laptop CPU; time grows linearly in weeks, seeds and
cells. The full set of runs behind the paper is generated as job lists:

1. Baseline tuning: `scripts/sim/tune_baselines.py`. The configurations used in the paper are
   included as `outputs/**/best.json` (`outputs/validation_remote/` for the validation round,
   `outputs/validation_local/` for the later runs, `outputs/batchC/` for the earlier sweeps).
2. Validation round (all cases, both tiers, six timing cells, five seeds, six weeks):
   `uv run python scripts/sim/cluster/jobs.py --boxes 1 --out outputs/jobs`, then
   `bash scripts/sim/cluster/run_box.sh outputs/jobs/box1 3 8` (phase 1 re-runs the tuning into
   `outputs/validation/`, phase 2 the sweeps); tables with
   `uv run python scripts/sim/validation_report.py`.
3. TAP runs on the paired streams (Table II, Figs. 3-4):
   `uv run python scripts/sim/runs_jobs.py`, then
   `uv run python scripts/sim/runs_runner.py --workers 6`; tables with
   `scripts/sim/runs_paired_table.py` and `scripts/sim/runs_cells_table.py`.

Steps 2 and 3 took several hundred CPU-hours in total (step 3 alone is a queue of about
14-19 h on a six-worker laptop). Every run is deterministic given its arguments and seed: the
smoke run above reproduces the corresponding rows of the original runs bit for bit.

### Theory

`uv run python scripts/theory/instantiate.py` and `scripts/theory/t5_*.py` instantiate the
bounds at the measured calibration and compare them with the simulation.

## Citation

```bibtex
@unpublished{wang2026tap,
  author = {Zhihao Wang and Min Ren and Yuanyuan Liu and Gaoyong Li and Ding Ding},
  title  = {Timetable-Aware Provisioning for {LLM} Teaching-Assistant Services},
  note   = {Manuscript submitted to IEEE Transactions on Services Computing},
  year   = {2026}
}
```

## License

Code: MIT (`LICENSE`). Derived data and result tables: CC BY 4.0 (`data/LICENSE-DATA`).
Third-party data keep their own terms (see `data/LICENSE-DATA`).
