# GPU measurements and replay (`scripts/q2`)

Measurement and replay code behind the capacity calibration, the lead times and the
real-machine validation of the paper. Running it needs a GPU machine with vLLM; every
script also has a `--dry-run` mode that runs on a CPU against `fake_server.py`.

| file | role |
|---|---|
| `coldstart_vllm.py` | replica start: launch `vllm serve`, timestamp every start-up stage |
| `instance_acquire.py` | instance acquisition: milestone log on the instance and a report mode |
| `concurrency_curve.py` | closed-loop load generator: concurrency -> TTFT / throughput |
| `prefix_warmup.py` | shared course-material prefix, warm-up vs no warm-up |
| `summarize.py` | aggregate into `q2_summary.csv`, `T3_calibration.csv` and one figure |
| `compare_calibrations.py` | compare calibration tables |
| `make_replay_trace.py`, `replay_client.py`, `replay_compare.py` | replay of one measured classroom burst and its comparison with the simulator |
| `make_bell_trace.py`, `replica_pool.py`, `pool_*.py`, `bell_window.py` | the policy-driven replica pool and its comparison with the simulator |
| `vendor_policies.py`, `simpolicies/` | the simulator's feedback policies, vendored for a machine without the `pilot` package |
| `remote/` | stage driver on the GPU machine (`stages.py`, `run_all.sh`, `bootstrap.sh`) and the local orchestrator (`orchestrate.py`) |
| `results/` | the calibration and pool CSVs the simulator and the paper read |

## Dry run (CPU only)

```bash
python scripts/q2/coldstart_vllm.py    --dry-run --out scripts/q2/results/dryrun
python scripts/q2/concurrency_curve.py --dry-run --out scripts/q2/results/dryrun
python scripts/q2/prefix_warmup.py     --dry-run --out scripts/q2/results/dryrun
python scripts/q2/summarize.py         --dry-run
```

No number produced by a dry run is a measurement; every record carries `"dry_run": true`.

## Real run

Connection details are never stored in the code. The orchestrator takes them as arguments
or environment variables, and the operator states what happens to the machine after a
successful run, after a failure and at the budget cap:

```bash
export Q2_HOST=<host> Q2_PORT=<port> Q2_USER=<user> Q2_REMOTE_ROOT=<dir on the machine>
python scripts/q2/remote/orchestrate.py --local-out scripts/q2/results/b1_remote \
    --cap-hours 14.5 --on-done keep --on-fail keep --on-cap keep
```

On the machine itself the stages can be run directly with
`bash scripts/q2/remote/run_all.sh --root <dir> --on-done keep --on-fail keep --on-cap keep`.

## Calibration files used by the simulator

* `results/b4_remote/b4/ladder/<regime>/T3_calibration.csv`: capacity ladder (cases c1-c5 use
  `rag_shared_4000`, `rag_unique_2000` and `rag_unique_4000`).
* `results/b1_remote/b1/T3_calibration.csv`: the 7B calibration used by the theory scripts.
* `results/b5_remote/...`: single-replica calibrations and the per-minute pool records of the
  real-machine rounds.
