"""B' step 2: the bell window export and the pool comparison, on a synthetic stream.

``scripts/q2/bell_window.py``, ``make_bell_trace.py`` and ``pool_compare.py`` are CLIs rather
than a package, so ``scripts/q2`` goes on ``sys.path`` first.  The stream here is a small
hand-made ``ArrivalStream`` (three bells, in- and off-class requests), so every test runs in
seconds; the real export is exercised by running the script itself.  The pool's results are a
fixture written the way design section 5 describes them, so ``pool_compare`` reads, compares and
draws end to end without a box.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

PILOT = Path(__file__).resolve().parents[1]
REPO = PILOT
for _extra in (PILOT / "scripts" / "q2", PILOT / "scripts" / "sim"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import bell_window as bw  # noqa: E402
import make_bell_trace as mbt  # noqa: E402
import pool_compare as pc  # noqa: E402
from replay_client import load_trace  # noqa: E402  (bell_window makes it importable sans requests)
from pilot.sim import RunConfig, WorkloadConfig  # noqa: E402
from pilot.sim.workload import IN_CLASS, OFF_CLASS, ArrivalStream  # noqa: E402

BELL = 40
SPEC = bw.WindowSpec(bell_min=BELL, before_min=6, after_min=9)  # a 15-minute window, horizon 25
SESSION_COLUMNS = ["session_id", "section_id", "college", "week", "weekday", "bell_min",
                   "duration_min", "roster", "trigger_offset_min", "realised_start_min"]


def synthetic_stream(seed: int = 0, horizon: int = 90) -> ArrivalStream:
    """Three bells (10, 40, 70), a burst of in-class requests around each, off-class noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for bell, n_sessions in ((10, 1), (BELL, 3), (70, 1)):
        for _ in range(n_sessions):
            sid = len(rows)
            rows.append((sid, 100 + sid, "X", 1, 1, bell, 45.0, 30, -5.0, float(bell - 4)))
    sessions = pd.DataFrame(rows, columns=SESSION_COLUMNS)
    t, sid, stu = [], [], []
    for row in sessions.itertuples(index=False):
        first = (row.bell_min - 5) * 60.0 + rng.exponential(200.0, size=24)
        for j, f in enumerate(first):
            for k in range(2):  # two requests per student
                t.append(f + k * rng.exponential(170.0))
                sid.append(int(row.session_id))
                stu.append(int(row.session_id) * 100 + j)
    n_in = len(t)
    off_t = np.sort(rng.uniform(0.0, horizon * 60.0, size=2 * horizon))
    t_all = np.concatenate([np.asarray(t), off_t])
    kind = np.concatenate([np.full(n_in, IN_CLASS, dtype=np.int8),
                           np.full(off_t.size, OFF_CLASS, dtype=np.int8)])
    session = np.concatenate([np.asarray(sid, dtype=np.int64), np.full(off_t.size, -1)])
    student = np.concatenate([np.asarray(stu, dtype=np.int64), np.full(off_t.size, -1)])
    keep = t_all < horizon * 60.0
    order = np.argsort(t_all[keep], kind="stable")
    t_all, kind, session, student = (a[keep][order] for a in (t_all, kind, session, student))
    prompt = rng.choice([50, 400, 1020, 2800, 7930], size=t_all.size)
    output = rng.choice([1, 16, 64, 128, 210, 500, 1000], size=t_all.size)
    minute = np.minimum((t_all // 60.0).astype(np.int64), horizon - 1)
    lam_in = np.bincount(minute[kind == IN_CLASS], minlength=horizon).astype(float)
    lam_off = np.bincount(minute[kind == OFF_CLASS], minlength=horizon).astype(float)
    hat_in = np.convolve(lam_in, [0.25, 0.5, 0.25], mode="same")
    return ArrivalStream(t_all, kind, session, prompt, output, lam_in, lam_off, hat_in, lam_off.copy(),
                         sessions, horizon, lam_stale_in=None, student_id=student,
                         lam_hat_learn_in=np.roll(hat_in, 1), lam_hat_learn_prior_in=np.roll(hat_in, 2))


@pytest.fixture(scope="module")
def stream() -> ArrivalStream:
    return synthetic_stream()


@pytest.fixture(scope="module")
def window(stream: ArrivalStream) -> ArrivalStream:
    return bw.cut_window(stream, SPEC)


def test_cut_window_keeps_the_windows_in_class_requests_shifted_to_t0(stream, window) -> None:
    lo, hi = SPEC.start_min * 60.0, SPEC.end_min * 60.0
    expected = (stream.kind == IN_CLASS) & (stream.t_arrive_s >= lo) & (stream.t_arrive_s < hi)
    assert window.n_requests == int(expected.sum()) > 0
    assert (window.kind == IN_CLASS).all()
    assert window.t_arrive_s.min() >= 0.0 and window.t_arrive_s.max() < SPEC.duration_min * 60.0
    assert np.all(np.diff(window.t_arrive_s) >= 0.0)
    np.testing.assert_allclose(window.t_arrive_s, np.round(stream.t_arrive_s[expected] - lo, 3))
    assert window.horizon_min == SPEC.horizon_min == 25
    assert window.lam_in.size == 25 and window.lam_in.sum() == window.n_requests
    assert window.lam_in[SPEC.duration_min:].sum() == 0.0  # the tail carries no arrival
    assert not window.lam_off.any() and not window.lam_hat_off.any()
    np.testing.assert_allclose(window.lam_hat_in, stream.lam_hat_in[SPEC.start_min: SPEC.start_min + 25])
    np.testing.assert_allclose(window.lam_hat_learn_prior_in,
                               stream.lam_hat_learn_prior_in[SPEC.start_min: SPEC.start_min + 25])
    at_bell = window.sessions[window.sessions["bell_min"] == SPEC.before_min]
    assert len(at_bell) == 3  # the bell's sessions, re-timed so the bell is minute 6
    assert set(window.sessions["session_id"]) >= set(window.session_id)
    assert (window.sessions["bell_min"] != 70 - SPEC.start_min).all()  # a later bell is not touched


def test_trace_round_trips_through_the_client(window, tmp_path: Path) -> None:
    trace = bw.trace_frame(window)
    assert list(trace.columns) == ["offset_s", "student", "prompt_chars", "reply_chars"]
    assert trace["prompt_chars"].max() <= 8700
    path = tmp_path / "trace.csv"
    trace.to_csv(path, index=False, float_format="%.3f")
    reqs = load_trace(path, scale=1, max_offset=SPEC.duration_min * 60, seed=0)
    assert len(reqs) == window.n_requests
    wanted = np.clip(window.output_tokens, bw.OUTPUT_TOKEN_FLOOR, bw.OUTPUT_TOKEN_CAP)
    assert [r.idx for r in reqs] == list(range(len(reqs)))
    assert [r.max_tokens for r in reqs] == wanted.tolist()  # the client sends the intended count
    np.testing.assert_allclose([r.offset_s for r in reqs], window.t_arrive_s)
    aligned = bw.with_client_lengths(window, trace)
    assert aligned.output_tokens.tolist() == [r.max_tokens for r in reqs]
    assert aligned.prompt_tokens.tolist() == [bw.prompt_tokens_of(r.idx, r.prompt_chars) for r in reqs]
    assert aligned.prompt_tokens.max() <= bw.PROMPT_TOKEN_CAP + 30
    rebuilt = bw.stream_from_records(trace.assign(idx=np.arange(len(trace))), SPEC.horizon_min)
    np.testing.assert_array_equal(rebuilt.output_tokens, aligned.output_tokens)
    np.testing.assert_array_equal(rebuilt.prompt_tokens, aligned.prompt_tokens)
    np.testing.assert_allclose(rebuilt.t_arrive_s, aligned.t_arrive_s)
    np.testing.assert_allclose(rebuilt.lam_in, aligned.lam_in)


def test_reply_chars_inverse_covers_the_client_clip() -> None:
    for tokens in range(bw.OUTPUT_TOKEN_FLOOR, bw.OUTPUT_TOKEN_CAP + 1):
        chars = bw.reply_chars_for(tokens)
        assert bw.output_tokens_of(chars) == tokens
        assert abs(chars - tokens * 1.45) <= 1.0


def test_thin_window_keeps_a_bernoulli_share_and_scales_the_forecasts(window) -> None:
    thin = bw.thin_window(window, 0.5, seed=7)
    n = window.n_requests
    assert 0.3 * n <= thin.n_requests <= 0.7 * n  # 131 draws at p = 0.5
    assert thin.lam_in.sum() == thin.n_requests and thin.horizon_min == window.horizon_min
    assert np.all(np.diff(thin.t_arrive_s) >= 0.0) and set(thin.t_arrive_s) <= set(window.t_arrive_s)
    np.testing.assert_allclose(thin.lam_hat_in, 0.5 * window.lam_hat_in)
    np.testing.assert_allclose(thin.lam_hat_learn_prior_in, 0.5 * window.lam_hat_learn_prior_in)
    np.testing.assert_allclose(thin.lam_hat_learn_in, 0.5 * window.lam_hat_learn_in)
    assert thin.lam_stale_in is None and not thin.lam_off.any()
    assert thin.sessions.equals(window.sessions)
    assert bw.thin_window(window, 1.0) is window
    again = bw.thin_window(window, 0.5, seed=7)
    np.testing.assert_array_equal(again.t_arrive_s, thin.t_arrive_s)  # the seed fixes the draw


def test_capped_service_uses_the_measured_ttft_at_k() -> None:
    if not bw.CALIB_CSV.exists():
        pytest.skip("calibration card missing")
    service = bw.capped_service(1.0)
    at_k = bw.measured_ttft_p95(bw.CALIB_CSV, bw.MODEL_ROW, bw.MAX_CONCURRENCY)
    assert service.max_concurrency == 10 and service.lead_min == 1.0
    assert service.ttft_at_k_s == at_k and 0.0 < at_k < bw.SLO_S  # the card's N = 10 value, not the SLO
    assert service.ttft_floor_s == bw.measured_ttft_p95(bw.CALIB_CSV, bw.MODEL_ROW, 1)
    with pytest.raises(ValueError, match="N=99"):
        bw.measured_ttft_p95(bw.CALIB_CSV, bw.MODEL_ROW, 99)


@pytest.fixture(scope="module")
def inputs(stream: ArrivalStream, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The generator's whole output for the synthetic window (contexts, plans, predictions)."""
    if not bw.CALIB_CSV.exists() or not all(p.exists() for p in bw.TUNED.values()):
        pytest.skip("calibration card or tuned best.json files missing")
    base = RunConfig(workload=WorkloadConfig(weeks=(1,)), seed=3, root=str(REPO))
    out = tmp_path_factory.mktemp("q2") / "pool_inputs"
    mbt.build_inputs(stream, SPEC, base, out, week=1, mu_ref=300.0)  # f = mu_cap / 300 < 1
    return out


def test_generator_writes_every_input_in_the_design_formats(inputs: Path) -> None:
    names = {p.name for p in inputs.glob("*")}
    assert {"bell_c3_w3.csv", "bell_c3_w3.json", "plan_learnprior.json", "plan_offline.json",
            "sim_predictions.json", "pool_runs.json"} <= names
    for tier in ("d_rep", "d_inst"):
        for policy in ("velocity", "reactive", "plan"):
            ctx = json.loads((inputs / f"context_{tier}_{policy}.json").read_text())
            assert ctx["lead_min"] == int(bw.LEAD_MIN[tier]) and ctx["horizon_min"] == 15
            assert ctx["x_max"] == int(bw.X_MAX) and ctx["min_replicas"] == 1 and ctx["mu"] > 0
            assert {"c_run", "beta_start", "p_in", "p_off", "slo_in_s", "x_max"} <= set(ctx["cost"])
            assert ctx["cost"]["c_run"] == ctx["cost"]["beta_start"] == 0.0333
            assert abs(ctx["cost"]["p_in"] * ctx["mu"] / ctx["cost"]["c_run"] - 30.0) < 1e-9
            assert {"min_replicas", "headroom", "hpa_up_util", "velocity_window_min"} <= set(ctx["policy"])
            assert {"max_concurrency", "decode_tokens_per_s", "per_stream_tokens_per_s",
                    "ttft_floor_s", "ttft_at_k_s"} <= set(ctx["service"])
            assert ctx["service"]["max_concurrency"] == 10 and ctx["service"]["lead_min"] == bw.LEAD_MIN[tier]
            assert ctx["service"]["ttft_at_k_s"] < bw.SLO_S  # the measured value at N = 10
            pctx = bw.context_of(ctx)  # the box-side reconstruction of design 5.6
            assert pctx.mu == ctx["mu"] and pctx.lead_min == ctx["lead_min"]
    for stem in ("plan_learnprior", "plan_offline"):
        plan = json.loads((inputs / f"{stem}.json").read_text())
        assert plan["label"] == stem and plan["lead_min"] in (1, 8)
        assert len(plan["x"]) == SPEC.plan_len == 15 + 8 + 1
        assert all(isinstance(v, int) and 1 <= v <= int(bw.X_MAX) for v in plan["x"])
    runs = json.loads((inputs / "pool_runs.json").read_text())
    assert runs["trace"] == "pool_inputs/bell_c3_w3.csv" and runs["duration_min"] == 15
    assert runs["max_offset_s"] == 900 and runs["head_s"] == 240 and runs["abandon_s"] == runs["give_up_s"] == 60.0
    assert runs["calib_multi"] == {"replicas": int(bw.X_MAX), "levels": "8,16,24,32,40", "duration_s": 30}
    assert runs["max_replicas"] == int(bw.X_MAX) and runs["gpu_frac"] == bw.GPU_FRAC
    assert [r["label"] for r in runs["runs"]] == [r.label for r in mbt.RUNS]
    for row in runs["runs"]:
        assert row["context"].startswith("pool_inputs/context_") and row["lead_gate_min"] in (0, 8)
        assert ("plan" in row) == (row["policy"] == "plan")
    predictions = json.loads((inputs / "sim_predictions.json").read_text())
    assert list(predictions) == [r.label for r in mbt.RUNS]
    offline = json.loads((inputs / "plan_offline.json").read_text())["x"]
    for label, pred in predictions.items():
        assert len(pred["x"]) == SPEC.horizon_min and len(pred["share_slo"]) == 15
        assert 0.0 <= pred["slo_in"] <= 1.0 and pred["replica_minutes"] == sum(pred["x"][:15])
        assert pred["x"][0] == 1 and max(pred["x"]) <= int(bw.X_MAX)  # both worlds start from min_replicas
    assert predictions["plan_offline"]["x"][:15] == offline[:15]
    meta = json.loads((inputs / "bell_c3_w3.json").read_text())
    assert meta["n_requests"] == len(pd.read_csv(inputs / "bell_c3_w3.csv"))
    assert meta["lengths"]["output_token_cap"] == 320 and len(meta["lam_in"]) == 15
    thinning = meta["thinning"]
    assert thinning["seed"] == bw.THIN_SEED and thinning["mu_72b_req_min"] == 300.0
    assert abs(thinning["f"] - meta["mu_cap_req_min"] / 300.0) < 1e-12 and 0.0 < thinning["f"] < 1.0
    assert thinning["kept"] == meta["n_requests"] and thinning["dropped"] > 0
    assert thinning["kept"] + thinning["dropped"] == bw.cut_window(synthetic_stream(), SPEC).n_requests
    assert sum(meta["lam_in"]) == meta["n_requests"] and meta["service"]["max_concurrency"] == 10
    assert meta["service"]["ttft_at_k_s"] == bw.measured_ttft_p95(bw.CALIB_CSV, bw.MODEL_ROW, 10)
    learn = json.loads((inputs / "plan_learnprior.json").read_text())
    assert predictions["plan_learnprior_d_inst"]["x"][:15] == learn["x"][:15]
    assert predictions["plan_learnprior_d_rep"]["x"] == predictions["plan_learnprior_d_inst"]["x"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pool_results(inputs: Path, results: Path, labels: List[str], manifest_ok: bool = True
                       ) -> None:
    """A pool's output for ``labels`` in the box's own formats (``replica_pool.py``,
    ``pool_control.MINUTE_COLUMNS``, ``pool_replicas.Replica.record``, ``vendor_policies``'
    MANIFEST), with the admitted trajectory following the simulator's prediction."""
    predictions = json.loads((inputs / "sim_predictions.json").read_text())
    runs = {r["label"]: r for r in json.loads((inputs / "pool_runs.json").read_text())["runs"]}
    trace = pd.read_csv(inputs / "bell_c3_w3.csv")
    rng = np.random.default_rng(1)
    src = PILOT / "src" / "pilot" / "sim"
    sources: Dict[str, Dict[str, object]] = {
        name: {"path": f"src/pilot/sim/{'' if name == 'config.py' else 'policies/'}{name}",
               "sha256": _sha(src / name if name == "config.py" else src / "policies" / name)}
        for name in ("config.py", "base.py", "reactive.py", "velocity.py")}
    if not manifest_ok:
        sources["velocity.py"]["sha256"] = "0" * 64
    manifest = {"schema": "q2.simpolicies.manifest.v1", "generator": "scripts/q2/vendor_policies.py",
                "generated_at": "2026-09-22T00:00:00+0000", "sources": sources,
                "generated": {name: {"sha256": "1" * 64, "source": entry["path"],
                                     "source_sha256": entry["sha256"], "rewrites": []}
                              for name, entry in sources.items()}}
    t0 = 1_800_000_000.0
    for label in labels:
        pred = predictions[label]
        d = results / "runs" / label
        d.mkdir(parents=True)
        ttft = np.where(rng.random(len(trace)) < 0.9, 0.3, 3.0)
        ttft[-1], ttft[-2] = 65.0, np.nan  # one wait beyond the simulator's rule, one give-up
        with (d / f"{label}.jsonl").open("w") as fh:
            for i, row in trace.iterrows():
                fh.write(json.dumps({
                    "idx": int(i), "student": int(row["student"]), "offset_s": float(row["offset_s"]),
                    "prompt_chars": int(row["prompt_chars"]),
                    "max_tokens": bw.output_tokens_of(int(row["reply_chars"])),
                    "ttft_s": None if np.isnan(ttft[i]) else float(ttft[i]), "attempts": 1,
                    "error": None, "dry_run": True}) + "\n")
        n = pred["duration_min"]
        x = np.asarray(pred["x"][:n], dtype=float)
        lead = int(runs[label]["lead_gate_min"]) or 1
        minutes = pd.DataFrame({"t": np.arange(n), "ts": t0 + 60.0 * np.arange(n),
                                "arrivals": pred["arrivals"], "tokens": pred["tokens"],
                                "in_flight": 0, "admitted": x, "installed_t": x,
                                "s": np.arange(n) + lead, "target_s": x})
        ctx = json.loads((inputs.parent / runs[label]["context"]).read_text())
        if runs[label]["policy"] in pc.FEEDBACK:
            minutes["target_s"] = pc.recompute_targets(runs[label]["policy"], ctx, minutes)
        ups = np.maximum(np.diff(x, prepend=x[0]), 0)
        minutes["launched_ids"] = [";".join(str(k) for k in range(int(u))) for u in ups]
        minutes["drained_ids"] = ""
        minutes["cancelled_ids"] = ""
        minutes.to_csv(d / "pool_minutes.csv", index=False)
        with (d / "pool_events.jsonl").open("w") as fh:
            fh.write(json.dumps({"ts": t0 - 180, "minute": -3, "replica_id": 0, "port": 8101,
                                 "event": "launch", "detail": None, "dry_run": True}) + "\n")
        (d / "pool_meta.json").write_text(json.dumps({
            "schema": "q2.pool_meta.v1", "dry_run": True, "policy": runs[label]["policy"],
            "lead_min": lead, "lead_gate_min": runs[label]["lead_gate_min"], "t0_wallclock": t0,
            "context": ctx, "simpolicies_manifest": manifest}))
        for rid, launch in ((0, t0 - 180.0), (1, t0 + 130.0)):
            (d / f"replica_{rid}_server.json").write_text(json.dumps({
                "schema": "q2.pool.replica.v1", "replica_id": rid, "state": "admitted",
                "launch_wallclock": launch, "health_wallclock": launch + 38.0,
                "ready_wallclock": launch + 41.0,
                "admit_wallclock": launch + (41.0 if rid == 0 else 50.0),
                "drain_wallclock": None, "stop_wallclock": None, "health_after_s": 38.0,
                "ready_after_s": 41.0, "warmup_s": 3.0, "gate_wait_s": 0.0 if rid == 0 else 9.0,
                "dry_run": True}))


def test_pool_compare_reads_compares_and_draws(inputs: Path, tmp_path: Path) -> None:
    results = tmp_path / "pool"
    write_pool_results(inputs, results, ["velocity_d_rep", "plan_offline"])
    out = tmp_path / "out"
    started = time.perf_counter()
    summary = pc.compare_all(results, inputs, out)
    assert time.perf_counter() - started < 60.0
    assert {"pool_compare.csv", "pool_minutes_compare.csv", "f10b_pool_validation.pdf",
            "f10b_pool_validation.png"} <= {p.name for p in out.glob("*")}
    assert list(summary["label"]) == ["plan_offline", "velocity_d_rep"]
    per_minute = pd.read_csv(out / "pool_minutes_compare.csv")
    assert len(per_minute) == 2 * 15
    for _, row in summary.iterrows():
        assert row["dry_run"] and row["manifest_ok"]
        assert row["meas_gave_up"] == 1 and row["meas_over_abandon"] == 1
        assert 0.0 <= row["meas_share_slo"] <= 1.0 and 0.0 <= row["simB_share_slo"] <= 1.0
        assert row["x_equal_minutes"] == 15  # the fixture's pool followed the simulator's x
        assert row["meas_replica_minutes"] == row["simA_replica_minutes"]
        assert row["cold_start_n"] == 2 and row["cold_start_median_s"] == 41.0
        assert row["gate_wait_median_s"] == 4.5
    velocity = summary.set_index("label").loc["velocity_d_rep"]
    assert velocity["targets_checked"] == 15 and velocity["targets_mismatched"] == 0
    assert summary.set_index("label").loc["plan_offline", "targets_checked"] == 0
    part = per_minute[per_minute["label"] == "velocity_d_rep"]
    assert part["target_recomputed"].notna().all()
    busy = part["arrivals_sim"] > 0  # a minute without arrivals has no share to compare
    assert busy.any() and part.loc[busy, ["share_meas", "share_simA", "share_simB"]].notna().all().all()
    assert (part["arrivals_meas"] == part["arrivals_sim"]).all()


def test_pool_compare_flags_a_changed_source_and_missing_columns(inputs: Path, tmp_path: Path) -> None:
    results = tmp_path / "pool"
    write_pool_results(inputs, results, ["reactive_d_inst"], manifest_ok=False)
    meta = json.loads((results / "runs/reactive_d_inst/pool_meta.json").read_text())
    ok, bad = pc.manifest_check(meta)
    assert ok is False and bad == ["src/pilot/sim/policies/velocity.py"]
    assert pc.manifest_check({"label": "x"}) == (None, [])
    real = PILOT / "scripts/q2/simpolicies/MANIFEST.json"
    if real.exists():  # the vendored package on disk must match the sources it was copied from
        assert pc.manifest_check({"simpolicies_manifest": json.loads(real.read_text())}) == (True, [])
    minutes = results / "runs/reactive_d_inst/pool_minutes.csv"
    pd.read_csv(minutes).drop(columns=["tokens"]).to_csv(minutes, index=False)
    with pytest.raises(ValueError, match="tokens"):
        pc.load_minutes(minutes, 15)
