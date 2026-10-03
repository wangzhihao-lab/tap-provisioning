"""Service model: the concurrency -> TTFT curve, priority, timeouts and the T3 loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pilot.sim import ServiceConfig
from pilot.sim.service import load_service_config, replay
from pilot.sim.workload import IN_CLASS, OFF_CLASS, ArrivalStream

PILOT = Path(__file__).resolve().parents[1]
T3_CSV = PILOT / "scripts" / "q2" / "results" / "dryrun" / "T3_calibration.csv"


def burst_stream(n: int, horizon_min: int = 10, kind: int = IN_CLASS,
                 output_tokens: int = 4000, at_s: float = 0.0) -> ArrivalStream:
    """``n`` requests arriving together, long enough that none finishes during the burst."""
    t = np.full(n, at_s, dtype=float)
    return ArrivalStream(
        t_arrive_s=t, kind=np.full(n, kind, dtype=np.int8),
        session_id=np.full(n, -1, dtype=np.int64),
        prompt_tokens=np.full(n, 512, dtype=np.int64),
        output_tokens=np.full(n, output_tokens, dtype=np.int64),
        lam_in=np.zeros(horizon_min), lam_off=np.zeros(horizon_min),
        lam_hat_in=np.zeros(horizon_min), lam_hat_off=np.zeros(horizon_min),
        sessions=pd.DataFrame(columns=["session_id", "bell_min"]), horizon_min=horizon_min)


@pytest.mark.parametrize("n", [1, 8, 16, 32])
def test_ttft_matches_the_calibration_table(n: int) -> None:
    """Acceptance (d): the model reproduces the concurrency -> TTFT p95 table it was given.

    The table is still the PLACEHOLDER of ServiceConfig; re-point this at the batch B curve once
    T3 lands.  The k-th request of a simultaneous burst is admitted at concurrency k, so the burst
    sweeps the whole curve in one replay.
    """
    cfg = ServiceConfig()
    result = replay(burst_stream(n), np.ones(10), cfg)
    served = result.ttft_s[result.served]
    assert served.size == n
    expected = np.array([cfg.ttft_p95_s(k) for k in range(1, n + 1)])
    assert np.allclose(np.sort(served), expected, rtol=0.10)
    assert served.max() == pytest.approx(cfg.ttft_p95_s(n), rel=0.10)


def test_capacity_is_k_per_replica() -> None:
    """Only ``K * replicas`` streams run at once; the rest wait."""
    cfg = ServiceConfig(abandon_in_s=1e9)
    result = replay(burst_stream(80), np.ones(10), cfg)
    assert int(result.served.sum()) == 80
    late = np.sort(result.ttft_s[result.served])[-10:]
    assert late.min() > cfg.ttft_at_k_s  # the last arrivals queued behind a full replica


def test_zero_replicas_time_everything_out() -> None:
    """With no warm replica every request abandons and none is served."""
    cfg = ServiceConfig(abandon_in_s=30.0)
    result = replay(burst_stream(20, horizon_min=5), np.zeros(5), cfg)
    assert int(result.served.sum()) == 0
    assert int(result.timed_out.sum()) == 20


def test_in_class_requests_have_priority() -> None:
    """A full pool serves the in-class queue first."""
    off = burst_stream(40, kind=OFF_CLASS, output_tokens=6000)
    ins = burst_stream(8, kind=IN_CLASS, output_tokens=6000, at_s=1.0)
    stream = ArrivalStream(
        t_arrive_s=np.concatenate([off.t_arrive_s, ins.t_arrive_s]),
        kind=np.concatenate([off.kind, ins.kind]),
        session_id=np.concatenate([off.session_id, ins.session_id]),
        prompt_tokens=np.concatenate([off.prompt_tokens, ins.prompt_tokens]),
        output_tokens=np.concatenate([off.output_tokens, ins.output_tokens]),
        lam_in=off.lam_in, lam_off=off.lam_off, lam_hat_in=off.lam_hat_in,
        lam_hat_off=off.lam_hat_off, sessions=off.sessions, horizon_min=off.horizon_min)
    cfg = ServiceConfig(max_concurrency=32, abandon_in_s=1e9, abandon_off_s=1e9)
    result = replay(stream, np.ones(10), cfg)
    is_in = stream.kind == IN_CLASS
    assert np.nanmax(result.ttft_s[is_in]) < np.nanmax(result.ttft_s[~is_in])


def test_priority_off_serves_in_arrival_order() -> None:
    """With the priority switched off, in-class requests wait behind earlier off-class ones."""
    off = burst_stream(40, kind=OFF_CLASS, output_tokens=6000)
    ins = burst_stream(8, kind=IN_CLASS, output_tokens=6000, at_s=1.0)
    stream = ArrivalStream(
        t_arrive_s=np.concatenate([off.t_arrive_s, ins.t_arrive_s]),
        kind=np.concatenate([off.kind, ins.kind]),
        session_id=np.concatenate([off.session_id, ins.session_id]),
        prompt_tokens=np.concatenate([off.prompt_tokens, ins.prompt_tokens]),
        output_tokens=np.concatenate([off.output_tokens, ins.output_tokens]),
        lam_in=off.lam_in, lam_off=off.lam_off, lam_hat_in=off.lam_hat_in,
        lam_hat_off=off.lam_hat_off, sessions=off.sessions, horizon_min=off.horizon_min)
    cfg = ServiceConfig(max_concurrency=32, abandon_in_s=1e9, abandon_off_s=1e9, in_class_priority=False)
    result = replay(stream, np.ones(10), cfg)
    is_in = stream.kind == IN_CLASS
    assert np.nanmin(result.ttft_s[is_in]) > np.nanmax(result.ttft_s[~is_in][:32])


def test_rejection_when_the_queue_is_full() -> None:
    """Beyond ``max_queue_in`` a request is turned away at the door rather than queued."""
    cfg = ServiceConfig(max_queue_in=5, abandon_in_s=1e9)
    result = replay(burst_stream(60, output_tokens=8000), np.zeros(10), cfg)
    assert int(result.rejected.sum()) == 55
    assert int(result.timed_out.sum()) == 5


def test_t3_loader_reads_the_calibration_csv() -> None:
    """A T3 CSV replaces the placeholder numbers; K is where TTFT p95 reaches the SLO."""
    if not T3_CSV.exists():
        pytest.skip("no T3 calibration CSV in the tree")
    cfg = load_service_config(T3_CSV, model="Qwen/Qwen2.5-7B-Instruct", slo_in_s=2.0)
    assert cfg.source == str(T3_CSV)
    assert cfg.model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.max_concurrency > 20  # the dry-run curve is still flat at N = 20
    assert cfg.ttft_at_k_s == pytest.approx(2.0)
    assert 0.0 < cfg.ttft_floor_s < 2.0
    assert cfg.decode_tokens_per_s > 1000.0
    assert cfg.lead_min > 0.0
    assert cfg.ttft_p95_s(1) == pytest.approx(cfg.ttft_floor_s)
    assert cfg.ttft_p95_s(cfg.max_concurrency) == pytest.approx(2.0)


def test_requests_per_min_is_monotone() -> None:
    """Longer answers mean fewer requests per minute per replica."""
    cfg = ServiceConfig()
    assert cfg.requests_per_min(100.0) > cfg.requests_per_min(500.0) > 0.0
    assert cfg.stream_tokens_per_s(1) >= cfg.stream_tokens_per_s(32)
