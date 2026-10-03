"""The sweep's stream cache: every stream built once, dropped after the last cell that needs it.

A sweep whose cells only move prices or policy knobs replays each cached stream in several cells
(the epsilon sweep repeats its error grid at every price ratio), so a stream has to stay cached
until its workload's last cell; an offset sweep gives every cell its own workload, so holding all
of them only costs memory (six cells x five seeds of an institution-wide six-week run is several
GB).  Neither may change a row.
"""

import weakref
from pathlib import Path
from typing import List

import pytest

from pilot.sim import RunConfig, SweepConfig, WorkloadConfig
from pilot.sim import sweeps

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def slice_present() -> None:
    """Skip when the timetable slice has not been built."""
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def _base() -> RunConfig:
    return RunConfig(workload=WorkloadConfig(college="college_04", weeks=(1,), adoption=0.4,
                                            max_sections=6), seed=0, root=str(REPO))


def test_a_workload_that_recurs_is_not_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two price ratios x two forecast scales: two streams, each built once, four cells of rows."""
    built: List[tuple] = []
    real = sweeps.build_stream

    def counting(cfg):
        built.append((cfg.workload, cfg.seed))
        return real(cfg)

    monkeypatch.setattr(sweeps, "build_stream", counting)
    grid = SweepConfig(price_ratio=(6.0, 18.0), forecast_scale=(1.0, 2.0), bell_shift_frac=(),
                       p_cancel=())
    result = sweeps.run_sweep("epsilon", _base(), grid, seeds=[3], policies=["offline"])
    assert len(built) == 2 and len(set(built)) == 2
    assert len(result.frame[result.frame["week"] == 0]) == 4


def test_an_offset_sweep_holds_one_stream_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """At every build only the previous stream, still bound in the loop, is alive."""
    refs: List[weakref.ref] = []
    alive_at_build: List[int] = []
    real = sweeps.build_stream

    def tracking(cfg):
        alive_at_build.append(sum(r() is not None for r in refs))
        stream = real(cfg)
        refs.append(weakref.ref(stream))
        return stream

    monkeypatch.setattr(sweeps, "build_stream", tracking)
    grid = SweepConfig(trigger_offset=SweepConfig().trigger_offset[:2])
    result = sweeps.run_sweep("offset", _base(), grid, seeds=[3, 4, 5], policies=["offline"])
    assert alive_at_build == [0, 1, 1, 1, 1, 1]  # without eviction: [0, 1, 2, 3, 4, 5]
    assert len(result.frame[result.frame["week"] == 0]) == 6
