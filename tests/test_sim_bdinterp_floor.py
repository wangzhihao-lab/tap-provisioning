"""Under a base tier, BDINTERP-d must still abandon a grossly wrong advice.

The phase test compares the two branches' accumulated costs; the always-on replica is a sunk cost
that both pay, and leaving it inside the comparison pinned the ratio near one (batch C attempt 3:
the robust branch never engaged even at a 20x over-forecast).  The slot cost now excludes it.
"""

from pathlib import Path

import pytest

from pilot.sim import PolicyConfig, RunConfig, ServiceConfig, WorkloadConfig, engine
from pilot.sim.sweeps import with_price_ratio

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def guard() -> None:
    if not (REPO / WorkloadConfig().slots_csv).exists():
        pytest.skip("timetable slice missing; run scripts/t2/run_t2.py first")


def _runs(forecast_scale: float, trust: float) -> dict:
    cfg = RunConfig(workload=WorkloadConfig(college="college_18", weeks=(1,), adoption=1.0,
                                            sigma=0.7, forecast_scale=forecast_scale),
                    service=ServiceConfig(lead_min=0.5),
                    policy=PolicyConfig(min_replicas=1, trust=trust), seed=11, root=str(REPO))
    stream = engine.build_stream(cfg)
    cfg = with_price_ratio(cfg, engine.fluid_mu(cfg, stream), 6.0)
    return engine.run_all(cfg, policies=("calendar", "bdinterp", "lcp", "offline"), stream=stream)


def test_robust_branch_engages_under_a_gross_over_forecast_with_a_base_tier() -> None:
    runs = _runs(forecast_scale=20.0, trust=1.0)
    ftp, bd, lcp = (runs[k].metrics.total_cost for k in ("calendar", "bdinterp", "lcp"))
    base = 0.0333 * 10080  # the always-on replica for one week, paid by every policy
    assert ftp - base > 2.0 * (lcp - base), "the advice must be grossly worse than LCP above the tier"
    assert bd < 0.6 * ftp, "BDINTERP-d at eps = 1 must leave a twenty-fold over-forecast behind"
    assert bd < 1.1 * lcp


def test_advice_is_followed_when_it_is_good() -> None:
    runs = _runs(forecast_scale=1.0, trust=1.0)
    ftp, bd = (runs[k].metrics.total_cost for k in ("calendar", "bdinterp"))
    assert bd == pytest.approx(ftp, rel=0.02)
