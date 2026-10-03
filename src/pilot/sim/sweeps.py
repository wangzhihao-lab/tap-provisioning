"""Parameter grids of the simulator design, one CSV per sweep.

Eight grids: synchrony ``sigma``, calendar error ``epsilon``, lead time ``d``, the BDINTERP-d trust
knob (column ``trust``, the ``epsilon`` of BDINTERP-d), the planning margin ``margin``, adoption
``a``, the empirical offset pools ``pool`` and ``rho`` for the supplementary F9.  Every cell replays the *same* arrival stream
through every policy, and streams are cached by ``(WorkloadConfig, seed)`` so that a sweep which
only moves a policy or service knob (``d``, ``trust``, ``margin``) stays paired across its cells
too.  A cached stream is dropped as soon as the last cell that uses its workload is done with it
(the cell list is known before the loop starts), so a sweep whose every cell has its own workload -
the offset sweep - holds one stream at a time instead of all of them; no stream is built twice.

A policy may be named ``"<key>@<eps>"`` (for example ``bdinterp@0.5``): the policy is then run
with that BDINTERP-d trust knob instead of the cell's, and the ``policy`` column keeps the full
spec while ``trust`` records the value used.  F6 needs policy 7 at three trust values on one error
axis, and this is how it gets them without tripling the grid.  A policy class with a
``fixed_trust`` attribute (``tap``) runs at that value when no ``@eps`` is given, so its ``trust``
column is the value it used rather than the cell's.  Policies that report per-run diagnostics
(a ``diagnostics()`` method, ``tap``) get one row per run in ``SweepResult.diagnostics``.

Every row carries the measured synchrony of its stream - ``sigma2_measured`` (the median over
sessions of the share of per-student first arrivals within 2 min of the session's first arrival)
and ``sigma2_pooled`` (the same share pooled over students) - so that F5 is drawn against the
synchrony a cell realised rather than against ``sigma``, which is only the weight the two offset
pools are mixed with.  Every row also carries the measured calendar error of its run, ``eta`` and
``eta_norm`` (``workload.forecast_error``), so F6 can put the policies on an error axis that was
observed rather than assumed, and the dimensionless price ratio it was priced with, ``p_in * mu / c_run`` - the second
axis of F6 and the parameter that decides whether policy 7's robust branch is ever worth entering.
``price_ratio`` is the value the cell asked for (the facet key, absent when that axis is off) and
``price_ratio_real`` the one the run realised.  Rows come in two kinds, told apart by ``week``:
``week = 0`` is the whole run and ``week = k`` is teaching week ``k`` (``metrics.weekly_rows``).
The offline and informed optima are whole-run quantities; a weekly ratio divides by the whole-run
oracle's cost restricted to that week.
"""

import logging
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from . import PolicyFactory
from .config import LEAD_TIERS, CostConfig, RunConfig, SweepConfig, with_lead_tier
from .engine import DEFAULT_POLICIES, build_stream, fluid_mu, make_context, run_policy
from .metrics import measure_synchrony, weekly_rows
from .workload import ArrivalStream, forecast_error

logger = logging.getLogger(__name__)

Cell = Tuple[Dict[str, object], RunConfig]
Builder = Callable[[RunConfig, SweepConfig, bool], Iterator[Cell]]


def parse_spec(name: str) -> Tuple[str, Optional[float]]:
    """``"bdinterp@0.5"`` -> ``("bdinterp", 0.5)``; a plain name keeps the cell's own trust."""
    base, _, eps = name.partition("@")
    return base, (float(eps) if eps else None)


def spec_trust(name: str) -> Tuple[str, Optional[float]]:
    """``parse_spec`` plus the policy's own ``fixed_trust`` for a spec without ``@eps``."""
    base, eps = parse_spec(name)
    if eps is None:
        eps = getattr(PolicyFactory(base), "fixed_trust", None)
    return base, eps


@dataclass(frozen=True)
class SweepResult:
    """Long-format rows of one sweep, and the per-run diagnostics of the policies that report any."""

    name: str
    frame: pd.DataFrame
    diagnostics: Optional[pd.DataFrame] = None

    def write(self, path) -> None:
        """Write the sweep to ``path`` (parent directories are created)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(self.frame))


def _sigma(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """The synchrony mixture weight w (``WorkloadConfig.sigma``), the knob behind F5's x axis.

    The grid is in w - the share of students drawn from the prompted offset pool - because that is
    what the generator takes.  What F5 plots is ``sigma2_measured``, the synchrony every cell
    actually realised, which ``run_sweep`` measures on the cell's own stream; the quick grid is the
    two endpoints, whose measured sigma_2 are the two data anchors (0.70 sign-ins, 0.24
    ASSISTments).
    """
    grid = (0.0, 1.0) if quick else sweep.sigma
    for sigma in grid:
        yield {"sigma": sigma}, replace(base, workload=replace(base.workload, sigma=sigma))


# the value of each calendar-error knob at which it contributes no error at all
EXACT: Dict[str, float] = {"forecast_scale": 1.0, "bell_shift_frac": 0.0, "p_cancel": 0.0}
# cell key carrying the *requested* price ratio (the facet key of F6); every row also records the
# realised ``p_in * mu / c_run`` under ``PRICE_RATIO_REAL``, which is the measured quantity and is
# the only one available when the axis is off, where it drifts with mu from cell to cell
PRICE_RATIO: str = "price_ratio"
PRICE_RATIO_REAL: str = "price_ratio_real"


def price_ratio_of(cost: CostConfig, mu: float) -> float:
    """The dimensionless price ratio ``p_in * mu / c_run``: penalty per replica-minute."""
    return cost.p_in * mu / max(1e-9, cost.c_run)


def with_price_ratio(cfg: RunConfig, mu: float, ratio: Optional[float]) -> RunConfig:
    """Re-price a cell so that ``p_in * mu / c_run`` equals ``ratio``; ``None`` leaves it alone.

    ``mu`` is the fluid capacity of one replica, which depends on the mean output length of the
    stream, so a *dimensionless* price can only be honoured once the cell's stream exists - hence
    this lives next to the loop rather than in the grid builder.  ``p_off / p_in`` is preserved, so
    the base config (or ``--p-off``) keeps deciding how much cheaper an off-class miss is.
    """
    if ratio is None:
        return cfg
    cost = cfg.cost
    share = cost.p_off / cost.p_in if cost.p_in > 0 else 0.0
    p_in = ratio * cost.c_run / max(1e-9, mu)
    return replace(cfg, cost=replace(cost, p_in=p_in, p_off=p_in * share))


def _epsilon(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """Calendar error x price ratio (F6, F8): one error knob at a time, at each price ratio.

    The three error knobs are independent error sources - a size bias on every booked session
    (``forecast_scale``), a timing error on a fraction of them (``bell_shift_frac``) and sessions
    the plan books that never happen (``p_cancel``).  Sweeping one while the others sit at
    ``EXACT`` keeps the axis clean; roster noise and the attendance band stay at their realistic
    defaults, so even the exact cell has the residual error a real forecast would have, and that
    residual is measured rather than assumed (see ``eta_norm``).

    The second axis is ``SweepConfig.price_ratio``, the dimensionless ``p_in * mu / c_run``: the
    whole error grid is replayed at each ratio on the *same* arrival streams (the ratio moves only
    the cost config, which is not part of the stream cache key).  An empty tuple switches the axis
    off and keeps the base prices, which is the pre-two-axis behaviour.
    """
    grids: Dict[str, Tuple[float, ...]] = {
        "forecast_scale": (0.5, 1.0, 2.0, 20.0) if quick else sweep.forecast_scale,
        "bell_shift_frac": (0.0, 0.3) if quick else sweep.bell_shift_frac,
        "p_cancel": (0.0, 0.08) if quick else sweep.p_cancel,
    }
    ratios: Tuple[Optional[float], ...] = tuple(sweep.price_ratio) or (None,)
    for ratio in ratios:
        for knob, grid in grids.items():
            for value in grid:
                wl = replace(base.workload, **{**EXACT, knob: value})
                params: Dict[str, object] = {
                    "error_knob": knob, "error_value": value,
                    "forecast_scale": wl.forecast_scale,
                    "bell_shift_frac": wl.bell_shift_frac, "p_cancel": wl.p_cancel}
                if ratio is not None:
                    params[PRICE_RATIO] = ratio
                yield params, replace(base, workload=wl)


def _lead(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """The three provisioning tiers of the simulator design (F7).

    A tier is three numbers, not one: the availability delay ``lead_min``, the commitment block the
    capacity has to be constant on and the billing block each replica-block is charged for.  The
    two rented tiers differ only in their lead time; the reserved tier shares the instance lead
    time and adds an hour of commitment and an hour of billing.
    """
    tiers = ("d_rep", "d_inst", "d_res") if quick else sweep.lead_tiers
    for tier in tiers:
        spec = LEAD_TIERS[tier]
        yield ({"lead_tier": tier, "lead_min": spec.lead_min,
                "commit_block_min": spec.commit_block_min,
                "billing_block_min": spec.billing_block_min},
               replace(base, service=with_lead_tier(base.service, tier)))


def _trust(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """BDINTERP-d's epsilon (F6).  Only policy 7 reacts to it: 0 is the calendar, large is LCP."""
    grid = (0.10, 1.00) if quick else sweep.trust
    for lam in grid:
        yield {"trust": lam}, replace(base, policy=replace(base.policy, trust=lam))


def _margin(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """Planning margin kappa: how much more than the forecast policies 6 and 7 book."""
    grid = (1.0, 1.5) if quick else sweep.plan_margin
    for kappa in grid:
        yield ({"plan_margin": kappa},
               replace(base, policy=replace(base.policy, plan_margin=kappa)))


def _adoption(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """Adoption a: how many sections use the service (F9)."""
    grid = (0.05, 0.20) if quick else sweep.adoption
    for a in grid:
        yield {"adoption": a}, replace(base, workload=replace(base.workload, adoption=a))


def _pool(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """Which empirical offset pools the arrival mixture is drawn from.

    The mixture weight ``w`` stays where the cell left it; only the two pools change, so this axis
    asks how much of a result depends on the arrival shape of one data source.  ``signin`` is the
    measured pair (classroom sign-ins, ASSISTments in-class) and ``ai`` the AI-assistant pair of
    batch A, whose prompted side is sharper (2-minute share 0.90 against 0.70).
    """
    grid = sweep.offset_sources or ("signin",)
    for source in grid:
        yield ({"offset_source": source},
               replace(base, workload=replace(base.workload, offset_source=source)))


def _offset(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """Trigger timing (batch A's early bursts): realised offset vs what the planners assume.

    Each cell is ``(realised mean, per-session sd, per-section sd, per-bell common sd, forecast
    offset, forecast sd)``; see ``SweepConfig.trigger_offset`` for the three correlation
    structures and the two planner types.
    """
    grid = ((0.0,) * 6, (-8.7, 5.3, 4.0, 0.0, 0.0, 0.0), (-8.7, 0.0, 0.0, 5.3, -8.7, 5.3)) \
        if quick else sweep.trigger_offset
    for mean, sd, course_sd, common_sd, forecast, forecast_sd in grid:
        yield ({"trigger_offset_mean": mean, "trigger_offset_sd": sd,
                "trigger_offset_course_sd": course_sd, "trigger_offset_common_sd": common_sd,
                "forecast_offset": forecast, "forecast_offset_sd": forecast_sd},
               replace(base, workload=replace(base.workload, trigger_offset_mean_min=mean,
                                              trigger_offset_sd_min=sd,
                                              trigger_offset_course_sd_min=course_sd,
                                              trigger_offset_common_sd_min=common_sd,
                                              forecast_offset_min=forecast,
                                              forecast_offset_sd_min=forecast_sd)))


def _rho(base: RunConfig, sweep: SweepConfig, quick: bool) -> Iterator[Cell]:
    """Requests per arriving student (F9)."""
    grid = (1.0, 4.0) if quick else sweep.rho
    for r in grid:
        yield {"rho": r}, replace(base, workload=replace(base.workload, rho_mean=r))


SWEEPS: Dict[str, Builder] = {
    "sigma": _sigma,
    "epsilon": _epsilon,
    "lead": _lead,
    "trust": _trust,
    "margin": _margin,
    "adoption": _adoption,
    "rho": _rho,
    "pool": _pool,
    "offset": _offset,
}


def run_sweep(name: str, base: RunConfig, sweep: Optional[SweepConfig] = None,
              seeds: Optional[Sequence[int]] = None,
              policies: Optional[Sequence[str]] = None, quick: bool = False) -> SweepResult:
    """Run one grid and return its rows (one row per cell x seed x policy)."""
    try:
        builder = SWEEPS[name]
    except KeyError as exc:
        raise KeyError(f"unknown sweep '{name}'; known: {sorted(SWEEPS)}") from exc
    sweep = sweep or SweepConfig()
    seeds = list(seeds if seeds is not None else ((0, 1) if quick else sweep.seeds))
    names = list(policies or DEFAULT_POLICIES)
    cache: Dict[Tuple, Tuple[ArrivalStream, Dict[str, object]]] = {}
    rows: List[Dict[str, object]] = []
    diag: List[Dict[str, object]] = []
    cells = list(builder(base, sweep, quick))
    last_use = {(cell.workload, cell.root): idx for idx, (_, cell) in enumerate(cells)}
    for idx, (params, cell) in enumerate(cells):
        ratio = params.get(PRICE_RATIO)  # requested; ``shared`` re-records the realised value
        for seed in seeds:
            cfg = replace(cell, seed=seed)
            key = (cfg.workload, seed, cfg.root)
            if key not in cache:
                stream = build_stream(cfg)
                sync = measure_synchrony(stream)
                cache[key] = (stream, {"sigma2_measured": sync["sigma2_median"],
                                       "sigma2_pooled": sync["sigma2_pooled"],
                                       "sync_sessions": sync["n_sessions"]})
            stream, measured = cache[key]
            cfg = with_price_ratio(cfg, fluid_mu(cfg, stream), ratio)
            ctx = make_context(cfg, stream)
            oracle = run_policy("offline", cfg, stream, ctx)
            denom = oracle.metrics.total_cost
            eta, eta_norm = forecast_error(stream.lam_in, stream.lam_off, stream.lam_hat_in,
                                           stream.lam_hat_off)
            weeks = cfg.workload.weeks
            oracle_weekly = [float(r["total_cost"]) for r in
                             weekly_rows("offline", seed, oracle.x, stream, oracle.service, ctx.mu,
                                         cfg.cost, weeks,
                                         billing_block_min=cfg.service.billing_block_min)]
            shared = {"sweep": name, "seed": seed, "lead_min": cfg.service.lead_min,
                      "commit_block_min": cfg.service.commit_block_min,
                      "billing_block_min": cfg.service.billing_block_min,
                      "trust": cfg.policy.trust, "sigma": cfg.workload.sigma, **measured,
                      "adoption": cfg.workload.adoption, "rho": cfg.workload.rho_mean,
                      "plan_margin": cfg.policy.plan_margin,
                      "college": cfg.workload.college or "ALL",
                      "weeks": cfg.workload.n_weeks, "mu": ctx.mu,
                      "eta": eta, "eta_norm": eta_norm, "offline_cost": denom,
                      "p_in": cfg.cost.p_in, "p_off": cfg.cost.p_off,
                      PRICE_RATIO_REAL: price_ratio_of(cfg.cost, ctx.mu)}
            for spec in names:
                policy_key, eps = spec_trust(spec)
                run_ctx = ctx if eps is None else make_context(cfg, stream, trust=eps)
                run = (oracle if policy_key == "offline" and eps is None
                       else run_policy(policy_key, cfg, stream, run_ctx, denom))
                metrics = run.metrics.to_row()
                metrics["cost_ratio"] = (run.metrics.total_cost / denom if denom > 0 else
                                         float("nan"))
                tag = {"policy": spec, "trust": run_ctx.trust}
                rows.append({**params, **shared, "week": 0, **metrics, **tag})
                if run.diagnostics:
                    diag.append({**params, "seed": seed, **run.diagnostics, **tag})
                for week in weekly_rows(policy_key, seed, run.x, stream, run.service, ctx.mu,
                                        cfg.cost, weeks, oracle_weekly,
                                        cfg.service.billing_block_min):
                    rows.append({**params, **shared, **week, **tag})
            logger.info("%s cell %s seed %d done (eta_norm %.3f, p_in %.4g)", name, params, seed,
                        eta_norm, cfg.cost.p_in)
            if last_use[(cell.workload, cell.root)] == idx:
                del cache[key]  # no later cell replays this workload
    return SweepResult(name, pd.DataFrame(rows), pd.DataFrame(diag) if diag else None)


__all__ = ["EXACT", "PRICE_RATIO", "PRICE_RATIO_REAL", "SWEEPS", "Cell", "SweepResult",
           "parse_spec", "price_ratio_of", "run_sweep", "spec_trust", "with_price_ratio"]
