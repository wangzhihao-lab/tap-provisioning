"""Command line entry point.

Examples::

    pilot q1 --source mock
    pilot q1 --source assistments --path data/raw/assistments_2012_13.csv
    pilot q1 --source burstgpt --path data/raw/BurstGPT_1.csv
    pilot q1 --source xxt_signin --paths data/raw/signin_*.xlsx
    pilot q1 --source mock --source burstgpt --path data/raw/BurstGPT_1.csv
    pilot mock --out outputs/mock_arrivals.csv

``--path`` is consumed, in order, by the sources that read a single file; ``--paths`` by the
sources that read a list of files. Sources that need no file (mock) consume neither.
"""

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import AnalysisConfig, LoaderConfig, MockConfig, PredictConfig, SessionDetectConfig
from .q1 import run_q1
from .sources import SOURCE_FACTORY, SourceFactory
from .sources.base import ArrivalSource
from .sources.mock import MockSource

logger = logging.getLogger("pilot")

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# matplotlib's font machinery logs every glyph it subsets at INFO level
_QUIET_LOGGERS = ("matplotlib", "fontTools", "PIL")


def build_sources(names: Sequence[str], single: Sequence[str],
                  multi: Sequence[Sequence[str]]) -> Dict[str, ArrivalSource]:
    """Instantiate the requested sources, handing each one the files it needs."""
    singles = list(single)
    multis = [list(group) for group in multi]
    built: Dict[str, ArrivalSource] = {}
    for name in names:
        cls = SourceFactory(name)
        if cls.n_paths == 0:
            src: ArrivalSource = cls()
        elif cls.n_paths == 1:
            if not singles:
                raise ValueError(f"source '{name}' needs --path <file>")
            src = cls(Path(singles.pop(0)))
        else:
            if not multis:
                raise ValueError(f"source '{name}' needs --paths <file> [<file> ...]")
            src = cls([Path(p) for p in multis.pop(0)])
        key = name if name not in built else f"{name}_{sum(k.startswith(name) for k in built) + 1}"
        built[key] = src
    for leftover, flag in ((singles, "--path"), (multis, "--paths")):
        if leftover:
            raise ValueError(f"{len(leftover)} unused {flag} argument(s): {leftover}")
    return built


def _add_q1(sub: argparse._SubParsersAction) -> None:
    analysis, predict, loader = AnalysisConfig(), PredictConfig(), LoaderConfig()
    p = sub.add_parser("q1", help="burst shape and calendar predictability, source by source")
    p.add_argument("--source", action="append", required=True,
                   choices=sorted(SOURCE_FACTORY), help="repeatable")
    p.add_argument("--path", action="append", default=[], help="file for a single-file source")
    p.add_argument("--paths", action="append", nargs="+", default=[],
                   help="files for a multi-file source")
    p.add_argument("--out", type=Path, default=Path("outputs/q1"))
    p.add_argument("--bin-minutes", type=int, default=analysis.bin_minutes)
    p.add_argument("--horizon-min", type=int, default=predict.horizon_min)
    p.add_argument("--max-rows", type=int, default=loader.max_rows,
                   help="0 reads the whole file")


def _add_mock(sub: argparse._SubParsersAction) -> None:
    cfg = MockConfig()
    p = sub.add_parser("mock", help="write a synthetic calendar-driven arrival frame")
    p.add_argument("--out", type=Path, default=Path("outputs/mock_arrivals.csv"))
    p.add_argument("--weeks", type=int, default=cfg.weeks)
    p.add_argument("--seed", type=int, default=cfg.seed)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the ``pilot`` console script."""
    parser = argparse.ArgumentParser(prog="pilot", description=__doc__.splitlines()[0])
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)
    _add_q1(sub)
    _add_mock(sub)
    return parser


def _run_q1(args: argparse.Namespace) -> int:
    analysis = dataclasses.replace(AnalysisConfig(), bin_minutes=args.bin_minutes,
                                   forecast_horizon_min=args.horizon_min)
    predict = dataclasses.replace(PredictConfig(), bin_minutes=args.bin_minutes,
                                  horizon_min=args.horizon_min)
    loader = dataclasses.replace(LoaderConfig(), max_rows=args.max_rows)
    sources = build_sources(args.source, args.path, args.paths)
    for src in sources.values():
        if isinstance(getattr(src, "cfg", None), LoaderConfig):
            src.cfg = loader
    summary = run_q1(sources, args.out, analysis, predict, SessionDetectConfig())
    logger.info("%d source(s) summarised in %s", len(summary), args.out / "summary.md")
    return 0


def _run_mock(args: argparse.Namespace) -> int:
    cfg = dataclasses.replace(MockConfig(), weeks=args.weeks, seed=args.seed)
    df = MockSource(cfg).load()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %d arrivals to %s (%s to %s)", len(df), args.out, df["ts"].min(),
                df["ts"].max())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=_LOG_FORMAT)
    for noisy in _QUIET_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    handlers = {"q1": _run_q1, "mock": _run_mock}
    try:
        return handlers[args.command](args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
