"""Arrival-source registry: every source yields a DataFrame with columns ts, group, kind."""

import logging
from typing import Callable, Dict, Type

from .base import ArrivalSource

logger = logging.getLogger(__name__)

SOURCE_FACTORY: Dict[str, Type[ArrivalSource]] = {}


def register_source(name: str) -> Callable[[Type[ArrivalSource]], Type[ArrivalSource]]:
    """Register an ArrivalSource implementation under ``name``."""

    def decorator(cls: Type[ArrivalSource]) -> Type[ArrivalSource]:
        SOURCE_FACTORY[name] = cls
        logger.debug("registered source %s", name)
        return cls

    return decorator


def SourceFactory(name: str) -> Type[ArrivalSource]:
    """Return the source class registered under ``name``."""
    try:
        return SOURCE_FACTORY[name]
    except KeyError as exc:
        raise KeyError(f"unknown source '{name}'; known: {sorted(SOURCE_FACTORY)}") from exc


from . import assistments, generic_traces, mock, xuexitong  # noqa: E402,F401  (registration side effects)

__all__ = ["ArrivalSource", "SourceFactory", "register_source", "SOURCE_FACTORY"]
