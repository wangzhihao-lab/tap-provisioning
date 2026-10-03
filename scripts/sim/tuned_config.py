"""Tuned ``PolicyConfig`` values on disk: the JSON contract the tuner and the sweeps share.

``scripts/sim/tune_baselines.py`` writes ``best.json``; ``scripts/sim/run_sweep.py`` reads it back
through ``--tuned`` (or takes a flat override through ``--policy-config``) and applies the values to
the base ``PolicyConfig``.  The schema lives here, in one module, so that the writer and the reader
cannot drift apart and the round trip is testable on its own.

Layout of ``best.json``::

    {"policies": {"velocity": {"config": {"velocity_window_min": 10, ...}, ...},
                  "reactive": {"config": {"hpa_up_util": 0.5, ...}, ...}},
     "tuning_seeds": [0, 1], "heldout_seed": 2, ...}

Only ``policies.<name>.config`` is read back; everything else in the file is provenance.  Two
sections may not disagree about a field: the two grids are disjoint by construction (``velocity_*``
against ``hpa_*``), so a collision is a bug rather than a precedence question and it raises.

A flat override (``--policy-config``) is either inline JSON or a path to a JSON object, and its
keys are ``PolicyConfig`` field names.  Unknown names raise instead of being ignored silently, and
every value is coerced to the field's declared type, so ``{"velocity_window_min": 10.0}`` from a
JSON round trip still produces an ``int``.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, get_type_hints

PILOT = Path(__file__).resolve().parents[2]
if str(PILOT / "src") not in sys.path:  # the scripts run without the package being installed
    sys.path.insert(0, str(PILOT / "src"))

from pilot.sim import PolicyConfig  # noqa: E402

logger = logging.getLogger(__name__)

POLICIES_KEY = "policies"
CONFIG_KEY = "config"
_TRUE = {"true", "yes", "1", "on"}
_FALSE = {"false", "no", "0", "off"}


def field_hints() -> Dict[str, object]:
    """Resolved type hint of every ``PolicyConfig`` field, in declaration order."""
    hints = get_type_hints(PolicyConfig)
    return {f.name: hints[f.name] for f in fields(PolicyConfig)}


def _coerce_scalar(name: str, hint: object, value: object) -> object:
    """One value coerced to ``hint``; ``bool`` is checked before ``int`` (it is a subclass)."""
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
            return bool(value)
        raise TypeError(f"{name}: expected a boolean, got {value!r}")
    if hint is int:
        as_float = float(value)  # type: ignore[arg-type]
        if as_float != int(as_float):
            raise TypeError(f"{name}: expected a whole number, got {value!r}")
        return int(as_float)
    if hint is float:
        return float(value)  # type: ignore[arg-type]
    if hint is str:
        return str(value)
    raise TypeError(f"{name}: unsupported field type {hint!r}")


def coerce_field(name: str, value: object) -> object:
    """``value`` coerced to the declared type of ``PolicyConfig.<name>``.

    ``Optional[str]`` (``forecast_warm_start``) is the only union in the dataclass, and ``None`` is
    always legal for it; everything else is a plain ``bool`` / ``int`` / ``float`` / ``str``.
    """
    hints = field_hints()
    if name not in hints:
        raise KeyError(f"unknown PolicyConfig field '{name}'; known: {sorted(hints)}")
    hint = hints[name]
    if value is None:
        if hint is Optional[str]:
            return None
        raise TypeError(f"{name}: null is not a legal value for {hint!r}")
    return _coerce_scalar(name, str if hint is Optional[str] else hint, value)


def policy_values(mapping: Mapping[str, object]) -> Dict[str, object]:
    """Validate and coerce a flat mapping of ``PolicyConfig`` field names."""
    return {str(k): coerce_field(str(k), v) for k, v in mapping.items()}


def apply_policy_values(base: PolicyConfig, values: Mapping[str, object]) -> PolicyConfig:
    """``base`` with the validated ``values`` applied; an empty mapping returns ``base``."""
    clean = policy_values(values)
    return replace(base, **clean) if clean else base


def read_tuned(path: Path) -> Dict[str, object]:
    """Merge ``policies.<name>.config`` of a ``best.json`` into one flat override dict."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sections = payload.get(POLICIES_KEY)
    if not isinstance(sections, dict) or not sections:
        raise ValueError(f"{path}: no '{POLICIES_KEY}' section to read")
    merged: Dict[str, object] = {}
    for name, section in sections.items():
        values = policy_values((section or {}).get(CONFIG_KEY, {}))
        clash = {k: (merged[k], v) for k, v in values.items()
                 if k in merged and merged[k] != v}
        if clash:
            raise ValueError(f"{path}: policy sections disagree about {sorted(clash)}")
        merged.update(values)
        logger.debug("tuned %s: %s", name, values)
    return merged


def parse_policy_config(text: str) -> Dict[str, object]:
    """``--policy-config``: inline JSON object, or a path to a file holding one."""
    candidate = Path(text)
    raw = candidate.read_text(encoding="utf-8") if candidate.exists() else text
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--policy-config expects a JSON object of PolicyConfig fields")
    return policy_values(parsed)


def policy_overrides(tuned: Optional[Path], inline: Optional[str]) -> Dict[str, object]:
    """Overrides from ``--tuned`` and ``--policy-config``; the inline flag wins on a collision."""
    out: Dict[str, object] = {}
    if tuned:
        out.update(read_tuned(Path(tuned)))
    if inline:
        out.update(parse_policy_config(inline))
    return out


__all__ = ["CONFIG_KEY", "POLICIES_KEY", "apply_policy_values", "coerce_field", "field_hints",
           "parse_policy_config", "policy_overrides", "policy_values", "read_tuned"]
