"""Cell-C runs that put every central policy on the laptop's streams (paired with TAP).

The cloud boxes and this laptop generate different streams from the same arguments (JOBS_README,
"Platform"), so the paper's central cell-C comparisons can only pair TAP with a baseline when both
ran here.  For each central case x tier (the R1 groups), this derives - from the ``--policies`` and
``--tuned`` of every original in the group, not from a hand-kept list - which policy variants exist
on the cloud but not on the laptop, and adds one run per tuned file that still has missing
variants.  A *variant* is the policy spec alone, except for ``reactive`` and ``velocity``, the only
policies that read the knobs the tuning sets (``tune_baselines.KNOBS``), whose variant also carries
those knob values; two ``best.json`` files with the same values give the same variant.  ``tap`` is
the R1 job's and ``offline`` rides along in every run as the stream anchor.

Each run reuses the argument list of the original that used its tuned file with the most policies,
with ``--offset-cells C``, ``--policies offline <missing>`` and its own ``--out``; its check demands
offline rows equal to every laptop mirror (the R1 job's output included) and records the cloud gap.
The SLO-floor variants of the dispersed tuning (``tuning*_r30_slo95disp``) are on the laptop already
(``sensitivity_sweeps/*_lead_d{1,1.5,8}``); if section 3's resource rule selects another configuration,
``s3_eval_*`` reruns them on the laptop, cell C included.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Set, Tuple

from runs_groups import PILOT, Original, drop_flags, flag_value
from pilot.sim import PolicyConfig
from pilot.sim.sweeps import parse_spec
from tune_baselines import KNOBS
from tuned_config import read_tuned

TUNED_POLICIES: Tuple[str, ...] = tuple(KNOBS)  # reactive, velocity: the only readers of the knobs
Variant = Tuple[str, Optional[str]]  # (policy spec, JSON of its tuned knob values or None)


def policies_of(argv: Sequence[str]) -> List[str]:
    """The ``--policies`` list of an argument list."""
    if "--policies" not in argv:
        return []
    out: List[str] = []
    for tok in argv[list(argv).index("--policies") + 1:]:
        if tok.startswith("--"):
            break
        out.append(tok)
    return out


def knob_key(policy: str, tuned: Optional[str]) -> str:
    """JSON of the knob values ``policy`` runs with under ``--tuned tuned`` (defaults without)."""
    values = read_tuned(PILOT / tuned) if tuned else {}
    base = PolicyConfig()
    return json.dumps({k: values.get(k, getattr(base, k)) for k in KNOBS[policy]}, sort_keys=True)


def variants(orig: Original) -> Set[Variant]:
    """Policy variants an original produced."""
    tuned = flag_value(orig.argv, "--tuned")
    out: Set[Variant] = set()
    for spec in policies_of(orig.argv):
        base, _ = parse_spec(spec)
        out.add((spec, knob_key(base, tuned) if base in TUNED_POLICIES else None))
    return out


def missing_runs(mirrors: Sequence[Original]) -> List[Tuple[Original, List[str]]]:
    """``(base original, missing policy specs)`` per tuned file, the richest original first."""
    cloud = [o for o in mirrors if not o.laptop]
    have: Set[Variant] = {("offline", None), ("tap", None)}
    for orig in mirrors:
        if orig.laptop:
            have |= variants(orig)
    runs: Dict[str, Tuple[Original, List[str]]] = {}
    placed: Set[Variant] = set()
    for orig in sorted(cloud, key=lambda o: (-len(policies_of(o.argv)), o.out)):  # richest first
        tuned = flag_value(orig.argv, "--tuned") or "-"
        for var in sorted(variants(orig) - have - placed, key=lambda v: v[0]):
            runs.setdefault(tuned, (orig, []))[1].append(var[0])  # base: first = richest user
            placed.add(var)
    return [runs[k] for k in sorted(runs, key=lambda k: -len(runs[k][1]))]


def paired_argv(base: Original, specs: Sequence[str], out_dir: str) -> List[str]:
    """The base original's arguments on cell C with the missing policies and its own output."""
    argv = drop_flags(base.argv, ["--policies", "--out", "--offset-cells"])
    return [*argv, "--offset-cells", "C", "--policies", "offline", *specs, "--out", out_dir]


__all__ = ["TUNED_POLICIES", "knob_key", "missing_runs", "paired_argv", "policies_of", "variants"]
