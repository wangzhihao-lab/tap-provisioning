#!/usr/bin/env python3
"""What GPU we actually got, and which ``vllm serve`` arguments that allows.

Batch B was planned for an A100 80 GB but is rented on whatever is available, so
the serve arguments are derived from the measured VRAM instead of being baked in:

=================  ====================================================
reported VRAM      serve arguments
>= 70 GiB          ``--max-model-len 4096 --gpu-memory-utilization 0.90``
(nominal 80 GB)    the RUNBOOK values
>= 36 GiB          ``--max-model-len 2048 --gpu-memory-utilization 0.92``
(nominal 40-48 GB) A100 40 GB, L20 / L40S 48 GB; all three models fit
<  36 GiB          7B only; 14B bf16 and 32B AWQ refused, reason logged
=================  ====================================================

The cuts sit below the nominal capacities on purpose: ``nvidia-smi`` reports
usable memory, so a "40 GB" A100 comes back as about 39.6 GiB and an "80 GB" one
as about 79.2 GiB.  A literal 40 GiB cut would push a 40 GB card into the 7B-only
tier.

``--max-model-len`` / ``--gpu-mem-util`` on the driver override the tier, so a
known-good pair can always be forced.  The chosen values land in ``env.json``
and in the header line of every stage log.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from layout import RemoteLayout, iso_now

logger = logging.getLogger("q2.remote.hardware")

LARGE_VRAM_GB = 70.0   # nominal 80 GB cards report about 79.2 GiB
MID_VRAM_GB = 36.0     # nominal 40 GB cards report about 39.6 GiB
SMALL_MODEL_SLUGS = ("qwen2.5-7b",)


@dataclass(frozen=True)
class GpuTier:
    """The serve arguments the measured hardware allows, and why."""

    name: str
    vram_gb: float | None
    gpu_name: str
    max_model_len: int
    gpu_memory_utilization: float
    allow_large_models: bool
    reason: str

    @property
    def vram_text(self) -> str:
        return "unknown" if self.vram_gb is None else f"{self.vram_gb:.1f}GB"

    def header(self) -> str:
        return (f"gpu={self.gpu_name} vram={self.vram_text} tier={self.name} "
                f"max_model_len={self.max_model_len} "
                f"gpu_memory_utilization={self.gpu_memory_utilization:g} "
                f"large_models={'yes' if self.allow_large_models else 'no'}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.name, "gpu_name": self.gpu_name, "vram_gb": self.vram_gb,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "large_models_allowed": self.allow_large_models, "reason": self.reason,
        }


RUNBOOK_TIER = GpuTier("80g", None, "unknown", 4096, 0.90, True,
                       "RUNBOOK default (planned A100 80 GB); not probed")


def query_gpu() -> tuple[float | None, str]:
    """(total VRAM in GB, name) of the largest visible GPU; (None, 'unknown') if none."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("nvidia-smi unavailable (%s)", type(exc).__name__)
        return None, "unknown"
    if proc.returncode != 0:
        logger.warning("nvidia-smi exit %d: %s", proc.returncode, proc.stderr.strip()[:200])
        return None, "unknown"
    best_mib, best_name = 0.0, "unknown"
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mib = float(parts[0])
        except ValueError:
            continue
        if mib > best_mib:
            best_mib, best_name = mib, parts[1]
    if best_mib <= 0:
        return None, "unknown"
    return best_mib / 1024.0, best_name


def choose_tier(
    vram_gb: float | None, gpu_name: str = "unknown",
    max_model_len: int | None = None, gpu_mem_util: float | None = None,
) -> GpuTier:
    """Pick the serve arguments for the hardware we were actually given."""
    if vram_gb is None:
        tier = replace(RUNBOOK_TIER, gpu_name=gpu_name,
                       reason="no VRAM reading from nvidia-smi; assuming the RUNBOOK values")
    elif vram_gb >= LARGE_VRAM_GB:
        tier = GpuTier("80g", vram_gb, gpu_name, 4096, 0.90, True,
                       f"{vram_gb:.0f} GB VRAM: the RUNBOOK A100 80 GB plan fits unchanged")
    elif vram_gb >= MID_VRAM_GB:
        tier = GpuTier("40g", vram_gb, gpu_name, 2048, 0.92, True,
                       f"{vram_gb:.0f} GB VRAM (A100 40 GB / L20 / L40S class): a 2k context "
                       "leaves KV-cache room for 14B bf16 and 32B AWQ on one card")
    else:
        tier = GpuTier("small", vram_gb, gpu_name, 2048, 0.92, False,
                       f"only {vram_gb:.0f} GB VRAM: 14B bf16 (~28 GB of weights) and 32B AWQ "
                       "(~19 GB plus KV cache) cannot be served here, so batch B measures 7B "
                       "only and table T3 gets one measured size")
    if max_model_len is not None:
        tier = replace(tier, max_model_len=max_model_len,
                       reason=f"{tier.reason}; --max-model-len {max_model_len} forced on the CLI")
    if gpu_mem_util is not None:
        tier = replace(tier, gpu_memory_utilization=gpu_mem_util,
                       reason=f"{tier.reason}; --gpu-mem-util {gpu_mem_util:g} forced on the CLI")
    logger.warning("GPU tier: %s", tier.header())
    logger.warning("reason: %s", tier.reason)
    return tier


def detect_tier(max_model_len: int | None = None, gpu_mem_util: float | None = None) -> GpuTier:
    vram_gb, gpu_name = query_gpu()
    return choose_tier(vram_gb, gpu_name, max_model_len, gpu_mem_util)


def record_environment(layout: RemoteLayout, tier: GpuTier, models: Sequence[str]) -> Path:
    """Merge the GPU tier and the chosen serve arguments into ``env.json``."""
    path = layout.results / "env.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            payload = loaded if isinstance(loaded, dict) else {"bootstrap": loaded}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("env.json unreadable (%s); writing a fresh one", exc)
    payload["gpu_tier"] = tier.as_dict()
    payload["serve_args_recorded_at"] = iso_now()
    payload["models_measured"] = list(models)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("recorded GPU tier in %s", path)
    return path
