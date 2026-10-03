#!/usr/bin/env python3
"""The paths and stage list the box and the laptop have to agree on.

A leaf module on purpose: ``stages.py`` (box), ``measurements.py`` (box) and
``orchestrate.py`` (laptop) all import it, so there is exactly one definition of
where a marker, a log or ``STATUS`` lives.  Everything hangs off ``--root``
(the working directory on the GPU box, a temp directory in a dry run).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass(frozen=True)
class RemoteLayout:
    root: Path = Path("q2_box")
    out_name: str = "b1"

    @property
    def scripts(self) -> Path:
        """The q2 folder as rsynced onto the box."""
        return self.root / "q2"

    @property
    def remote_dir(self) -> Path:
        return self.scripts / "remote"

    @property
    def results(self) -> Path:
        """Everything the laptop pulls lives under here."""
        return self.root / "q2_results"

    @property
    def out(self) -> Path:
        return self.results / self.out_name

    @property
    def logs(self) -> Path:
        return self.results / "logs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def venv_python(self) -> Path:
        return self.root / "venv" / "bin" / "python"

    @property
    def status(self) -> Path:
        return self.out / "STATUS"

    @property
    def heartbeat(self) -> Path:
        return self.results / "heartbeat.txt"

    @property
    def pulled(self) -> Path:
        """Written by the laptop once the results are verified on disk."""
        return self.results / "PULLED"

    @property
    def bootstrap_done(self) -> Path:
        return self.results / "bootstrap.done"

    @property
    def watchdog_pid(self) -> Path:
        return self.results / "watchdog.pid"

    @property
    def run_all_pid(self) -> Path:
        return self.results / "run_all.pid"

    @property
    def no_shutdown_note(self) -> Path:
        return self.results / "DRY_RUN_NO_SHUTDOWN.txt"

    def stage_log(self, stage: str) -> Path:
        return self.logs / f"stage_{stage}.log"

    def done(self, stage: str) -> Path:
        return self.results / f"stage_{stage}.done"

    def failed(self, stage: str) -> Path:
        return self.results / f"stage_{stage}.failed"


@dataclass(frozen=True)
class StageSpec:
    name: str
    timeout_s: float


STAGES = (
    StageSpec("check_patterns", 15 * 60),
    StageSpec("coldstart", 60 * 60),
    StageSpec("concurrency", 75 * 60),
    StageSpec("prefix", 45 * 60),
    StageSpec("summarize", 5 * 60),
    StageSpec("replay", 100 * 60),  # batch B': burst replay (F10); run with --only replay
    StageSpec("shortprompt", 60 * 60),  # batch B3: classroom-length prompts, 7B and 32B; --only shortprompt
    StageSpec("ladder", 240 * 60),  # batch B4: model ladder x prompt regimes (short / RAG shared / RAG unique)
    StageSpec("calib_single", 30 * 60),  # B' step two, phase 1: the capped replica alone; --only calib_single
    StageSpec("pool", 8 * 3600),  # batch B' step two: policy-driven replica pool; --only pool
)
STAGE_NAMES = tuple(spec.name for spec in STAGES)
