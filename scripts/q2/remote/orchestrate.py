#!/usr/bin/env python3
"""Laptop side of batch B: push, bootstrap, launch, keep pulling, then switch the box off.

The box is treated as expendable.  Results are rsynced down every ``--poll-seconds``
and never deleted locally, so a box that dies mid-run costs at most one poll
interval of data.  The loop ends when the box says ``DONE`` or ``FAILED``, when its
heartbeat goes stale (hang), or when ssh fails five times in a row (box gone).
It then does a checksum pull, compares a per-file manifest, and only if everything
matches writes the ``PULLED`` marker and halts the box.

    python remote/orchestrate.py --host <host> --port <port> --user <user> \\
        --remote-root <dir on the box> --local-out results/b1_remote --cap-hours 14.5 \\
        --on-done keep --on-fail keep --on-cap keep
    python remote/orchestrate.py --dry-run --local-out results/b1_remote_dryrun \\
        --remote-root /tmp/q2_fake            # whole flow, no box, no shutdown

Exit codes: 0 DONE, 2 FAILED, 3 hang, 4 box unreachable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from layout import STAGE_NAMES, RemoteLayout
from transport import (
    FileEntry, LocalTransport, ManifestReport, RemoteProbe, SshTransport, Transport,
    compare_manifests, local_manifest, manifest_command, parse_manifest, parse_probe,
    probe_command,
)

logger = logging.getLogger("q2.remote.orchestrate")

EXIT_CODES = {"done": 0, "failed": 2, "hang": 3, "unreachable": 4}
PUSH_EXCLUDES = ("results/", "__pycache__/", ".DS_Store", ".ruff_cache/", ".pytest_cache/")


@dataclass(frozen=True)
class OrchestratorConfig:
    local_out: Path
    q2_dir: Path
    remote_root: Path = Path("q2_box")
    out_name: str = "b1"
    cap_hours: float = 14.5
    poll_s: float = 120.0
    stale_heartbeat_s: float = 20 * 60.0
    max_ssh_failures: int = 5
    max_wait_s: float = 16 * 3600.0
    max_model_len: int | None = None
    gpu_mem_util: float | None = None
    remote_python: str | None = None
    skip_bootstrap: bool = False
    dry_run: bool = False
    only: str | None = None  # comma-separated stage names; None = every stage
    models: str | None = None  # comma-separated slugs for bootstrap to download; None = all
    on_done: str = "keep"  # operator's instruction after a verified pull: halt | keep
    on_fail: str = "keep"  # after a failed stage or a hang: halt | keep
    on_cap: str = "keep"  # when the box's budget cap fires: halt | keep
    fail_grace_min: float = 30.0  # on-fail halt: minutes to wait for a relaunch before halting


@dataclass(frozen=True)
class PollState:
    status: str | None = None
    heartbeat_age_s: float | None = None
    ssh_failures: int = 0


def terminal_reason(
    state: PollState, stale_heartbeat_s: float = 20 * 60.0, max_ssh_failures: int = 5
) -> str | None:
    """Why the watch loop should stop, or None to keep waiting."""
    if state.ssh_failures >= max_ssh_failures:
        return "unreachable"
    if state.status == "DONE":
        return "done"
    if state.status == "FAILED":
        return "failed"
    if state.heartbeat_age_s is not None and state.heartbeat_age_s > stale_heartbeat_s:
        return "hang"
    return None


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig, transport: Transport) -> None:
        self.cfg = cfg
        self.tp = transport
        self.layout = RemoteLayout(root=cfg.remote_root, out_name=cfg.out_name)
        self.probe: RemoteProbe | None = None
        self.report: ManifestReport | None = None
        self.shutdown_note = "not attempted"

    # ---------------------------------------------------------------- steps
    def push_scripts(self) -> None:
        mkdirs = " ".join(shlex.quote(str(p)) for p in
                          (self.layout.scripts, self.layout.out, self.layout.logs))
        self.tp.run(f"mkdir -p {mkdirs}")
        result = self.tp.push(self.cfg.q2_dir, str(self.layout.scripts), PUSH_EXCLUDES)
        if not result.ok:
            raise SystemExit(f"rsync of {self.cfg.q2_dir} failed: {result.tail()}\n"
                             "Nothing was started, so no box is billing yet.")
        logger.info("pushed %s -> %s", self.cfg.q2_dir, self.layout.scripts)

    def bootstrap(self) -> None:
        if self.cfg.skip_bootstrap:
            logger.info("--skip-bootstrap given")
            return
        marker = self.tp.run(f"test -f {shlex.quote(str(self.layout.bootstrap_done))}")
        if marker.ok:
            logger.info("bootstrap.done already on the box; skipping")
            return
        script = shlex.quote(str(self.layout.remote_dir / "bootstrap.sh"))
        cmd = f"bash {script} --root {shlex.quote(str(self.cfg.remote_root))}"
        if self.cfg.dry_run:
            cmd += " --dry-run"
        if self.cfg.models:
            cmd += f" --models {shlex.quote(self.cfg.models)}"
        logger.info("bootstrap (this installs vllm and downloads the weights of %s)",
                    self.cfg.models or "all four models, ~65 GB")
        result = self.tp.run(cmd, timeout_s=3 * 3600)
        if not result.ok:
            raise SystemExit(
                f"bootstrap failed (exit {result.returncode}): {result.tail(10)}\n"
                f"Read {self.layout.results}/env.json (arch_problems, fix_hint) and "
                f"{self.layout.logs}/bootstrap.log.\n"
                "The box is left running on purpose so the fix does not cost another "
                "65 GB download -- it is still billing, and no watchdog is armed yet. "
                "Re-run this command after fixing, or release the instance."
            )
        logger.info("bootstrap finished: %s", result.tail(1) or "ok")

    def launch(self) -> None:
        state = self.poll_once()
        if state.status in ("DONE", "FAILED"):
            logger.info("run already finished on the box (STATUS=%s); pulling only", state.status)
            return
        pid_file = shlex.quote(str(self.layout.run_all_pid))
        alive = self.tp.run(f"test -f {pid_file} && kill -0 \"$(cat {pid_file})\" 2>/dev/null")
        if alive.ok:
            logger.info("run_all already running on the box; attaching to it")
            return
        # a PULLED marker left by an earlier batch would let the box halt before this run is
        # pulled; the driver waits for a fresh one written after the final pull
        self.tp.run(f"rm -f {shlex.quote(str(self.layout.pulled))}")
        script = shlex.quote(str(self.layout.remote_dir / "run_all.sh"))
        cmd = (f"bash {script} --root {shlex.quote(str(self.cfg.remote_root))} "
               f"--cap-hours {self.cfg.cap_hours:g} --out-name {shlex.quote(self.cfg.out_name)}")
        if self.cfg.only:
            cmd += f" --only {shlex.quote(self.cfg.only)}"
        cmd += (f" --on-done {self.cfg.on_done} --on-fail {self.cfg.on_fail}"
                f" --on-cap {self.cfg.on_cap}")
        if self.cfg.remote_python:
            cmd += f" --python {shlex.quote(self.cfg.remote_python)}"
        if self.cfg.max_model_len is not None:
            cmd += f" --max-model-len {self.cfg.max_model_len}"
        if self.cfg.gpu_mem_util is not None:
            cmd += f" --gpu-mem-util {self.cfg.gpu_mem_util:g}"
        if self.cfg.dry_run:
            cmd += " --dry-run"
        result = self.tp.run(cmd, timeout_s=300)
        if not result.ok:
            raise SystemExit(
                f"could not launch run_all.sh: {result.tail()}\n"
                "The box is still running and still billing; nothing is measuring."
            )
        logger.info("launched: %s", result.tail(1) or "ok")

    def poll_once(self) -> PollState:
        result = self.tp.run(
            probe_command(str(self.layout.results), f"{self.cfg.out_name}/STATUS"), timeout_s=120
        )
        self.probe = parse_probe(result)
        return PollState(status=self.probe.status, heartbeat_age_s=self.probe.heartbeat_age_s)

    def watch(self) -> str:
        """Pull, probe, log, repeat until a terminal condition."""
        failures = 0
        started_at = time.monotonic()
        deadline = started_at + self.cfg.max_wait_s
        while True:
            pulled = self.tp.pull(str(self.layout.results), self.cfg.local_out)
            state = self.poll_once()
            probe = self.probe
            failures = 0 if (probe is not None and probe.reachable) else failures + 1
            # A driver that never wrote a heartbeat at all is aged from our launch: a
            # launch that failed silently must not keep the laptop waiting for hours.
            since_launch = state.heartbeat_age_s is None
            age_s = time.monotonic() - started_at if since_launch else state.heartbeat_age_s
            state = PollState(state.status, age_s, failures)
            logger.info(
                "status=%s stage=%s %s=%.0fs markers=%d pull=%s ssh_failures=%d",
                state.status or "-", (probe.stage if probe else None) or "-",
                "no_heartbeat_for" if since_launch else "heartbeat", age_s or 0.0,
                len(probe.markers) if probe else 0, "ok" if pulled.ok else pulled.tail(1),
                failures,
            )
            reason = terminal_reason(state, self.cfg.stale_heartbeat_s, self.cfg.max_ssh_failures)
            if reason is not None:
                logger.warning("terminal condition: %s", reason)
                return reason
            if time.monotonic() > deadline:
                logger.error("laptop-side wait of %.1f h elapsed; treating as a hang",
                             self.cfg.max_wait_s / 3600)
                return "hang"
            time.sleep(self.cfg.poll_s)

    def final_pull(self) -> ManifestReport:
        result = self.tp.pull(str(self.layout.results), self.cfg.local_out, extra=("-c",))
        if not result.ok:
            logger.error("final checksum rsync failed: %s", result.tail())
        remote = self.tp.run(manifest_command(str(self.layout.results)), timeout_s=1800)
        remote_entries: dict[str, FileEntry] = parse_manifest(remote.stdout) if remote.ok else {}
        if not remote.ok:
            logger.error("remote manifest failed (exit %d): %s", remote.returncode,
                         remote.tail())
        local_entries = local_manifest(self.cfg.local_out)
        self.report = compare_manifests(remote_entries, local_entries,
                                        remote_unavailable=not remote.ok)
        self._write_manifest_json(remote_entries, local_entries, self.report)
        return self.report

    def _write_manifest_json(
        self, remote: dict[str, FileEntry], local: dict[str, FileEntry], report: ManifestReport
    ) -> None:
        payload = {
            "schema": "q2.pull_manifest.v1",
            "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "transport": self.tp.label,
            "remote_results_dir": str(self.layout.results),
            "local_out": str(self.cfg.local_out),
            "dry_run": self.cfg.dry_run,
            "remote_files": report.remote_files, "local_files": report.local_files,
            "remote_bytes": report.remote_bytes, "local_bytes": report.local_bytes,
            "ok": report.ok,
            "missing_locally": list(report.missing_locally),
            "extra_locally": list(report.extra_locally),
            "size_mismatch": list(report.size_mismatch),
            "hash_mismatch": list(report.hash_mismatch),
            "files": [
                {
                    "path": rel,
                    "remote_size": remote[rel].size if rel in remote else None,
                    "remote_sha256": remote[rel].sha256 if rel in remote else None,
                    "local_size": local[rel].size if rel in local else None,
                    "local_sha256": local[rel].sha256 if rel in local else None,
                }
                for rel in sorted(set(remote) | set(local))
            ],
        }
        path = self.cfg.local_out / "PULL_MANIFEST.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        logger.info("wrote %s", path)

    def _relaunched_on_box(self) -> bool:
        """A new driver appeared on the box after the failure (someone fixed and relaunched)."""
        pid_file = shlex.quote(str(self.layout.run_all_pid))
        status = shlex.quote(str(self.layout.out / "STATUS"))
        probe = self.tp.run(f"test -f {pid_file} && kill -0 \"$(cat {pid_file})\" 2>/dev/null "
                            f"&& test ! -f {status}")
        return probe.ok

    def _release_after_failure(self, reason: str, report: ManifestReport) -> None:
        """The operator's --on-fail instruction, with the data protected first.

        ``keep``: the box stays up (the operator fixes and relaunches on the same machine).
        ``halt``: the data was already pulled (``final_pull``); wait ``fail_grace_min`` minutes for
        a relaunch on the box, and if none appears pull once more, write PULLED and power the box
        off - a stuck box with nobody attending must not keep billing, and nothing is lost that was
        on the box up to the last pull.
        """
        if self.cfg.on_fail != "halt":
            self.shutdown_note = (f"{reason}: box kept (--on-fail keep; "
                                  f"at the {self.cfg.cap_hours:g} h budget --on-cap {self.cfg.on_cap})")
            logger.error("run %s; data pulled (verified=%s); NOT writing PULLED; box kept "
                         "(--on-fail keep)", reason, report.ok)
            return
        grace_s = max(0.0, self.cfg.fail_grace_min * 60.0)
        logger.error("run %s; data pulled (verified=%s); --on-fail halt: waiting %.0f min for a "
                     "relaunch on the box before halting it", reason, report.ok, grace_s / 60)
        limit = time.monotonic() + grace_s
        while time.monotonic() < limit:
            if self._relaunched_on_box():
                self.shutdown_note = f"{reason}: relaunched on the box within the grace period; not halted"
                logger.warning("a new driver is running on the box; leaving it alone")
                return
            time.sleep(min(60.0, max(1.0, limit - time.monotonic())))
        final = self.final_pull()
        if not final.ok and not self.cfg.dry_run:
            self.shutdown_note = (f"{reason}: grace period over, but the last pull did not verify; "
                                  "not halted, data first")
            logger.error("last pull did not verify; NOT halting (data first)")
            return
        marker = shlex.quote(str(self.layout.pulled))
        self.tp.run(f"date +%Y-%m-%dT%H:%M:%S%z > {marker}")
        if self.cfg.dry_run:
            self.shutdown_note = f"{reason}: grace period over (--dry-run: would halt)"
            logger.warning("--dry-run: would halt the box now after the grace period")
            return
        halt = self.tp.run("sync; shutdown -h now", timeout_s=60)
        self.shutdown_note = (f"{reason}: no relaunch within {self.cfg.fail_grace_min:g} min; "
                              f"data pulled and verified; halted (exit code {halt.returncode})")

    def release_box(self, reason: str, report: ManifestReport) -> None:
        if reason in ("failed", "hang"):
            self._release_after_failure(reason, report)
            return
        if reason == "unreachable":
            self.shutdown_note = "skipped (box unreachable; check the provider console)"
            return
        if not report.ok:
            why = "remote manifest unavailable" if report.remote_unavailable else "file check failed"
            self.shutdown_note = (f"skipped ({why}; box kept for manual handling; "
                                  "the box's own watchdog halts it after 30 min)")
            logger.error("results not verified (%s); NOT writing PULLED, NOT shutting down", why)
            return
        marker = shlex.quote(str(self.layout.pulled))
        wrote = self.tp.run(f"date +%Y-%m-%dT%H:%M:%S%z > {marker}")
        if not wrote.ok:
            self.shutdown_note = f"skipped (could not write the PULLED marker: {wrote.tail(1)})"
            return
        if self.cfg.dry_run:
            self.shutdown_note = "skipped (--dry-run: would run `sync; shutdown -h now`)"
            logger.warning("--dry-run: would run `sync; shutdown -h now` on %s", self.tp.label)
            return
        if self.cfg.on_done != "halt":
            self.shutdown_note = "box kept (--on-done keep)"
            logger.warning("results pulled and verified; box left running (--on-done keep)")
            return
        halt = self.tp.run("sync; shutdown -h now", timeout_s=60)
        self.shutdown_note = (f"ran `sync; shutdown -h now` (exit code {halt.returncode})"
                              if halt.ok or halt.returncode == 255
                              else f"command returned {halt.returncode}: {halt.tail(1)}; check the provider console")

    # ------------------------------------------------------------- reporting
    def summary(self, reason: str, report: ManifestReport | None) -> str:
        probe = self.probe
        markers = set(probe.markers) if probe else set()
        stage_bits = []
        names = tuple(n.strip() for n in self.cfg.only.split(",")) if self.cfg.only else STAGE_NAMES
        for name in names:
            if f"stage_{name}.done" in markers:
                stage_bits.append(f"{name}[done]")
            elif f"stage_{name}.failed" in markers:
                stage_bits.append(f"{name}[failed]")
            else:
                stage_bits.append(f"{name}[not run]")
        verdicts = {"done": "every stage finished (STATUS=DONE)",
                    "failed": "a stage failed (STATUS=FAILED; see the log tail at the end of STATUS)",
                    "hang": "heartbeat older than 20 min; treated as a hang",
                    "unreachable": "five consecutive connection failures; box treated as gone"}
        lines = [
            "",
            "========== remote run summary ==========",
            f"verdict: {verdicts[reason]}",
            f"transport: {self.tp.label}",
            f"stages: {'  '.join(stage_bits)}",
        ]
        if report is not None and report.remote_unavailable:
            lines.append(
                f"file check: remote manifest unavailable; "
                f"{report.local_files} files / {report.local_bytes / 1e6:.1f} MB kept locally, "
                "not verified file by file"
            )
        elif report is not None:
            lines.append(
                f"file check: remote {report.remote_files} files / {report.remote_bytes / 1e6:.1f} MB, "
                f"local {report.local_files} files / {report.local_bytes / 1e6:.1f} MB, "
                f"{'sha256 all equal' if report.ok else 'sha256 mismatches'}"
            )
            for label, items in (("missing locally", report.missing_locally),
                                 ("size mismatch", report.size_mismatch),
                                 ("hash mismatch", report.hash_mismatch)):
                if items:
                    lines.append(f"  {label} ({len(items)}): {', '.join(items[:5])}"
                                 + ("…" if len(items) > 5 else ""))
        lines += [
            f"data: {self.cfg.local_out}",
            f"  measurements: {self.cfg.local_out / self.cfg.out_name}"
            f" (coldstart/ concurrency/ prefix/ q2_summary.csv T3_calibration.csv figures)",
            f"  logs: {self.cfg.local_out / 'logs'}",
            f"  manifest: {self.cfg.local_out / 'PULL_MANIFEST.json'}",
            f"box: {self.shutdown_note}",
            "resume: run the same command again (finished stages are skipped via stage_*.done)",
            "=======================================",
        ]
        return "\n".join(lines)

    def run(self) -> int:
        logger.info("batch B orchestrator: %s -> %s", self.tp.label, self.cfg.local_out)
        self.cfg.local_out.mkdir(parents=True, exist_ok=True)
        self.push_scripts()
        self.bootstrap()
        self.launch()
        reason = self.watch()
        report = self.final_pull()
        self.release_box(reason, report)
        print(self.summary(reason, report))
        return EXIT_CODES[reason]


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    default_q2 = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=os.environ.get("Q2_HOST"),
                   help="box address, or $Q2_HOST (required without --dry-run)")
    p.add_argument("--port", type=int, default=_env_int("Q2_PORT"),
                   help="box port, or $Q2_PORT (required without --dry-run)")
    p.add_argument("--user", default=os.environ.get("Q2_USER"),
                   help="login user on the box, or $Q2_USER (required without --dry-run)")
    p.add_argument("--local-out", type=Path, default=default_q2 / "results" / "b1_remote")
    p.add_argument("--q2-dir", type=Path, default=default_q2, help="folder rsynced to the box")
    p.add_argument("--remote-root", default=os.environ.get("Q2_REMOTE_ROOT"),
                   help="working directory on the box, or $Q2_REMOTE_ROOT (a temp dir with --dry-run)")
    p.add_argument("--out-name", default="b1")
    p.add_argument("--cap-hours", type=float, default=14.5)
    p.add_argument("--poll-seconds", type=float, default=None, help="default 120 (5 in dry runs)")
    p.add_argument("--stale-heartbeat-minutes", type=float, default=20.0)
    p.add_argument("--max-ssh-failures", type=int, default=5)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="force the vllm --max-model-len instead of the detected GPU tier")
    p.add_argument("--gpu-mem-util", type=float, default=None,
                   help="force the vllm --gpu-memory-utilization")
    p.add_argument("--remote-python", default=None,
                   help="interpreter on the box; default <remote-root>/venv/bin/python")
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument("--models", default=None,
                   help="comma-separated model slugs for bootstrap to download (default: all four)")
    p.add_argument("--on-done", choices=("halt", "keep"), required=True,
                   help="operator's instruction after a verified pull")
    p.add_argument("--on-fail", choices=("halt", "keep"), required=True,
                   help="operator's instruction after a failed stage or a hang")
    p.add_argument("--on-cap", choices=("halt", "keep"), required=True,
                   help="operator's instruction when the box's --cap-hours budget fires")
    p.add_argument("--fail-grace-min", type=float, default=30.0,
                   help="with --on-fail halt: minutes to wait for a relaunch before halting")
    p.add_argument("--only", default=None,
                   help="comma-separated stages to run (e.g. replay); default: all")
    p.add_argument("--dry-run", action="store_true",
                   help="run the whole flow locally against fake_server.py; never shuts down")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _build(argv: list[str] | None) -> tuple[OrchestratorConfig, Transport]:
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(message)s")
    if not a.dry_run:
        missing = [n for n, v in (("--host", a.host), ("--port", a.port), ("--user", a.user),
                                  ("--remote-root", a.remote_root)) if not v]
        if missing:
            raise SystemExit(f"{', '.join(missing)} required (or use --dry-run)")
    root = a.remote_root or "/tmp/q2_fake"
    cfg = OrchestratorConfig(
        local_out=a.local_out.expanduser().resolve(),
        q2_dir=a.q2_dir.expanduser().resolve(),
        remote_root=Path(root).expanduser(),
        out_name=a.out_name, cap_hours=a.cap_hours,
        poll_s=a.poll_seconds if a.poll_seconds is not None else (5.0 if a.dry_run else 120.0),
        stale_heartbeat_s=a.stale_heartbeat_minutes * 60.0,
        max_ssh_failures=a.max_ssh_failures,
        max_wait_s=(a.cap_hours + 1.5) * 3600.0,
        max_model_len=a.max_model_len, gpu_mem_util=a.gpu_mem_util,
        remote_python=a.remote_python,
        skip_bootstrap=a.skip_bootstrap, dry_run=a.dry_run, only=a.only,
        on_done=a.on_done, on_fail=a.on_fail, on_cap=a.on_cap, models=a.models,
        fail_grace_min=a.fail_grace_min,
    )
    transport: Transport = LocalTransport() if a.dry_run else SshTransport(
        host=str(a.host), port=a.port, user=a.user
    )
    return cfg, transport


def main(argv: list[str] | None = None) -> int:
    cfg, transport = _build(argv)
    return Orchestrator(cfg, transport).run()


if __name__ == "__main__":
    raise SystemExit(main())
