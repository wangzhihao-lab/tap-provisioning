#!/usr/bin/env python3
"""Laptop side: how to reach the box (ssh/rsync) and how to verify what came back.

The real transport is key-based ssh only: ``BatchMode=yes`` turns a password
prompt into an immediate failure instead of a hang, which is what the orchestrator
counts towards "box gone".  ``LocalTransport`` runs the same command strings
through ``bash -c`` and the same rsync invocations between two local directories,
so ``orchestrate.py --dry-run`` exercises the whole flow without a box.

Also here: the one-shot remote status probe and the file manifests that decide
whether the results are safely on the laptop before anything is switched off.
"""

from __future__ import annotations

import hashlib
import logging
import shlex
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("q2.remote.transport")

SSH_CONNECTION_ERROR = 255
NOT_STARTED = 9  # our own exit code for "the results directory is not there yet"
SKIP_NAMES = ("PULL_MANIFEST.json", ".DS_Store")


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def connection_failed(self) -> bool:
        """ssh could not talk to the box at all (vs. a command that ran and failed)."""
        return self.returncode in (SSH_CONNECTION_ERROR, -1)

    def tail(self, n_lines: int = 3) -> str:
        text = (self.stderr or self.stdout).strip()
        return " | ".join(text.splitlines()[-n_lines:])


def _run(argv: Sequence[str], timeout_s: float | None) -> CommandResult:
    logger.debug("$ %s", " ".join(shlex.quote(part) for part in argv))
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True,
                              timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return CommandResult(tuple(argv), -1, "", f"timed out after {timeout_s}s")
    except OSError as exc:
        return CommandResult(tuple(argv), -1, "", f"{type(exc).__name__}: {exc}")
    return CommandResult(tuple(argv), proc.returncode, proc.stdout, proc.stderr)


class Transport(ABC):
    """Everything the orchestrator needs from a box, real or faked."""

    @abstractmethod
    def run(self, command: str, timeout_s: float | None = 120.0) -> CommandResult:
        """Run a shell command string on the box."""

    @abstractmethod
    def push(self, local_dir: Path, remote_dir: str,
             excludes: Sequence[str] = ()) -> CommandResult:
        """rsync ``local_dir/`` to ``remote_dir/``."""

    @abstractmethod
    def pull(self, remote_dir: str, local_dir: Path,
             extra: Sequence[str] = ()) -> CommandResult:
        """rsync ``remote_dir/`` down to ``local_dir`` without ever deleting locally."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable description for the logs."""


def _rsync_argv(rsync_bin: str, extra: Sequence[str], src: str, dst: str,
                excludes: Sequence[str], ssh_cmd: str | None) -> list[str]:
    argv = [rsync_bin, "-az", "--partial", *extra]
    if ssh_cmd is not None:
        argv += ["-e", ssh_cmd]
    for pattern in excludes:
        argv += ["--exclude", pattern]
    return [*argv, src, dst]


@dataclass(frozen=True)
class LocalTransport(Transport):
    """Dry-run stand-in: the 'box' is a directory on this machine."""

    rsync_bin: str = "rsync"
    shell: str = "bash"

    def run(self, command: str, timeout_s: float | None = 120.0) -> CommandResult:
        return _run([self.shell, "-c", command], timeout_s)

    def push(self, local_dir: Path, remote_dir: str,
             excludes: Sequence[str] = ()) -> CommandResult:
        Path(remote_dir).mkdir(parents=True, exist_ok=True)
        argv = _rsync_argv(self.rsync_bin, (), f"{local_dir}/", f"{remote_dir}/", excludes, None)
        return _run(argv, 600.0)

    def pull(self, remote_dir: str, local_dir: Path,
             extra: Sequence[str] = ()) -> CommandResult:
        local_dir.mkdir(parents=True, exist_ok=True)
        argv = _rsync_argv(self.rsync_bin, extra, f"{remote_dir}/", f"{local_dir}/", (), None)
        return _run(argv, 1800.0)

    @property
    def label(self) -> str:
        return "local (dry run)"


@dataclass(frozen=True)
class SshTransport(Transport):
    """The real thing: key-based ssh, no password prompt, no interactive host check."""

    host: str
    port: int
    user: str = "root"
    ssh_bin: str = "ssh"
    rsync_bin: str = "rsync"

    @property
    def _opts(self) -> tuple[str, ...]:
        return ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-p", str(self.port))

    @property
    def _ssh_for_rsync(self) -> str:
        return " ".join(shlex.quote(part) for part in (self.ssh_bin, *self._opts))

    @property
    def _target(self) -> str:
        return f"{self.user}@{self.host}"

    def run(self, command: str, timeout_s: float | None = 120.0) -> CommandResult:
        return _run([self.ssh_bin, *self._opts, self._target, command], timeout_s)

    def push(self, local_dir: Path, remote_dir: str,
             excludes: Sequence[str] = ()) -> CommandResult:
        argv = _rsync_argv(self.rsync_bin, (), f"{local_dir}/",
                           f"{self._target}:{remote_dir}/", excludes, self._ssh_for_rsync)
        return _run(argv, 900.0)

    def pull(self, remote_dir: str, local_dir: Path,
             extra: Sequence[str] = ()) -> CommandResult:
        local_dir.mkdir(parents=True, exist_ok=True)
        argv = _rsync_argv(self.rsync_bin, extra, f"{self._target}:{remote_dir}/",
                           f"{local_dir}/", (), self._ssh_for_rsync)
        return _run(argv, 3600.0)

    @property
    def label(self) -> str:
        return f"ssh {self._target} -p {self.port}"


@dataclass(frozen=True)
class RemoteProbe:
    """One round trip: clock, STATUS, heartbeat and the marker files."""

    reachable: bool
    started: bool
    now_epoch: int | None = None
    status: str | None = None
    heartbeat_epoch: int | None = None
    stage: str | None = None
    markers: tuple[str, ...] = ()
    error: str | None = None

    @property
    def heartbeat_age_s(self) -> float | None:
        if self.now_epoch is None or self.heartbeat_epoch is None:
            return None
        return float(self.now_epoch - self.heartbeat_epoch)


def probe_command(results_dir: str, status_rel: str) -> str:
    """One shell command, so a poll costs a single ssh round trip."""
    quoted = shlex.quote(results_dir)
    return (
        f"cd {quoted} 2>/dev/null || exit {NOT_STARTED}; "
        "printf '@@NOW\\n'; date +%s; "
        f"printf '@@STATUS\\n'; cat {shlex.quote(status_rel)} 2>/dev/null; "
        "printf '@@HEARTBEAT\\n'; cat heartbeat.txt 2>/dev/null; "
        "printf '@@MARKERS\\n'; "
        "ls -1 . 2>/dev/null | grep -E '^(stage_.*\\.(done|failed)|bootstrap\\.done|PULLED)$'; "
        "printf '@@END\\n'"
    )


def _sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith("@@"):
            current = line[2:].strip().lower()
            out.setdefault(current, [])
            continue
        if current and line.strip():
            out[current].append(line.rstrip())
    return out


def parse_probe(result: CommandResult) -> RemoteProbe:
    if result.connection_failed:
        return RemoteProbe(reachable=False, started=False, error=result.tail())
    if result.returncode == NOT_STARTED:
        return RemoteProbe(reachable=True, started=False, error="results directory not created yet")
    if not result.ok:
        return RemoteProbe(reachable=True, started=False,
                           error=f"probe exit {result.returncode}: {result.tail()}")
    sections = _sections(result.stdout)
    now = sections.get("now") or []
    status = sections.get("status") or []
    beat = (sections.get("heartbeat") or [""])[0].split("\t")
    epoch = int(beat[1]) if len(beat) > 1 and beat[1].isdigit() else None
    return RemoteProbe(
        reachable=True, started=True,
        now_epoch=int(now[0]) if now and now[0].strip().isdigit() else None,
        status=status[0].strip() if status else None,
        heartbeat_epoch=epoch,
        stage=beat[2] if len(beat) > 2 else None,
        markers=tuple(sections.get("markers") or ()),
    )


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ManifestReport:
    remote_files: int
    local_files: int
    remote_bytes: int
    local_bytes: int
    missing_locally: tuple[str, ...] = ()
    extra_locally: tuple[str, ...] = ()
    size_mismatch: tuple[str, ...] = ()
    hash_mismatch: tuple[str, ...] = ()
    remote_unavailable: bool = False

    @property
    def ok(self) -> bool:
        """Extra local files are fine; anything missing, different or unverified is not.

        An unreadable remote manifest must never look like a match: that verdict
        is what releases the box.
        """
        return not (self.remote_unavailable or self.missing_locally
                    or self.size_mismatch or self.hash_mismatch)


def manifest_command(results_dir: str) -> str:
    """size, sha256 and relative path of every file, computed on the box."""
    return (
        f"cd {shlex.quote(results_dir)} || exit {NOT_STARTED}; "
        "H=sha256sum; command -v sha256sum >/dev/null 2>&1 || H='shasum -a 256'; "
        "find . -type f ! -name PULL_MANIFEST.json ! -name .DS_Store | LC_ALL=C sort | "
        "while IFS= read -r f; do s=$(wc -c < \"$f\" | tr -d ' '); "
        "h=$($H \"$f\" | cut -d' ' -f1); "
        "printf '%s\\t%s\\t%s\\n' \"$s\" \"$h\" \"${f#./}\"; done"
    )


def parse_manifest(text: str) -> dict[str, FileEntry]:
    entries: dict[str, FileEntry] = {}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        size, digest, rel = int(parts[0]), parts[1].strip(), parts[2]
        entries[rel] = FileEntry(rel, size, digest)
    return entries


def local_manifest(local_dir: Path) -> dict[str, FileEntry]:
    entries: dict[str, FileEntry] = {}
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        rel = path.relative_to(local_dir).as_posix()
        entries[rel] = FileEntry(rel, path.stat().st_size, digest.hexdigest())
    return entries


def compare_manifests(
    remote: Mapping[str, FileEntry], local: Mapping[str, FileEntry],
    remote_unavailable: bool = False,
) -> ManifestReport:
    missing = tuple(sorted(set(remote) - set(local)))
    extra = tuple(sorted(set(local) - set(remote)))
    shared = sorted(set(remote) & set(local))
    size_bad = tuple(rel for rel in shared if remote[rel].size != local[rel].size)
    hash_bad = tuple(
        rel for rel in shared
        if rel not in size_bad and remote[rel].sha256 and local[rel].sha256
        and remote[rel].sha256 != local[rel].sha256
    )
    return ManifestReport(
        remote_files=len(remote), local_files=len(local),
        remote_bytes=sum(entry.size for entry in remote.values()),
        local_bytes=sum(entry.size for entry in local.values()),
        missing_locally=missing, extra_locally=extra,
        size_mismatch=size_bad, hash_mismatch=hash_bad,
        remote_unavailable=remote_unavailable,
    )
