"""Batch-B remote automation: stage sequencing, terminal conditions, manifest compare.

Pure logic only: no network, no ssh, no subprocesses, no GPU.  The modules under
test live in ``scripts/q2/remote`` (standalone scripts, not an installed package),
so the directory goes on ``sys.path`` first.
"""

import sys
import time
from pathlib import Path

import pytest

REMOTE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "q2" / "remote"
if str(REMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DIR))

import hardware  # noqa: E402
import measurements  # noqa: E402
import orchestrate  # noqa: E402
import stages  # noqa: E402
import transport  # noqa: E402
from layout import RemoteLayout, StageSpec  # noqa: E402


# --------------------------------------------------------------- stage sequencing
def _layout(tmp_path: Path) -> RemoteLayout:
    layout = RemoteLayout(root=tmp_path)
    layout.results.mkdir(parents=True, exist_ok=True)
    layout.logs.mkdir(parents=True, exist_ok=True)
    return layout


THREE = (StageSpec("first", 60), StageSpec("second", 60), StageSpec("third", 60))


def test_sequence_runs_every_stage_and_writes_done_markers(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    seen: list[str] = []
    result = stages.run_stage_sequence(THREE, lambda spec: seen.append(spec.name), layout)
    assert result.ok
    assert seen == ["first", "second", "third"]
    assert result.ran == ("first", "second", "third")
    assert all(layout.done(name).exists() for name in ("first", "second", "third"))
    assert not any(layout.failed(name).exists() for name in ("first", "second", "third"))


def test_sequence_skips_stages_already_marked_done(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.done("first").write_text("earlier run\n", encoding="utf-8")
    seen: list[str] = []
    result = stages.run_stage_sequence(THREE, lambda spec: seen.append(spec.name), layout)
    assert seen == ["second", "third"]           # the resumable part: no re-run, no re-spend
    assert result.skipped == ("first",)
    assert result.ran == ("second", "third")
    assert layout.done("first").read_text(encoding="utf-8") == "earlier run\n"


def test_sequence_stops_at_the_first_failure_and_records_it(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    seen: list[str] = []

    def runner(spec: StageSpec) -> None:
        seen.append(spec.name)
        if spec.name == "second":
            raise RuntimeError("vllm died")

    result = stages.run_stage_sequence(THREE, runner, layout)
    assert seen == ["first", "second"]           # "third" never starts
    assert not result.ok
    assert result.failed == "second"
    assert "vllm died" in (result.error or "")
    assert layout.done("first").exists()
    assert layout.failed("second").exists()
    assert "vllm died" in layout.failed("second").read_text(encoding="utf-8")
    assert not layout.done("second").exists()
    assert not layout.done("third").exists() and not layout.failed("third").exists()


def test_stage_failure_status_carries_the_log_tail(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.stage_log("concurrency").write_text(
        "\n".join(f"line {i}" for i in range(200)) + "\n", encoding="utf-8"
    )
    result = stages.SequenceResult(ran=("coldstart",), failed="concurrency", error="exit 1")
    assert stages.write_status(layout, result, tail_lines=50) == "FAILED"
    text = layout.status.read_text(encoding="utf-8")
    assert text.startswith("FAILED\n")
    assert "stage=concurrency" in text
    assert "line 199" in text and "line 150" in text and "line 149" not in text


def test_done_status_lists_the_stages(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    result = stages.SequenceResult(ran=("coldstart", "prefix"), skipped=("check_patterns",))
    assert stages.write_status(layout, result) == "DONE"
    text = layout.status.read_text(encoding="utf-8")
    assert "stages_ran=coldstart,prefix" in text
    assert "stages_skipped=check_patterns" in text


# ----------------------------------------------------------- terminal conditions
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (orchestrate.PollState(status="DONE", heartbeat_age_s=12.0), "done"),
        (orchestrate.PollState(status="FAILED", heartbeat_age_s=12.0), "failed"),
        (orchestrate.PollState(status=None, heartbeat_age_s=20 * 60 + 1), "hang"),
        (orchestrate.PollState(status=None, heartbeat_age_s=None, ssh_failures=5),
         "unreachable"),
        (orchestrate.PollState(status=None, heartbeat_age_s=45.0), None),
        (orchestrate.PollState(status=None, heartbeat_age_s=None), None),
        (orchestrate.PollState(status=None, heartbeat_age_s=60.0, ssh_failures=4), None),
    ],
)
def test_terminal_reason(state: orchestrate.PollState, expected: str | None) -> None:
    assert orchestrate.terminal_reason(state) == expected


def test_terminal_reason_prefers_unreachable_over_a_stale_status() -> None:
    state = orchestrate.PollState(status="DONE", heartbeat_age_s=10.0, ssh_failures=5)
    assert orchestrate.terminal_reason(state) == "unreachable"


def test_every_terminal_reason_has_an_exit_code() -> None:
    assert orchestrate.EXIT_CODES == {"done": 0, "failed": 2, "hang": 3, "unreachable": 4}


def test_probe_parsing_reads_status_heartbeat_and_markers() -> None:
    stdout = (
        "@@NOW\n1789000600\n"
        "@@STATUS\nDONE\nfinished_at=2026-09-17T10:00:00+0800\n"
        "@@HEARTBEAT\n2026-09-17T09:59:00+0800\t1789000540\tsummarize\n"
        "@@MARKERS\nstage_coldstart.done\nstage_prefix.failed\n"
        "@@END\n"
    )
    probe = transport.parse_probe(transport.CommandResult(("ssh",), 0, stdout, ""))
    assert probe.reachable and probe.started
    assert probe.status == "DONE"
    assert probe.stage == "summarize"
    assert probe.heartbeat_age_s == 60.0
    assert probe.markers == ("stage_coldstart.done", "stage_prefix.failed")


def test_probe_distinguishes_a_dead_connection_from_a_box_that_has_not_started() -> None:
    gone = transport.parse_probe(transport.CommandResult(("ssh",), 255, "", "ssh: connect"))
    assert not gone.reachable
    not_yet = transport.parse_probe(transport.CommandResult(("ssh",), transport.NOT_STARTED,
                                                            "", ""))
    assert not_yet.reachable and not not_yet.started and not_yet.status is None


# ---------------------------------------------------------------------- manifests
def _entry(path: str, size: int, digest: str) -> transport.FileEntry:
    return transport.FileEntry(path, size, digest)


def test_manifest_comparison_flags_a_size_mismatch() -> None:
    remote = {
        "b1/coldstart/coldstart.jsonl": _entry("b1/coldstart/coldstart.jsonl", 4096, "a" * 64),
        "b1/q2_summary.csv": _entry("b1/q2_summary.csv", 512, "b" * 64),
    }
    local = {
        "b1/coldstart/coldstart.jsonl": _entry("b1/coldstart/coldstart.jsonl", 2048, "a" * 64),
        "b1/q2_summary.csv": _entry("b1/q2_summary.csv", 512, "b" * 64),
    }
    report = transport.compare_manifests(remote, local)
    assert not report.ok
    assert report.size_mismatch == ("b1/coldstart/coldstart.jsonl",)
    assert report.hash_mismatch == ()          # a size mismatch is not double-reported
    assert report.missing_locally == ()
    assert report.remote_bytes == 4608 and report.local_bytes == 2560


def test_manifest_comparison_flags_missing_and_hash_mismatches() -> None:
    remote = {"a": _entry("a", 10, "a" * 64), "b": _entry("b", 10, "b" * 64)}
    local = {"a": _entry("a", 10, "c" * 64), "extra": _entry("extra", 1, "d" * 64)}
    report = transport.compare_manifests(remote, local)
    assert report.missing_locally == ("b",)
    assert report.hash_mismatch == ("a",)
    assert report.extra_locally == ("extra",)
    assert not report.ok


def test_manifest_comparison_accepts_an_identical_tree() -> None:
    entries = {"b1/STATUS": _entry("b1/STATUS", 7, "e" * 64)}
    report = transport.compare_manifests(entries, dict(entries))
    assert report.ok
    assert report.remote_files == report.local_files == 1


def test_manifest_parsing_ignores_junk_lines() -> None:
    text = "4096\tdeadbeef\tb1/coldstart/coldstart.jsonl\nrsync: warning\n\n12\tcafe\tb1/STATUS\n"
    entries = transport.parse_manifest(text)
    assert set(entries) == {"b1/coldstart/coldstart.jsonl", "b1/STATUS"}
    assert entries["b1/STATUS"].size == 12


# ------------------------------------------------------- log patterns & GPU tiers
def test_check_log_patterns_reports_the_unmatched_milestone() -> None:
    patterns = {
        "weights_loaded": (r"Model loading took", r"Loading model weights took"),
        "cuda_graph_done": (r"Graph capturing finished",),
        "server_listening": (r"Application startup complete",),
    }
    lines = [
        "INFO [gpu_model_runner.py:2178] Model loading took 14.99 GiB, 11.5 seconds",
        "INFO [gpu_model_runner.py:2178] Model loading took 14.99 GiB, 11.6 seconds",
        "INFO:     Application startup complete.",
    ]
    report = measurements.check_log_patterns(lines, patterns)
    assert report.unmatched == ("cuda_graph_done",)
    assert report.matched["weights_loaded"] == (r"Model loading took",)
    assert report.counts[r"Model loading took"] == 2
    assert report.counts[r"Graph capturing finished"] == 0


@pytest.mark.parametrize(
    ("vram_gb", "tier", "max_len", "util", "large"),
    [
        (84.0, "80g", 4096, 0.90, True),      # RTX PRO 6000D (Blackwell), the rented box
        (79.2, "80g", 4096, 0.90, True),
        (70.0, "80g", 4096, 0.90, True),
        (47.5, "40g", 2048, 0.92, True),
        (39.6, "40g", 2048, 0.92, True),      # a nominal 40 GB A100 reports ~39.6 GiB
        (31.7, "small", 2048, 0.92, False),   # nominal 32 GB V100
        (23.9, "small", 2048, 0.92, False),
        (None, "80g", 4096, 0.90, True),
    ],
)
def test_gpu_tier_picks_serve_args_from_vram(
    vram_gb: float | None, tier: str, max_len: int, util: float, large: bool
) -> None:
    chosen = hardware.choose_tier(vram_gb, "test-gpu")
    assert (chosen.name, chosen.max_model_len, chosen.allow_large_models) == (tier, max_len, large)
    assert chosen.gpu_memory_utilization == util
    assert chosen.reason


def test_the_rented_blackwell_box_lands_in_the_runbook_tier() -> None:
    chosen = hardware.choose_tier(84.0, "NVIDIA RTX PRO 6000D")
    assert (chosen.name, chosen.max_model_len, chosen.gpu_memory_utilization) == ("80g", 4096, 0.9)
    assert chosen.allow_large_models
    assert len(measurements.select_models(chosen)) == len(measurements.DEFAULT_MODELS)  # 72B-AWQ joined in B4


def test_cli_overrides_beat_the_detected_tier() -> None:
    chosen = hardware.choose_tier(47.5, "L40S", max_model_len=1024, gpu_mem_util=0.85)
    assert chosen.max_model_len == 1024
    assert chosen.gpu_memory_utilization == 0.85
    assert "forced on the CLI" in chosen.reason


def test_a_small_card_drops_the_models_it_cannot_serve() -> None:
    small = hardware.choose_tier(24.0, "RTX 4090")
    kept = measurements.select_models(small)
    assert [model.slug for model in kept] == ["qwen2.5-7b"]
    roomy = hardware.choose_tier(47.5, "L40S")
    assert len(measurements.select_models(roomy)) == len(measurements.DEFAULT_MODELS)


# --------------------------------------------------- the watch loop (no network)
class _ScriptedTransport(transport.Transport):
    """Canned probe replies; push and pull always succeed and touch nothing."""

    def __init__(self, replies: list[transport.CommandResult]) -> None:
        self.replies = list(replies)
        self.commands: list[str] = []

    def run(self, command: str, timeout_s: float | None = 120.0) -> transport.CommandResult:
        self.commands.append(command)
        if self.replies:
            return self.replies.pop(0)
        return transport.CommandResult(("fake",), 0, "", "")

    def push(self, local_dir: Path, remote_dir: str,
             excludes: "tuple[str, ...]" = ()) -> transport.CommandResult:
        return transport.CommandResult(("rsync", "push"), 0, "", "")

    def pull(self, remote_dir: str, local_dir: Path,
             extra: "tuple[str, ...]" = ()) -> transport.CommandResult:
        time.sleep(0.002)          # so "no heartbeat since launch" is strictly positive
        return transport.CommandResult(("rsync", "pull"), 0, "", "")

    @property
    def label(self) -> str:
        return "scripted"


def _probe_reply(status: str = "", heartbeat_epoch: int | None = None,
                 now_epoch: int = 1789000600) -> transport.CommandResult:
    beat = "" if heartbeat_epoch is None else f"2026-09-17T10:00:00+0800\t{heartbeat_epoch}\tprefix"
    stdout = (f"@@NOW\n{now_epoch}\n@@STATUS\n{status}\n@@HEARTBEAT\n{beat}\n"
              "@@MARKERS\nstage_coldstart.done\n@@END\n")
    return transport.CommandResult(("ssh",), 0, stdout, "")


def _cfg(tmp_path: Path, **kwargs: object) -> orchestrate.OrchestratorConfig:
    base = {"local_out": tmp_path / "out", "q2_dir": tmp_path, "remote_root": tmp_path / "box",
            "poll_s": 0.0, "stale_heartbeat_s": 300.0}
    base.update(kwargs)
    return orchestrate.OrchestratorConfig(**base)  # type: ignore[arg-type]


def test_watch_stops_on_done_after_a_few_polls(tmp_path: Path) -> None:
    tp = _ScriptedTransport([
        _probe_reply(heartbeat_epoch=1789000590),
        _probe_reply(heartbeat_epoch=1789000595),
        _probe_reply("DONE", heartbeat_epoch=1789000599),
    ])
    assert orchestrate.Orchestrator(_cfg(tmp_path), tp).watch() == "done"
    assert len(tp.commands) == 3


def test_watch_calls_a_stale_heartbeat_a_hang(tmp_path: Path) -> None:
    tp = _ScriptedTransport([_probe_reply(heartbeat_epoch=1789000000)])  # 600 s old
    assert orchestrate.Orchestrator(_cfg(tmp_path), tp).watch() == "hang"


def test_watch_gives_up_when_the_driver_never_writes_a_heartbeat(tmp_path: Path) -> None:
    """A launch that failed silently must not hold the laptop for the whole budget."""
    tp = _ScriptedTransport([_probe_reply() for _ in range(3)])
    assert orchestrate.Orchestrator(_cfg(tmp_path, stale_heartbeat_s=0.0), tp).watch() == "hang"


def test_watch_reports_a_vanished_box_as_unreachable(tmp_path: Path) -> None:
    gone = transport.CommandResult(("ssh",), 255, "", "ssh: connect to host port 22: timed out")
    tp = _ScriptedTransport([gone] * 5)
    assert orchestrate.Orchestrator(_cfg(tmp_path), tp).watch() == "unreachable"
    assert len(tp.commands) == 5


def test_an_unreadable_remote_manifest_never_counts_as_verified() -> None:
    """The ok verdict is what releases a rented box, so unverified must not be ok."""
    local = {"b1/STATUS": _entry("b1/STATUS", 7, "e" * 64)}
    report = transport.compare_manifests({}, local, remote_unavailable=True)
    assert not report.ok
    assert report.remote_unavailable
    assert transport.compare_manifests({}, local).ok        # genuinely empty remote is fine
