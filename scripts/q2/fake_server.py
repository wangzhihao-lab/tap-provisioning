#!/usr/bin/env python3
"""Tiny OpenAI-compatible fake inference server used by the Q2 ``--dry-run`` paths.

It is a crude analytical stand-in for a vLLM replica: a fixed overhead, a prefill
term proportional to the prompt length (discounted on a prefix-cache hit), a
per-token decode term, and a queueing inflation factor driven by the number of
in-flight requests.  It exists so that the request / SSE-parse / aggregation path
of ``concurrency_curve.py`` and ``prefix_warmup.py`` can be exercised on a
CPU-only laptop.  No number produced against this server is a measurement.

Standalone::

    python fake_server.py --port 8001 --capacity 16 --time-scale 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("q2.fake_server")

CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class FakeServerConfig:
    """Parameters of the analytical serving model."""

    host: str = "localhost"
    port: int = 0
    model: str = "fake/Qwen2.5-7B-Instruct"
    capacity: int = 16
    base_ttft_s: float = 0.045
    prefill_s_per_ktok: float = 0.055
    decode_s_per_token: float = 0.012
    prefix_cache: bool = True
    prefix_hit_factor: float = 0.12
    prefix_block_tokens: int = 512
    jitter: float = 0.12
    error_rate: float = 0.0
    time_scale: float = 1.0
    seed: int = 20260917


class _ServerState:
    """Mutable server-side state shared by the handler threads."""

    def __init__(self, seed: int) -> None:
        self.lock = threading.Lock()
        self.inflight = 0
        self.prefix_blocks: set[str] = set()
        self.rng = random.Random(seed)

    def enter(self) -> int:
        with self.lock:
            self.inflight += 1
            return self.inflight

    def leave(self) -> None:
        with self.lock:
            self.inflight -= 1

    def jitter(self, scale: float) -> float:
        with self.lock:
            return self.rng.gauss(1.0, scale)

    def prefix_hit(self, key: str) -> bool:
        with self.lock:
            hit = key in self.prefix_blocks
            self.prefix_blocks.add(key)
            return hit


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _prompt_text(body: dict[str, Any]) -> str:
    if "prompt" in body:
        prompt = body["prompt"]
        return prompt if isinstance(prompt, str) else "\n".join(map(str, prompt))
    parts = [str(m.get("content", "")) for m in body.get("messages", [])]
    return "\n".join(parts)


def _prefix_key(text: str, block_tokens: int) -> str:
    head = text[: block_tokens * CHARS_PER_TOKEN]
    return hashlib.blake2b(head.encode("utf-8"), digest_size=8).hexdigest()


def _plan(cfg: FakeServerConfig, state: _ServerState, text: str) -> tuple[float, float, bool]:
    """Return (ttft_s, decode_delay_per_token_s, prefix_hit)."""
    n_tok = _estimate_tokens(text)
    hit = cfg.prefix_cache and state.prefix_hit(_prefix_key(text, cfg.prefix_block_tokens))
    prefill = cfg.prefill_s_per_ktok * (n_tok / 1000.0)
    if hit:
        prefill *= cfg.prefix_hit_factor
    inflight = max(1, state.inflight)
    queue = 1.0 + max(0.0, inflight - 1.0) / float(cfg.capacity)
    noise = max(0.5, state.jitter(cfg.jitter))
    ttft = (cfg.base_ttft_s + prefill) * queue * noise * cfg.time_scale
    decode = cfg.decode_s_per_token * math.sqrt(queue) * cfg.time_scale
    return ttft, decode, hit


def _make_handler(cfg: FakeServerConfig, state: _ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "fake-vllm/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib hook
            logger.debug("fake-server %s", fmt % args)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path in ("/v1/models", "/models"):
                self._send_json(200, {"object": "list", "data": [{"id": cfg.model}]})
                return
            self._send_json(404, {"error": {"message": f"no route {path}"}})

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/completions", "/v1/chat/completions"):
                self._send_json(404, {"error": {"message": f"no route {path}"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": {"message": "bad json"}})
                return
            state.enter()
            try:
                self._serve(path, body)
            finally:
                state.leave()

        def _serve(self, path: str, body: dict[str, Any]) -> None:
            if cfg.error_rate > 0.0 and random.random() < cfg.error_rate:
                self._send_json(503, {"error": {"message": "simulated overload"}})
                return
            text = _prompt_text(body)
            max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 64)
            ttft, decode, hit = _plan(cfg, state, text)
            n_out = max(1, int(max_tokens * 0.92))
            chat = path.endswith("chat/completions")
            if body.get("stream"):
                self._stream(chat, ttft, decode, n_out, hit)
            else:
                time.sleep(ttft + decode * n_out)
                self._send_json(200, _final_body(cfg.model, chat, n_out, _estimate_tokens(text)))

        def _stream(self, chat: bool, ttft: float, decode: float, n_out: int, hit: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Fake-Prefix-Hit", "1" if hit else "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            time.sleep(ttft)
            created = int(time.time())
            for i in range(n_out):
                if i:
                    time.sleep(decode)
                chunk = _chunk_body(cfg.model, chat, created, f"tok{i} ", None)
                if not self._write_sse(chunk):
                    return
            self._write_sse(_chunk_body(cfg.model, chat, created, "", "length"))
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def _write_sse(self, payload: dict[str, Any]) -> bool:
            try:
                self.wfile.write(b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return False
            return True

    return Handler


def _chunk_body(
    model: str, chat: bool, created: int, content: str, finish: str | None
) -> dict[str, Any]:
    if chat:
        delta = {"content": content}
        choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish}
        obj = "chat.completion.chunk"
    else:
        choice = {"index": 0, "text": content, "finish_reason": finish}
        obj = "text_completion"
    return {"id": "fake-0", "object": obj, "created": created, "model": model, "choices": [choice]}


def _final_body(model: str, chat: bool, n_out: int, n_in: int) -> dict[str, Any]:
    text = " ".join(f"tok{i}" for i in range(n_out))
    if chat:
        choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "length",
        }
        obj = "chat.completion"
    else:
        choice = {"index": 0, "text": text, "finish_reason": "length"}
        obj = "text_completion"
    usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}
    return {
        "id": "fake-0",
        "object": obj,
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
        "usage": usage,
    }


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 512


def start_fake_server(cfg: FakeServerConfig) -> tuple[_Server, threading.Thread, str]:
    """Start the fake server on a background thread and return (server, thread, base_url)."""
    state = _ServerState(cfg.seed)
    server = _Server((cfg.host, cfg.port), _make_handler(cfg, state))
    thread = threading.Thread(target=server.serve_forever, name="fake-server", daemon=True)
    thread.start()
    base_url = f"http://{cfg.host}:{server.server_address[1]}"
    logger.info("fake server listening on %s (model=%s)", base_url, cfg.model)
    return server, thread, base_url


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=FakeServerConfig.host)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default=FakeServerConfig.model)
    parser.add_argument("--capacity", type=int, default=FakeServerConfig.capacity)
    parser.add_argument("--time-scale", type=float, default=FakeServerConfig.time_scale)
    parser.add_argument("--error-rate", type=float, default=FakeServerConfig.error_rate)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=FakeServerConfig.seed)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = FakeServerConfig(
        host=args.host,
        port=args.port,
        model=args.model,
        capacity=args.capacity,
        time_scale=args.time_scale,
        error_rate=args.error_rate,
        prefix_cache=not args.no_prefix_cache,
        seed=args.seed,
    )
    server, _thread, base_url = start_fake_server(cfg)
    logger.info("serving at %s; Ctrl-C to stop", base_url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
