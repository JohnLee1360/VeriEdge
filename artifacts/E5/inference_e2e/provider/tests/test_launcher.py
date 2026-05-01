from __future__ import annotations

import importlib.util
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


def _load_module():
    path = Path(__file__).resolve().parents[1] / "launcher.py"
    spec = importlib.util.spec_from_file_location("provider_launcher", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_aggregate_question_results():
    mod = _load_module()
    rows = [
        {"status": "success", "ttft_s": 0.4, "otps_tok_s": 12.0},
        {"status": "success", "ttft_s": 0.6, "otps_tok_s": 14.0},
        {"status": "failed", "ttft_s": None, "otps_tok_s": None},
    ]
    agg = mod.aggregate_question_results(rows)
    assert agg["ttft_p50_s"] == 0.5
    assert agg["otps_p50_tok_s"] == 13.0
    assert agg["question_success_count"] == 2
    assert agg["question_fail_count"] == 1


def test_launcher_health_endpoint():
    mod = _load_module()
    cfg = mod.LauncherConfig(
        provider_node_id="jlmini_2",
        provider_ip="192.168.31.159",
        exo_endpoint="http://127.0.0.1:52415",
        private_key_path=Path("/tmp/fake.pem"),
        model_id="mlx-community/Qwen3-0.6B-8bit",
        timeout_sec=3,
    )
    token_counter = mod.TokenCounter(cfg.model_id)
    server = mod.LauncherServer(("127.0.0.1", 0), cfg, token_counter)
    try:
        thread = mod.threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        resp = mod.httpx.get(f"http://127.0.0.1:{server.server_port}/health", timeout=3)
        assert resp.status_code == 200
        payload = json.loads(resp.text)
        assert payload["provider_node_id"] == "jlmini_2"
    finally:
        server.shutdown()
        server.server_close()


class _StreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN001, D401
        return

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(20):
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n')
            self.wfile.flush()
            time.sleep(0.2)


class _CallbackRetryHandler(BaseHTTPRequestHandler):
    attempts = 0

    def log_message(self, fmt, *args):  # noqa: ANN001, D401
        return

    def do_POST(self):  # noqa: N802
        if self.path != "/task-result":
            self.send_response(404)
            self.end_headers()
            return
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).attempts += 1
        if type(self).attempts == 1:
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')


def test_run_single_question_wall_clock_timeout():
    mod = _load_module()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        start = time.time()
        result = mod.asyncio.run(
            mod.run_single_question(
                entry_url=f"http://127.0.0.1:{server.server_port}",
                model_id="mlx-community/Qwen3-0.6B-8bit",
                prompt="hello",
                token_counter=mod.TokenCounter("mlx-community/Qwen3-0.6B-8bit"),
                timeout_sec=1,
                task_id="task-1",
                question_index=1,
                prompt_id=11,
            )
        )
        elapsed = time.time() - start
        assert result["status"] == "failed"
        assert result["failure_type"] == "timeout"
        assert elapsed < 4.0
    finally:
        server.shutdown()
        server.server_close()


def test_post_callback_with_retry():
    mod = _load_module()
    _CallbackRetryHandler.attempts = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CallbackRetryHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ok = mod.post_callback_with_retry(
            callback_url=f"http://127.0.0.1:{server.server_port}/task-result",
            payload={"task_id": "task-1"},
            timeout_sec=2,
            task_id="task-1",
            max_attempts=3,
            retry_delay_sec=0.05,
        )
        assert ok is True
        assert _CallbackRetryHandler.attempts == 2
    finally:
        server.shutdown()
        server.server_close()
