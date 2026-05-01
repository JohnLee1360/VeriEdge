from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "health_check.py"
    spec = importlib.util.spec_from_file_location("provider_health_check", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "mlx-community/Qwen3-0.6B-8bit"}]}).encode("utf-8"))
            return

        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                    }
                }
            ]
        }
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def log_message(self, format, *args):
        return


def _run(server: ThreadingHTTPServer):
    server.serve_forever(poll_interval=0.01)


def test_run_health_and_probe_success():
    mod = _load_module()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=_run, args=(server,), daemon=True)
    t.start()

    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        result = mod.run(endpoint=endpoint, timeout_sec=3.0, model="mlx-community/Qwen3-0.6B-8bit", probe=True)
        assert result["ok"] is True
        assert result["checks"]["models"]["ok"] is True
        assert result["checks"]["probe"]["ok"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_run_unhealthy_endpoint():
    mod = _load_module()
    result = mod.run(endpoint="http://127.0.0.1:9", timeout_sec=0.5, model=None, probe=False)
    assert result["ok"] is False
