from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from hetero_cluster import cluster_nodes_from_config, load_cluster_config, resolve_node
from hetero_qwen_common import QwenShardRunner, parse_torch_dtype, select_torch_device
from hetero_transport import decode_array, encode_array


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one Qwen shard for heterogeneous THC/TSTC capture")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--node-name", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--start-layer", type=int, default=-1)
    parser.add_argument("--end-layer", type=int, default=-1)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--quantization-bits", type=int, default=8)
    parser.add_argument("--quantization-group-size", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--first-shard", action="store_true")
    parser.add_argument("--last-shard", action="store_true")
    parser.add_argument("--cluster-file", default="")
    parser.add_argument("--local-node", default="")
    return parser.parse_args()


def _resolve_runtime_args(args: argparse.Namespace) -> dict[str, Any]:
    resolved = {
        "host": str(args.host).strip() or "0.0.0.0",
        "port": int(args.port) if int(args.port) > 0 else 8311,
        "node_name": str(args.node_name).strip(),
        "model_id": str(args.model_id).strip() or "Qwen/Qwen3-0.6B",
        "start_layer": int(args.start_layer),
        "end_layer": int(args.end_layer),
        "checkpoint": str(args.checkpoint).strip(),
        "device": str(args.device).strip(),
        "torch_dtype": str(args.torch_dtype).strip(),
        "quantization": str(args.quantization).strip() or "none",
        "quantization_bits": int(args.quantization_bits),
        "quantization_group_size": int(args.quantization_group_size),
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
        "first_shard": bool(args.first_shard),
        "last_shard": bool(args.last_shard),
    }

    if str(args.cluster_file).strip():
        if not str(args.local_node).strip():
            raise ValueError("--local-node is required when --cluster-file is set")
        cluster_cfg = load_cluster_config(Path(str(args.cluster_file)).expanduser().resolve())
        node = resolve_node(cluster_nodes_from_config(cluster_cfg), str(args.local_node))
        resolved.update(
            {
                "host": str(node.get("bind_host", resolved["host"])).strip() or resolved["host"],
                "port": int(node.get("port", resolved["port"])),
                "node_name": str(node["node_name"]),
                "model_id": str(cluster_cfg.get("model_id", resolved["model_id"])),
                "start_layer": int(node["start_layer"]),
                "end_layer": int(node["end_layer"]),
                "checkpoint": str(node["checkpoint"]),
                "device": str(node.get("device", resolved["device"])),
                "torch_dtype": str(node.get("torch_dtype", resolved["torch_dtype"])),
                "quantization": str(node.get("quantization", resolved["quantization"])),
                "quantization_bits": int(node.get("quantization_bits", resolved["quantization_bits"])),
                "quantization_group_size": int(
                    node.get("quantization_group_size", resolved["quantization_group_size"])
                ),
                "local_files_only": bool(node.get("local_files_only", resolved["local_files_only"])),
                "trust_remote_code": bool(node.get("trust_remote_code", resolved["trust_remote_code"])),
                "first_shard": bool(node.get("first_shard", False)),
                "last_shard": bool(node.get("last_shard", False)),
            }
        )

    required = ["node_name", "checkpoint"]
    missing = [name for name in required if not resolved[name]]
    if missing:
        raise ValueError(f"missing required server fields: {missing}")
    if resolved["start_layer"] < 0 or resolved["end_layer"] < 0:
        raise ValueError("start-layer and end-layer must be provided")
    return resolved


class _Handler(BaseHTTPRequestHandler):
    runner: QwenShardRunner
    node_name: str

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            if self.path == "/ping":
                self._send_json({"ok": True, "node_name": self.node_name, "backend": self.runner.backend_label})
                return

            if self.path == "/reset_session":
                self.runner.reset_session(str(payload["session_id"]))
                self._send_json({"ok": True, "node_name": self.node_name})
                return

            if self.path not in {"/prefill", "/decode"}:
                self._send_json({"ok": False, "error": f"unsupported path: {self.path}"}, status=404)
                return

            hidden_payload = payload.get("hidden_states")
            hidden_states_np = decode_array(hidden_payload) if hidden_payload is not None else None
            out = self.runner.run(
                session_id=str(payload["session_id"]),
                input_ids=list(payload["input_ids"]) if payload.get("input_ids") is not None else None,
                hidden_states_np=hidden_states_np,
                position_ids=[int(v) for v in payload["position_ids"]],
                cache_position=[int(v) for v in payload["cache_position"]],
            )
            self._send_json(
                {
                    "ok": True,
                    "node_name": self.node_name,
                    "checkpoint": out["checkpoint"],
                    "shape": out["shape"],
                    "backend": out["backend"],
                    "device": out["device"],
                    "dtype": out["dtype"],
                    "tensor": encode_array(out["tensor"]),
                }
            )
        except Exception as exc:  # pragma: no cover - server path
            self._send_json({"ok": False, "error": repr(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def main() -> None:
    args = _parse_args()
    runtime = _resolve_runtime_args(args)
    runner = QwenShardRunner(
        model_id=str(runtime["model_id"]),
        start_layer=int(runtime["start_layer"]),
        end_layer=int(runtime["end_layer"]),
        checkpoint=str(runtime["checkpoint"]),
        is_first=bool(runtime["first_shard"]),
        is_last=bool(runtime["last_shard"]),
        device=select_torch_device(str(runtime["device"])),
        dtype=parse_torch_dtype(str(runtime["torch_dtype"])),
        quantization=str(runtime["quantization"]),
        quantization_bits=int(runtime["quantization_bits"]),
        quantization_group_size=int(runtime["quantization_group_size"]),
        local_files_only=bool(runtime["local_files_only"]),
        trust_remote_code=bool(runtime["trust_remote_code"]),
    )
    _Handler.runner = runner
    _Handler.node_name = str(runtime["node_name"])
    server = ThreadingHTTPServer((str(runtime["host"]), int(runtime["port"])), _Handler)
    print(
        f"Serving {runtime['node_name']} on http://{runtime['host']}:{runtime['port']} "
        f"checkpoint={runtime['checkpoint']} device={runtime['device']} quant={runtime['quantization']}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
