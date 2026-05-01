from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "artifacts" / "_shared" / "thc" / "src"))

from hetero_cluster import cluster_nodes_from_config, load_cluster_config, resolve_node


def _cluster_payload() -> dict[str, object]:
    return {
        "model_id": "Qwen/Qwen3-0.6B",
        "nodes": [
            {
                "node_name": "jlmini_3",
                "host": "192.168.0.12",
                "port": 8311,
                "checkpoint": "C1",
                "start_layer": 0,
                "end_layer": 7,
                "device": "mps",
                "quantization": "metal_8bit",
                "first_shard": True,
                "last_shard": False,
            },
            {
                "node_name": "linux124",
                "host": "172.31.100.124",
                "port": 8311,
                "checkpoint": "C2",
                "start_layer": 8,
                "end_layer": 15,
                "device": "cuda",
                "quantization": "bitsandbytes_8bit",
                "first_shard": False,
                "last_shard": False,
            },
            {
                "node_name": "jlmini_2",
                "host": "192.168.0.11",
                "port": 8311,
                "checkpoint": "C3",
                "start_layer": 16,
                "end_layer": 23,
                "device": "mps",
                "quantization": "none",
                "first_shard": False,
                "last_shard": True,
            },
        ],
    }


class HeteroClusterTests(unittest.TestCase):
    def test_load_cluster_config_accepts_valid_layout(self) -> None:
        payload = _cluster_payload()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cluster.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            data = load_cluster_config(path)
        nodes = cluster_nodes_from_config(data)
        self.assertEqual([node["checkpoint"] for node in nodes], ["C1", "C2", "C3"])

    def test_resolve_node_returns_named_entry(self) -> None:
        node = resolve_node(cluster_nodes_from_config(_cluster_payload()), "linux124")
        self.assertEqual(node["device"], "cuda")
        self.assertEqual(node["checkpoint"], "C2")

    def test_non_contiguous_layers_are_rejected(self) -> None:
        payload = _cluster_payload()
        payload["nodes"][1]["start_layer"] = 9  # type: ignore[index]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cluster.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "start_layer"):
                load_cluster_config(path)

    def test_quantization_must_match_device(self) -> None:
        payload = _cluster_payload()
        payload["nodes"][1]["quantization"] = "metal_8bit"  # type: ignore[index]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cluster.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires device=mps"):
                load_cluster_config(path)

    def test_metal_8bit_is_rejected_on_non_first_shard(self) -> None:
        payload = _cluster_payload()
        payload["nodes"][2]["quantization"] = "metal_8bit"  # type: ignore[index]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cluster.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only supported on the first shard"):
                load_cluster_config(path)

    def test_float32_and_bfloat16_variants_are_accepted(self) -> None:
        payload = _cluster_payload()
        payload["nodes"][1]["torch_dtype"] = "bfloat16"  # type: ignore[index]
        payload["nodes"][1]["quantization"] = "none"  # type: ignore[index]
        payload["nodes"][2]["torch_dtype"] = "float32"  # type: ignore[index]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cluster.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            data = load_cluster_config(path)
        self.assertEqual(data["nodes"][1]["torch_dtype"], "bfloat16")
