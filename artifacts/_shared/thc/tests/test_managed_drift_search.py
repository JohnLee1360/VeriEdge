from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from managed_drift_search import _completed, _remote_runtime_dir, _runtime_cluster_payloads


def _cluster_payload() -> dict[str, object]:
    return {
        "model_id": "Qwen/Qwen3-0.6B",
        "nodes": [
            {
                "node_name": "jlmini_3",
                "host": "192.168.31.51",
                "port": 8311,
                "checkpoint": "C1",
                "start_layer": 0,
                "end_layer": 7,
                "device": "mps",
                "torch_dtype": "float16",
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
                "torch_dtype": "float16",
                "quantization": "none",
                "first_shard": False,
                "last_shard": False,
            },
            {
                "node_name": "jlmini_2",
                "host": "192.168.31.159",
                "port": 8311,
                "checkpoint": "C3",
                "start_layer": 16,
                "end_layer": 23,
                "device": "mps",
                "torch_dtype": "float32",
                "quantization": "none",
                "first_shard": False,
                "last_shard": True,
            },
        ],
    }


class ManagedDriftSearchTests(unittest.TestCase):
    def test_runtime_clusters_rewrite_local_capture_endpoints(self) -> None:
        capture_cluster, remote_cluster = _runtime_cluster_payloads(
            _cluster_payload(),
            mac_host="192.168.31.83",
            local_port=18312,
            linux_tunnel_port=18311,
        )

        capture_nodes = {str(node["node_name"]): node for node in capture_cluster["nodes"]}
        self.assertEqual(str(capture_nodes["jlmini_2"]["host"]), "127.0.0.1")
        self.assertEqual(int(capture_nodes["jlmini_2"]["port"]), 18312)
        self.assertEqual(str(capture_nodes["linux124"]["host"]), "127.0.0.1")
        self.assertEqual(int(capture_nodes["linux124"]["port"]), 18311)
        self.assertEqual(str(capture_nodes["jlmini_3"]["host"]), "192.168.31.83")

        remote_nodes = {str(node["node_name"]): node for node in remote_cluster["nodes"]}
        self.assertEqual(str(remote_nodes["jlmini_2"]["host"]), "192.168.31.159")
        self.assertEqual(int(remote_nodes["jlmini_2"]["port"]), 18312)
        self.assertEqual(str(remote_nodes["linux124"]["host"]), "172.31.100.124")

    def test_completed_only_accepts_terminal_statuses(self) -> None:
        self.assertTrue(_completed({"status": "zero_delta"}))
        self.assertTrue(_completed({"status": "t5_complete"}))
        self.assertFalse(_completed({"status": "failed"}))
        self.assertFalse(_completed({"status": "running_calibration"}))

    def test_remote_runtime_dir_is_stable_for_external_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "managed_run"
            expected = "/remote/repo/artifacts/_shared/thc/output/managed_runtime/managed_run/d8_pA_c2_fp16_c3_fp32"
            self.assertEqual(
                _remote_runtime_dir(output_dir, "d8_pA_c2_fp16_c3_fp32", "/remote/repo"),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
