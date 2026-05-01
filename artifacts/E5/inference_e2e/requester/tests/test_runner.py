from __future__ import annotations

import importlib.util
import json
import math
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


def _load_runner_module():
    runner_path = Path(__file__).resolve().parents[1] / "runner.py"
    spec = importlib.util.spec_from_file_location("task_runner", runner_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_task_manifest():
    mod = _load_runner_module()
    manifest = mod.build_task_manifest(
        task_id="task-001",
        model_id="mlx-community/Qwen3-0.6B-8bit",
        prompts=[(3, "alpha"), (8, "beta")],
        requester_id="johnlee",
    )
    assert manifest["task_id"] == "task-001"
    assert manifest["task_type"] == "text"
    assert manifest["question_count"] == 2
    assert manifest["requester_id"] == "johnlee"
    assert manifest["prompts"][0] == {"prompt_id": 3, "content": "alpha"}


def test_build_dispatch_payload():
    mod = _load_runner_module()
    payload = mod.build_dispatch_payload(
        task_id="task-001",
        cid="bafy123",
        encrypted_task_key="cipher",
        entry_url="http://192.168.31.159:52415",
        callback_url="http://192.168.31.189:18081/task-result",
        gateway_url="http://192.168.31.189:8080",
        model_id="mlx-community/Qwen3-0.6B-8bit",
        instance_id="inst-1",
    )
    assert payload["task_id"] == "task-001"
    assert payload["cid"] == "bafy123"
    assert payload["callback_url"].endswith("/task-result")
    assert payload["instance_id"] == "inst-1"


def test_aggregate_task_result():
    mod = _load_runner_module()
    task_result = {
        "task_id": "task-001",
        "instance_id": "inst-1",
        "provider_node_id": "jlmini_2",
        "aggregate_stats": {
            "download_s": 1.2,
            "task_latency_s": 10.0,
            "question_success_count": 2,
            "question_fail_count": 1,
        },
        "question_results": [
            {"ttft_s": 0.4, "otps_tok_s": 11.0, "latency_s": 1.0},
            {"ttft_s": 0.6, "otps_tok_s": 13.0, "latency_s": 1.2},
            {"ttft_s": None, "otps_tok_s": None, "latency_s": 0.9},
        ],
    }
    row = mod.aggregate_task_result(
        task_result=task_result,
        network="WAN",
        instance_node_count=2,
        phase="main",
        task_index_in_cell=3,
        question_count=50,
    )
    assert row["network"] == "WAN"
    assert row["instance_node_count"] == 2
    assert row["phase"] == "main"
    assert row["task_index_in_cell"] == 3
    assert row["download_s_per_task"] == 1.2
    assert row["task_latency_s_per_task"] == 10.0
    assert row["question_latency_s_per_q"] == 0.2
    assert row["ttft_p50_s_per_task"] == 0.5
    assert row["otps_p50_tok_s_per_task"] == 12.0
    assert row["question_success_count_per_task"] == 2
    assert row["question_fail_count_per_task"] == 1


def test_aggregate_cell_summary():
    mod = _load_runner_module()
    rows = [
        {
            "download_s_per_task": 1.0,
            "task_latency_s_per_task": 10.0,
            "question_latency_s_per_q": 0.2,
            "ttft_p50_s_per_task": 0.5,
            "otps_p50_tok_s_per_task": 12.0,
            "question_success_count_per_task": 50,
            "question_fail_count_per_task": 0,
        },
        {
            "download_s_per_task": 2.0,
            "task_latency_s_per_task": 14.0,
            "question_latency_s_per_q": 0.28,
            "ttft_p50_s_per_task": 0.7,
            "otps_p50_tok_s_per_task": 10.0,
            "question_success_count_per_task": 48,
            "question_fail_count_per_task": 2,
        },
    ]
    summary = mod.aggregate_cell_summary(task_rows=rows, network="LAN", instance_node_count=1)
    assert summary["instance_node_count"] == 1
    assert summary["mean_download_s_per_task"] == 1.5
    assert summary["mean_task_latency_s_per_task"] == 12.0
    assert math.isclose(summary["mean_question_latency_s_per_q"], 0.24)
    assert summary["mean_ttft_p50_s"] == 0.6
    assert summary["mean_otps_p50_tok_s"] == 11.0
    assert summary["sum_question_success_count"] == 98
    assert summary["sum_question_fail_count"] == 2
    assert summary["completed_task_count"] == 2


def test_expected_instance_ids_for_node_count():
    mod = _load_runner_module()
    cfg = {
        "external_exo": {
            "expected_instance_ids_by_node_count": {
                "1": ["inst-a"],
                "2": ["inst-b"],
            }
        }
    }
    assert mod.expected_instance_ids_for_node_count(cfg, 1) == ["inst-a"]
    assert mod.expected_instance_ids_for_node_count(cfg, 2) == ["inst-b"]


def test_callback_server_roundtrip():
    mod = _load_runner_module()
    server, thread = mod.start_callback_server("http://127.0.0.1:18091/task-result")
    assert thread.is_alive()
    try:
        server.store({"task_id": "task-xyz", "value": 1})
        got = mod.wait_for_task_callback(server, "task-xyz", timeout_sec=1)
        assert got["value"] == 1

        start = time.time()
        try:
            mod.wait_for_task_callback(server, "missing-task", timeout_sec=0)
        except TimeoutError:
            pass
        else:  # pragma: no cover
            raise AssertionError("Expected TimeoutError")
        assert time.time() - start < 1.0
    finally:
        mod.stop_callback_server(server)


def test_validate_external_instance_state():
    mod = _load_runner_module()
    state = {
        "nodeNetwork": {
            "jlmini_1": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
            "jlmini_2": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
            "jlmini_3": {"interfaces": [{"ipAddress": "192.168.31.83"}]},
        },
        "instances": {
            "inst-1": {
                "shardAssignments": {
                    "modelId": "mlx-community/Qwen3-0.6B-8bit",
                    "nodeToRunner": {
                        "jlmini_2": "runner-0",
                        "jlmini_1": "runner-1",
                        "jlmini_3": "runner-2",
                    },
                    "runnerToShard": {
                        "runner-0": {"deviceRank": 0, "startLayer": 0},
                        "runner-1": {"deviceRank": 1, "startLayer": 10},
                        "runner-2": {"deviceRank": 2, "startLayer": 20},
                    },
                }
            }
        },
    }
    validated = mod.validate_external_instance_state(
        state=state,
        model_id="mlx-community/Qwen3-0.6B-8bit",
        expected_instance_node_count=3,
        expected_instance_ids=["inst-1"],
        provider_urls=[
            "http://192.168.31.52:52415",
            "http://192.168.31.159:52415",
            "http://192.168.31.83:52415",
        ],
    )
    assert validated["selected_instance_id"] == "inst-1"
    assert validated["first_shard_node_id"] == "jlmini_2"
    assert validated["first_shard_ip"] == "192.168.31.159"
    assert validated["ordered_instances"][0]["instance_id"] == "inst-1"
    assert validated["ordered_instances"][0]["instance_node_count"] == 3


def test_validate_external_instance_state_filters_by_node_count():
    mod = _load_runner_module()
    state = {
        "nodeNetwork": {
            "jlmini_1": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
            "jlmini_2": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
            "jlmini_3": {"interfaces": [{"ipAddress": "192.168.31.83"}]},
        },
        "instances": {
            "inst-1": {
                "shardAssignments": {
                    "modelId": "mlx-community/Qwen3-0.6B-8bit",
                    "nodeToRunner": {
                        "jlmini_2": "runner-0",
                        "jlmini_1": "runner-1",
                        "jlmini_3": "runner-2",
                    },
                    "runnerToShard": {
                        "runner-0": {"deviceRank": 0, "startLayer": 0},
                        "runner-1": {"deviceRank": 1, "startLayer": 10},
                        "runner-2": {"deviceRank": 2, "startLayer": 20},
                    },
                }
            },
            "inst-extra": {
                "shardAssignments": {
                    "modelId": "mlx-community/Qwen3-0.6B-8bit",
                    "nodeToRunner": {
                        "jlmini_2": "runner-10",
                    },
                    "runnerToShard": {
                        "runner-10": {"deviceRank": 0, "startLayer": 0},
                    },
                }
            },
        },
    }
    validated = mod.validate_external_instance_state(
        state=state,
        model_id="mlx-community/Qwen3-0.6B-8bit",
        expected_instance_node_count=3,
        expected_instance_ids=["inst-1"],
        provider_urls=[
            "http://192.168.31.52:52415",
            "http://192.168.31.159:52415",
            "http://192.168.31.83:52415",
        ],
    )
    assert validated["selected_instance_id"] == "inst-1"
    assert [item["instance_id"] for item in validated["ordered_instances"]] == ["inst-1"]


def test_wait_for_validated_external_instance_state_requires_stable_repeats():
    mod = _load_runner_module()
    states = [
        {
            "nodeNetwork": {
                "jlmini_1": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
                "jlmini_2": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
            },
            "instances": {},
        },
        {
            "nodeNetwork": {
                "jlmini_1": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
                "jlmini_2": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
            },
            "instances": {
                "inst-1": {
                    "shardAssignments": {
                        "modelId": "mlx-community/Qwen3-0.6B-8bit",
                        "nodeToRunner": {
                            "jlmini_2": "runner-0",
                        },
                        "runnerToShard": {
                            "runner-0": {"deviceRank": 0, "startLayer": 0},
                        },
                    }
                }
            },
        },
        {
            "nodeNetwork": {
                "jlmini_1": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
                "jlmini_2": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
            },
            "instances": {
                "inst-1": {
                    "shardAssignments": {
                        "modelId": "mlx-community/Qwen3-0.6B-8bit",
                        "nodeToRunner": {
                            "jlmini_2": "runner-0",
                        },
                        "runnerToShard": {
                            "runner-0": {"deviceRank": 0, "startLayer": 0},
                        },
                    }
                }
            },
        },
    ]

    async def fake_fetch_state(entry_url: str, timeout_sec: int):
        _ = entry_url, timeout_sec
        current = states.pop(0) if len(states) > 1 else states[0]
        return current

    original_fetch_state = mod.fetch_state
    mod.fetch_state = fake_fetch_state
    try:
        validated, state = mod.asyncio.run(
            mod.wait_for_validated_external_instance_state(
                entry_url="http://127.0.0.1:52415",
                timeout_sec=1,
                model_id="mlx-community/Qwen3-0.6B-8bit",
                expected_instance_node_count=1,
                expected_instance_ids=["inst-1"],
                provider_urls=["http://192.168.31.159:52415"],
                poll_sec=1,
                consecutive_successes=2,
            )
        )
    finally:
        mod.fetch_state = original_fetch_state

    assert validated["selected_instance_id"] == "inst-1"
    assert state["instances"]["inst-1"]["shardAssignments"]["modelId"] == "mlx-community/Qwen3-0.6B-8bit"


def test_build_smoke_prompt_sets_round_robin():
    mod = _load_runner_module()
    smoke_prompts = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
    instances = [
        {"instance_id": "inst-a", "first_shard": {"node_id": "n1", "ip": "192.168.31.52"}},
        {"instance_id": "inst-b", "first_shard": {"node_id": "n2", "ip": "192.168.31.159"}},
    ]
    sets = mod._build_smoke_prompt_sets(smoke_prompts, instances, 3)
    assert [item[0]["instance_id"] for item in sets] == ["inst-a", "inst-b"]
    assert [prompt_id for prompt_id, _ in sets[0][1]] == [1, 2, 3]
    assert [prompt_id for prompt_id, _ in sets[1][1]] == [4, 5, 1]


class _HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN001, ANN002, D401
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path == "/version":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"Version":"0.0.0"}')
            return
        self.send_response(404)
        self.end_headers()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(server):
    server.serve_forever()


def test_run_aux_service_checks():
    mod = _load_runner_module()
    callback_server, _ = mod.start_callback_server(f"http://127.0.0.1:{_free_port()}/task-result")
    gateway_port = _free_port()
    api_port = _free_port()
    launcher_port = _free_port()
    gateway_server = ThreadingHTTPServer(("127.0.0.1", gateway_port), _HealthHandler)
    api_server = ThreadingHTTPServer(("127.0.0.1", api_port), _HealthHandler)
    launcher_server = ThreadingHTTPServer(("127.0.0.1", launcher_port), _HealthHandler)
    threads = [
        Thread(target=_serve, args=(gateway_server,), daemon=True),
        Thread(target=_serve, args=(api_server,), daemon=True),
        Thread(target=_serve, args=(launcher_server,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        rows = mod.asyncio.run(
            mod.run_aux_service_checks(
                requester_callback_url=f"http://127.0.0.1:{callback_server.server_port}/task-result",
                ipfs_api_url=f"http://127.0.0.1:{api_port}",
                ipfs_gateway_url=f"http://127.0.0.1:{gateway_port}",
                launcher_urls=[f"http://127.0.0.1:{launcher_port}"],
                timeout_sec=3,
            )
        )
        assert all(row["ok"] for row in rows), json.dumps(rows, indent=2)
    finally:
        mod.stop_callback_server(callback_server)
        for server in [gateway_server, api_server, launcher_server]:
            server.shutdown()
            server.server_close()


def test_ensure_preflight_checks_ok_reports_failed_health_and_aux():
    mod = _load_runner_module()
    try:
        mod.ensure_preflight_checks_ok(
            health_checks=[
                {"endpoint": "http://192.168.31.159:52415", "ok": False, "detail": "ConnectError: failed"},
            ],
            aux_checks=[
                {
                    "service": "provider_launcher",
                    "endpoint": "http://192.168.31.159:18080",
                    "ok": False,
                    "detail": "ConnectError: failed",
                },
            ],
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "health checks failed" in message
        assert "aux checks failed" in message
        assert "192.168.31.159:52415" in message
        assert "192.168.31.159:18080" in message
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError for failed preflight checks")
