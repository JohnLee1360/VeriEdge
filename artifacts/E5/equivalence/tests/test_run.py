from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equivalence_common as common  # noqa: E402
import run_1device  # noqa: E402
import run_2device  # noqa: E402
import run_3device  # noqa: E402


def _cluster_cfg() -> dict:
    return {
        "experiment": {
            "seed": 42,
            "cluster_stabilization_timeout_sec": 1,
            "cluster_stabilization_poll_sec": 0.0,
            "cluster_stabilization_consecutive_successes": 1,
            "instance_ready_wait_sec": 1,
            "instance_ready_poll_sec": 0.0,
        },
        "dataset": {"sample_size": 1, "hf_dataset": "gsm8k", "subset": "main", "split": "test"},
        "model": {"model_id": "mlx-community/Qwen3-0.6B-8bit"},
        "decoding": {"timeout_sec": 5, "temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "stream": False},
    }


def test_summarize_state_for_model_handles_member_ips():
    state = {
        "nodeNetwork": {
            "n1": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
        },
        "instances": {
            "inst-1": {
                "MlxRingInstance": {
                    "instanceId": "inst-1",
                    "shardAssignments": {
                        "modelId": "mlx-community/Qwen3-0.6B-8bit",
                        "nodeToRunner": {"n1": "r1"},
                        "runnerToShard": {"r1": {"PipelineShardMetadata": {"deviceRank": 0, "startLayer": 0}}},
                    },
                }
            }
        },
    }
    summary = common.summarize_state_for_model(state, "mlx-community/Qwen3-0.6B-8bit")
    assert summary["cluster_primary_ips"] == ["192.168.31.159"]
    assert summary["instances"][0]["member_ips"] == ["192.168.31.159"]


def test_build_initial_results_single_setting():
    cfg = _cluster_cfg()
    sample_rows = [
        {
            "sample_id": 0,
            "dataset_index": 10,
            "question": "q",
            "gold_answer_text": "#### 5",
            "gold_final_answer": "5",
        }
    ]
    results = common.build_initial_results(
        cfg,
        setting_name="1device",
        namespace="equiv-1device",
        label="1-device exo",
        entry_url="http://192.168.31.159:52415",
        expected_ips=["192.168.31.159"],
        sample_rows=sample_rows,
    )
    assert results["setting"] == "1device"
    assert results["questions"][0]["result"] is None
    assert results["summary"]["accuracy"] is None


def test_select_preview_for_expected_ips():
    state = {
        "nodeNetwork": {
            "n1": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
            "n3": {"interfaces": [{"ipAddress": "192.168.31.83"}]},
        }
    }
    previews = [
        {
            "model_id": "mlx-community/Qwen3-0.6B-8bit",
            "error": None,
            "instance": {
                "MlxRingInstance": {
                    "shardAssignments": {
                        "nodeToRunner": {"n1": "r1", "n3": "r3"},
                        "runnerToShard": {
                            "r1": {"PipelineShardMetadata": {"deviceRank": 0, "startLayer": 0}},
                            "r3": {"PipelineShardMetadata": {"deviceRank": 1, "startLayer": 12}},
                        },
                    }
                }
            },
        }
    ]
    selected = common.select_preview_for_expected_ips(
        previews,
        state=state,
        model_id="mlx-community/Qwen3-0.6B-8bit",
        expected_ips=["192.168.31.52", "192.168.31.83"],
    )
    assert selected["instance"]["MlxRingInstance"]["shardAssignments"]["nodeToRunner"] == {"n1": "r1", "n3": "r3"}


def test_select_preview_for_expected_ips_prefers_stable_order_when_duplicate_matches():
    state = {
        "nodeNetwork": {
            "n1": {"interfaces": [{"ipAddress": "192.168.31.52"}]},
        }
    }
    previews = [
        {
            "model_id": "mlx-community/Qwen3-0.6B-8bit",
            "command_id": "cmd-b",
            "error": None,
            "instance": {
                "MlxRingInstance": {
                    "shardAssignments": {
                        "nodeToRunner": {"n1": "r1"},
                        "runnerToShard": {
                            "r1": {"PipelineShardMetadata": {"deviceRank": 0, "startLayer": 0}},
                        },
                    }
                }
            },
        },
        {
            "model_id": "mlx-community/Qwen3-0.6B-8bit",
            "command_id": "cmd-a",
            "error": None,
            "instance": {
                "MlxRingInstance": {
                    "shardAssignments": {
                        "nodeToRunner": {"n1": "r1"},
                        "runnerToShard": {
                            "r1": {"PipelineShardMetadata": {"deviceRank": 0, "startLayer": 0}},
                        },
                    }
                }
            },
        },
    ]
    selected = common.select_preview_for_expected_ips(
        previews,
        state=state,
        model_id="mlx-community/Qwen3-0.6B-8bit",
        expected_ips=["192.168.31.52"],
    )
    assert selected["command_id"] == "cmd-a"


def test_run_1device_wait_for_cluster_ready_success(monkeypatch):
    cfg = _cluster_cfg()
    setting_cfg = {
        "namespace": "equiv-1device",
        "entry_url": "http://192.168.31.159:52415",
        "target_nodes": [{"ip": "192.168.31.159"}],
    }
    state = {"nodeNetwork": {"n1": {"interfaces": [{"ipAddress": "192.168.31.159"}]}}}

    async def fake_fetch_state(entry_url: str, timeout_sec: int):
        assert entry_url.startswith("http://192.168.31.159")
        return state

    async def fake_fetch_models(entry_url: str, timeout_sec: int):
        return ["mlx-community/Qwen3-0.6B-8bit"]

    monkeypatch.setattr(run_1device, "fetch_state", fake_fetch_state)
    monkeypatch.setattr(run_1device, "fetch_models", fake_fetch_models)
    report = asyncio.run(run_1device.wait_for_cluster_ready(cfg, setting_cfg))
    assert report["cluster_primary_ips"] == ["192.168.31.159"]


def test_instance_runners_ready_accepts_idle():
    assert common.instance_runners_ready({"r1": "RunnerIdle"}) is True


def test_blocking_task_snapshot_ignores_download_tasks():
    state = {
        "tasks": {
            "t1": {
                "DownloadModel": {
                    "taskId": "t1",
                    "taskStatus": "Running",
                    "instanceId": "inst-1",
                }
            },
            "t2": {
                "CreateRunner": {
                    "taskId": "t2",
                    "taskStatus": "Running",
                    "instanceId": "inst-1",
                }
            },
        }
    }
    assert common.blocking_task_snapshot(state, "inst-1") == [
        {
            "task_id": "t2",
            "task_type": "CreateRunner",
            "task_status": "Running",
            "command_id": "",
        }
    ]


def test_run_1device_wait_for_cluster_ready_rejects_extra_node(monkeypatch):
    cfg = _cluster_cfg()
    cfg["experiment"]["cluster_stabilization_timeout_sec"] = 0
    setting_cfg = {
        "namespace": "equiv-1device",
        "entry_url": "http://192.168.31.159:52415",
        "target_nodes": [{"ip": "192.168.31.159"}],
    }
    state = {
        "nodeNetwork": {
            "n1": {"interfaces": [{"ipAddress": "192.168.31.159"}]},
            "n2": {"interfaces": [{"ipAddress": "192.168.31.83"}]},
        }
    }

    async def fake_fetch_state(entry_url: str, timeout_sec: int):
        return state

    async def fake_fetch_models(entry_url: str, timeout_sec: int):
        return ["mlx-community/Qwen3-0.6B-8bit"]

    monkeypatch.setattr(run_1device, "fetch_state", fake_fetch_state)
    monkeypatch.setattr(run_1device, "fetch_models", fake_fetch_models)
    try:
        asyncio.run(run_1device.wait_for_cluster_ready(cfg, setting_cfg))
    except TimeoutError as exc:
        assert "unexpected cluster ip set" in str(exc)
    else:
        raise AssertionError("expected wait_for_cluster_ready to fail")


def test_run_1device_process_label_and_runtime_paths():
    node_cfg = {"node_tag": "jlmini_2", "ip": "192.168.31.159", "exo_dir": "/tmp/exo"}
    label = run_1device.process_label("equiv-1device", node_cfg)
    paths = run_1device._runtime_paths(node_cfg, "equiv-1device")
    assert label == "exo:equiv-1device:jlmini_2"
    assert str(paths["pid_file"]).endswith("tmp/equivalence/equiv-1device/exo.pid")


def test_run_2device_sync_code_builds_rsync_commands(monkeypatch):
    setting_cfg = {
        "sync_relative_paths": ["artifacts/E5/equivalence"],
        "target_nodes": [
            {"ssh_target": "Mac1", "project_root": "/Users/jlmini_1/repo/paper/bc-ra-paper"},
            {"ssh_target": "Mac3", "project_root": "/Users/jlmini_3/repo/paper/bc-ra-paper"},
        ],
    }
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, text=None, capture_output=None, check=None):
        calls.append(list(cmd))
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    monkeypatch.setattr(run_2device.subprocess, "run", fake_run)
    run_2device.sync_code({}, setting_cfg)
    assert calls[0][0] == "rsync"
    assert "Mac1:/Users/jlmini_1/repo/paper/bc-ra-paper/artifacts/E5/equivalence/" in calls[0][-1]
    assert "Mac3:/Users/jlmini_3/repo/paper/bc-ra-paper/artifacts/E5/equivalence/" in calls[1][-1]


def test_run_2device_verify_exo_env_checks_each_remote_path(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "external_exo": {
                    "git": {"commit": "abc", "branch": "main"},
                    "files": {"flake_lock_sha256": "xyz", "uv_lock_sha256": "", "python_version_sha256": "", "pyproject_sha256": ""},
                    "python": {"python_version_file": "", "requires_python": "", "uv_required_version": ""},
                    "project": {"package_version": ""},
                    "nix": {"nixpkgs_rev": ""},
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = {"experiment": {"freeze_manifest_path": str(manifest)}}
    setting_cfg = {
        "target_nodes": [
            {"ssh_target": "Mac1", "exo_dir": "/Users/jlmini_1/repo/paper/third_party/exo", "ip": "192.168.31.52"},
            {"ssh_target": "Mac3", "exo_dir": "/Users/jlmini_3/repo/paper/third_party/exo", "ip": "192.168.31.83"},
        ]
    }
    seen: list[tuple[str, str]] = []

    def fake_collect_remote(target: str, exo_dir: str):
        seen.append((target, exo_dir))
        return {"commit": "abc", "flake_lock_sha256": "xyz"}

    monkeypatch.setattr(run_2device.verify_mod, "collect_remote", fake_collect_remote)
    run_2device.verify_exo_env(cfg, setting_cfg)
    assert seen == [
        ("Mac1", "/Users/jlmini_1/repo/paper/third_party/exo"),
        ("Mac3", "/Users/jlmini_3/repo/paper/third_party/exo"),
    ]


def test_run_2device_and_1device_namespaces_do_not_conflict():
    node_a = {"node_tag": "jlmini_2", "ip": "192.168.31.159", "exo_dir": "/tmp/exo"}
    node_b = {"node_tag": "jlmini_1", "ip": "192.168.31.52"}
    assert run_1device.process_label("equiv-1device", node_a) != run_2device.process_label("equiv-2device", node_b)


def test_run_3device_verify_exo_env_uses_local_and_remote(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "external_exo": {
                    "git": {"commit": "abc", "branch": "main"},
                    "files": {"flake_lock_sha256": "xyz", "uv_lock_sha256": "", "python_version_sha256": "", "pyproject_sha256": ""},
                    "python": {"python_version_file": "", "requires_python": "", "uv_required_version": ""},
                    "project": {"package_version": ""},
                    "nix": {"nixpkgs_rev": ""},
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = {"experiment": {"freeze_manifest_path": str(manifest)}}
    setting_cfg = {
        "target_nodes": [
            {"access": "local", "exo_dir": "/Users/jlmini_2/repo/paper/third_party/exo", "ip": "192.168.31.159"},
            {"ssh_target": "Mac1", "exo_dir": "/Users/jlmini_1/repo/paper/third_party/exo", "ip": "192.168.31.52"},
        ]
    }
    seen: list[tuple[str, str]] = []

    def fake_collect_local(exo_dir: Path):
        seen.append(("local", str(exo_dir)))
        return {"commit": "abc", "flake_lock_sha256": "xyz"}

    def fake_collect_remote(target: str, exo_dir: str):
        seen.append((target, exo_dir))
        return {"commit": "abc", "flake_lock_sha256": "xyz"}

    monkeypatch.setattr(run_3device.verify_mod, "collect_local", fake_collect_local)
    monkeypatch.setattr(run_3device.verify_mod, "collect_remote", fake_collect_remote)
    run_3device.verify_exo_env(cfg, setting_cfg)
    assert seen == [
        ("local", "/Users/jlmini_2/repo/paper/third_party/exo"),
        ("Mac1", "/Users/jlmini_1/repo/paper/third_party/exo"),
    ]
