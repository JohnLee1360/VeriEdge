#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from equivalence_common import (  # noqa: E402
    build_prompt,
    blocking_task_snapshot,
    ensure_output_root,
    find_matching_instance,
    infer_routed_instance,
    instance_runners_ready,
    load_config,
    load_or_create_sample,
    load_or_initialize_results,
    output_paths,
    persist_results,
    primary_cluster_ips,
    record_question_result,
    runner_status_snapshot,
    select_preview_for_expected_ips,
    summarize_state_for_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1-device EXO equivalence experiment")
    parser.add_argument("--config", default=str(ROOT / "config.example.yaml"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--keep-cluster", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    return parser.parse_args()


def _setting_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg["one_device"])


def _node_cfg(setting_cfg: Dict[str, Any]) -> Dict[str, Any]:
    nodes = list(setting_cfg["target_nodes"])
    if len(nodes) != 1:
        raise ValueError("one_device.target_nodes must contain exactly one node")
    return dict(nodes[0])


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _runtime_paths(node_cfg: Dict[str, Any], namespace: str) -> Dict[str, Path]:
    exo_dir = Path(str(node_cfg["exo_dir"])).expanduser().resolve()
    runtime_dir = exo_dir / "tmp" / "equivalence" / namespace
    node_tag = str(node_cfg.get("node_tag", node_cfg.get("ip", "local")))
    return {
        "runtime_dir": runtime_dir,
        "pid_file": runtime_dir / "exo.pid",
        "log_file": runtime_dir / "exo.log",
        "process_label": runtime_dir / f"process_{node_tag}.label",
    }


def process_label(namespace: str, node_cfg: Dict[str, Any]) -> str:
    return f"exo:{namespace}:{str(node_cfg.get('node_tag', node_cfg.get('ip', 'local')))}"


def _write_process_label(paths: Dict[str, Path], label: str) -> None:
    paths["runtime_dir"].mkdir(parents=True, exist_ok=True)
    paths["process_label"].write_text(label, encoding="utf-8")


def _read_process_label(paths: Dict[str, Path]) -> str:
    if not paths["process_label"].exists():
        return ""
    return paths["process_label"].read_text(encoding="utf-8").strip()


def verify_exo_env(cfg: Dict[str, Any], setting_cfg: Dict[str, Any]) -> None:
    manifest = Path(str(cfg["experiment"]["freeze_manifest_path"])).expanduser().resolve()
    verify_script = ROOT.parent / "inference_e2e" / "requester" / "verify_exo_env.py"
    exo_dir = Path(str(_node_cfg(setting_cfg)["exo_dir"])).expanduser().resolve()
    completed = subprocess.run(
        [sys.executable, str(verify_script), "--manifest", str(manifest), "--exo-dir", str(exo_dir)],
        cwd=str(ROOT.parent.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or completed.stderr.strip() or "verify_exo_env failed")


def stop_local_cluster(setting_cfg: Dict[str, Any]) -> None:
    node_cfg = _node_cfg(setting_cfg)
    namespace = str(setting_cfg["namespace"])
    paths = _runtime_paths(node_cfg, namespace)
    label = _read_process_label(paths) or process_label(namespace, node_cfg)
    if paths["pid_file"].exists():
        try:
            pid = int(paths["pid_file"].read_text(encoding="utf-8").strip())
        except ValueError:
            pid = 0
        if pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        paths["pid_file"].unlink(missing_ok=True)
    subprocess.run(["bash", "-lc", f"pkill -f {_quote(label)} || true"], check=False, cwd=str(ROOT.parent.parent))


def start_local_cluster(setting_cfg: Dict[str, Any], *, dry_run: bool) -> Optional[subprocess.Popen[bytes]]:
    node_cfg = _node_cfg(setting_cfg)
    namespace = str(setting_cfg["namespace"])
    exo_home = str(setting_cfg["exo_home"])
    paths = _runtime_paths(node_cfg, namespace)
    label = process_label(namespace, node_cfg)
    _write_process_label(paths, label)
    paths["runtime_dir"].mkdir(parents=True, exist_ok=True)
    inner = f"exec -a {_quote(label)} nix run ."
    env = os.environ.copy()
    env["EXO_LIBP2P_NAMESPACE"] = namespace
    env["EXO_HOME"] = exo_home
    env["EXO_OFFLINE"] = "true"
    env.setdefault("PYTHONUNBUFFERED", "1")
    if dry_run:
        print(
            json.dumps(
                {
                    "cwd": str(Path(str(node_cfg["exo_dir"])).expanduser().resolve()),
                    "env": {
                        "EXO_LIBP2P_NAMESPACE": namespace,
                        "EXO_HOME": exo_home,
                        "EXO_OFFLINE": "true",
                    },
                    "command": ["bash", "-lc", inner],
                    "log_file": str(paths["log_file"]),
                    "pid_file": str(paths["pid_file"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return None
    stream = paths["log_file"].open("ab", buffering=0)
    proc = subprocess.Popen(
        ["bash", "-lc", inner],
        cwd=str(Path(str(node_cfg["exo_dir"])).expanduser().resolve()),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    paths["pid_file"].write_text(str(proc.pid), encoding="utf-8")
    return proc


async def fetch_json(url: str, timeout_sec: int) -> Any:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def fetch_state(entry_url: str, timeout_sec: int) -> Dict[str, Any]:
    data = await fetch_json(entry_url.rstrip("/") + "/state", timeout_sec)
    if not isinstance(data, dict):
        raise RuntimeError("state endpoint returned non-object payload")
    return data


async def fetch_models(entry_url: str, timeout_sec: int) -> List[str]:
    data = await fetch_json(entry_url.rstrip("/") + "/v1/models", timeout_sec)
    model_ids: List[str] = []
    if isinstance(data, dict):
        for item in data.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                model_ids.append(str(item["id"]))
    return model_ids


async def fetch_previews(entry_url: str, model_id: str, timeout_sec: int) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.get(entry_url.rstrip("/") + "/instance/previews", params={"model_id": model_id})
        resp.raise_for_status()
        data = resp.json()
    previews = data.get("previews", [])
    if not isinstance(previews, list):
        raise RuntimeError("instance preview response missing previews list")
    return [dict(preview) for preview in previews if isinstance(preview, dict)]


async def create_instance(entry_url: str, instance_payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.post(entry_url.rstrip("/") + "/instance", json={"instance": instance_payload})
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("create instance response was not an object")
    return data


async def delete_instance(entry_url: str, instance_id: str, timeout_sec: int) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.delete(entry_url.rstrip("/") + f"/instance/{instance_id}")
        resp.raise_for_status()


async def run_completion(
    *,
    entry_url: str,
    model_id: str,
    messages: List[Dict[str, str]],
    decoding_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": float(decoding_cfg["temperature"]),
        "top_p": float(decoding_cfg["top_p"]),
        "max_tokens": int(decoding_cfg["max_tokens"]),
        "stream": bool(decoding_cfg["stream"]),
    }
    if "reasoning_effort" in decoding_cfg and decoding_cfg["reasoning_effort"] is not None:
        payload["reasoning_effort"] = str(decoding_cfg["reasoning_effort"])
    if "enable_thinking" in decoding_cfg and decoding_cfg["enable_thinking"] is not None:
        payload["enable_thinking"] = bool(decoding_cfg["enable_thinking"])
    timeout_sec = int(decoding_cfg["timeout_sec"])
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.post(entry_url.rstrip("/") + "/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    choices = data.get("choices", [])
    content = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            content = str(message.get("content", "") or "")
    return {
        "request_payload": payload,
        "response_body": data,
        "output_text": content,
        "usage": data.get("usage", {}),
        "finish_reason": choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None,
    }


async def collect_state_views(member_urls: Sequence[str], timeout_sec: int) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    for endpoint in member_urls:
        row: Dict[str, Any] = {"endpoint": endpoint, "ok": False, "cluster_primary_ips": [], "detail": ""}
        try:
            state = await fetch_state(endpoint, timeout_sec)
            row["ok"] = True
            row["cluster_primary_ips"] = primary_cluster_ips(state)
            row["detail"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["detail"] = f"{type(exc).__name__}: {exc}"
        views.append(row)
    return views


async def wait_for_cluster_ready(cfg: Dict[str, Any], setting_cfg: Dict[str, Any]) -> Dict[str, Any]:
    experiment_cfg = cfg["experiment"]
    timeout_sec = int(experiment_cfg["cluster_stabilization_timeout_sec"])
    poll_sec = float(experiment_cfg["cluster_stabilization_poll_sec"])
    consecutive_successes = int(experiment_cfg["cluster_stabilization_consecutive_successes"])
    decoding_timeout = int(cfg["decoding"]["timeout_sec"])
    expected_ips = sorted(str(node["ip"]) for node in setting_cfg["target_nodes"])
    member_urls = [f"http://{str(node['ip'])}:52415" for node in setting_cfg["target_nodes"]]
    entry_url = str(setting_cfg["entry_url"])
    model_id = str(cfg["model"]["model_id"])

    deadline = time.monotonic() + timeout_sec
    stable_hits = 0
    last_signature: Optional[str] = None
    last_error = "not_started"
    while True:
        try:
            state = await fetch_state(entry_url, decoding_timeout)
            observed_ips = primary_cluster_ips(state)
            if observed_ips != expected_ips:
                raise RuntimeError(f"unexpected cluster ip set: observed={observed_ips} expected={expected_ips}")
            member_views = await collect_state_views(member_urls, decoding_timeout)
            if not all(bool(view["ok"]) for view in member_views):
                raise RuntimeError(f"member state view failed: {json.dumps(member_views, ensure_ascii=False)}")
            if any(sorted(view["cluster_primary_ips"]) != expected_ips for view in member_views):
                raise RuntimeError(f"member state views disagreed: {json.dumps(member_views, ensure_ascii=False)}")
            models = await fetch_models(entry_url, decoding_timeout)
            if model_id not in models:
                raise RuntimeError(f"model not visible from /v1/models: {model_id}")
            signature = json.dumps(
                {
                    "cluster_ips": observed_ips,
                    "member_views": member_views,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature == last_signature:
                stable_hits += 1
            else:
                last_signature = signature
                stable_hits = 1
            if stable_hits >= consecutive_successes:
                return {
                    "entry_state": state,
                    "state_views": member_views,
                    "cluster_primary_ips": observed_ips,
                }
            last_error = f"state not stable yet ({stable_hits}/{consecutive_successes})"
        except Exception as exc:  # noqa: BLE001
            stable_hits = 0
            last_signature = None
            last_error = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for 1-device cluster readiness: {last_error}")
        await asyncio.sleep(poll_sec)


async def wait_for_matching_instance(cfg: Dict[str, Any], setting_cfg: Dict[str, Any]) -> Dict[str, Any]:
    experiment_cfg = cfg["experiment"]
    timeout_sec = int(experiment_cfg["cluster_stabilization_timeout_sec"])
    poll_sec = float(experiment_cfg["cluster_stabilization_poll_sec"])
    consecutive_successes = int(experiment_cfg["cluster_stabilization_consecutive_successes"])
    decoding_timeout = int(cfg["decoding"]["timeout_sec"])
    entry_url = str(setting_cfg["entry_url"])
    model_id = str(cfg["model"]["model_id"])
    expected_ips = sorted(str(node["ip"]) for node in setting_cfg["target_nodes"])

    deadline = time.monotonic() + timeout_sec
    stable_hits = 0
    last_instance_id: Optional[str] = None
    last_error = "not_started"
    while True:
        try:
            state = await fetch_state(entry_url, decoding_timeout)
            matched = find_matching_instance(state=state, model_id=model_id, expected_ips=expected_ips)
            if matched is None:
                raise RuntimeError("matching instance not visible yet")
            current_instance_id = str(matched["instance_id"])
            if current_instance_id == last_instance_id:
                stable_hits += 1
            else:
                last_instance_id = current_instance_id
                stable_hits = 1
            if stable_hits >= consecutive_successes:
                return {"state": state, **matched}
            last_error = f"instance not stable yet ({stable_hits}/{consecutive_successes})"
        except Exception as exc:  # noqa: BLE001
            stable_hits = 0
            last_instance_id = None
            last_error = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for matching 1-device instance: {last_error}")
        await asyncio.sleep(poll_sec)


async def wait_for_instance_ready(cfg: Dict[str, Any], setting_cfg: Dict[str, Any], instance_id: str) -> Dict[str, Any]:
    model_id = str(cfg["model"]["model_id"])
    timeout_sec = int(cfg["decoding"]["timeout_sec"])
    poll_sec = float(cfg["experiment"]["instance_ready_poll_sec"])
    wait_budget_sec = float(cfg["experiment"]["instance_ready_wait_sec"])
    entry_url = str(setting_cfg["entry_url"])
    deadline = time.monotonic() + wait_budget_sec
    last_snapshot: Optional[Dict[str, Any]] = None
    while True:
        state = await fetch_state(entry_url, timeout_sec)
        summary = summarize_state_for_model(state, model_id)
        matched = next((item for item in summary["instances"] if item["instance_id"] == instance_id), None)
        if matched is None:
            raise RuntimeError(f"instance disappeared while waiting for readiness: {instance_id}")
        active_tasks = blocking_task_snapshot(state, instance_id)
        runner_statuses = runner_status_snapshot(state, instance_id)
        ready = (not active_tasks) and instance_runners_ready(runner_statuses)
        last_snapshot = {
            "instance_id": instance_id,
            "state_summary": summary,
            "matched_instance": matched,
            "active_tasks": active_tasks,
            "runner_statuses": runner_statuses,
            "instance_ready": ready,
        }
        if ready:
            return last_snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for ready 1-device instance: {json.dumps(last_snapshot, ensure_ascii=False)}")
        await asyncio.sleep(poll_sec)


async def ensure_instance(cfg: Dict[str, Any], setting_cfg: Dict[str, Any]) -> Dict[str, Any]:
    timeout_sec = int(cfg["decoding"]["timeout_sec"])
    entry_url = str(setting_cfg["entry_url"])
    model_id = str(cfg["model"]["model_id"])
    expected_ips = sorted(str(node["ip"]) for node in setting_cfg["target_nodes"])

    state = await fetch_state(entry_url, timeout_sec)
    existing = find_matching_instance(state=state, model_id=model_id, expected_ips=expected_ips)
    if existing is not None:
        return {
            "instance_id": str(existing["instance_id"]),
            "instance": existing["instance"],
            "member_ips": list(existing["member_ips"]),
            "member_node_ids": list(existing["member_node_ids"]),
            "instance_source": "reused_existing",
            "preview": None,
            "created_command_id": None,
        }

    previews = await fetch_previews(entry_url, model_id, timeout_sec)
    preview = select_preview_for_expected_ips(previews, state=state, model_id=model_id, expected_ips=expected_ips)
    created = await create_instance(entry_url, dict(preview["instance"]), timeout_sec)
    matched = await wait_for_matching_instance(cfg, setting_cfg)
    return {
        "instance_id": str(matched["instance_id"]),
        "instance": matched["instance"],
        "member_ips": list(matched["member_ips"]),
        "member_node_ids": list(matched["member_node_ids"]),
        "instance_source": "created_from_preview",
        "preview": preview,
        "created_command_id": created.get("command_id"),
    }


async def run_questions(
    cfg: Dict[str, Any],
    setting_cfg: Dict[str, Any],
    results: Dict[str, Any],
    results_path: Path,
    instance_id: str,
) -> None:
    model_id = str(cfg["model"]["model_id"])
    decoding_cfg = dict(cfg["decoding"])
    entry_url = str(setting_cfg["entry_url"])
    label = str(setting_cfg["label"])
    namespace = str(setting_cfg["namespace"])
    preflight_summary = dict(results["cluster_evidence"]["before_run"])
    for question_row in results["questions"]:
        if question_row.get("result") is not None:
            continue
        await wait_for_instance_ready(cfg, setting_cfg, instance_id)
        messages = build_prompt(cfg, str(question_row["question"]))
        before_state = await fetch_state(entry_url, int(decoding_cfg["timeout_sec"]))
        completion = await run_completion(
            entry_url=entry_url,
            model_id=model_id,
            messages=messages,
            decoding_cfg=decoding_cfg,
        )
        after_state = await fetch_state(entry_url, int(decoding_cfg["timeout_sec"]))
        routing = infer_routed_instance(before_state=before_state, after_state=after_state, model_id=model_id)
        record_question_result(
            question_row=question_row,
            label=label,
            entry_url=entry_url,
            namespace=namespace,
            instance_id=instance_id,
            completion=completion,
            preflight_summary=preflight_summary,
            routing_observation=routing,
        )
        persist_results(results, results_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    setting_cfg = _setting_cfg(cfg)
    paths = output_paths(cfg, "1device")
    ensure_output_root(paths)

    verify_exo_env(cfg, setting_cfg)

    proc: Optional[subprocess.Popen[bytes]] = None
    created_instance_id: Optional[str] = None
    created_instance = False
    try:
        stop_local_cluster(setting_cfg)
        proc = start_local_cluster(setting_cfg, dry_run=args.dry_run)
        if args.dry_run:
            raise SystemExit(0)

        cluster_report = asyncio.run(wait_for_cluster_ready(cfg, setting_cfg))
        check_report = {
            "ok": True,
            "setting": "1device",
            "cluster_namespace": setting_cfg["namespace"],
            "entry_url": setting_cfg["entry_url"],
            "state_views": cluster_report["state_views"],
            "entry_state_summary": summarize_state_for_model(cluster_report["entry_state"], str(cfg["model"]["model_id"])),
        }
        paths["check_report"].write_text(json.dumps(check_report, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.check_only:
            print(json.dumps({"check_report": str(paths["check_report"])}, ensure_ascii=False, indent=2))
            return

        instance_info = asyncio.run(ensure_instance(cfg, setting_cfg))
        created_instance_id = str(instance_info["instance_id"])
        created_instance = instance_info["instance_source"] == "created_from_preview"
        ready_snapshot = asyncio.run(wait_for_instance_ready(cfg, setting_cfg, created_instance_id))

        sample_rows = load_or_create_sample(cfg, paths["sample"])
        expected_ips = sorted(str(node["ip"]) for node in setting_cfg["target_nodes"])
        results = load_or_initialize_results(
            cfg,
            setting_name="1device",
            namespace=str(setting_cfg["namespace"]),
            label=str(setting_cfg["label"]),
            entry_url=str(setting_cfg["entry_url"]),
            expected_ips=expected_ips,
            sample_rows=sample_rows,
            results_path=paths["results"],
            resume=args.resume,
        )
        results["cluster_evidence"]["before_run"] = summarize_state_for_model(cluster_report["entry_state"], str(cfg["model"]["model_id"]))
        results["cluster_evidence"]["state_views"] = list(cluster_report["state_views"])
        results["instance_evidence"].update(
            {
                "instance_id": created_instance_id,
                "instance_source": instance_info["instance_source"],
                "preview": instance_info["preview"],
                "member_ips": instance_info["member_ips"],
                "member_node_ids": instance_info["member_node_ids"],
                "ready_snapshot": ready_snapshot,
                "created_command_id": instance_info["created_command_id"],
            }
        )
        persist_results(results, paths["results"])
        asyncio.run(run_questions(cfg, setting_cfg, results, paths["results"], created_instance_id))
        final_state = asyncio.run(fetch_state(str(setting_cfg["entry_url"]), int(cfg["decoding"]["timeout_sec"])))
        results["cluster_evidence"]["after_run"] = summarize_state_for_model(final_state, str(cfg["model"]["model_id"]))
        persist_results(results, paths["results"])
        print(json.dumps({"results": str(paths["results"])}, ensure_ascii=False, indent=2))
    finally:
        if created_instance and created_instance_id and not args.keep_instance and not args.dry_run:
            try:
                asyncio.run(delete_instance(str(setting_cfg["entry_url"]), created_instance_id, int(cfg["decoding"]["timeout_sec"])))
            except Exception:
                pass
        if not args.keep_cluster and not args.dry_run:
            stop_local_cluster(setting_cfg)
        if proc is not None and proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
