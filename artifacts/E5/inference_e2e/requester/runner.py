#!/usr/bin/env python3
"""Task-level EXO feasibility benchmark runner."""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx
import yaml

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from common import (  # noqa: E402
    canonical_json_bytes,
    dedupe_preserve,
    expand_prompt_id_spec,
    parse_prompts_from_markdown,
    select_prompts_by_ids,
    percentile_or_nan,
    sha256_hex,
    utc_iso_now,
    write_json,
    write_jsonl,
)
from exo_state_utils import (  # noqa: E402
    first_shard_provider,
    host_from_url,
    instance_node_count,
    iter_model_instances,
    node_ip_map,
)

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_delta_text(event_obj: Dict[str, Any]) -> str:
    if "choices" in event_obj and isinstance(event_obj["choices"], list):
        chunks: List[str] = []
        for choice in event_obj["choices"]:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    chunks.append(content)
            text = choice.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks)
    token_obj = event_obj.get("token")
    if isinstance(token_obj, dict):
        text = token_obj.get("text")
        if isinstance(text, str):
            return text
    return ""


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connect_error"
    if isinstance(
        exc,
        (
            httpx.ReadError,
            httpx.WriteError,
            httpx.CloseError,
            httpx.ProtocolError,
        ),
    ):
        return "stream_interrupted"
    return "other"


class CallbackServer(ThreadingHTTPServer):
    def __init__(self, server_address: Tuple[str, int], callback_path: str):
        super().__init__(server_address, CallbackHandler)
        self.callback_path = callback_path or "/task-result"
        self.results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def store(self, payload: Dict[str, Any]) -> None:
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("callback payload missing task_id")
        with self._lock:
            self.results[task_id] = payload

    def pop_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.results.pop(task_id, None)


class CallbackHandler(BaseHTTPRequestHandler):
    server: CallbackServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != self.server.callback_path:
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            self.server.store(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as exc:  # noqa: BLE001
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"))


def start_callback_server(callback_url: str) -> Tuple[CallbackServer, threading.Thread]:
    parsed = urlparse(callback_url)
    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8000
    server = CallbackServer((host, port), parsed.path or "/task-result")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_callback_server(server: CallbackServer) -> None:
    server.shutdown()
    server.server_close()


async def fetch_state(entry_url: str, timeout_sec: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.get(entry_url.rstrip("/") + "/state")
        resp.raise_for_status()
        return resp.json()


def runtime_cluster_stability_config(cfg: Dict[str, Any]) -> Dict[str, int]:
    runtime_cfg = cfg.get("runtime", {})
    return {
        "timeout_sec": int(runtime_cfg.get("cluster_stabilization_timeout_sec", 180)),
        "poll_sec": max(1, int(runtime_cfg.get("cluster_stabilization_poll_sec", 2))),
        "consecutive_successes": max(1, int(runtime_cfg.get("cluster_stabilization_consecutive_successes", 3))),
    }


def build_task_manifest(
    *,
    task_id: str,
    model_id: str,
    prompts: List[Tuple[int, str]],
    requester_id: str,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "task_type": "text",
        "model_id": model_id,
        "question_count": len(prompts),
        "created_at": utc_iso_now(),
        "requester_id": requester_id,
        "prompts": [{"prompt_id": prompt_id, "content": text} for prompt_id, text in prompts],
    }


async def upload_task_package_to_ipfs(
    *,
    api_url: str,
    package_bytes: bytes,
    file_name: str,
    timeout_sec: int,
) -> str:
    files = {"file": (file_name, package_bytes, "application/octet-stream")}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.post(api_url.rstrip("/") + "/add", files=files, params={"pin": "true"})
        resp.raise_for_status()
        lines = [line for line in resp.text.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty IPFS add response")
        payload = json.loads(lines[-1])
        cid = payload.get("Hash")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"Invalid IPFS add response: {payload}")
        return cid


def aggregate_task_result(
    *,
    task_result: Dict[str, Any],
    network: str,
    instance_node_count: int,
    phase: str,
    task_index_in_cell: int,
    question_count: int,
) -> Dict[str, Any]:
    aggregate = task_result.get("aggregate_stats", {})
    ttfts = [float(row["ttft_s"]) for row in task_result.get("question_results", []) if row.get("ttft_s") is not None]
    otps = [
        float(row["otps_tok_s"])
        for row in task_result.get("question_results", [])
        if row.get("otps_tok_s") is not None and row.get("latency_s") is not None and row.get("ttft_s") is not None
    ]
    return {
        "network": network,
        "instance_node_count": instance_node_count,
        "phase": phase,
        "task_index_in_cell": task_index_in_cell,
        "task_id": task_result["task_id"],
        "instance_id": task_result.get("instance_id", ""),
        "provider_node_id": task_result.get("provider_node_id", ""),
        "question_count": question_count,
        "download_s_per_task": float(aggregate.get("download_s", float("nan"))),
        "task_latency_s_per_task": float(aggregate.get("task_latency_s", float("nan"))),
        "question_latency_s_per_q": float(aggregate.get("task_latency_s", float("nan"))) / max(question_count, 1),
        "ttft_p50_s_per_task": percentile_or_nan(ttfts, 50),
        "otps_p50_tok_s_per_task": percentile_or_nan(otps, 50),
        "question_success_count_per_task": int(aggregate.get("question_success_count", 0)),
        "question_fail_count_per_task": int(aggregate.get("question_fail_count", 0)),
    }


def aggregate_cell_summary(
    *,
    task_rows: Sequence[Dict[str, Any]],
    network: str,
    instance_node_count: int,
) -> Dict[str, Any]:
    if not task_rows:
        raise ValueError("task_rows must not be empty")
    download_values = [float(row["download_s_per_task"]) for row in task_rows]
    task_latency_values = [float(row["task_latency_s_per_task"]) for row in task_rows]
    question_latency_values = [float(row["question_latency_s_per_q"]) for row in task_rows]
    ttft_values = [float(row["ttft_p50_s_per_task"]) for row in task_rows]
    otps_values = [float(row["otps_p50_tok_s_per_task"]) for row in task_rows]
    return {
        "instance_node_count": instance_node_count,
        "network": network,
        "mean_task_latency_s_per_task": sum(task_latency_values) / len(task_latency_values),
        "mean_question_latency_s_per_q": sum(question_latency_values) / len(question_latency_values),
        "mean_download_s_per_task": sum(download_values) / len(download_values),
        "mean_ttft_p50_s": sum(ttft_values) / len(ttft_values),
        "mean_otps_p50_tok_s": sum(otps_values) / len(otps_values),
        "sum_question_success_count": sum(int(row["question_success_count_per_task"]) for row in task_rows),
        "sum_question_fail_count": sum(int(row["question_fail_count_per_task"]) for row in task_rows),
        "completed_task_count": len(task_rows),
    }


async def run_health_checks(entry_url: str, provider_urls: Sequence[str], timeout_sec: int) -> List[Dict[str, Any]]:
    endpoints = dedupe_preserve([entry_url, *provider_urls])
    timeout = httpx.Timeout(min(timeout_sec, 10))
    rows: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for endpoint in endpoints:
            row = {
                "endpoint": endpoint,
                "ok": False,
                "http_status": None,
                "detail": "",
            }
            try:
                resp = await client.get(endpoint.rstrip("/") + "/v1/models")
                row["http_status"] = int(resp.status_code)
                row["ok"] = resp.status_code == 200
                row["detail"] = "ok" if row["ok"] else f"/v1/models={resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                row["detail"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def _tcp_reachable(url: str, timeout_sec: int) -> Tuple[bool, str]:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        return False, "invalid_url"
    try:
        with socket.create_connection((host, port), timeout=min(timeout_sec, 5)):
            return True, "ok"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def run_aux_service_checks(
    *,
    requester_callback_url: str,
    ipfs_api_url: str,
    ipfs_gateway_url: str,
    launcher_urls: Sequence[str],
    timeout_sec: int,
    include_requester_callback: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if include_requester_callback:
        callback_ok, callback_detail = _tcp_reachable(requester_callback_url, timeout_sec)
        rows.append(
            {
                "service": "requester_callback",
                "endpoint": requester_callback_url,
                "ok": callback_ok,
                "detail": callback_detail,
            }
        )
    gateway_ok, gateway_detail = _tcp_reachable(ipfs_gateway_url, timeout_sec)
    rows.append(
        {
            "service": "ipfs_gateway",
            "endpoint": ipfs_gateway_url,
            "ok": gateway_ok,
            "detail": gateway_detail,
        }
    )
    timeout = httpx.Timeout(min(timeout_sec, 10))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        try:
            resp = await client.post(ipfs_api_url.rstrip("/") + "/version")
            rows.append(
                {
                    "service": "ipfs_api",
                    "endpoint": ipfs_api_url,
                    "ok": resp.status_code == 200,
                    "detail": f"/version={resp.status_code}",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "service": "ipfs_api",
                    "endpoint": ipfs_api_url,
                    "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
        for launcher_url in dedupe_preserve(list(launcher_urls)):
            row = {
                "service": "provider_launcher",
                "endpoint": launcher_url,
                "ok": False,
                "detail": "",
            }
            try:
                resp = await client.get(launcher_url.rstrip("/") + "/health")
                row["ok"] = resp.status_code == 200
                row["detail"] = f"/health={resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                row["detail"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def failing_check_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if not bool(row.get("ok"))]


def format_failed_checks(rows: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        endpoint = str(row.get("endpoint", "")).strip()
        service = str(row.get("service", row.get("endpoint", "check"))).strip() or "check"
        detail = str(row.get("detail", "")).strip()
        if endpoint and detail:
            parts.append(f"{service} {endpoint}: {detail}")
        elif endpoint:
            parts.append(f"{service} {endpoint}")
        elif detail:
            parts.append(f"{service}: {detail}")
        else:
            parts.append(service)
    return "; ".join(parts)


async def collect_preflight_checks(
    *,
    entry_url: str,
    provider_urls: Sequence[str],
    requester_callback_url: str,
    ipfs_api_url: str,
    ipfs_gateway_url: str,
    launcher_urls: Sequence[str],
    timeout_sec: int,
    include_requester_callback: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    health_checks = await run_health_checks(entry_url, provider_urls, timeout_sec)
    aux_checks = await run_aux_service_checks(
        requester_callback_url=requester_callback_url,
        ipfs_api_url=ipfs_api_url,
        ipfs_gateway_url=ipfs_gateway_url,
        launcher_urls=launcher_urls,
        timeout_sec=timeout_sec,
        include_requester_callback=include_requester_callback,
    )
    return {
        "health_checks": health_checks,
        "aux_checks": aux_checks,
    }


async def collect_cluster_state_views(
    *,
    provider_urls: Sequence[str],
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    timeout = httpx.Timeout(min(timeout_sec, 10))
    views: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for endpoint in dedupe_preserve(list(provider_urls)):
            row: Dict[str, Any] = {
                "endpoint": endpoint,
                "ok": False,
                "detail": "",
                "node_count": 0,
                "node_ids": [],
                "state_ips": [],
            }
            try:
                resp = await client.get(endpoint.rstrip("/") + "/state")
                resp.raise_for_status()
                state = resp.json()
                node_network = state.get("nodeNetwork")
                if not isinstance(node_network, dict):
                    raise ValueError("state missing nodeNetwork")
                node_ids = sorted(str(node_id) for node_id in node_network.keys())
                state_ips = sorted(
                    {
                        str(item.get("ipAddress"))
                        for info in node_network.values()
                        if isinstance(info, dict)
                        for item in info.get("interfaces", [])
                        if isinstance(item, dict) and item.get("ipAddress")
                    }
                )
                row.update(
                    {
                        "ok": True,
                        "detail": "ok",
                        "node_count": len(node_ids),
                        "node_ids": node_ids,
                        "state_ips": state_ips,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["detail"] = f"{type(exc).__name__}: {exc}"
            views.append(row)
    return views


def ensure_cluster_not_isolated(
    *,
    state_views: Sequence[Dict[str, Any]],
    expected_provider_count: int,
) -> None:
    healthy_views = [dict(view) for view in state_views if bool(view.get("ok"))]
    if len(healthy_views) < max(2, expected_provider_count):
        return
    if not all(int(view.get("node_count", 0)) == 1 for view in healthy_views):
        return

    singleton_ids = {
        str(view.get("node_ids", [""])[0])
        for view in healthy_views
        if isinstance(view.get("node_ids"), list) and view.get("node_ids")
    }
    if len(singleton_ids) < 2:
        return

    details = "; ".join(
        f"{view['endpoint']} sees node_ids={view.get('node_ids', [])} ips={view.get('state_ips', [])}"
        for view in healthy_views
    )
    raise RuntimeError(
        "provider endpoints are healthy but EXO cluster is isolated; "
        "each provider /state only sees itself instead of a shared cluster. "
        + details
    )


def ensure_preflight_checks_ok(
    *,
    health_checks: Sequence[Dict[str, Any]],
    aux_checks: Sequence[Dict[str, Any]],
) -> None:
    failed_health = failing_check_rows(health_checks)
    failed_aux = failing_check_rows(aux_checks)
    if not failed_health and not failed_aux:
        return

    details: List[str] = []
    if failed_health:
        details.append("health checks failed: " + format_failed_checks(failed_health))
    if failed_aux:
        details.append("aux checks failed: " + format_failed_checks(failed_aux))
    raise RuntimeError("; ".join(details))


def validate_external_instance_state(
    *,
    state: Dict[str, Any],
    model_id: str,
    expected_instance_node_count: int,
    expected_instance_ids: Optional[Sequence[str]],
    provider_urls: Sequence[str],
) -> Dict[str, Any]:
    model_instances = iter_model_instances(state, model_id)
    matching_instances = [
        (instance_id, instance)
        for instance_id, instance in model_instances
        if instance_node_count(instance) == expected_instance_node_count
    ]
    if not matching_instances:
        raise RuntimeError(
            f"No active instances for {model_id} matched instance_node_count={expected_instance_node_count}; "
            f"available_counts={[instance_node_count(instance) for _, instance in model_instances]}"
        )
    expected_ids_normalized = [str(item).lower() for item in (expected_instance_ids or [])]
    instance_by_id = {instance_id.lower(): (instance_id, instance) for instance_id, instance in matching_instances}
    if expected_ids_normalized:
        missing_ids = [item for item in expected_ids_normalized if item not in instance_by_id]
        if missing_ids:
            raise RuntimeError(
                f"Expected instance ids {expected_ids_normalized} for {model_id} with "
                f"instance_node_count={expected_instance_node_count}, missing {missing_ids}"
            )

    ip_map = node_ip_map(state)
    state_ips = sorted({ip for ips in ip_map.values() for ip in ips})
    missing_ips = [host_from_url(url) for url in provider_urls if host_from_url(url) not in state_ips]
    if missing_ips:
        raise RuntimeError(f"Providers missing in /state: {missing_ips}")

    ordered_instances: List[Tuple[str, Dict[str, Any]]] = []
    if expected_ids_normalized:
        ordered_instances = [instance_by_id[item] for item in expected_ids_normalized]
    else:
        ordered_instances = sorted(matching_instances, key=lambda item: item[0])

    selected_instance_id, selected_instance = ordered_instances[0]
    first_node_id, first_ip = first_shard_provider(selected_instance, state)
    if first_node_id is None or first_ip is None:
        raise RuntimeError(f"Unable to determine first-shard provider for instance {selected_instance_id}")

    return {
        "selected_instance_id": selected_instance_id,
        "selected_instance": selected_instance,
        "ordered_instances": [
            {
                "instance_id": instance_id,
                "instance_node_count": instance_node_count(instance),
                "node_count": instance_node_count(instance),
                "first_shard": {
                    "node_id": first_shard_provider(instance, state)[0],
                    "ip": first_shard_provider(instance, state)[1],
                },
            }
            for instance_id, instance in ordered_instances
        ],
        "first_shard_node_id": first_node_id,
        "first_shard_ip": first_ip,
        "model_instances_summary": [
            {
                "instance_id": instance_id,
                "instance_node_count": instance_node_count(instance),
                "node_count": instance_node_count(instance),
            }
            for instance_id, instance in sorted(model_instances, key=lambda item: item[0])
        ],
        "state_ips": state_ips,
    }


async def wait_for_validated_external_instance_state(
    *,
    entry_url: str,
    timeout_sec: int,
    model_id: str,
    expected_instance_node_count: int,
    expected_instance_ids: Optional[Sequence[str]],
    provider_urls: Sequence[str],
    poll_sec: int = 2,
    consecutive_successes: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    deadline = time.monotonic() + max(1, timeout_sec)
    stable_hits = 0
    last_signature: Optional[Tuple[str, str, Tuple[str, ...]]] = None
    last_error: Optional[str] = None

    while True:
        try:
            state = await fetch_state(entry_url, timeout_sec=min(timeout_sec, 10))
            validated = validate_external_instance_state(
                state=state,
                model_id=model_id,
                expected_instance_node_count=expected_instance_node_count,
                expected_instance_ids=expected_instance_ids,
                provider_urls=provider_urls,
            )
            signature = (
                str(validated["selected_instance_id"]),
                str(validated["first_shard_ip"]),
                tuple(str(item["instance_id"]) for item in validated["ordered_instances"]),
            )
            if signature == last_signature:
                stable_hits += 1
            else:
                last_signature = signature
                stable_hits = 1
            if stable_hits >= consecutive_successes:
                return validated, state
            last_error = (
                f"state changed before reaching stability threshold "
                f"({stable_hits}/{consecutive_successes})"
            )
        except Exception as exc:  # noqa: BLE001
            stable_hits = 0
            last_signature = None
            last_error = f"{type(exc).__name__}: {exc}"

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for stable EXO cluster state; "
                f"last_error={last_error or 'unknown'}"
            )
        await asyncio.sleep(max(1, poll_sec))


def build_dispatch_payload(
    *,
    task_id: str,
    cid: str,
    encrypted_task_key: str,
    entry_url: str,
    callback_url: str,
    gateway_url: str,
    model_id: str,
    instance_id: str,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "cid": cid,
        "encrypted_task_key": encrypted_task_key,
        "entry_url": entry_url,
        "callback_url": callback_url,
        "gateway_url": gateway_url,
        "model_id": model_id,
        "instance_id": instance_id,
    }


def load_task_schedule(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "cells" not in payload:
        raise ValueError(f"Invalid task schedule: {path}")
    return payload


def _resolve_task_prompts(
    prompts: Sequence[Tuple[int, str]],
    task_spec: Dict[str, Any],
) -> List[Tuple[int, str]]:
    prompt_ids = expand_prompt_id_spec(task_spec)
    return select_prompts_by_ids(prompts, prompt_ids=prompt_ids)


def iter_scheduled_cells(
    schedule: Dict[str, Any],
    *,
    network_profiles: Sequence[str],
    instance_node_counts: Sequence[int],
) -> List[Dict[str, Any]]:
    allowed_networks = {str(item) for item in network_profiles}
    allowed_counts = {int(item) for item in instance_node_counts}
    selected: List[Dict[str, Any]] = []
    for cell in schedule.get("cells", []):
        network = str(cell.get("network", ""))
        instance_node_count_value = int(cell.get("instance_node_count", 0))
        if network in allowed_networks and instance_node_count_value in allowed_counts:
            selected.append(cell)
    return selected


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_task_status(
    task_dir: Path,
    *,
    status: str,
    network: str,
    instance_node_count: int,
    phase: str,
    task_index_in_cell: int,
    task_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "status": status,
        "network": network,
        "instance_node_count": instance_node_count,
        "phase": phase,
        "task_index_in_cell": task_index_in_cell,
        "task_id": task_id,
    }
    if extra:
        payload.update(extra)
    write_json(task_dir / "task_status.json", payload)


def maybe_resume_completed_task(task_dir: Path) -> Optional[Dict[str, Any]]:
    status_payload = read_json_if_exists(task_dir / "task_status.json")
    summary_payload = read_json_if_exists(task_dir / "task_summary.json")
    if (
        status_payload
        and status_payload.get("status") == "completed"
        and summary_payload is not None
    ):
        return summary_payload
    return None


def expected_instance_ids_for_node_count(cfg: Dict[str, Any], instance_node_count: int) -> List[str]:
    external = cfg.get("external_exo", {})
    raw = external.get("expected_instance_ids_by_node_count", {})
    if not isinstance(raw, dict):
        return []
    values = raw.get(str(instance_node_count), raw.get(instance_node_count, []))
    if not isinstance(values, list):
        return []
    return [str(item).lower() for item in values]


def configured_instance_node_counts(cfg: Dict[str, Any], networks: Sequence[str]) -> List[int]:
    matrix_cfg = cfg.get("matrix", {})
    by_network = matrix_cfg.get("instance_node_counts_by_network", {})
    if isinstance(by_network, dict) and networks:
        counts: List[int] = []
        for network in networks:
            values = by_network.get(str(network))
            if not isinstance(values, list):
                continue
            counts.extend(int(item) for item in values)
        if counts:
            return sorted(set(counts))
    return [int(x) for x in matrix_cfg.get("instance_node_counts", [])]


async def dispatch_task_to_first_shard(
    *,
    launcher_url: str,
    payload: Dict[str, Any],
    timeout_sec: int,
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.post(launcher_url.rstrip("/") + "/launch-task", json=payload)
        resp.raise_for_status()
        return resp.json()


def wait_for_task_callback(
    callback_server: CallbackServer,
    task_id: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        result = callback_server.pop_result(task_id)
        if result is not None:
            return result
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for callback for task_id={task_id}")


def execute_task(
    *,
    prompts: Sequence[Tuple[int, str]],
    entry_url: str,
    requester_callback_url: str,
    ipfs_api_url: str,
    ipfs_gateway_url: str,
    providers_cfg: Sequence[Dict[str, Any]],
    selected_instance_id: str,
    first_node_id: str,
    first_ip: str,
    callback_server: CallbackServer,
    task_dir: Path,
    network: str,
    instance_node_count: int,
    phase: str,
    task_index_in_cell: int,
    timeout_sec: int,
    task_id: str,
    model_id: str,
    requester_id: str,
) -> Dict[str, Any]:
    from crypto_utils import (
        encrypt_bytes_aes_gcm,
        encrypt_task_key_for_provider,
        generate_task_key,
    )

    resumed = maybe_resume_completed_task(task_dir)
    if resumed is not None:
        return resumed

    provider_cfg = None
    for item in providers_cfg:
        if str(item["ip"]) == first_ip:
            provider_cfg = item
            break
    if provider_cfg is None:
        raise RuntimeError(f"No provider config found for first-shard IP {first_ip}")

    task_dir.mkdir(parents=True, exist_ok=True)
    write_task_status(
        task_dir,
        status="running",
        network=network,
        instance_node_count=instance_node_count,
        phase=phase,
        task_index_in_cell=task_index_in_cell,
        task_id=task_id,
        extra={"started_at": utc_iso_now()},
    )

    manifest = build_task_manifest(
        task_id=task_id,
        model_id=model_id,
        prompts=list(prompts),
        requester_id=requester_id,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    task_key = generate_task_key()
    encrypted_package = encrypt_bytes_aes_gcm(manifest_bytes, task_key)
    package_payload = {
        "task_id": task_id,
        "encryption": {
            "scheme": "AES-256-GCM",
            "nonce_b64": encrypted_package["nonce_b64"],
        },
        "ciphertext_b64": encrypted_package["ciphertext_b64"],
    }
    package_bytes = canonical_json_bytes(package_payload)
    cid = asyncio.run(
        upload_task_package_to_ipfs(
            api_url=ipfs_api_url,
            package_bytes=package_bytes,
            file_name=f"{task_id}.json",
            timeout_sec=timeout_sec,
        )
    )
    encrypted_task_key = encrypt_task_key_for_provider(task_key, Path(str(provider_cfg["public_key_path"])))
    dispatch_payload = build_dispatch_payload(
        task_id=task_id,
        cid=cid,
        encrypted_task_key=encrypted_task_key,
        entry_url=entry_url,
        callback_url=requester_callback_url,
        gateway_url=ipfs_gateway_url,
        model_id=model_id,
        instance_id=selected_instance_id,
    )

    write_json(task_dir / "task_manifest.json", manifest)
    write_json(
        task_dir / "dispatch_record.json",
        {
            "network": network,
            "instance_node_count": instance_node_count,
            "phase": phase,
            "task_index_in_cell": task_index_in_cell,
            "instance_id": selected_instance_id,
            "first_shard_node_id": first_node_id,
            "first_shard_ip": first_ip,
            "cid": cid,
            "encrypted_package_sha256": sha256_hex(package_bytes),
            "dispatch_payload": dispatch_payload,
        },
    )

    dispatch_resp = asyncio.run(
        dispatch_task_to_first_shard(
            launcher_url=str(provider_cfg["launcher_url"]),
            payload=dispatch_payload,
            timeout_sec=timeout_sec,
        )
    )
    write_json(task_dir / "launcher_ack.json", dispatch_resp)
    task_result = wait_for_task_callback(callback_server, task_id, timeout_sec)
    write_json(task_dir / "task_result.json", task_result)
    write_jsonl(task_dir / "question_results.jsonl", [row for row in task_result.get("question_results", [])])
    task_summary = aggregate_task_result(
        task_result=task_result,
        network=network,
        instance_node_count=instance_node_count,
        phase=phase,
        task_index_in_cell=task_index_in_cell,
        question_count=len(prompts),
    )
    write_json(task_dir / "task_summary.json", task_summary)
    write_task_status(
        task_dir,
        status="completed",
        network=network,
        instance_node_count=instance_node_count,
        phase=phase,
        task_index_in_cell=task_index_in_cell,
        task_id=task_id,
        extra={
            "finished_at": utc_iso_now(),
            "instance_id": selected_instance_id,
            "provider_node_id": task_summary.get("provider_node_id", ""),
        },
    )
    return task_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-level EXO feasibility benchmark runner")
    parser.add_argument("--config", default="artifacts/E5/inference_e2e/requester/config.example.yaml")
    parser.add_argument("--network-profile", default="", help="Only run one network profile from config (LAN or WAN)")
    parser.add_argument("--instance-node-count", type=int, default=0, help="Only run one node-count cell from config")
    parser.add_argument("--smoke", action="store_true", help="Only run the fixed smoke task for the selected cell(s)")
    parser.add_argument("--check-only", action="store_true", help="Only run black-box preflight and write snapshots")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--output-dir", default="", help="Use an explicit output directory instead of a timestamped run dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    entry_url = str(cfg["endpoints"]["entry_url"])
    requester_callback_url = str(cfg["endpoints"]["requester_callback_url"])
    providers_cfg = list(cfg.get("providers", []))
    provider_urls = [f"http://{item['ip']}:{int(item.get('exo_port', 52415))}" for item in providers_cfg]

    network_profiles = list(cfg["matrix"]["network_profiles"])
    if args.network_profile:
        if args.network_profile not in network_profiles:
            raise ValueError(f"Unknown network profile: {args.network_profile}")
        network_profiles = [args.network_profile]

    instance_node_counts = configured_instance_node_counts(cfg, network_profiles)
    if args.instance_node_count:
        if args.instance_node_count not in instance_node_counts:
            raise ValueError(f"Unknown instance_node_count: {args.instance_node_count}")
        instance_node_counts = [args.instance_node_count]

    task_cfg = cfg["task"]
    formal_question_count = int(task_cfg.get("question_count", 20))
    smoke_question_count = int(task_cfg.get("smoke_question_count", 5))
    prompt_file = Path(str(task_cfg["prompt_file"]))
    prompts = parse_prompts_from_markdown(prompt_file)
    schedule_path = Path(str(task_cfg.get("schedule_path", "artifacts/E5/inference_e2e/requester/task_schedule.json")))
    schedule = load_task_schedule(schedule_path)

    runtime_cfg = cfg.get("runtime", {})
    timeout_sec = int(runtime_cfg.get("timeout_sec", 180))
    cluster_stability = runtime_cluster_stability_config(cfg)
    output_root = Path(str(runtime_cfg.get("output_root", "artifacts/E5/inference_e2e/requester/output")))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        run_id = output_dir.name
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ipfs_cfg = cfg["ipfs"]
    ipfs_api_url = str(ipfs_cfg["api_url"])
    ipfs_gateway_url = str(ipfs_cfg["gateway_url"])

    callback_server, _ = start_callback_server(requester_callback_url)
    summary_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    selected_cells = iter_scheduled_cells(
        schedule,
        network_profiles=network_profiles,
        instance_node_counts=instance_node_counts,
    )
    if not selected_cells:
        raise ValueError("No scheduled cells matched the requested network_profiles/instance_node_counts")
    single_cell_mode = len(selected_cells) == 1

    print("=" * 72)
    print("BC-RA Task-level EXO Feasibility Benchmark")
    print("=" * 72)
    print(f"run_id                : {run_id}")
    print(f"entry_url             : {entry_url}")
    print(f"requester_callback_url: {requester_callback_url}")
    print(f"network_profiles      : {network_profiles}")
    print(f"instance_node_counts  : {instance_node_counts}")
    print(f"formal_question_count : {formal_question_count}")
    print(f"smoke_question_count  : {smoke_question_count}")
    print(f"schedule_path         : {schedule_path}")
    print(f"output_dir            : {output_dir}")

    try:
        preflight = asyncio.run(
            collect_preflight_checks(
                entry_url=entry_url,
                provider_urls=provider_urls,
                requester_callback_url=requester_callback_url,
                ipfs_api_url=ipfs_api_url,
                ipfs_gateway_url=ipfs_gateway_url,
                launcher_urls=[str(item["launcher_url"]) for item in providers_cfg],
                timeout_sec=timeout_sec,
            )
        )
        health_checks = preflight["health_checks"]
        pd.DataFrame(health_checks).to_csv(output_dir / "health_checks.csv", index=False)
        aux_checks = preflight["aux_checks"]
        pd.DataFrame(aux_checks).to_csv(output_dir / "aux_service_checks.csv", index=False)
        ensure_preflight_checks_ok(health_checks=health_checks, aux_checks=aux_checks)
        state_views = asyncio.run(
            collect_cluster_state_views(
                provider_urls=provider_urls,
                timeout_sec=timeout_sec,
            )
        )
        write_json(output_dir / "cluster_state_views.json", {"state_views": state_views})
        ensure_cluster_not_isolated(
            state_views=state_views,
            expected_provider_count=len(provider_urls),
        )

        for cell in selected_cells:
            network = str(cell["network"])
            instance_node_count_value = int(cell["instance_node_count"])
            cell_id = str(cell.get("cell_id", f"{network.lower()}_n{instance_node_count_value}"))
            expected_instance_ids = expected_instance_ids_for_node_count(cfg, instance_node_count_value)
            cell_dir = output_dir if (single_cell_mode and args.output_dir) else (output_dir / cell_id)
            tasks_root = cell_dir / "tasks"
            tasks_root.mkdir(parents=True, exist_ok=True)
            main_task_rows: List[Dict[str, Any]] = []
            smoke_spec = dict(schedule.get("smoke", {}))
            if not smoke_spec:
                raise ValueError("task schedule missing smoke section")
            smoke_prompts = _resolve_task_prompts(prompts, smoke_spec)
            if len(smoke_prompts) != smoke_question_count:
                raise ValueError(
                    f"Smoke prompt count mismatch: expected {smoke_question_count}, got {len(smoke_prompts)}"
                )
            scheduled_tasks = list(cell.get("tasks", []))
            if not args.smoke and len(scheduled_tasks) != int(task_cfg.get("tasks_per_cell", 5)):
                raise ValueError(f"Expected 5 main tasks in schedule for {cell_id}, got {len(scheduled_tasks)}")

            write_json(
                cell_dir / "cell_manifest.json",
                {
                    "cell_id": cell_id,
                    "network": network,
                    "instance_node_count": instance_node_count_value,
                    "smoke_prompt_ids": [prompt_id for prompt_id, _ in smoke_prompts],
                    "main_tasks": [
                        {
                            "task_index_in_cell": int(task.get("task_index_in_cell", idx + 1)),
                            "prompt_ids": expand_prompt_id_spec(task),
                        }
                        for idx, task in enumerate(scheduled_tasks)
                    ],
                },
            )
            write_json(
                cell_dir / "cell_status.json",
                {
                    "status": "preflight_started",
                    "network": network,
                    "instance_node_count": instance_node_count_value,
                    "started_at": utc_iso_now(),
                },
            )
            try:
                validated, state = asyncio.run(
                    wait_for_validated_external_instance_state(
                        entry_url=entry_url,
                        timeout_sec=int(cluster_stability["timeout_sec"]),
                        model_id=str(cfg["model"]["model_id"]),
                        expected_instance_node_count=instance_node_count_value,
                        expected_instance_ids=expected_instance_ids,
                        provider_urls=provider_urls,
                        poll_sec=int(cluster_stability["poll_sec"]),
                        consecutive_successes=int(cluster_stability["consecutive_successes"]),
                    )
                )
                selected_instance_id = str(validated["selected_instance_id"])
                first_node_id = str(validated["first_shard_node_id"])
                first_ip = str(validated["first_shard_ip"])
                cell_dir.mkdir(parents=True, exist_ok=True)
                write_json(
                    cell_dir / "state_snapshot.json",
                    {
                        "network": network,
                        "instance_node_count": instance_node_count_value,
                        "selected_instance_id": selected_instance_id,
                        "expected_instance_ids": expected_instance_ids,
                        "ordered_instances": validated["ordered_instances"],
                        "first_shard_node_id": first_node_id,
                        "first_shard_ip": first_ip,
                        "model_instances_summary": validated["model_instances_summary"],
                        "raw_state": state,
                    },
                )
                if args.check_only:
                    write_json(
                        cell_dir / "cell_status.json",
                        {
                            "status": "preflight_passed",
                            "network": network,
                            "instance_node_count": instance_node_count_value,
                            "finished_at": utc_iso_now(),
                            "selected_instance_id": selected_instance_id,
                            "first_shard_ip": first_ip,
                        },
                    )
                    continue

                print(f"\n[CELL] network={network} instance_node_count={instance_node_count_value} first_shard={first_ip}")
                smoke_instance_info = validated["ordered_instances"][0]
                smoke_summary = execute_task(
                    prompts=smoke_prompts,
                    entry_url=entry_url,
                    requester_callback_url=requester_callback_url,
                    ipfs_api_url=ipfs_api_url,
                    ipfs_gateway_url=ipfs_gateway_url,
                    providers_cfg=providers_cfg,
                    selected_instance_id=str(smoke_instance_info["instance_id"]),
                    first_node_id=str(smoke_instance_info["first_shard"]["node_id"]),
                    first_ip=str(smoke_instance_info["first_shard"]["ip"]),
                    callback_server=callback_server,
                    task_dir=tasks_root / "smoke_01",
                    network=network,
                    instance_node_count=instance_node_count_value,
                    phase="smoke",
                    task_index_in_cell=0,
                    timeout_sec=timeout_sec,
                    task_id=f"{run_id}-{cell_id}-smoke",
                    model_id=str(cfg["model"]["model_id"]),
                    requester_id=str(cfg["requester"]["requester_id"]),
                )
                smoke_summaries = [smoke_summary]
                write_json(cell_dir / "smoke_summaries.json", smoke_summaries)
                write_json(cell_dir / "smoke_summary.json", smoke_summary)
                write_json(
                    cell_dir / "cell_status.json",
                    {
                        "status": "smoke_completed",
                        "network": network,
                        "instance_node_count": instance_node_count_value,
                        "finished_at": utc_iso_now(),
                        "smoke_task_ids": [item["task_id"] for item in smoke_summaries],
                    },
                )

                if args.smoke:
                    summary_rows.extend(smoke_summaries)
                    task_rows.extend(smoke_summaries)
                    continue

                validated, state = asyncio.run(
                    wait_for_validated_external_instance_state(
                        entry_url=entry_url,
                        timeout_sec=int(cluster_stability["timeout_sec"]),
                        model_id=str(cfg["model"]["model_id"]),
                        expected_instance_node_count=instance_node_count_value,
                        expected_instance_ids=expected_instance_ids,
                        provider_urls=provider_urls,
                        poll_sec=int(cluster_stability["poll_sec"]),
                        consecutive_successes=int(cluster_stability["consecutive_successes"]),
                    )
                )
                write_json(
                    cell_dir / "state_snapshot_after_smoke.json",
                    {
                        "network": network,
                        "instance_node_count": instance_node_count_value,
                        "selected_instance_id": str(validated["selected_instance_id"]),
                        "expected_instance_ids": expected_instance_ids,
                        "ordered_instances": validated["ordered_instances"],
                        "first_shard_node_id": str(validated["first_shard_node_id"]),
                        "first_shard_ip": str(validated["first_shard_ip"]),
                        "model_instances_summary": validated["model_instances_summary"],
                        "raw_state": state,
                    },
                )

                for idx, task in enumerate(scheduled_tasks, start=1):
                    target_instance_info = validated["ordered_instances"][0]
                    selected_instance_id = str(target_instance_info["instance_id"])
                    first_node_id = str(target_instance_info["first_shard"]["node_id"])
                    first_ip = str(target_instance_info["first_shard"]["ip"])
                    selected_prompts = _resolve_task_prompts(prompts, task)
                    if len(selected_prompts) != formal_question_count:
                        raise ValueError(
                            f"Formal task prompt count mismatch for {cell_id} task {idx}: "
                            f"expected {formal_question_count}, got {len(selected_prompts)}"
                        )
                    task_id = str(task.get("task_id", f"{run_id}-{cell_id}-task{idx:02d}"))
                    task_summary = execute_task(
                        prompts=selected_prompts,
                        entry_url=entry_url,
                        requester_callback_url=requester_callback_url,
                        ipfs_api_url=ipfs_api_url,
                        ipfs_gateway_url=ipfs_gateway_url,
                        providers_cfg=providers_cfg,
                        selected_instance_id=selected_instance_id,
                        first_node_id=first_node_id,
                        first_ip=first_ip,
                        callback_server=callback_server,
                        task_dir=tasks_root / f"main_{idx:02d}",
                        network=network,
                        instance_node_count=instance_node_count_value,
                        phase="main",
                        task_index_in_cell=idx,
                        timeout_sec=timeout_sec,
                        task_id=task_id,
                        model_id=str(cfg["model"]["model_id"]),
                        requester_id=str(cfg["requester"]["requester_id"]),
                    )
                    main_task_rows.append(task_summary)
                    task_rows.append(task_summary)

                task_df = pd.DataFrame(main_task_rows)
                task_df.to_csv(cell_dir / "summary_by_task.csv", index=False)
                cell_summary = aggregate_cell_summary(
                    task_rows=main_task_rows,
                    network=network,
                    instance_node_count=instance_node_count_value,
                )
                cell_summary["cell_id"] = cell_id
                pd.DataFrame([cell_summary]).to_csv(cell_dir / "summary_by_cell.csv", index=False)
                summary_rows.append(cell_summary)
                write_json(
                    cell_dir / "cell_status.json",
                    {
                        "status": "completed",
                        "network": network,
                        "instance_node_count": instance_node_count_value,
                        "finished_at": utc_iso_now(),
                        "completed_task_count": len(main_task_rows),
                        "smoke_task_ids": [item["task_id"] for item in smoke_summaries],
                    },
                )
                print(
                    "  "
                    f"mean_task_latency_s_per_task={cell_summary['mean_task_latency_s_per_task']:.3f} "
                    f"mean_question_latency_s_per_q={cell_summary['mean_question_latency_s_per_q']:.3f} "
                    f"completed_task_count={cell_summary['completed_task_count']}"
                )
            except Exception as exc:  # noqa: BLE001
                write_json(
                    cell_dir / "cell_status.json",
                    {
                        "status": "failed",
                        "network": network,
                        "instance_node_count": instance_node_count_value,
                        "failed_at": utc_iso_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise

        summary_path = output_dir / "summary_by_cell.csv"
        tasks_path = output_dir / "summary_by_task.csv"
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["network", "instance_node_count"])
            summary_df.to_csv(summary_path, index=False)
        else:
            pd.DataFrame().to_csv(summary_path, index=False)
        task_df = pd.DataFrame(task_rows)
        if not task_df.empty:
            task_df.to_csv(tasks_path, index=False)
        else:
            pd.DataFrame().to_csv(tasks_path, index=False)

        comparison_script = Path(__file__).resolve().with_name("make_comparison_table.py")
        subprocess_cmd = [sys.executable, str(comparison_script), "--input", str(summary_path)]
        if subprocess_cmd and comparison_script.exists() and not args.check_only and not summary_df.empty and not args.smoke:
            import subprocess

            subprocess.run(subprocess_cmd, check=False)

        if not args.no_plot and not args.check_only and not summary_df.empty and not args.smoke:
            plot_script = Path(__file__).resolve().with_name("plot.py")
            if plot_script.exists():
                import subprocess

                subprocess.run(
                    [sys.executable, str(plot_script), "--input", str(summary_path), "--output-dir", str(output_dir)],
                    check=False,
                )

        save_cfg = dict(cfg)
        save_cfg["effective"] = {
            "network_profiles": network_profiles,
            "instance_node_counts": instance_node_counts,
            "formal_question_count": formal_question_count,
            "smoke_question_count": smoke_question_count,
            "timeout_sec": timeout_sec,
            "schedule_path": str(schedule_path),
        }
        with (output_dir / "run_config_snapshot.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(save_cfg, f, sort_keys=False, allow_unicode=False)

        print("\nDone.")
        print(f"- summary_by_cell.csv : {summary_path}")
        print(f"- summary_by_task.csv : {tasks_path}")
    finally:
        stop_callback_server(callback_server)


if __name__ == "__main__":
    main()
