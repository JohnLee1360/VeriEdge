#!/usr/bin/env python3
"""Lightweight first-shard task launcher for EXO providers."""
from __future__ import annotations

import argparse
import asyncio
import math
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from common import percentile_or_nan, utc_iso_now  # noqa: E402
from crypto_utils import decrypt_bytes_aes_gcm, decrypt_task_key_from_request  # noqa: E402
from exo_state_utils import first_shard_provider, iter_model_instances  # noqa: E402


@dataclass
class LauncherConfig:
    provider_node_id: str
    provider_ip: str
    exo_endpoint: str
    private_key_path: Path
    model_id: str
    timeout_sec: int
    max_tokens: int


class TokenCounter:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()))


def log_event(event: str, **fields: Any) -> None:
    payload = {"ts": utc_iso_now(), "event": event}
    payload.update({key: value for key, value in fields.items() if value is not None})
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


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


async def _run_single_question_inner(
    *,
    entry_url: str,
    model_id: str,
    prompt: str,
    token_counter: TokenCounter,
    timeout_sec: int,
    max_tokens: int,
) -> Dict[str, Any]:
    start_perf = time.perf_counter()
    start_ts = utc_iso_now()
    first_token_perf: Optional[float] = None
    output_parts: List[str] = []
    http_status: Optional[int] = None
    failure_type: Optional[str] = None

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }

    stream_timeout = httpx.Timeout(
        connect=timeout_sec,
        write=timeout_sec,
        pool=timeout_sec,
        read=min(float(timeout_sec), 15.0),
    )

    try:
        async with httpx.AsyncClient(timeout=stream_timeout, trust_env=False) as client:
            async with client.stream("POST", entry_url.rstrip("/") + "/v1/chat/completions", json=payload) as resp:
                http_status = int(resp.status_code)
                if resp.status_code != 200:
                    _ = await resp.aread()
                    raise RuntimeError("http_error")
                saw_done = False
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        saw_done = True
                        break
                    event_obj = json.loads(data_str)
                    piece = _extract_delta_text(event_obj)
                    if piece:
                        if first_token_perf is None:
                            first_token_perf = time.perf_counter()
                        output_parts.append(piece)

        end_perf = time.perf_counter()
        output_text = "".join(output_parts)
        if not saw_done and not output_text.strip():
            failure_type = "stream_interrupted"
        elif not output_text.strip() or first_token_perf is None:
            failure_type = "empty_output"
        output_tokens = token_counter.count(output_text) if failure_type is None else 0
        ttft_s = (first_token_perf - start_perf) if first_token_perf is not None else None
        latency_s = end_perf - start_perf
        otps = None
        if failure_type is None and first_token_perf is not None and output_tokens > 0:
            denom = end_perf - first_token_perf
            if denom > 0:
                otps = output_tokens / denom
        return {
            "status": "success" if failure_type is None else "failed",
            "latency_s": latency_s,
            "ttft_s": ttft_s,
            "output_tokens": output_tokens,
            "otps_tok_s": otps,
            "output_text": output_text,
            "failure_type": failure_type,
            "failure_detail": None,
            "http_status": http_status,
            "start_ts": start_ts,
            "end_ts": utc_iso_now(),
        }
    except httpx.ReadTimeout as exc:
        end_perf = time.perf_counter()
        output_text = "".join(output_parts)
        if first_token_perf is not None and output_text.strip():
            output_tokens = token_counter.count(output_text)
            ttft_s = first_token_perf - start_perf
            denom = end_perf - first_token_perf
            otps = output_tokens / denom if denom > 0 else None
            return {
                "status": "success",
                "latency_s": end_perf - start_perf,
                "ttft_s": ttft_s,
                "output_tokens": output_tokens,
                "otps_tok_s": otps,
                "output_text": output_text,
                "failure_type": None,
                "failure_detail": f"{type(exc).__name__}: stream ended by inactivity after output",
                "http_status": http_status,
                "start_ts": start_ts,
                "end_ts": utc_iso_now(),
            }
        failure = classify_exception(exc)
        return {
            "status": "failed",
            "latency_s": end_perf - start_perf,
            "ttft_s": None,
            "output_tokens": 0,
            "otps_tok_s": None,
            "output_text": "",
            "failure_type": failure,
            "failure_detail": f"{type(exc).__name__}: {exc}",
            "http_status": http_status,
            "start_ts": start_ts,
            "end_ts": utc_iso_now(),
        }
    except Exception as exc:  # noqa: BLE001
        end_perf = time.perf_counter()
        failure = classify_exception(exc)
        if isinstance(exc, RuntimeError) and str(exc) == "http_error":
            failure = "http_error"
        return {
            "status": "failed",
            "latency_s": end_perf - start_perf,
            "ttft_s": None,
            "output_tokens": 0,
            "otps_tok_s": None,
            "output_text": "",
            "failure_type": failure,
            "failure_detail": f"{type(exc).__name__}: {exc}",
            "http_status": http_status,
            "start_ts": start_ts,
            "end_ts": utc_iso_now(),
        }


async def run_single_question(
    *,
    entry_url: str,
    model_id: str,
    prompt: str,
    token_counter: TokenCounter,
    timeout_sec: int,
    task_id: str = "",
    question_index: int = 0,
    prompt_id: Optional[int] = None,
    max_tokens: int = 64,
) -> Dict[str, Any]:
    question_timeout_sec = max(1.0, float(min(timeout_sec, 120)))
    log_event(
        "question_start",
        task_id=task_id,
        question_index=question_index,
        prompt_id=prompt_id,
        timeout_sec=question_timeout_sec,
        prompt_chars=len(prompt),
    )
    start_perf = time.perf_counter()
    start_ts = utc_iso_now()
    try:
        result = await asyncio.wait_for(
            _run_single_question_inner(
                entry_url=entry_url,
                model_id=model_id,
                prompt=prompt,
                token_counter=token_counter,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
            ),
            timeout=question_timeout_sec,
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        end_perf = time.perf_counter()
        result = {
            "status": "failed",
            "latency_s": end_perf - start_perf,
            "ttft_s": None,
            "output_tokens": 0,
            "otps_tok_s": None,
            "output_text": "",
            "failure_type": "timeout",
            "failure_detail": f"{type(exc).__name__}: question exceeded wall-clock timeout of {question_timeout_sec}s",
            "http_status": None,
            "start_ts": start_ts,
            "end_ts": utc_iso_now(),
        }
    log_event(
        "question_end",
        task_id=task_id,
        question_index=question_index,
        prompt_id=prompt_id,
        status=result.get("status"),
        failure_type=result.get("failure_type"),
        latency_s=round(float(result.get("latency_s", 0.0)), 3),
        http_status=result.get("http_status"),
    )
    return result


async def fetch_state(entry_url: str, timeout_sec: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.get(entry_url.rstrip("/") + "/state")
        resp.raise_for_status()
        return resp.json()


def aggregate_question_results(question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ttfts = [float(row["ttft_s"]) for row in question_results if row.get("ttft_s") is not None]
    otps = [float(row["otps_tok_s"]) for row in question_results if row.get("otps_tok_s") is not None]
    success_count = sum(1 for row in question_results if row.get("status") == "success")
    fail_count = sum(1 for row in question_results if row.get("status") != "success")
    return {
        "ttft_p50_s": percentile_or_nan(ttfts, 50),
        "otps_p50_tok_s": percentile_or_nan(otps, 50),
        "question_success_count": success_count,
        "question_fail_count": fail_count,
    }


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def post_callback_with_retry(
    *,
    callback_url: str,
    payload: Dict[str, Any],
    timeout_sec: int,
    task_id: str,
    max_attempts: int = 3,
    retry_delay_sec: float = 1.0,
) -> bool:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.post(callback_url, json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            log_event(
                "callback_success",
                task_id=task_id,
                attempt=attempt,
                status_code=resp.status_code,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log_event(
                "callback_failure",
                task_id=task_id,
                attempt=attempt,
                error=f"{type(exc).__name__}: {exc}",
            )
            if attempt < max_attempts:
                time.sleep(retry_delay_sec)
    if last_error is not None:
        return False
    return True


def process_launch(payload: Dict[str, Any], cfg: LauncherConfig, token_counter: TokenCounter) -> None:
    task_id = str(payload["task_id"])
    entry_url = str(payload["entry_url"])
    callback_url = str(payload["callback_url"])
    gateway_url = str(payload["gateway_url"])
    model_id = str(payload["model_id"])
    instance_id = str(payload["instance_id"])
    download_started_ts = utc_iso_now()
    download_started_perf = time.perf_counter()
    log_event(
        "task_start",
        task_id=task_id,
        instance_id=instance_id,
        model_id=model_id,
        provider_node_id=cfg.provider_node_id,
    )

    try:
        state_obj = asyncio.run(fetch_state(entry_url, cfg.timeout_sec))
        wrapped = state_obj.get("instances", {})
        if not isinstance(wrapped, dict) or instance_id not in wrapped:
            raise RuntimeError(f"instance_id not found in /state: {instance_id}")
        instance = wrapped[instance_id]
        first_node_id, first_ip = first_shard_provider(instance if isinstance(instance, dict) else {}, state_obj)
        if first_ip != cfg.provider_ip:
            raise RuntimeError(
                f"provider {cfg.provider_ip} is not first-shard for {instance_id}; expected {first_ip}"
            )

        package_resp = httpx.get(
            gateway_url.rstrip("/") + f"/ipfs/{payload['cid']}",
            timeout=cfg.timeout_sec,
        )
        package_resp.raise_for_status()
        encrypted_package = package_resp.json()
        task_key = decrypt_task_key_from_request(
            str(payload["encrypted_task_key"]),
            cfg.private_key_path,
        )
        manifest_bytes = decrypt_bytes_aes_gcm(
            encrypted_package["ciphertext_b64"],
            encrypted_package["encryption"]["nonce_b64"],
            task_key,
        )
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        download_finished_perf = time.perf_counter()
        download_finished_ts = utc_iso_now()

        question_results: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(manifest["prompts"], start=1):
            prompt_id = int(prompt["prompt_id"])
            log_event(
                "question_dispatch",
                task_id=task_id,
                question_index=idx,
                prompt_id=prompt_id,
                instance_id=instance_id,
            )
            result = asyncio.run(
                run_single_question(
                    entry_url=entry_url,
                    model_id=model_id,
                    prompt=str(prompt["content"]),
                    token_counter=token_counter,
                    timeout_sec=cfg.timeout_sec,
                    task_id=task_id,
                    question_index=idx,
                    prompt_id=prompt_id,
                    max_tokens=cfg.max_tokens,
                )
            )
            result["question_index"] = idx
            result["prompt_id"] = prompt_id
            result["prompt_text"] = str(prompt["content"])
            question_results.append(result)

        task_finished_perf = time.perf_counter()
        task_finished_ts = utc_iso_now()
        aggregate = aggregate_question_results(question_results)
        aggregate.update(
            {
                "download_s": download_finished_perf - download_started_perf,
                "task_latency_s": task_finished_perf - download_started_perf,
                "task_throughput_tps": 1.0 / max(task_finished_perf - download_started_perf, 1e-9),
                "question_throughput_qps": len(question_results) / max(task_finished_perf - download_started_perf, 1e-9),
            }
        )
        callback_payload = {
            "task_id": task_id,
            "instance_id": instance_id,
            "provider_node_id": cfg.provider_node_id,
            "download_started_ts": download_started_ts,
            "download_finished_ts": download_finished_ts,
            "task_finished_ts": task_finished_ts,
            "question_results": question_results,
            "aggregate_stats": aggregate,
        }
        callback_payload = sanitize_for_json(callback_payload)
        callback_ok = post_callback_with_retry(
            callback_url=callback_url,
            payload=callback_payload,
            timeout_sec=cfg.timeout_sec,
            task_id=task_id,
        )
        log_event(
            "task_complete",
            task_id=task_id,
            instance_id=instance_id,
            provider_node_id=cfg.provider_node_id,
            question_count=len(question_results),
            callback_ok=callback_ok,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(
            "task_exception",
            task_id=task_id,
            instance_id=instance_id,
            provider_node_id=cfg.provider_node_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        error_payload = {
            "task_id": task_id,
            "instance_id": instance_id,
            "provider_node_id": cfg.provider_node_id,
            "download_started_ts": download_started_ts,
            "download_finished_ts": utc_iso_now(),
            "task_finished_ts": utc_iso_now(),
            "question_results": [],
            "aggregate_stats": {
                "download_s": 0.0,
                "task_latency_s": 0.0,
                "task_throughput_tps": 0.0,
                "question_throughput_qps": 0.0,
                "ttft_p50_s": float("nan"),
                "otps_p50_tok_s": float("nan"),
                "question_success_count": 0,
                "question_fail_count": 0,
            },
            "error": f"{type(exc).__name__}: {exc}",
        }
        error_payload = sanitize_for_json(error_payload)
        callback_ok = post_callback_with_retry(
            callback_url=callback_url,
            payload=error_payload,
            timeout_sec=cfg.timeout_sec,
            task_id=task_id,
        )
        log_event(
            "task_error_reported",
            task_id=task_id,
            instance_id=instance_id,
            provider_node_id=cfg.provider_node_id,
            callback_ok=callback_ok,
        )


class LauncherServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], cfg: LauncherConfig, token_counter: TokenCounter):
        super().__init__(server_address, LauncherHandler)
        self.cfg = cfg
        self.token_counter = token_counter


class LauncherHandler(BaseHTTPRequestHandler):
    server: LauncherServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "ok": True,
                    "provider_node_id": self.server.cfg.provider_node_id,
                    "provider_ip": self.server.cfg.provider_ip,
                }
            ).encode("utf-8")
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/launch-task":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            worker = threading.Thread(
                target=process_launch,
                args=(payload, self.server.cfg, self.server.token_counter),
                daemon=True,
            )
            worker.start()
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "accepted": True,
                        "task_id": payload.get("task_id"),
                        "provider_node_id": self.server.cfg.provider_node_id,
                    }
                ).encode("utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"accepted": False, "error": str(exc)}).encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provider-side first-shard launcher")
    parser.add_argument("--host", default=os.environ.get("LAUNCHER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAUNCHER_PORT", "18080")))
    parser.add_argument(
        "--provider-node-id",
        default=os.environ.get("PROVIDER_NODE_ID", "provider"),
    )
    parser.add_argument("--provider-ip", default=os.environ.get("PROVIDER_IP", "127.0.0.1"))
    parser.add_argument("--exo-endpoint", default=os.environ.get("EXO_ENDPOINT", "http://127.0.0.1:52415"))
    parser.add_argument("--private-key-path", default=os.environ.get("PROVIDER_PRIVATE_KEY_PATH", ""))
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "mlx-community/Qwen3-0.6B-8bit"))
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("LAUNCHER_TIMEOUT_SEC", "180")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LAUNCHER_MAX_TOKENS", "64")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.private_key_path:
        raise SystemExit("Missing PROVIDER_PRIVATE_KEY_PATH / --private-key-path")
    cfg = LauncherConfig(
        provider_node_id=str(args.provider_node_id),
        provider_ip=str(args.provider_ip),
        exo_endpoint=str(args.exo_endpoint),
        private_key_path=Path(str(args.private_key_path)),
        model_id=str(args.model_id),
        timeout_sec=int(args.timeout_sec),
        max_tokens=max(1, int(args.max_tokens)),
    )
    token_counter = TokenCounter(cfg.model_id)
    server = LauncherServer((str(args.host), int(args.port)), cfg, token_counter)
    print(f"launcher listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
