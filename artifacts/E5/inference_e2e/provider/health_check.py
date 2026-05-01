#!/usr/bin/env python3
"""Provider endpoint health check and tiny probe for exo/OpenAI-compatible APIs."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_models(endpoint: str, timeout_sec: float) -> Tuple[bool, int | None, str | None]:
    url = endpoint.rstrip("/") + "/v1/models"
    try:
        resp = httpx.get(url, timeout=timeout_sec)
        if resp.status_code == 200:
            return True, resp.status_code, None
        return False, resp.status_code, f"models_status_{resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def check_health(endpoint: str, timeout_sec: float) -> Tuple[bool, int | None, str | None]:
    url = endpoint.rstrip("/") + "/health"
    try:
        resp = httpx.get(url, timeout=timeout_sec)
        if resp.status_code < 500:
            return True, resp.status_code, None
        return False, resp.status_code, f"health_status_{resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def probe_inference(endpoint: str, model: str, timeout_sec: float) -> Dict[str, Any]:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in one word."}],
        "stream": False,
        "max_tokens": 8,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    out: Dict[str, Any] = {
        "ok": False,
        "http_status": None,
        "latency_ms": None,
        "error": None,
        "response_preview": "",
    }
    try:
        resp = httpx.post(url, json=payload, timeout=timeout_sec)
        out["http_status"] = int(resp.status_code)
        out["latency_ms"] = (time.perf_counter() - start) * 1000.0

        if resp.status_code != 200:
            out["error"] = f"probe_status_{resp.status_code}"
            return out

        data = resp.json()
        content = ""
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        content = message["content"]
                    elif isinstance(first.get("text"), str):
                        content = first["text"]
        out["response_preview"] = (content or "")[:120]
        out["ok"] = True
        return out

    except Exception as exc:  # noqa: BLE001
        out["latency_ms"] = (time.perf_counter() - start) * 1000.0
        out["error"] = str(exc)
        return out


def run(endpoint: str, timeout_sec: float, model: Optional[str], probe: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "timestamp": utc_now(),
        "endpoint": endpoint,
        "timeout_sec": timeout_sec,
        "checks": {},
        "ok": False,
    }

    ok_models, status_models, err_models = check_models(endpoint, timeout_sec)
    result["checks"]["models"] = {
        "ok": ok_models,
        "http_status": status_models,
        "error": err_models,
    }

    ok_health, status_health, err_health = check_health(endpoint, timeout_sec)
    result["checks"]["health"] = {
        "ok": ok_health,
        "http_status": status_health,
        "error": err_health,
    }

    if probe:
        if not model:
            raise ValueError("--model is required when --probe is set")
        result["checks"]["probe"] = probe_inference(endpoint, model, timeout_sec)

    core_ok = ok_models or ok_health
    probe_ok = True
    if probe:
        probe_ok = bool(result["checks"]["probe"]["ok"])

    result["ok"] = core_ok and probe_ok
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provider endpoint health/probe checker")
    parser.add_argument("--endpoint", required=True, help="Provider endpoint base URL, e.g. http://127.0.0.1:52415")
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--model", default="")
    parser.add_argument("--probe", action="store_true", help="Send one tiny non-stream inference request")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        endpoint=args.endpoint,
        timeout_sec=float(args.timeout_sec),
        model=args.model or None,
        probe=bool(args.probe),
    )

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(result, ensure_ascii=True))

    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
