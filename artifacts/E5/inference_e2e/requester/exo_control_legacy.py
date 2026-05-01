#!/usr/bin/env python3
"""Legacy EXO control helpers kept outside the default black-box path."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import httpx

from exo_state_utils import iter_model_instances


async def fetch_state(entry_url: str, timeout_sec: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.get(entry_url.rstrip("/") + "/state")
        resp.raise_for_status()
        return resp.json()


async def wait_for_model_instance_count(
    entry_url: str,
    model_id: str,
    expected_count: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_state: Dict[str, Any] = {}
    while time.time() < deadline:
        last_state = await fetch_state(entry_url, timeout_sec=min(10, timeout_sec))
        instances = iter_model_instances(last_state, model_id)
        if len(instances) == expected_count:
            return last_state
        await asyncio.sleep(1.0)
    raise TimeoutError(
        f"Timed out waiting for {expected_count} instances of {model_id}; "
        f"last_count={len(iter_model_instances(last_state, model_id))}"
    )


async def delete_model_instances(entry_url: str, model_id: str, timeout_sec: int) -> None:
    state = await fetch_state(entry_url, timeout_sec)
    instances = iter_model_instances(state, model_id)
    if not instances:
        return
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        for instance_id, _ in instances:
            resp = await client.delete(entry_url.rstrip("/") + f"/instance/{instance_id}")
            resp.raise_for_status()
    await wait_for_model_instance_count(entry_url, model_id, 0, timeout_sec)


async def _place_instance(entry_url: str, payload: Dict[str, Any], timeout_sec: int) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        resp = await client.post(entry_url.rstrip("/") + "/place_instance", json=payload)
        resp.raise_for_status()


async def _placement_fallback_create(entry_url: str, payload: Dict[str, Any], timeout_sec: int) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec), trust_env=False) as client:
        placement_resp = await client.get(entry_url.rstrip("/") + "/instance/placement", params=payload)
        placement_resp.raise_for_status()
        placement = placement_resp.json()
        create_resp = await client.post(entry_url.rstrip("/") + "/instance", json={"instance": placement})
        create_resp.raise_for_status()


async def ensure_instance_count(
    entry_url: str,
    model_id: str,
    instance_count: int,
    min_nodes: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    await delete_model_instances(entry_url, model_id, timeout_sec)
    payload = {
        "model_id": model_id,
        "sharding": "Pipeline",
        "instance_meta": "MlxRing",
        "min_nodes": int(min_nodes),
    }
    for _ in range(instance_count):
        await _place_instance(entry_url, payload, timeout_sec)
    try:
        return await wait_for_model_instance_count(entry_url, model_id, instance_count, timeout_sec)
    except TimeoutError:
        await delete_model_instances(entry_url, model_id, timeout_sec)
        for _ in range(instance_count):
            await _placement_fallback_create(entry_url, payload, timeout_sec)
        return await wait_for_model_instance_count(entry_url, model_id, instance_count, timeout_sec)
