from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


def host_from_url(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").strip()


def unwrap_tagged(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and len(obj) == 1:
        key, value = next(iter(obj.items()))
        if isinstance(key, str) and key[:1].isupper() and isinstance(value, dict):
            return value
    if isinstance(obj, dict):
        return obj
    return {}


def node_ip_map(state_obj: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    node_network = state_obj.get("nodeNetwork")
    if not isinstance(node_network, dict):
        return out
    for node_id, info in node_network.items():
        ips: List[str] = []
        if isinstance(info, dict):
            for item in info.get("interfaces", []):
                if isinstance(item, dict):
                    ip = item.get("ipAddress")
                    if isinstance(ip, str) and ip:
                        ips.append(ip)
        if ips:
            out[str(node_id)] = sorted(ips, key=_ip_preference_key)
    return out


def _ip_preference_key(ip: str) -> Tuple[int, str]:
    if ip.startswith("192.168."):
        return (0, ip)
    if ip.startswith("10.") or ip.startswith("172."):
        return (1, ip)
    if ip.startswith("127.") or ip == "::1" or ip.startswith("fe80::") or "%" in ip:
        return (3, ip)
    return (2, ip)


def iter_model_instances(state_obj: Dict[str, Any], model_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    instances = state_obj.get("instances")
    if not isinstance(instances, dict):
        return []
    matches: List[Tuple[str, Dict[str, Any]]] = []
    for instance_id, wrapped in instances.items():
        inst = unwrap_tagged(wrapped)
        assignments = inst.get("shardAssignments", {})
        if isinstance(assignments, dict) and assignments.get("modelId") == model_id:
            matches.append((str(instance_id), inst))
    return matches


def instance_node_count(instance: Dict[str, Any]) -> int:
    assignments = instance.get("shardAssignments", {})
    if not isinstance(assignments, dict):
        return 0
    node_to_runner = assignments.get("nodeToRunner")
    return len(node_to_runner) if isinstance(node_to_runner, dict) else 0


def _unwrap_shard(shard_obj: Any) -> Dict[str, Any]:
    return unwrap_tagged(shard_obj)


def first_shard_node_id(instance: Dict[str, Any]) -> Optional[str]:
    instance = unwrap_tagged(instance)
    assignments = instance.get("shardAssignments", {})
    if not isinstance(assignments, dict):
        return None
    node_to_runner = assignments.get("nodeToRunner")
    runner_to_shard = assignments.get("runnerToShard")
    if not isinstance(node_to_runner, dict) or not isinstance(runner_to_shard, dict):
        return None

    candidates: List[Tuple[int, str]] = []
    for node_id, runner_id in node_to_runner.items():
        shard = _unwrap_shard(runner_to_shard.get(runner_id))
        if not shard:
            continue
        device_rank = int(shard.get("deviceRank", 1 << 30))
        start_layer = shard.get("startLayer")
        if start_layer == 0:
            candidates.append((device_rank, str(node_id)))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    fallback: List[Tuple[int, str]] = []
    for node_id, runner_id in node_to_runner.items():
        shard = _unwrap_shard(runner_to_shard.get(runner_id))
        if not shard:
            continue
        device_rank = int(shard.get("deviceRank", 1 << 30))
        fallback.append((device_rank, str(node_id)))
    if fallback:
        return min(fallback, key=lambda item: item[0])[1]
    return None


def first_shard_provider(instance: Dict[str, Any], state_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    instance = unwrap_tagged(instance)
    node_id = first_shard_node_id(instance)
    if node_id is None:
        return None, None
    ip_map = node_ip_map(state_obj)
    ips = ip_map.get(node_id, [])
    return node_id, (ips[0] if ips else None)
