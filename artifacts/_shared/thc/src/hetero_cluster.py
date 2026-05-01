from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_ORDER = ["C1", "C2", "C3"]


def load_cluster_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("cluster file must be a JSON object")
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("cluster file must contain a nodes list")
    validate_cluster_nodes(nodes)
    return data


def validate_cluster_nodes(nodes: list[dict[str, Any]]) -> None:
    if len(nodes) != 3:
        raise ValueError("cluster file must contain exactly 3 nodes")

    checkpoints = [str(node["checkpoint"]) for node in nodes]
    if checkpoints != EXPECTED_CHECKPOINT_ORDER:
        raise ValueError("cluster nodes must be ordered as C1, C2, C3")

    names = [str(node["node_name"]) for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("cluster node_name values must be unique")

    hosts = [str(node["host"]) for node in nodes]
    if len(hosts) != len(set((host, int(node["port"])) for host, node in zip(hosts, nodes))):
        raise ValueError("cluster host/port pairs must be unique")

    expected_start = 0
    for index, node in enumerate(nodes):
        start_layer = int(node["start_layer"])
        end_layer = int(node["end_layer"])
        if start_layer != expected_start:
            raise ValueError(
                f"node {node['node_name']} start_layer={start_layer} does not match expected {expected_start}"
            )
        if end_layer < start_layer:
            raise ValueError(
                f"node {node['node_name']} end_layer={end_layer} must be >= start_layer={start_layer}"
            )
        expected_start = end_layer + 1

        checkpoint = str(node["checkpoint"])
        expected_checkpoint = EXPECTED_CHECKPOINT_ORDER[index]
        if checkpoint != expected_checkpoint:
            raise ValueError(
                f"node index {index} checkpoint={checkpoint} does not match expected {expected_checkpoint}"
            )

        if checkpoint == "C1" and not bool(node.get("first_shard", False)):
            raise ValueError("C1 node must set first_shard=true")
        if checkpoint == "C3" and not bool(node.get("last_shard", False)):
            raise ValueError("C3 node must set last_shard=true")
        if checkpoint == "C2":
            if bool(node.get("first_shard", False)) or bool(node.get("last_shard", False)):
                raise ValueError("C2 node must not set first_shard/last_shard")

        device = str(node.get("device", "")).lower()
        quantization = str(node.get("quantization", "none")).lower()
        if quantization == "metal_8bit" and device != "mps":
            raise ValueError("metal_8bit quantization requires device=mps")
        if quantization == "metal_8bit" and not bool(node.get("first_shard", False)):
            raise ValueError("metal_8bit quantization is only supported on the first shard in the current MPS runtime")
        if quantization == "bitsandbytes_8bit" and device != "cuda":
            raise ValueError("bitsandbytes_8bit quantization requires device=cuda")


def cluster_nodes_from_config(data: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("cluster config nodes must be a list")
    validate_cluster_nodes(nodes)
    return [dict(node) for node in nodes]


def resolve_node(nodes: list[dict[str, Any]], local_node: str) -> dict[str, Any]:
    for node in nodes:
        if str(node["node_name"]) == local_node:
            return dict(node)
    raise ValueError(f"local node not found in cluster config: {local_node}")
