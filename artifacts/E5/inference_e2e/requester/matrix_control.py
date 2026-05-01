#!/usr/bin/env python3
"""Low-token control-plane runner for the EXO requester experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx
import yaml


def load_structured(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-token control-plane runner for EXO experiment")
    parser.add_argument("--config", default="artifacts/E5/inference_e2e/requester/config.example.yaml")
    parser.add_argument("--batch-dir", default="", help="Stable output directory for this control-plane run")
    parser.add_argument("--network-profile", action="append", default=[])
    parser.add_argument("--instance-node-count", type=int, action="append", default=[])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--provision-instances", action="store_true")
    parser.add_argument("--allow-wan", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def configured_instance_node_counts(cfg: Dict[str, Any], networks: List[str]) -> List[int]:
    matrix_cfg = cfg.get("matrix", {})
    by_network = matrix_cfg.get("instance_node_counts_by_network", {})
    if isinstance(by_network, dict) and networks:
        counts: List[int] = []
        for network in networks:
            values = by_network.get(network)
            if not isinstance(values, list):
                continue
            counts.extend(int(item) for item in values)
        if counts:
            return sorted(set(counts))
    return [int(x) for x in matrix_cfg.get("instance_node_counts", [])]


def exo_request(cfg: Dict[str, Any], method: str, path: str, **kwargs: Any) -> Any:
    base = str(cfg["endpoints"]["entry_url"]).rstrip("/")
    timeout_sec = min(int(cfg.get("runtime", {}).get("timeout_sec", 180)), 30)
    resp = httpx.request(method, base + path, timeout=timeout_sec, **kwargs)
    resp.raise_for_status()
    return resp.json() if resp.text.strip() else None


def create_instance(cfg: Dict[str, Any], node_count: int) -> str:
    placement = exo_request(
        cfg,
        "GET",
        "/instance/placement",
        params={"model_id": str(cfg["model"]["model_id"]), "min_nodes": node_count},
    )
    ring = placement.get("MlxRingInstance", {})
    instance_id = str(ring.get("instanceId", "")).strip()
    if not instance_id:
        raise RuntimeError(f"invalid placement response for node_count={node_count}: {placement}")
    exo_request(cfg, "POST", "/instance", json={"instance": placement})
    for _ in range(120):
        state = exo_request(cfg, "GET", "/state")
        if instance_id in state.get("instances", {}):
            return instance_id
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for instance {instance_id}")


def delete_instance(cfg: Dict[str, Any], instance_id: str) -> None:
    try:
        exo_request(cfg, "DELETE", f"/instance/{instance_id}")
    except Exception:
        return
    for _ in range(60):
        state = exo_request(cfg, "GET", "/state")
        if instance_id not in state.get("instances", {}):
            return
        time.sleep(1)


def selected_cells(cfg: Dict[str, Any], args: argparse.Namespace, repo_root: Path) -> List[Dict[str, Any]]:
    schedule_path = Path(str(cfg["task"].get("schedule_path", "artifacts/E5/inference_e2e/requester/task_schedule.json")))
    if not schedule_path.is_absolute():
        schedule_path = repo_root / schedule_path
    schedule = load_structured(schedule_path)
    selected_networks = list(args.network_profile or list(cfg["matrix"]["network_profiles"]))
    allowed_networks = set(selected_networks)
    explicit_counts = [int(x) for x in args.instance_node_count]
    network_count_map = cfg.get("matrix", {}).get("instance_node_counts_by_network", {})
    cells: List[Dict[str, Any]] = []
    for cell in schedule.get("cells", []):
        network = str(cell.get("network", ""))
        node_count = int(cell.get("instance_node_count", 0))
        if network not in allowed_networks:
            continue
        if explicit_counts:
            allowed_counts = set(explicit_counts)
        else:
            configured = network_count_map.get(network, configured_instance_node_counts(cfg, [network]))
            allowed_counts = {int(x) for x in configured}
        if node_count not in allowed_counts:
            continue
        if network == "WAN" and not args.allow_wan:
            continue
        cells.append(
            {
                "cell_id": str(cell.get("cell_id", f"{network.lower()}_n{node_count}")),
                "network": network,
                "instance_node_count": node_count,
            }
        )
    return cells


def make_batch_dir(cfg: Dict[str, Any], requested: str) -> Path:
    if requested:
        return Path(requested)
    root = Path(str(cfg.get("runtime", {}).get("output_root", "artifacts/E5/inference_e2e/requester/output")))
    return root / datetime.now().strftime("control_%Y%m%d_%H%M%S")


def update_status(status_path: Path, **kwargs: Any) -> Dict[str, Any]:
    payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    payload.update(kwargs)
    payload["updated_at"] = now_iso()
    write_json(status_path, payload)
    return payload


def maybe_skip_completed(cell_dir: Path) -> bool:
    control_status = cell_dir / "control_status.json"
    summary_path = cell_dir / "summary_by_cell.csv"
    if not control_status.exists() or not summary_path.exists():
        return False
    try:
        payload = load_structured(control_status)
    except Exception:
        return False
    return payload.get("status") == "completed"


def build_temp_config(
    cfg: Dict[str, Any],
    *,
    network: str,
    node_count: int,
    instance_id: str,
    dst: Path,
) -> None:
    payload = json.loads(json.dumps(cfg))
    payload["external_exo"]["expected_instance_ids_by_node_count"] = {str(node_count): [instance_id]}
    payload["matrix"]["network_profiles"] = [network]
    payload["matrix"]["instance_node_counts"] = [node_count]
    write_yaml(dst, payload)


def run_logged(cmd: List[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{now_iso()}] CMD {' '.join(cmd)}\n")
        logf.flush()
        completed = subprocess.run(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, check=False)
        logf.write(f"[{now_iso()}] EXIT {completed.returncode}\n")
        return int(completed.returncode)


def run_hook_commands(
    commands: List[str],
    *,
    cwd: Path,
    log_path: Path,
    env_overrides: Dict[str, str],
) -> None:
    if not commands:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for command in commands:
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"\n[{now_iso()}] HOOK {command}\n")
            logf.flush()
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(cwd),
                env={**os.environ, **env_overrides},
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
            logf.write(f"[{now_iso()}] HOOK_EXIT {completed.returncode}\n")
            if completed.returncode != 0:
                raise RuntimeError(f"hook command failed: {command}")


def derive_wan_ports(cfg: Dict[str, Any]) -> str:
    ports = set()
    ports.add(urlparse(str(cfg["endpoints"]["entry_url"])).port or 52415)
    ports.add(urlparse(str(cfg["endpoints"]["requester_callback_url"])).port or 18081)
    ports.add(urlparse(str(cfg["ipfs"]["gateway_url"])).port or 8080)
    for provider in cfg.get("providers", []):
        launcher_port = urlparse(str(provider.get("launcher_url", ""))).port
        if launcher_port is not None:
            ports.add(launcher_port)
    return ",".join(str(port) for port in sorted(ports))


def apply_network_shaping(cfg: Dict[str, Any], script_dir: Path, network: str, cell_dir: Path) -> None:
    netem_script = script_dir / "netem_macos.sh"
    log_path = cell_dir / "network_control.log"
    if network == "LAN":
        rc = run_logged(["sudo", "-n", str(netem_script), "reset"], cwd=script_dir.parents[3], log_path=log_path)
        # LAN should still be runnable even if this shell cannot refresh sudo tickets.
        # In that case we log the reset failure and continue with no shaping applied.
        return

    target_spec = str(cfg.get("network_profiles", {}).get("WAN", {}).get("target_spec", "")).strip()
    if not target_spec:
        raise RuntimeError("WAN target_spec is empty in config")
    ports_csv = derive_wan_ports(cfg)
    rc = run_logged(
        ["sudo", "-n", str(netem_script), "apply", "--ports", ports_csv, "--target-spec", target_spec],
        cwd=script_dir.parents[3],
        log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError("failed to apply WAN shaping; ensure 'sudo -v' is active on requester host")


def control_plane_commands(cfg: Dict[str, Any], key: str, network: str) -> List[str]:
    control_plane = cfg.get("control_plane", {})
    if key == "before_network_commands":
        mapping = control_plane.get(key, {})
        values = mapping.get(network, []) if isinstance(mapping, dict) else []
    else:
        values = control_plane.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if str(item).strip()]


def combine_outputs(batch_dir: Path, python_bin: str, script_dir: Path, *, no_plot: bool) -> None:
    combine_code = """
from pathlib import Path
import pandas as pd
import sys
batch_dir = Path(sys.argv[1])
frames = [pd.read_csv(p) for p in sorted(batch_dir.glob('cells/*/summary_by_cell.csv'))]
task_frames = [pd.read_csv(p) for p in sorted(batch_dir.glob('cells/*/summary_by_task.csv'))]
if frames:
    pd.concat(frames, ignore_index=True).sort_values(['network', 'instance_node_count']).to_csv(batch_dir / 'summary_by_cell.csv', index=False)
else:
    pd.DataFrame().to_csv(batch_dir / 'summary_by_cell.csv', index=False)
if task_frames:
    pd.concat(task_frames, ignore_index=True).to_csv(batch_dir / 'summary_by_task.csv', index=False)
else:
    pd.DataFrame().to_csv(batch_dir / 'summary_by_task.csv', index=False)
"""
    subprocess.run([python_bin, "-c", combine_code, str(batch_dir)], check=True)
    subprocess.run(
        [
            python_bin,
            str(script_dir / "make_comparison_table.py"),
            "--input",
            str(batch_dir / "summary_by_cell.csv"),
            "--output-csv",
            str(batch_dir / "comparison_table.csv"),
            "--output-md",
            str(batch_dir / "comparison_table.md"),
        ],
        check=False,
    )
    if not no_plot:
        subprocess.run(
            [
                python_bin,
                str(script_dir / "plot.py"),
                "--input",
                str(batch_dir / "summary_by_cell.csv"),
                "--output-dir",
                str(batch_dir),
            ],
            check=False,
        )


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_structured(cfg_path)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    batch_dir = make_batch_dir(cfg, args.batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    cells = selected_cells(cfg, args, repo_root)
    if not cells:
        raise SystemExit("no cells selected for control-plane run")

    manifest = {
        "created_at": now_iso(),
        "config_path": str(cfg_path),
        "cells": cells,
        "provision_instances": bool(args.provision_instances),
        "allow_wan": bool(args.allow_wan),
    }
    write_json(batch_dir / "batch_manifest.json", manifest)
    update_status(
        batch_dir / "batch_status.json",
        status="running",
        current_cell_id=None,
        completed_cells=[],
        failed_cell_id=None,
    )

    for cell in cells:
        cell_id = cell["cell_id"]
        network = cell["network"]
        node_count = int(cell["instance_node_count"])
        cell_dir = batch_dir / "cells" / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)

        if maybe_skip_completed(cell_dir):
            payload = update_status(batch_dir / "batch_status.json")
            completed = list(payload.get("completed_cells", []))
            if cell_id not in completed:
                completed.append(cell_id)
            update_status(batch_dir / "batch_status.json", completed_cells=completed)
            continue

        update_status(batch_dir / "batch_status.json", current_cell_id=cell_id)
        write_json(
            cell_dir / "control_status.json",
            {
                "status": "preparing",
                "network": network,
                "instance_node_count": node_count,
                "updated_at": now_iso(),
            },
        )

        instance_id = ""
        config_for_cell = cfg_path
        try:
            env_overrides = {
                "BCRA_CELL_ID": cell_id,
                "BCRA_NETWORK": network,
                "BCRA_INSTANCE_NODE_COUNT": str(node_count),
                "BCRA_BATCH_DIR": str(batch_dir.resolve()),
                "BCRA_CELL_DIR": str(cell_dir.resolve()),
            }
            run_hook_commands(
                control_plane_commands(cfg, "before_cell_commands", network),
                cwd=repo_root,
                log_path=cell_dir / "control_hooks.log",
                env_overrides=env_overrides,
            )
            run_hook_commands(
                control_plane_commands(cfg, "before_network_commands", network),
                cwd=repo_root,
                log_path=cell_dir / "control_hooks.log",
                env_overrides=env_overrides,
            )
            apply_network_shaping(cfg, script_dir, network, cell_dir)
            if args.provision_instances:
                instance_id = create_instance(cfg, node_count)
                write_json(
                    cell_dir / "instance_control.json",
                    {
                        "status": "created",
                        "instance_id": instance_id,
                        "instance_node_count": node_count,
                        "created_at": now_iso(),
                    },
                )
                config_for_cell = cell_dir / "control_config.yaml"
                build_temp_config(
                    cfg,
                    network=network,
                    node_count=node_count,
                    instance_id=instance_id,
                    dst=config_for_cell,
                )

            write_json(
                cell_dir / "control_status.json",
                {
                    "status": "running",
                    "network": network,
                    "instance_node_count": node_count,
                    "instance_id": instance_id,
                    "updated_at": now_iso(),
                },
            )
            rc = run_logged(
                [
                    args.python_bin,
                    str(script_dir / "runner.py"),
                    "--config",
                    str(config_for_cell),
                    "--network-profile",
                    network,
                    "--instance-node-count",
                    str(node_count),
                    "--output-dir",
                    str(cell_dir),
                    "--no-plot",
                ],
                cwd=repo_root,
                log_path=cell_dir / "runner.log",
            )
            if rc != 0:
                write_json(
                    cell_dir / "control_status.json",
                    {
                        "status": "failed",
                        "network": network,
                        "instance_node_count": node_count,
                        "instance_id": instance_id,
                        "returncode": rc,
                        "updated_at": now_iso(),
                    },
                )
                update_status(batch_dir / "batch_status.json", status="failed", failed_cell_id=cell_id)
                raise SystemExit(rc)

            payload = update_status(batch_dir / "batch_status.json")
            completed = list(payload.get("completed_cells", []))
            if cell_id not in completed:
                completed.append(cell_id)
            write_json(
                cell_dir / "control_status.json",
                {
                    "status": "completed",
                    "network": network,
                    "instance_node_count": node_count,
                    "instance_id": instance_id,
                    "updated_at": now_iso(),
                },
            )
            update_status(batch_dir / "batch_status.json", completed_cells=completed)
        finally:
            if args.provision_instances and instance_id:
                delete_instance(cfg, instance_id)
                write_json(
                    cell_dir / "instance_control.json",
                    {
                        "status": "deleted",
                        "instance_id": instance_id,
                        "instance_node_count": node_count,
                        "deleted_at": now_iso(),
                    },
                )

    if args.allow_wan:
        final_netem_log = batch_dir / "network_reset.log"
        run_logged(["sudo", "-n", str(script_dir / "netem_macos.sh"), "reset"], cwd=repo_root, log_path=final_netem_log)

    combine_outputs(batch_dir, args.python_bin, script_dir, no_plot=args.no_plot)
    update_status(batch_dir / "batch_status.json", status="completed", current_cell_id=None)


if __name__ == "__main__":
    main()
