#!/usr/bin/env python3
"""Low-token outer orchestrator: sync code, run LAN, prepare WAN, then run WAN."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-token sequencer for LAN->WAN EXO experiment")
    parser.add_argument("--config", default="artifacts/E5/inference_e2e/requester/config.yaml")
    parser.add_argument("--run-dir", default="", help="Top-level orchestration directory")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def run_logged(cmd: List[str], *, cwd: Path, log_path: Path, env: Dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{now_iso()}] CMD {' '.join(shlex.quote(x) for x in cmd)}\n")
        logf.flush()
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env={**os.environ, **(env or {})},
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
        logf.write(f"[{now_iso()}] EXIT {completed.returncode}\n")
        return int(completed.returncode)


def run_shell_hook(command: str, *, cwd: Path, log_path: Path, env: Dict[str, str]) -> None:
    rc = run_logged(["bash", "-lc", command], cwd=cwd, log_path=log_path, env=env)
    if rc != 0:
        raise RuntimeError(f"hook failed: {command}")


def update_status(path: Path, **kwargs: Any) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(kwargs)
    payload["updated_at"] = now_iso()
    write_json(path, payload)
    return payload


def ssh_cmd(target: str, remote_command: str) -> List[str]:
    return ["ssh", target, remote_command]


def requester_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    orch = cfg.get("orchestrator", {})
    requester = orch.get("requester", {})
    if not isinstance(requester, dict):
        raise ValueError("orchestrator.requester must be a mapping")
    return requester


def requester_repo(cfg: Dict[str, Any]) -> Path:
    path = requester_cfg(cfg).get("repo_root")
    if not path:
        raise ValueError("orchestrator.requester.repo_root is required")
    return Path(str(path))


def requester_host(cfg: Dict[str, Any]) -> str:
    target = requester_cfg(cfg).get("ssh_target")
    if not target:
        raise ValueError("orchestrator.requester.ssh_target is required")
    return str(target)


def requester_python(cfg: Dict[str, Any]) -> str:
    value = requester_cfg(cfg).get("python_bin", "python3")
    return str(value)


def configured_instance_node_counts(cfg: Dict[str, Any], network: str) -> List[int]:
    matrix_cfg = cfg.get("matrix", {})
    by_network = matrix_cfg.get("instance_node_counts_by_network", {})
    if isinstance(by_network, dict):
        values = by_network.get(network)
        if isinstance(values, list) and values:
            return [int(item) for item in values]
    return [int(x) for x in matrix_cfg.get("instance_node_counts", [])]


def resolve_local_path(repo_root: Path, path_like: str) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def remote_output_root(cfg: Dict[str, Any]) -> Path:
    output_root = Path(str(cfg.get("runtime", {}).get("output_root", "artifacts/E5/inference_e2e/output")))
    if output_root.is_absolute():
        return output_root
    return requester_repo(cfg) / output_root


def local_mirror_output_root(cfg: Dict[str, Any], repo_root: Path) -> Path:
    local_root = cfg.get("orchestrator", {}).get("local_results_root", "")
    if str(local_root).strip():
        return resolve_local_path(repo_root, str(local_root))
    output_root = str(cfg.get("runtime", {}).get("output_root", "artifacts/E5/inference_e2e/output"))
    return resolve_local_path(repo_root, output_root)


def sync_requester_code(cfg: Dict[str, Any], repo_root: Path, log_path: Path) -> None:
    target = f"{requester_host(cfg)}:{requester_repo(cfg) / 'artifacts/E5/inference_e2e/'}"
    src = str(repo_root / "artifacts/E5/inference_e2e/") + "/"
    rc = run_logged(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            ".venv",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".DS_Store",
            src,
            target,
        ],
        cwd=repo_root,
        log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError("failed to sync requester code to requester host")


def start_sudo_keepalive(cfg: Dict[str, Any], run_dir: Path, log_path: Path) -> None:
    host = requester_host(cfg)
    remote_dir = remote_output_root(cfg) / run_dir.name
    remote_cmd = (
        f"mkdir -p {shlex.quote(str(remote_dir))} && "
        f"nohup bash -lc 'while true; do sudo -n true; sleep 20; done' "
        f"> {shlex.quote(str(remote_dir / 'sudo_keepalive.log'))} 2>&1 < /dev/null & echo $!"
    )
    completed = subprocess.run(ssh_cmd(host, remote_cmd), cwd=str(run_dir), capture_output=True, text=True, check=False)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{now_iso()}] START_SUDO_KEEPALIVE host={host}\n")
        logf.write(completed.stdout)
        logf.write(completed.stderr)
    pid = completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 and completed.stdout.strip() else ""
    write_json(run_dir / "requester_sudo_keepalive.json", {"host": host, "pid": pid, "started_at": now_iso()})
    if completed.returncode != 0:
        raise RuntimeError("failed to start sudo keepalive on requester host")


def start_remote_matrix(
    cfg: Dict[str, Any],
    *,
    network: str,
    run_dir: Path,
    counts: List[int],
    no_plot: bool,
    log_path: Path,
) -> Path:
    host = requester_host(cfg)
    repo = requester_repo(cfg)
    pybin = requester_python(cfg)
    remote_batch_dir = remote_output_root(cfg) / run_dir.name / network.lower()
    cmd = [
        pybin,
        str(repo / "artifacts/E5/inference_e2e/requester/matrix_control.py"),
        "--config",
        str(repo / "artifacts/E5/inference_e2e/requester/config.yaml"),
        "--batch-dir",
        str(remote_batch_dir),
        "--network-profile",
        network,
        "--provision-instances",
    ]
    for count in counts:
        cmd.extend(["--instance-node-count", str(count)])
    if network == "WAN":
        cmd.append("--allow-wan")
    if no_plot:
        cmd.append("--no-plot")
    remote_cmd = (
        f"mkdir -p {shlex.quote(str(remote_batch_dir))} && "
        f"nohup {' '.join(shlex.quote(x) for x in cmd)} "
        f"> {shlex.quote(str(remote_batch_dir / 'launch.log'))} 2>&1 < /dev/null & echo $!"
    )
    completed = subprocess.run(ssh_cmd(host, remote_cmd), capture_output=True, text=True, check=False)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{now_iso()}] START_REMOTE_MATRIX network={network}\n")
        logf.write(completed.stdout)
        logf.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"failed to start remote {network} batch")
    pid = completed.stdout.strip().splitlines()[-1]
    write_json(
        run_dir / f"{network.lower()}_remote_process.json",
        {"host": host, "pid": pid, "batch_dir": str(remote_batch_dir), "started_at": now_iso()},
    )
    return remote_batch_dir


def read_remote_json(host: str, path: Path) -> Dict[str, Any]:
    cmd = ssh_cmd(host, f"cat {shlex.quote(str(path))}")
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"failed to read {path}")
    return json.loads(completed.stdout)


def wait_for_remote_batch(cfg: Dict[str, Any], batch_dir: Path, *, poll_sec: int, status_path: Path) -> Dict[str, Any]:
    host = requester_host(cfg)
    remote_status = batch_dir / "batch_status.json"
    while True:
        payload = read_remote_json(host, remote_status)
        update_status(status_path, remote_status=payload)
        if payload.get("status") in {"completed", "failed"}:
            return payload
        time.sleep(poll_sec)


def wan_prepare_commands(cfg: Dict[str, Any]) -> List[str]:
    orch = cfg.get("orchestrator", {})
    values = orch.get("wan_prepare_commands", [])
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if str(item).strip()]


def sync_remote_results_back(cfg: Dict[str, Any], repo_root: Path, run_dir: Path, log_path: Path) -> None:
    host = requester_host(cfg)
    remote_dir = remote_output_root(cfg) / run_dir.name
    local_dir = local_mirror_output_root(cfg, repo_root) / run_dir.name
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    rc = run_logged(
        ["rsync", "-az", f"{host}:{remote_dir}/", str(local_dir) + "/"],
        cwd=repo_root,
        log_path=log_path,
    )
    if rc != 0:
        raise RuntimeError("failed to sync remote results back to local mirror path")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    if args.run_dir:
        run_dir = resolve_local_path(repo_root, args.run_dir)
    else:
        run_dir = resolve_local_path(
            repo_root,
            str(cfg.get("runtime", {}).get("output_root", "artifacts/E5/inference_e2e/output")),
        ) / datetime.now().strftime("sequence_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "orchestrator.log"
    write_json(
        run_dir / "sequence_manifest.json",
        {
            "created_at": now_iso(),
            "config_path": str(cfg_path),
            "counts_by_network": {
                "LAN": configured_instance_node_counts(cfg, "LAN"),
                "WAN": configured_instance_node_counts(cfg, "WAN"),
            },
            "requester_host": requester_host(cfg),
            "requester_repo": str(requester_repo(cfg)),
        },
    )
    update_status(run_dir / "sequence_status.json", status="starting", phase="sync")

    exit_code = 0
    lan_status: Dict[str, Any] = {}
    wan_status: Dict[str, Any] = {}
    try:
        sync_requester_code(cfg, repo_root, log_path)
        start_sudo_keepalive(cfg, run_dir, log_path)

        update_status(run_dir / "sequence_status.json", status="running", phase="lan")
        lan_batch_dir = start_remote_matrix(
            cfg,
            network="LAN",
            run_dir=run_dir,
            counts=configured_instance_node_counts(cfg, "LAN"),
            no_plot=args.no_plot,
            log_path=log_path,
        )
        lan_status = wait_for_remote_batch(cfg, lan_batch_dir, poll_sec=args.poll_sec, status_path=run_dir / "sequence_status.json")
        if lan_status.get("status") != "completed":
            update_status(run_dir / "sequence_status.json", status="failed", phase="lan", lan_status=lan_status)
            exit_code = 1
        else:
            update_status(run_dir / "sequence_status.json", status="running", phase="wan_prepare", lan_status=lan_status)
            env = {
                "BCRA_SEQUENCE_DIR": str(run_dir.resolve()),
                "BCRA_REQUESTER_HOST": requester_host(cfg),
                "BCRA_REQUESTER_REPO": str(requester_repo(cfg)),
            }
            for command in wan_prepare_commands(cfg):
                run_shell_hook(command, cwd=repo_root, log_path=run_dir / "wan_prepare.log", env=env)

            update_status(run_dir / "sequence_status.json", status="running", phase="wan")
            wan_batch_dir = start_remote_matrix(
                cfg,
                network="WAN",
                run_dir=run_dir,
                counts=configured_instance_node_counts(cfg, "WAN"),
                no_plot=args.no_plot,
                log_path=log_path,
            )
            wan_status = wait_for_remote_batch(cfg, wan_batch_dir, poll_sec=args.poll_sec, status_path=run_dir / "sequence_status.json")
            if wan_status.get("status") != "completed":
                update_status(run_dir / "sequence_status.json", status="failed", phase="wan", wan_status=wan_status, lan_status=lan_status)
                exit_code = 1
            else:
                update_status(run_dir / "sequence_status.json", status="completed", phase="done", lan_status=lan_status, wan_status=wan_status)
    finally:
        sync_remote_results_back(cfg, repo_root, run_dir, log_path)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
