from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from drift_search import (
    DEFAULT_CLUSTER_TEMPLATE,
    DEFAULT_CONFIG_TEMPLATE,
    DEFAULT_PYTHON_BIN,
    _build_variants,
    _has_nonzero_delta,
    _load_json,
    _run_calibration,
    _run_capture,
    _run_t5,
    _variant_config,
    _write_json,
)
from t3_server_supervisor import DEFAULT_MAC_HOST, T3ServerSupervisor, _resolve_ssh_host


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAC_ALIAS = "Mac3"
DEFAULT_LINUX_ALIAS = "3090"
DEFAULT_MAC_REPO_ROOT = "/Users/jlmini_3/repo/paper/bc-ra-paper-exp_verification"
DEFAULT_LINUX_REPO_ROOT = "/home/hzh/repo/paper/bc-ra-paper-exp_verification"
DEFAULT_MAC_PYTHON_BIN = "/Users/jlmini_3/repo/paper/bc-ra-paper/.venv/bin/python3"
DEFAULT_LINUX_PYTHON_BIN = "/home/hzh/repo/paper/bc-ra-paper/.venv/bin/python3"
COMPLETED_STATUSES = {"zero_delta", "t5_complete"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Managed unattended drift-search runner with server orchestration")
    parser.add_argument("--config-template", default=str(DEFAULT_CONFIG_TEMPLATE))
    parser.add_argument("--cluster-template", default=str(DEFAULT_CLUSTER_TEMPLATE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON_BIN))
    parser.add_argument("--mac-alias", default=DEFAULT_MAC_ALIAS)
    parser.add_argument("--linux-alias", default=DEFAULT_LINUX_ALIAS)
    parser.add_argument("--mac-repo-root", default=DEFAULT_MAC_REPO_ROOT)
    parser.add_argument("--linux-repo-root", default=DEFAULT_LINUX_REPO_ROOT)
    parser.add_argument("--mac-python-bin", default=DEFAULT_MAC_PYTHON_BIN)
    parser.add_argument("--linux-python-bin", default=DEFAULT_LINUX_PYTHON_BIN)
    parser.add_argument("--execute", choices=["true", "false"], default="true")
    parser.add_argument("--resume", choices=["true", "false"], default="true")
    parser.add_argument("--calibration-runs", type=int, default=3)
    parser.add_argument("--runs-per-mode", type=int, default=10)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--fallback-decode-steps", type=int, default=16)
    parser.add_argument("--local-port", type=int, default=18312)
    parser.add_argument("--linux-tunnel-port", type=int, default=18311)
    parser.add_argument("--ping-timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def _default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "artifacts/_shared/thc/output" / f"{stamp}_managed_drift_search"


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _remote_runtime_dir(output_dir: Path, variant_label: str, remote_repo_root: str) -> str:
    safe_label = variant_label.replace("/", "_")
    return str(Path(remote_repo_root) / "artifacts/_shared/thc/output/managed_runtime" / output_dir.name / safe_label)

def _run_local(cmd: List[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True)


def _runtime_cluster_payloads(
    variant_cluster: Dict[str, Any],
    *,
    mac_host: str,
    local_port: int,
    linux_tunnel_port: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    local_capture = json.loads(json.dumps(variant_cluster, ensure_ascii=True))
    remote_runtime = json.loads(json.dumps(variant_cluster, ensure_ascii=True))

    for node in local_capture["nodes"]:
        if str(node["node_name"]) == "jlmini_3":
            node["host"] = str(mac_host)
        elif str(node["node_name"]) == "jlmini_2":
            node["host"] = "127.0.0.1"
            node["port"] = int(local_port)
        elif str(node["node_name"]) == "linux124":
            node["host"] = "127.0.0.1"
            node["port"] = int(linux_tunnel_port)

    for node in remote_runtime["nodes"]:
        if str(node["node_name"]) == "jlmini_2":
            node["port"] = int(local_port)

    return local_capture, remote_runtime


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _completed(row: Dict[str, Any]) -> bool:
    return str(row.get("status", "")) in COMPLETED_STATUSES


def _cmd_error_text(exc: BaseException) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        return "\n".join(
            [
                f"command failed: {' '.join(exc.cmd) if isinstance(exc.cmd, list) else str(exc.cmd)}",
                exc.stdout.strip(),
                exc.stderr.strip(),
            ]
        ).strip()
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _prepare_variant_runtime(
    *,
    variant_dir: Path,
    variant: Dict[str, Any],
    config_template: Dict[str, Any],
    mac_host: str,
    local_port: int,
    linux_tunnel_port: int,
) -> Dict[str, Any]:
    runtime_dir = variant_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_file = runtime_dir / "config.json"
    capture_cluster_file = runtime_dir / "cluster_capture_local.json"
    remote_cluster_file = runtime_dir / "cluster_server_remote.json"

    _write_json(config_file, _variant_config(config_template, int(variant["decode_steps"])))
    capture_cluster, remote_cluster = _runtime_cluster_payloads(
        variant["cluster"],
        mac_host=mac_host,
        local_port=local_port,
        linux_tunnel_port=linux_tunnel_port,
    )
    _write_json(capture_cluster_file, capture_cluster)
    _write_json(remote_cluster_file, remote_cluster)
    return {
        "runtime_dir": runtime_dir,
        "config_file": config_file,
        "capture_cluster_file": capture_cluster_file,
        "remote_cluster_file": remote_cluster_file,
    }


def _reset_variant_outputs(variant_dir: Path, calibration_runs: int) -> None:
    for run_index in range(1, calibration_runs + 1):
        _clean_dir(variant_dir / f"capture_calibration_run_{run_index}")
    _clean_dir(variant_dir / "capture_evaluation")
    _clean_dir(variant_dir / "delta_calibration")


def main() -> None:
    args = _parse_args()
    execute = args.execute == "true"
    resume = args.resume == "true"
    config_template = _load_json(Path(args.config_template).expanduser().resolve())
    cluster_template = _load_json(Path(args.cluster_template).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "managed_drift_search_manifest.json"
    previous_manifest = _load_manifest(manifest_path) if resume else {}
    previous_rows = {str(row.get("label")): row for row in previous_manifest.get("variants", [])}
    chosen_variant = previous_manifest.get("chosen_variant")
    manifest_rows: List[Dict[str, Any]] = []
    mac_host = _resolve_ssh_host(args.mac_alias, DEFAULT_MAC_HOST)

    round_decode_steps = [int(args.decode_steps)]
    if int(args.fallback_decode_steps) > int(args.decode_steps):
        round_decode_steps.append(int(args.fallback_decode_steps))

    def flush_manifest() -> None:
        _write_manifest(
            manifest_path,
            {
                "output_dir": str(output_dir),
                "git_head": _git_head(),
                "execute": execute,
                "resume": resume,
                "config_template": str(Path(args.config_template).expanduser().resolve()),
                "cluster_template": str(Path(args.cluster_template).expanduser().resolve()),
                "chosen_variant": chosen_variant,
                "variants": manifest_rows,
            },
        )

    supervisor = T3ServerSupervisor(
        local_python_bin=str(args.python_bin),
        mac_alias=args.mac_alias,
        linux_alias=args.linux_alias,
        mac_repo_root=args.mac_repo_root,
        linux_repo_root=args.linux_repo_root,
        mac_python_bin=args.mac_python_bin,
        linux_python_bin=args.linux_python_bin,
        local_port=int(args.local_port),
        linux_tunnel_port=int(args.linux_tunnel_port),
        ping_timeout_seconds=float(args.ping_timeout_seconds),
    )

    try:
        for round_index, decode_steps in enumerate(round_decode_steps, start=1):
            variants = _build_variants(cluster_template, decode_steps)
            round_nonzero = False
            for variant_index, variant in enumerate(variants, start=1):
                variant_dir = output_dir / f"{round_index:02d}_{variant_index:02d}_{variant['label']}"
                runtime = _prepare_variant_runtime(
                    variant_dir=variant_dir,
                    variant=variant,
                    config_template=config_template,
                    mac_host=mac_host,
                    local_port=int(args.local_port),
                    linux_tunnel_port=int(args.linux_tunnel_port),
                )
                existing = previous_rows.get(str(variant["label"]))
                if existing and _completed(existing):
                    manifest_rows.append(existing)
                    if str(existing.get("status")) == "t5_complete":
                        chosen_variant = existing
                        round_nonzero = True
                        break
                    flush_manifest()
                    continue

                _reset_variant_outputs(variant_dir, int(args.calibration_runs))

                remote_variant_dir_mac = _remote_runtime_dir(output_dir, str(variant["label"]), args.mac_repo_root)
                remote_variant_dir_linux = _remote_runtime_dir(output_dir, str(variant["label"]), args.linux_repo_root)
                remote_cluster_path_mac = str(Path(remote_variant_dir_mac) / "runtime/cluster_server_remote.json")
                remote_cluster_path_linux = str(Path(remote_variant_dir_linux) / "runtime/cluster_server_remote.json")
                local_log = variant_dir / "server_logs" / "jlmini_2.log"
                mac_log = str(Path(remote_variant_dir_mac) / "server_logs/jlmini_3.log")
                linux_log = str(Path(remote_variant_dir_linux) / "server_logs/linux124.log")

                row: Dict[str, Any] = {
                    "round": round_index,
                    "label": variant["label"],
                    "decode_steps": decode_steps,
                    "placement": variant["placement"],
                    "variant_dir": str(variant_dir),
                    "config_file": str(runtime["config_file"]),
                    "capture_cluster_file": str(runtime["capture_cluster_file"]),
                    "remote_cluster_file_mac": remote_cluster_path_mac,
                    "remote_cluster_file_linux": remote_cluster_path_linux,
                    "server_logs": {
                        "jlmini_2": str(local_log),
                        "jlmini_3": mac_log,
                        "linux124": linux_log,
                    },
                    "status": "prepared",
                }
                manifest_rows.append(row)
                flush_manifest()

                if not execute:
                    continue

                try:
                    row["status"] = "starting_servers"
                    flush_manifest()
                    row["server_pids"] = supervisor.up(
                        local_cluster_file=runtime["remote_cluster_file"],
                        remote_cluster_file_local=runtime["remote_cluster_file"],
                        remote_cluster_path_mac=remote_cluster_path_mac,
                        remote_cluster_path_linux=remote_cluster_path_linux,
                        local_log=local_log,
                        mac_log=mac_log,
                        linux_log=linux_log,
                    )

                    row["status"] = "running_calibration"
                    row["capture_roots"] = []
                    flush_manifest()
                    capture_roots: List[Path] = []
                    for run_index in range(1, int(args.calibration_runs) + 1):
                        capture_root = variant_dir / f"capture_calibration_run_{run_index}"
                        _run_capture(
                            str(args.python_bin),
                            runtime["config_file"],
                            runtime["capture_cluster_file"],
                            "calibration",
                            capture_root,
                        )
                        capture_roots.append(capture_root)
                        row["capture_roots"].append(str(capture_root))
                        flush_manifest()

                    row["status"] = "running_delta_calibration"
                    flush_manifest()
                    delta_map_file = _run_calibration(str(args.python_bin), variant_dir / "delta_calibration", capture_roots)
                    nonzero, max_delta = _has_nonzero_delta(delta_map_file)
                    row["delta_map_file"] = str(delta_map_file)
                    row["max_delta"] = max_delta

                    if nonzero:
                        row["status"] = "running_evaluation"
                        flush_manifest()
                        eval_root = variant_dir / "capture_evaluation"
                        _run_capture(
                            str(args.python_bin),
                            runtime["config_file"],
                            runtime["capture_cluster_file"],
                            "evaluation",
                            eval_root,
                        )
                        row["evaluation_capture_root"] = str(eval_root)
                        row["status"] = "running_t5"
                        flush_manifest()
                        row["t5_run_dir"] = _run_t5(
                            str(args.python_bin),
                            runtime["config_file"],
                            eval_root,
                            delta_map_file,
                            int(args.runs_per_mode),
                        )
                        row["status"] = "t5_complete"
                        chosen_variant = row
                        round_nonzero = True
                        flush_manifest()
                        break

                    row["status"] = "zero_delta"
                    flush_manifest()
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = _cmd_error_text(exc)
                    row["traceback"] = "".join(traceback.format_exception(exc))
                    flush_manifest()
                    raise
                finally:
                    supervisor.down()

            if round_nonzero:
                break
    finally:
        supervisor.down()
        flush_manifest()

    print(f"Managed drift search manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
