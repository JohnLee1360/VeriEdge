#!/usr/bin/env python3
"""Black-box readiness check for external EXO + provider launchers + Kubo."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

RUNNER_PATH = Path(__file__).resolve().with_name("runner.py")
LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import runner as runner_mod  # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether external EXO is ready for one cell")
    parser.add_argument("--config", default="artifacts/E5/inference_e2e/requester/config.example.yaml")
    parser.add_argument("--instance-node-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    entry_url = str(cfg["endpoints"]["entry_url"])
    requester_callback_url = str(cfg["endpoints"]["requester_callback_url"])
    providers_cfg = list(cfg.get("providers", []))
    provider_urls = [f"http://{item['ip']}:{int(item.get('exo_port', 52415))}" for item in providers_cfg]
    launcher_urls = [str(item["launcher_url"]) for item in providers_cfg]
    timeout_sec = int(cfg.get("runtime", {}).get("timeout_sec", 180))
    cluster_stability = runner_mod.runtime_cluster_stability_config(cfg)
    model_id = str(cfg["model"]["model_id"])
    ipfs_cfg = cfg["ipfs"]

    report: Dict[str, Any] = {"ok": False}
    try:
        preflight = asyncio.run(
            runner_mod.collect_preflight_checks(
                entry_url=entry_url,
                provider_urls=provider_urls,
                requester_callback_url=requester_callback_url,
                ipfs_api_url=str(ipfs_cfg["api_url"]),
                ipfs_gateway_url=str(ipfs_cfg["gateway_url"]),
                launcher_urls=launcher_urls,
                timeout_sec=timeout_sec,
                include_requester_callback=False,
            )
        )
        report["health_checks"] = preflight["health_checks"]
        report["aux_checks"] = preflight["aux_checks"]
        runner_mod.ensure_preflight_checks_ok(
            health_checks=report["health_checks"],
            aux_checks=report["aux_checks"],
        )
        report["state_views"] = asyncio.run(
            runner_mod.collect_cluster_state_views(
                provider_urls=provider_urls,
                timeout_sec=timeout_sec,
            )
        )
        runner_mod.ensure_cluster_not_isolated(
            state_views=report["state_views"],
            expected_provider_count=len(provider_urls),
        )
        validated, state = asyncio.run(
            runner_mod.wait_for_validated_external_instance_state(
                entry_url=entry_url,
                timeout_sec=int(cluster_stability["timeout_sec"]),
                model_id=model_id,
                expected_instance_node_count=int(args.instance_node_count),
                expected_instance_ids=runner_mod.expected_instance_ids_for_node_count(cfg, int(args.instance_node_count)),
                provider_urls=provider_urls,
                poll_sec=int(cluster_stability["poll_sec"]),
                consecutive_successes=int(cluster_stability["consecutive_successes"]),
            )
        )
        report["state_summary"] = {
            "selected_instance_id": validated["selected_instance_id"],
            "ordered_instances": validated["ordered_instances"],
            "first_shard_node_id": validated["first_shard_node_id"],
            "first_shard_ip": validated["first_shard_ip"],
            "model_instances_summary": validated["model_instances_summary"],
            "state_ips": validated["state_ips"],
            "stability": cluster_stability,
            "raw_state": state,
        }
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
