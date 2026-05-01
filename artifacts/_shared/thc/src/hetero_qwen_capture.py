from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from checkpoint_qwen import PREFILL_STAGE, DECODE_STAGE, active_stage_families, decode_num_steps, decode_stage_key, write_capture_bundle
from hetero_cluster import cluster_nodes_from_config, load_cluster_config
from hetero_qwen_common import configure_hf_endpoint, load_json, load_prompt_records, resolve_pretrained_source
from hetero_transport import decode_array, post_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Qwen checkpoints over a heterogeneous 3-node shard chain")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cluster-file", required=True)
    parser.add_argument("--split", default="calibration", choices=["calibration", "evaluation"])
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--probe-strategy", default="repeat_last_prompt_token")
    parser.add_argument("--decode-steps", type=int, default=0)
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    return load_json(path)


def _load_cluster(path: Path) -> List[Dict[str, Any]]:
    data = load_cluster_config(path)
    return cluster_nodes_from_config(data)


def _ping_node(node: Dict[str, Any]) -> Dict[str, Any]:
    return post_json(f"http://{node['host']}:{node['port']}/ping", {})


def _reset_session(nodes: List[Dict[str, Any]], session_id: str) -> None:
    for node in nodes:
        post_json(
            f"http://{node['host']}:{node['port']}/reset_session",
            {"session_id": session_id},
        )


def _resolve_probe_token_id(token_ids: List[int], probe_strategy: str, fallback: int) -> int:
    if probe_strategy == "repeat_last_prompt_token" and token_ids:
        return int(token_ids[-1])
    return int(fallback)


def _stage_positions(token_count: int, decode: bool, decode_step: int = 0) -> tuple[list[int], list[int]]:
    if decode:
        idx = token_count + int(decode_step) - 1
        return [idx], [idx]
    values = list(range(token_count))
    return values, values


def _checkpoint_row(
    *,
    prompt: Dict[str, str],
    stage: str,
    response: Dict[str, Any],
    provider_plan: List[str],
) -> Dict[str, Any]:
    return {
        "prompt_id": str(prompt["prompt_id"]),
        "split": str(prompt["split"]),
        "stage": "decode" if stage.startswith("decode") else "prefill",
        "stage_key": stage,
        "decode_step": int(response.get("decode_step", 0)),
        "checkpoint": str(response["checkpoint"]),
        "shape": list(response["shape"]),
        "provider_plan": provider_plan,
        "provider": str(response["node_name"]),
        "runtime": "hetero_qwen_torch_chain",
        "backend": str(response["backend"]),
        "device": str(response["device"]),
        "tensor_dtype": "float32",
    }


def main() -> None:
    args = _parse_args()
    config = _load_config(Path(args.config).expanduser().resolve())
    cluster_nodes = _load_cluster(Path(args.cluster_file).expanduser().resolve())
    provider_plan = [str(node["node_name"]) for node in cluster_nodes]
    dataset_path = Path(str(config["experiment"]["prompt_dataset"])).expanduser().resolve()
    prompts = load_prompt_records(dataset_path, args.split)
    if args.limit_prompts > 0:
        prompts = prompts[: int(args.limit_prompts)]
    if not prompts:
        raise ValueError(f"no prompts found for split={args.split}")

    qwen_cfg = dict(config["qwen"])
    model_id = str(qwen_cfg.get("hetero_model_id", qwen_cfg.get("torch_model_id", "Qwen/Qwen3-0.6B")))
    local_files_only = bool(qwen_cfg.get("hetero_local_files_only", False))
    trust_remote_code = bool(qwen_cfg.get("hetero_trust_remote_code", False))
    fallback_token = int(dict(config["experiment"].get("decode_probe", {})).get("fallback_token_id", 151643))
    active_stages = active_stage_families(config)
    decode_steps = int(args.decode_steps) if int(args.decode_steps) > 0 else decode_num_steps(config)
    hf_endpoint = str(qwen_cfg.get("hetero_hf_endpoint", "")).strip()
    configure_hf_endpoint(hf_endpoint)
    model_source = resolve_pretrained_source(model_id, local_files_only, hf_endpoint)

    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )

    for node in cluster_nodes:
        _ping_node(node)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path(str(config["experiment"]["output_root"])).expanduser().resolve() / f"{timestamp}_hetero_capture_{args.split}")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    export_dir = output_dir / "captures"
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        session_id = uuid.uuid4().hex
        try:
            token_ids = tokenizer.encode(str(prompt["text"]))
            if not token_ids:
                raise ValueError(f"tokenized prompt is empty for prompt_id={prompt['prompt_id']}")

            probe_token_id = _resolve_probe_token_id(token_ids, str(args.probe_strategy), fallback_token)
            bundle: Dict[str, Dict[str, Any]] = {}
            metadata_rows: List[Dict[str, Any]] = []

            stage_specs: List[tuple[str, bool, int, List[int]]] = []
            if PREFILL_STAGE in active_stages:
                stage_specs.append((PREFILL_STAGE, False, 0, token_ids))
            if DECODE_STAGE in active_stages:
                for step in range(1, decode_steps + 1):
                    stage_specs.append((decode_stage_key(step, decode_steps), True, step, [probe_token_id]))

            for stage, decode, decode_step, stage_tokens in stage_specs:
                bundle.setdefault(stage, {})
                position_ids, cache_position = _stage_positions(len(token_ids), decode, decode_step)
                request_stage = "decode" if decode else stage
                hidden_payload = None
                for index, node in enumerate(cluster_nodes):
                    payload = {
                        "session_id": session_id,
                        "position_ids": position_ids,
                        "cache_position": cache_position,
                        "input_ids": stage_tokens if index == 0 else None,
                        "hidden_states": hidden_payload,
                    }
                    response = post_json(f"http://{node['host']}:{node['port']}/{request_stage}", payload)
                    if not response.get("ok", False):
                        raise RuntimeError(f"node {node['node_name']} failed: {response}")
                    tensor = decode_array(dict(response["tensor"]))
                    response["decode_step"] = decode_step if decode else 0
                    bundle[stage][str(response["checkpoint"])] = tensor
                    metadata_rows.append(
                        _checkpoint_row(
                            prompt=prompt,
                            stage=stage,
                            response=response,
                            provider_plan=provider_plan,
                        )
                    )
                    hidden_payload = response["tensor"]

            npz_path, meta_path = write_capture_bundle(export_dir, prompt, bundle, metadata_rows)
            summary_rows.append(
                {
                    "prompt_id": str(prompt["prompt_id"]),
                    "split": str(prompt["split"]),
                    "runtime": "hetero_qwen_torch_chain",
                    "decode_steps": decode_steps,
                    "provider_plan": provider_plan,
                    "npz_path": str(npz_path),
                    "metadata_path": str(meta_path),
                }
            )
        finally:
            _reset_session(cluster_nodes, session_id)

    with (output_dir / "capture_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=True)

    print(f"Captured {len(summary_rows)} prompt(s) to {output_dir}")


if __name__ == "__main__":
    main()
