from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exo-backed distributed shard capture for THC/TSTC calibration"
    )
    parser.add_argument("--exo-root", default="~/repo/paper/third_party/exo")
    parser.add_argument("--cluster-file", required=True, help="JSON file listing cluster nodes in rank order")
    parser.add_argument("--local-node", required=True, help="Local node name matching one entry in cluster-file")
    parser.add_argument("--model-id", default="mlx-community/Qwen3-0.6B-8bit")
    parser.add_argument("--prompt-dataset", default="artifacts/_shared/thc/data/qwen_prompt_splits.jsonl")
    parser.add_argument("--split", default="calibration", choices=["calibration", "evaluation"])
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instance-id", default="thc-tstc-capture")
    parser.add_argument("--instructions", default="You are a helpful assistant.")
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def _ensure_exo_path(exo_root: str) -> None:
    exo_src = Path(exo_root).expanduser() / "src"
    if not exo_src.exists():
        raise FileNotFoundError(f"exo src path not found: {exo_src}")
    sys.path.insert(0, str(exo_src))


@dataclass
class CaptureContext:
    output_dir: Path
    prompt_id: str
    split: str
    node_name: str
    node_ip: str
    model_id: str
    world_size: int
    rank: int
    instance_id: str
    provider_plan: list[str]


_CURRENT_CAPTURE_CONTEXT: CaptureContext | None = None


def _set_capture_context(ctx: CaptureContext | None) -> None:
    global _CURRENT_CAPTURE_CONTEXT
    _CURRENT_CAPTURE_CONTEXT = ctx


def _checkpoint_name(rank: int) -> str:
    return f"C{rank + 1}"


def _write_local_capture(stage: str, checkpoint: str, array: np.ndarray) -> None:
    ctx = _CURRENT_CAPTURE_CONTEXT
    if ctx is None:
        return

    capture_dir = ctx.output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    npz_path = capture_dir / f"{ctx.prompt_id}.npz"
    metadata_path = ctx.output_dir / "checkpoint_metadata.jsonl"
    key = f"{stage}__{checkpoint}"

    existing: dict[str, np.ndarray] = {}
    if npz_path.exists():
        with np.load(npz_path) as data:
            existing = {name: data[name] for name in data.files}
    existing[key] = np.asarray(array, dtype=np.float32)
    np.savez_compressed(npz_path, **existing)

    row = {
        "prompt_id": ctx.prompt_id,
        "split": ctx.split,
        "stage": stage,
        "checkpoint": checkpoint,
        "shape": list(np.asarray(array).shape),
        "node_name": ctx.node_name,
        "node_ip": ctx.node_ip,
        "model_id": ctx.model_id,
        "world_size": ctx.world_size,
        "rank": ctx.rank,
        "instance_id": ctx.instance_id,
        "provider_plan": list(ctx.provider_plan),
        "runtime": "exo_pipeline",
        "capture_mode": "exo_pipeline",
    }
    with metadata_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _install_pipeline_capture_hook() -> None:
    import exo.worker.engines.mlx.auto_parallel as ap  # type: ignore

    if getattr(ap.PipelineLastLayer, "_thc_capture_installed", False):
        return

    def patched_call(self, x, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        cache = self.original_layer_signature.bind_partial(
            x, *args, **kwargs
        ).arguments.get("cache", None)

        output = self.original_layer(x, *args, **kwargs)
        stage = "prefill" if self.is_prefill else "decode"
        checkpoint = _checkpoint_name(int(self.r))
        captured = np.array(output.astype(ap.mx.float32), dtype=np.float32)
        _write_local_capture(stage, checkpoint, captured)

        if self.r != self.s - 1:
            output = ap.mx.distributed.send(output, (self.r + 1) % self.s, group=self.group)
            if cache is not None:
                cache.keys = ap.mx.depends(cache.keys, output)  # type: ignore[attr-defined]
            if self.is_prefill:
                ap.mx.eval(output)
                if cache is not None:
                    ap.mx.eval(cache.keys)  # type: ignore[arg-type]

        if not self.is_prefill:
            output = ap.mx.distributed.all_gather(output, group=self.group)[-output.shape[0] :]

        return output

    ap.PipelineLastLayer.__call__ = patched_call  # type: ignore[assignment]
    ap.PipelineLastLayer._thc_capture_installed = True


def _load_prompt_records(dataset_path: Path, split: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if str(row["split"]) != split:
                continue
            rows.append(
                {
                    "prompt_id": str(row["prompt_id"]),
                    "split": str(row["split"]),
                    "text": str(row["text"]),
                }
            )
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no prompts found for split={split}")
    return rows


def _load_cluster_nodes(cluster_file: Path) -> list[dict[str, Any]]:
    data = json.loads(cluster_file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("cluster file must be a non-empty JSON list")
    if len(data) != 3:
        raise ValueError("strict T2/T3 flow requires exactly 3 cluster nodes")
    nodes: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("each cluster node must be a JSON object")
        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"duplicate cluster node name: {name}")
        seen_names.add(name)
        ip = str(item["ip"])
        port = int(item.get("port", 52417))
        nodes.append({"name": name, "ip": ip, "port": port, "rank": index})
    return nodes


async def _load_model_card(model_id: str):  # type: ignore[no-untyped-def]
    from exo.shared.models.model_cards import ModelCard  # type: ignore
    from exo.shared.types.common import ModelId  # type: ignore

    return await ModelCard.load(ModelId(model_id))


def _ring_host_list(nodes: list[dict[str, Any]], local_rank: int):  # type: ignore[no-untyped-def]
    from exo.shared.types.common import Host  # type: ignore

    world_size = len(nodes)
    hosts = [Host(ip="198.51.100.0", port=52417) for _ in nodes]
    hosts[(local_rank - 1) % world_size] = Host(
        ip=nodes[(local_rank - 1) % world_size]["ip"],
        port=nodes[(local_rank - 1) % world_size]["port"],
    )
    hosts[(local_rank + 1) % world_size] = Host(
        ip=nodes[(local_rank + 1) % world_size]["ip"],
        port=nodes[(local_rank + 1) % world_size]["port"],
    )
    hosts[local_rank] = Host(ip="0.0.0.0", port=nodes[local_rank]["port"])
    return hosts


def _build_bound_instance(
    nodes: list[dict[str, Any]],
    local_node: dict[str, Any],
    model_id: str,
    instance_id: str,
    card: Any,
):
    from exo.shared.types.common import NodeId  # type: ignore
    from exo.shared.types.worker.instances import BoundInstance, InstanceId, MlxRingInstance  # type: ignore
    from exo.shared.types.worker.runners import RunnerId, ShardAssignments  # type: ignore
    from exo.shared.types.worker.shards import PipelineShardMetadata  # type: ignore

    world_size = len(nodes)
    if int(card.n_layers) % world_size != 0:
        raise ValueError(
            f"model n_layers={card.n_layers} is not divisible by world_size={world_size}"
        )
    runner_to_shard = {}
    node_to_runner = {}
    for node in nodes:
        i = int(node["rank"])
        runner_id = RunnerId(node["name"])
        shard = PipelineShardMetadata(
            model_card=card,
            device_rank=i,
            world_size=world_size,
            start_layer=(card.n_layers // world_size) * i,
            end_layer=min(card.n_layers, (card.n_layers // world_size) * (i + 1)),
            n_layers=min(card.n_layers, (card.n_layers // world_size) * (i + 1))
            - (card.n_layers // world_size) * i,
        )
        runner_to_shard[runner_id] = shard
        node_to_runner[NodeId(node["name"])] = runner_id

    instance = MlxRingInstance(
        instance_id=InstanceId(instance_id),
        ephemeral_port=int(local_node["port"]),
        hosts_by_node={NodeId(local_node["name"]): _ring_host_list(nodes, int(local_node["rank"]))},
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner=node_to_runner,
            runner_to_shard=runner_to_shard,
        ),
    )
    return BoundInstance(
        instance=instance,
        bound_runner_id=RunnerId(local_node["name"]),
        bound_node_id=NodeId(local_node["name"]),
    )


def _drain_generation(generator) -> list[str]:  # type: ignore[no-untyped-def]
    texts: list[str] = []
    for response in generator:
        text = getattr(response, "text", "")
        if text:
            texts.append(str(text))
    return texts


def main() -> None:
    args = _parse_args()
    _ensure_exo_path(args.exo_root)
    _install_pipeline_capture_hook()

    from exo.shared.types.common import ModelId  # type: ignore
    from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams  # type: ignore
    from exo.worker.engines.mlx.cache import KVPrefixCache  # type: ignore
    from exo.worker.engines.mlx.cache import encode_prompt  # type: ignore
    from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference  # type: ignore
    from exo.worker.engines.mlx.utils_mlx import apply_chat_template, initialize_mlx, load_mlx_items  # type: ignore

    cluster_nodes = _load_cluster_nodes(Path(args.cluster_file).expanduser().resolve())
    local_node = next((node for node in cluster_nodes if node["name"] == args.local_node), None)
    if local_node is None:
        raise ValueError(f"local node {args.local_node} not found in cluster file")

    prompt_dataset = Path(args.prompt_dataset).expanduser().resolve()
    prompt_records = _load_prompt_records(prompt_dataset, args.split, int(args.limit_prompts))
    model_card = asyncio.run(_load_model_card(args.model_id))
    bound_instance = _build_bound_instance(
        nodes=cluster_nodes,
        local_node=local_node,
        model_id=args.model_id,
        instance_id=args.instance_id,
        card=model_card,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    group = initialize_mlx(bound_instance)
    model, tokenizer = load_mlx_items(bound_instance, group)
    if args.warmup:
        warmup_inference(model=model, tokenizer=tokenizer, group=group)

    summary_rows: list[dict[str, Any]] = []
    for prompt in prompt_records:
        ctx = CaptureContext(
            output_dir=output_dir,
            prompt_id=prompt["prompt_id"],
            split=prompt["split"],
            node_name=str(local_node["name"]),
            node_ip=str(local_node["ip"]),
            model_id=args.model_id,
            world_size=len(cluster_nodes),
            rank=int(local_node["rank"]),
            instance_id=args.instance_id,
            provider_plan=[str(node["name"]) for node in cluster_nodes],
        )
        _set_capture_context(ctx)
        task = TextGenerationTaskParams(
            model=ModelId(args.model_id),
            instructions=str(args.instructions),
            input=[InputMessage(role="user", content=prompt["text"])],
            max_output_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
            stream=False,
            seed=int(args.seed),
        )
        rendered_prompt = apply_chat_template(tokenizer, task)
        prompt_tokens = encode_prompt(tokenizer, rendered_prompt)
        if len(prompt_tokens) - 1 > 8192:
            raise ValueError(
                f"prompt_id={prompt['prompt_id']} exceeds strict prefill bound: "
                f"{len(prompt_tokens) - 1} tokens > 8192"
            )
        generator = mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=rendered_prompt,
            kv_prefix_cache=KVPrefixCache(group),
            group=group,
        )
        generated = _drain_generation(generator)
        summary_rows.append(
            {
                "prompt_id": prompt["prompt_id"],
                "split": prompt["split"],
                "node_name": local_node["name"],
                "rank": local_node["rank"],
                "provider_plan": [str(node["name"]) for node in cluster_nodes],
                "runtime": "exo_pipeline",
                "generated_text": "".join(generated),
            }
        )
        _set_capture_context(None)

    with (output_dir / "capture_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=True)

    print(f"exo-backed capture complete: {output_dir}")


if __name__ == "__main__":
    main()
