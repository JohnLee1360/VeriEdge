from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from hetero_qwen_common import resolve_pretrained_source

PREFILL_STAGE = "prefill"
DECODE_STAGE = "decode"
STAGES = (PREFILL_STAGE, DECODE_STAGE)

_REAL_CAPTURE_CACHE: Dict[str, Tuple[Dict[str, Dict[str, np.ndarray]], str]] = {}
_MLX_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_TORCH_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def expand_model_path(model_id: str) -> str:
    return str(Path(model_id).expanduser())


def load_prompt_records(config: Dict[str, Any], split: str | None = None) -> List[Dict[str, str]]:
    dataset_path = Path(str(config["experiment"]["prompt_dataset"]))
    rows: List[Dict[str, str]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            rows.append(
                {
                    "prompt_id": str(row["prompt_id"]),
                    "split": str(row["split"]),
                    "text": str(row["text"]),
                }
            )

    wanted = split or str(config["experiment"].get("default_split", "evaluation"))
    return [row for row in rows if row["split"] == wanted]


def checkpoint_order(config: Dict[str, Any]) -> List[str]:
    return [str(shard["checkpoint"]) for shard in config["experiment"]["shards"]]


def checkpoint_provider_map(config: Dict[str, Any]) -> Dict[str, str]:
    return {str(shard["checkpoint"]): str(shard["provider"]) for shard in config["experiment"]["shards"]}


def active_stage_families(config: Dict[str, Any]) -> List[str]:
    exp_cfg = dict(config.get("experiment", {}))
    configured = exp_cfg.get("active_stages", [])
    if configured:
        stages = [str(stage).strip().lower() for stage in configured if str(stage).strip()]
        filtered = [stage for stage in stages if stage in STAGES]
        if filtered:
            return filtered
    return [PREFILL_STAGE, DECODE_STAGE]


def decode_num_steps(config: Dict[str, Any]) -> int:
    if DECODE_STAGE not in active_stage_families(config):
        return 0
    probe_cfg = dict(config["experiment"].get("decode_probe", {}))
    return max(0, int(probe_cfg.get("num_steps", 1)))


def decode_stage_key(step_index: int, total_steps: int) -> str:
    if total_steps <= 1:
        return DECODE_STAGE
    return f"decode_s{int(step_index):02d}"


def stage_family(stage_key: str) -> str:
    return DECODE_STAGE if str(stage_key).startswith("decode") else PREFILL_STAGE


def stage_decode_step(stage_key: str) -> int:
    normalized = str(stage_key)
    if normalized == DECODE_STAGE:
        return 1
    if normalized.startswith("decode_s"):
        return int(normalized.split("decode_s", 1)[1])
    return 0


def ordered_stage_keys(stage_keys: Iterable[str]) -> List[str]:
    return sorted(
        [str(value) for value in stage_keys],
        key=lambda key: (
            0 if stage_family(key) == PREFILL_STAGE else 1,
            stage_decode_step(key),
            key,
        ),
    )


def _cache_key(config: Dict[str, Any], prompt_text: str) -> str:
    key_obj = {
        "prompt": prompt_text,
        "shards": config["experiment"]["shards"],
        "model_id": config["qwen"]["model_id"],
        "active_stages": active_stage_families(config),
        "decode_num_steps": decode_num_steps(config),
        "enable_mlx": config["qwen"].get("enable_mlx", True),
        "enable_torch_fallback": config["qwen"].get("enable_torch_fallback", False),
        "local_files_only": config["qwen"].get("local_files_only", True),
        "download_if_missing": config["qwen"].get("download_if_missing", False),
    }
    return json.dumps(key_obj, sort_keys=True, ensure_ascii=True)


def _copy_bundle(bundle: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        stage: {checkpoint: np.array(tensor, copy=True) for checkpoint, tensor in stage_map.items()}
        for stage, stage_map in bundle.items()
    }


def clone_checkpoint_bundle(bundle: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    return _copy_bundle(bundle)


def _resolve_probe_token_id(token_ids: List[int], config: Dict[str, Any]) -> int:
    probe_cfg = dict(config["experiment"].get("decode_probe", {}))
    strategy = str(probe_cfg.get("strategy", "repeat_last_prompt_token"))
    if strategy == "repeat_last_prompt_token" and token_ids:
        return int(token_ids[-1])
    return int(probe_cfg.get("fallback_token_id", 0))


def _mock_capture(config: Dict[str, Any], prompt_text: str, seed: int) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
    qwen_cfg = dict(config["qwen"])
    hidden_dim = int(qwen_cfg.get("mock_hidden_dim", 128))
    prompt_len = max(4, min(32, len(prompt_text.split()) + 2))
    stages = active_stage_families(config)
    num_decode_steps = decode_num_steps(config)
    rng = np.random.default_rng(seed)

    prefill_state = rng.normal(0.0, 1.0, size=(1, prompt_len, hidden_dim)).astype(np.float32)
    decode_step_inputs = [prefill_state[:, -1:, :].copy() for _ in range(num_decode_steps)]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if PREFILL_STAGE in stages:
        out[PREFILL_STAGE] = {}
    if DECODE_STAGE in stages:
        for step in range(1, num_decode_steps + 1):
            out[decode_stage_key(step, num_decode_steps)] = {}

    for shard_idx, shard in enumerate(config["experiment"]["shards"]):
        checkpoint = str(shard["checkpoint"])
        scale = 0.08 + 0.015 * shard_idx
        w_prefill = rng.normal(0.0, scale, size=(hidden_dim, hidden_dim)).astype(np.float32)
        w_decode = rng.normal(0.0, scale, size=(hidden_dim, hidden_dim)).astype(np.float32)

        prefill_state = np.tanh(prefill_state @ w_prefill).astype(np.float32)
        if PREFILL_STAGE in stages:
            out[PREFILL_STAGE][checkpoint] = np.array(prefill_state, copy=True)

        if DECODE_STAGE in stages:
            decode_step_states: List[np.ndarray] = []
            for step in range(1, num_decode_steps + 1):
                step_state = np.tanh(np.asarray(decode_step_inputs[step - 1], dtype=np.float32) @ w_decode).astype(np.float32)
                decode_step_states.append(np.array(step_state, copy=True))
                out[decode_stage_key(step, num_decode_steps)][checkpoint] = np.array(step_state, copy=True)
            decode_step_inputs = decode_step_states

    return out, "qwen_mock"


def _mlx_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[Any, Any]:
    qwen_cfg = dict(config["qwen"])
    model_ref = expand_model_path(str(qwen_cfg["model_id"]))
    hf_endpoint = str(qwen_cfg.get("hf_endpoint", "")).strip()
    local_files_only = bool(qwen_cfg.get("local_files_only", True))
    download_if_missing = bool(qwen_cfg.get("download_if_missing", False))

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    cache_key = json.dumps(
        {
            "model_ref": model_ref,
            "local_files_only": local_files_only,
            "download_if_missing": download_if_missing,
        },
        sort_keys=True,
    )
    cached = _MLX_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if download_if_missing:
        from huggingface_hub import snapshot_download  # type: ignore

        model_ref = snapshot_download(
            repo_id=model_ref,
            local_files_only=local_files_only,
            resume_download=True,
        )

    from mlx_lm import load as mlx_load  # type: ignore

    model, tokenizer = mlx_load(model_ref)
    _MLX_MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def _capture_mlx(config: Dict[str, Any], prompt_text: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
    import mlx.core as mx  # type: ignore
    from mlx_lm.models.base import create_attention_mask  # type: ignore

    model, tokenizer = _mlx_model_and_tokenizer(config)
    shards = list(config["experiment"]["shards"])
    token_ids = tokenizer.encode(prompt_text)
    if not token_ids:
        raise ValueError("tokenized prompt is empty")

    stages = active_stage_families(config)
    probe_token_id = _resolve_probe_token_id(token_ids, config)
    num_decode_steps = decode_num_steps(config)
    capture_layers = {int(shard["end_layer"]): str(shard["checkpoint"]) for shard in shards}
    last_layer = max(capture_layers)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    def _run(tokens: List[int], stage_key: str, family: str) -> None:
        inputs = mx.array([tokens], dtype=mx.int32)
        hidden = model.model.embed_tokens(inputs)
        mask = create_attention_mask(hidden, cache=None)
        for layer_idx, layer in enumerate(model.layers):
            hidden = layer(hidden, mask, None)
            checkpoint = capture_layers.get(layer_idx)
            if checkpoint is None:
                if layer_idx >= last_layer:
                    break
                continue
            captured = hidden if family == PREFILL_STAGE else hidden[:, -1:, :]
            out.setdefault(stage_key, {})[checkpoint] = np.array(captured.astype(mx.float32), dtype=np.float32)
            if layer_idx >= last_layer:
                break

    if PREFILL_STAGE in stages:
        _run(token_ids, PREFILL_STAGE, PREFILL_STAGE)
    if DECODE_STAGE in stages:
        for step in range(1, num_decode_steps + 1):
            stage_key = decode_stage_key(step, num_decode_steps)
            _run(token_ids + [probe_token_id] * step, stage_key, DECODE_STAGE)
    return out, "qwen_mlx_real"


def _torch_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[Any, Any]:
    qwen_cfg = dict(config["qwen"])
    model_ref = expand_model_path(str(qwen_cfg["model_id"]))
    local_files_only = bool(qwen_cfg.get("local_files_only", True))
    model_source = resolve_pretrained_source(model_ref, local_files_only, str(qwen_cfg.get("hf_endpoint", "")).strip())
    cache_key = json.dumps({"model_ref": model_ref, "local_files_only": local_files_only}, sort_keys=True)
    cached = _TORCH_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype="auto",
        local_files_only=local_files_only,
    )
    model.eval()
    _TORCH_MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def _capture_torch(config: Dict[str, Any], prompt_text: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
    import torch  # type: ignore

    model, tokenizer = _torch_model_and_tokenizer(config)
    shards = list(config["experiment"]["shards"])
    capture_layers = {int(shard["end_layer"]): str(shard["checkpoint"]) for shard in shards}
    token_ids = tokenizer.encode(prompt_text)
    if not token_ids:
        raise ValueError("tokenized prompt is empty")
    stages = active_stage_families(config)
    probe_token_id = _resolve_probe_token_id(token_ids, config)
    num_decode_steps = decode_num_steps(config)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    def _run(tokens: List[int], stage_key: str, family: str) -> None:
        with torch.no_grad():
            encoded = {"input_ids": torch.tensor([tokens], dtype=torch.long)}
            outputs = model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        for layer_idx, checkpoint in capture_layers.items():
            hs = hidden_states[layer_idx + 1]
            captured = hs if family == PREFILL_STAGE else hs[:, -1:, :]
            # Torch may emit bfloat16 hidden states on Apple backends; convert before NumPy export.
            out.setdefault(stage_key, {})[checkpoint] = captured.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32)

    if PREFILL_STAGE in stages:
        _run(token_ids, PREFILL_STAGE, PREFILL_STAGE)
    if DECODE_STAGE in stages:
        for step in range(1, num_decode_steps + 1):
            stage_key = decode_stage_key(step, num_decode_steps)
            _run(token_ids + [probe_token_id] * step, stage_key, DECODE_STAGE)
    return out, "qwen_torch_real"


def capture_qwen_checkpoints(
    config: Dict[str, Any],
    prompt_record: Dict[str, str],
    seed: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[Dict[str, Any]], str]:
    qwen_cfg = dict(config["qwen"])
    allow_mock = bool(qwen_cfg.get("allow_mock_fallback", qwen_cfg.get("use_mock_if_unavailable", True)))
    use_mlx = bool(qwen_cfg.get("enable_mlx", True))
    use_torch = bool(qwen_cfg.get("enable_torch_fallback", False))
    prompt_text = str(prompt_record["text"])

    if not allow_mock:
        cache_key = _cache_key(config, prompt_text)
        cached = _REAL_CAPTURE_CACHE.get(cache_key)
        if cached is not None:
            bundle, runtime = cached
            return _copy_bundle(bundle), checkpoint_metadata(config, prompt_record, bundle, runtime), runtime

    exc: Exception | None = None
    bundle: Dict[str, Dict[str, np.ndarray]]
    runtime = ""
    if use_mlx:
        try:
            bundle, runtime = _capture_mlx(config, prompt_text)
            if not allow_mock:
                _REAL_CAPTURE_CACHE[_cache_key(config, prompt_text)] = (_copy_bundle(bundle), runtime)
            return _copy_bundle(bundle), checkpoint_metadata(config, prompt_record, bundle, runtime), runtime
        except Exception as error:  # pragma: no cover - backend-specific path
            exc = error

    if use_torch:
        try:
            bundle, runtime = _capture_torch(config, prompt_text)
            if not allow_mock:
                _REAL_CAPTURE_CACHE[_cache_key(config, prompt_text)] = (_copy_bundle(bundle), runtime)
            return _copy_bundle(bundle), checkpoint_metadata(config, prompt_record, bundle, runtime), runtime
        except Exception as error:  # pragma: no cover - backend-specific path
            exc = error

    if not allow_mock:
        raise RuntimeError(f"failed to capture real qwen checkpoints: {exc}") from exc

    bundle, runtime = _mock_capture(config, prompt_text, seed)
    return _copy_bundle(bundle), checkpoint_metadata(config, prompt_record, bundle, runtime), runtime


def checkpoint_metadata(
    config: Dict[str, Any],
    prompt_record: Dict[str, str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    runtime: str,
) -> List[Dict[str, Any]]:
    provider_map = checkpoint_provider_map(config)
    rows: List[Dict[str, Any]] = []
    for stage_key in ordered_stage_keys(bundle.keys()):
        family = stage_family(stage_key)
        decode_step = stage_decode_step(stage_key)
        for checkpoint in checkpoint_order(config):
            tensor = bundle[stage_key][checkpoint]
            rows.append(
                {
                    "prompt_id": str(prompt_record["prompt_id"]),
                    "split": str(prompt_record["split"]),
                    "stage": family,
                    "stage_key": stage_key,
                    "decode_step": decode_step,
                    "checkpoint": checkpoint,
                    "shape": list(np.asarray(tensor).shape),
                    "provider_plan": list(config["experiment"]["shard_plan"]),
                    "provider": provider_map[checkpoint],
                    "runtime": runtime,
                }
            )
    return rows


def write_capture_bundle(
    output_dir: Path,
    prompt_record: Dict[str, str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    metadata_rows: Iterable[Dict[str, Any]],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{prompt_record['prompt_id']}.npz"
    meta_path = output_dir / "checkpoint_metadata.jsonl"
    np.savez_compressed(
        npz_path,
        **{
            f"{stage_key}__{checkpoint}": np.asarray(tensor, dtype=np.float32)
            for stage_key, stage_map in bundle.items()
            for checkpoint, tensor in stage_map.items()
        },
    )
    with meta_path.open("a", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return npz_path, meta_path


def load_capture_bundle_for_prompt(
    capture_root: Path,
    prompt_id: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[Dict[str, Any]], str]:
    capture_dir = Path(capture_root) / "captures"
    npz_path = capture_dir / f"{prompt_id}.npz"
    if not npz_path.exists():
        raise ValueError(f"capture npz not found for prompt_id={prompt_id}: {npz_path}")

    bundle: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(npz_path) as data:
        for key in data.files:
            stage_key, checkpoint = key.split("__", 1)
            bundle.setdefault(stage_key, {})[checkpoint] = np.asarray(data[key], dtype=np.float32)

    metadata_rows: List[Dict[str, Any]] = []
    runtime = "capture_root"
    meta_path = capture_dir / "checkpoint_metadata.jsonl"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if str(row.get("prompt_id", "")) != str(prompt_id):
                    continue
                metadata_rows.append(dict(row))
        if metadata_rows:
            runtime = str(metadata_rows[0].get("runtime", runtime))

    return _copy_bundle(bundle), metadata_rows, runtime
