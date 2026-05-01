from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import torch


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_records(dataset_path: Path, split: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
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
    if not rows:
        raise ValueError(f"no prompts found for split={split}")
    return rows


def configure_hf_endpoint(explicit: str = "") -> str:
    endpoint = (
        explicit.strip()
        or os.environ.get("THC_HETERO_HF_ENDPOINT", "").strip()
        or os.environ.get("BC_RA_HF_ENDPOINT", "").strip()
        or os.environ.get("HF_ENDPOINT", "").strip()
        or "https://hf-mirror.com"
    )
    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint


@lru_cache(maxsize=None)
def resolve_pretrained_source(model_ref: str, local_files_only: bool, hf_endpoint: str = "") -> str:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if not local_files_only:
        return model_ref

    configure_hf_endpoint(hf_endpoint)

    from huggingface_hub import snapshot_download

    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_ref,
                local_files_only=True,
            )
        ).resolve()
    except Exception as exc:
        raise FileNotFoundError(
            f"local_files_only=true but cached snapshot for {model_ref!r} was not found"
        ) from exc
    return str(snapshot_path)


def parse_torch_dtype(name: str) -> Any:
    import torch

    normalized = name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported torch dtype: {name}")
    return mapping[normalized]


def select_torch_device(preference: str = "auto") -> Any:
    import torch

    normalized = preference.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")
    if normalized == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but not available")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device preference: {preference}")


def load_qwen_model(
    model_id: str,
    device: Any,
    dtype: Any,
    local_files_only: bool,
    trust_remote_code: bool,
    quantization: str,
    quantization_bits: int,
    quantization_group_size: int,
    hf_endpoint: str = "",
) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    configure_hf_endpoint(hf_endpoint)
    model_source = resolve_pretrained_source(model_id, local_files_only, hf_endpoint)
    config = AutoConfig.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if getattr(config, "model_type", "") == "qwen3" and getattr(config, "tie_word_embeddings", False):
        config.tie_word_embeddings = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    normalized_quant = quantization.strip().lower()
    load_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if normalized_quant in {"", "none"}:
        load_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **load_kwargs,
        )
        model.eval()
        model.to(device)
        return model, tokenizer

    if normalized_quant == "metal_8bit":
        if device.type != "mps":
            raise ValueError("metal_8bit quantization requires device=mps")
        from transformers import MetalConfig

        load_kwargs["device_map"] = "mps"
        load_kwargs["dtype"] = "auto"
        load_kwargs["quantization_config"] = MetalConfig(
            bits=int(quantization_bits),
            group_size=int(quantization_group_size),
        )
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        model.eval()
        return model, tokenizer

    if normalized_quant == "bitsandbytes_8bit":
        if device.type != "cuda":
            raise ValueError("bitsandbytes_8bit quantization requires device=cuda")
        from transformers import BitsAndBytesConfig

        load_kwargs["device_map"] = "auto"
        load_kwargs["dtype"] = "auto"
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
        model.eval()
        return model, tokenizer

    raise ValueError(f"unsupported quantization mode: {quantization}")


class QwenShardRunner:
    def __init__(
        self,
        *,
        model_id: str,
        start_layer: int,
        end_layer: int,
        checkpoint: str,
        is_first: bool,
        is_last: bool,
        device: Any,
        dtype: Any,
        local_files_only: bool,
        trust_remote_code: bool,
        quantization: str = "none",
        quantization_bits: int = 8,
        quantization_group_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.checkpoint = checkpoint
        self.is_first = bool(is_first)
        self.is_last = bool(is_last)
        self.device = device
        self.dtype = dtype
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self.quantization = str(quantization)
        self.quantization_bits = int(quantization_bits)
        self.quantization_group_size = int(quantization_group_size)
        self.model, self.tokenizer = load_qwen_model(
            model_id=model_id,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            quantization=self.quantization,
            quantization_bits=self.quantization_bits,
            quantization_group_size=self.quantization_group_size,
            hf_endpoint="",
        )
        self.inner = self.model.model
        self.layers = self.inner.layers[self.start_layer : self.end_layer + 1]
        self.sessions: dict[str, Any] = {}

    @property
    def backend_label(self) -> str:
        if self.device.type == "cuda":
            return "torch_cuda" if self.quantization in {"", "none"} else f"torch_cuda_{self.quantization}"
        if self.device.type == "mps":
            return "torch_mps" if self.quantization in {"", "none"} else f"torch_mps_{self.quantization}"
        return "torch_cpu" if self.quantization in {"", "none"} else f"torch_cpu_{self.quantization}"

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _cache_for_session(self, session_id: str) -> Any:
        from transformers import DynamicCache

        cache = self.sessions.get(session_id)
        if cache is None:
            cache = DynamicCache(config=self.inner.config)
            self.sessions[session_id] = cache
        return cache

    def _mask_mapping(
        self,
        hidden_states: Any,
        cache: Any,
        position_ids: Any,
        cache_position: Any,
    ) -> dict[str, Any]:
        from transformers.models.qwen3.modeling_qwen3 import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        mask_kwargs = {
            "config": self.inner.config,
            "input_embeds": hidden_states,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if getattr(self.inner, "has_sliding_layers", False):
            mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return mapping

    def _position_embeddings(
        self,
        hidden_states: Any,
        position_ids: Any,
    ) -> Any:
        return self.inner.rotary_emb(hidden_states, position_ids)

    def run(
        self,
        *,
        session_id: str,
        input_ids: list[int] | None,
        hidden_states_np: Any | None,
        position_ids: list[int],
        cache_position: list[int],
    ) -> dict[str, Any]:
        import torch

        cache = self._cache_for_session(session_id)
        with torch.inference_mode():
            position_ids_t = torch.tensor([position_ids], dtype=torch.long, device=self.device)
            cache_position_t = torch.tensor(cache_position, dtype=torch.long, device=self.device)

            if self.is_first:
                if input_ids is None:
                    raise ValueError("first shard requires input_ids")
                input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                hidden_states = self.inner.embed_tokens(input_ids_t)
            else:
                if hidden_states_np is None:
                    raise ValueError("non-first shard requires hidden_states")
                hidden_states = torch.from_numpy(hidden_states_np).to(device=self.device, dtype=self.dtype)

            mask_mapping = self._mask_mapping(hidden_states, cache, position_ids_t, cache_position_t)
            position_embeddings = self._position_embeddings(hidden_states, position_ids_t)

            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=mask_mapping[layer.attention_type],
                    position_embeddings=position_embeddings,
                    position_ids=position_ids_t,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position_t,
                )

            if self.is_last:
                hidden_states = self.inner.norm(hidden_states)

            out = hidden_states.detach().to(dtype=torch.float32).cpu().numpy()
        return {
            "checkpoint": self.checkpoint,
            "tensor": out,
            "shape": list(out.shape),
            "backend": self.backend_label,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
        }
