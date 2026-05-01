from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class HashConfig:
    mode: str
    seed_base: int = 2026
    delta_map: Dict[str, Dict[str, float]] = field(default_factory=dict)
    prefill_token_samples: int = 4
    prefill_channel_samples: int = 16
    decode_channel_samples: int = 32


def _normalize_mode(mode: str) -> str:
    return mode.lower()


def _hash_bytes(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _hash_tensor_thc(tensor: np.ndarray) -> bytes:
    arr = np.asarray(tensor, dtype=np.float32)
    return _hash_bytes(arr.tobytes(order="C"))


def _sample_indices(total_size: int, sample_count: int, seed: int) -> np.ndarray:
    if total_size <= 0:
        return np.array([], dtype=np.int32)
    k = min(max(1, int(sample_count)), total_size)
    if k == total_size:
        return np.arange(total_size, dtype=np.int32)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total_size, size=k, replace=False).astype(np.int32))
    return idx


def _delta_for(cfg: HashConfig, stage: str, checkpoint: str) -> float:
    delta = float(cfg.delta_map.get(stage, {}).get(checkpoint, 0.0))
    return delta


def _quantize(sampled: np.ndarray, delta: float) -> np.ndarray:
    return np.rint(np.asarray(sampled, dtype=np.float32) / delta).astype(np.int32)


def _append_sample_values(payload: bytearray, sampled: np.ndarray, delta: float) -> None:
    arr = np.asarray(sampled, dtype=np.float32).reshape(-1)
    if delta > 0:
        payload.extend(_quantize(arr, delta).astype(np.int32).tobytes(order="C"))
        return
    payload.extend(arr.astype(np.float32).tobytes(order="C"))


def _summary_prefill(tensor: np.ndarray, cfg: HashConfig, checkpoint: str, seed: int) -> bytes:
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"prefill tensor for {checkpoint} must be rank-3, got shape {arr.shape}")

    bsz, seq_len, hidden = arr.shape
    rows = arr.reshape(bsz * seq_len, hidden)
    token_idx = _sample_indices(rows.shape[0], cfg.prefill_token_samples, seed)
    delta = _delta_for(cfg, "prefill", checkpoint)

    payload = bytearray()
    payload.extend(np.asarray([len(token_idx)], dtype=np.int32).tobytes(order="C"))
    payload.extend(token_idx.astype(np.int32).tobytes(order="C"))

    for offset, row_idx in enumerate(token_idx):
        channel_idx = _sample_indices(hidden, cfg.prefill_channel_samples, seed + 1000 + offset + int(row_idx))
        payload.extend(np.asarray([len(channel_idx)], dtype=np.int32).tobytes(order="C"))
        payload.extend(channel_idx.astype(np.int32).tobytes(order="C"))
        _append_sample_values(payload, rows[int(row_idx), channel_idx], delta)

    return _hash_bytes(bytes(payload))


def _summary_decode(tensor: np.ndarray, cfg: HashConfig, checkpoint: str, seed: int) -> bytes:
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"decode tensor for {checkpoint} must be rank-3, got shape {arr.shape}")
    _, _, hidden = arr.shape

    channel_idx = _sample_indices(hidden, cfg.decode_channel_samples, seed)
    sampled = arr[:, 0, :][:, channel_idx]
    delta = _delta_for(cfg, "decode", checkpoint)
    payload = bytearray()
    payload.extend(channel_idx.astype(np.int32).tobytes(order="C"))
    _append_sample_values(payload, sampled.reshape(-1), delta)
    return _hash_bytes(payload)


def sample_descriptor(cfg: HashConfig) -> Dict[str, Dict[str, int]]:
    return {
        "prefill": {
            "token_samples": int(cfg.prefill_token_samples),
            "channel_samples": int(cfg.prefill_channel_samples),
            "sample_count": int(cfg.prefill_token_samples) * int(cfg.prefill_channel_samples),
        },
        "decode": {
            "channel_samples": int(cfg.decode_channel_samples),
            "sample_count": int(cfg.decode_channel_samples),
        },
    }


def delta_descriptor(cfg: HashConfig) -> Dict[str, Dict[str, float]]:
    return json.loads(json.dumps(cfg.delta_map, ensure_ascii=True))


def compute_hash_chain(
    checkpoints: Mapping[str, np.ndarray],
    checkpoint_order: Sequence[str],
    stage: str,
    cfg: HashConfig,
) -> List[str]:
    tensors = [np.asarray(checkpoints[name], dtype=np.float32) for name in checkpoint_order]
    if not tensors:
        return []

    mode = _normalize_mode(cfg.mode)
    chain: List[str] = []

    if mode == "thc":
        prev_hash = _hash_tensor_thc(tensors[0])
        chain.append(prev_hash.hex())
        for tensor in tensors[1:]:
            curr_tensor_hash = _hash_tensor_thc(tensor)
            prev_hash = _hash_bytes(prev_hash + curr_tensor_hash)
            chain.append(prev_hash.hex())
        return chain

    if mode == "tstc":
        summary_fn = _summary_prefill if stage == "prefill" else _summary_decode
        s0 = summary_fn(tensors[0], cfg, checkpoint_order[0], cfg.seed_base)
        prev_chain = _hash_bytes(s0)
        chain.append(prev_chain.hex())

        for index, tensor in enumerate(tensors[1:], start=1):
            checkpoint = checkpoint_order[index]
            s_k = summary_fn(tensor, cfg, checkpoint, cfg.seed_base + index)
            prev_chain = _hash_bytes(prev_chain + s_k)
            chain.append(prev_chain.hex())
        return chain

    raise ValueError(f"unsupported hash mode: {cfg.mode}")


def first_mismatch_index(reference: Sequence[str], candidate: Sequence[str]) -> Optional[int]:
    max_len = max(len(reference), len(candidate))
    for idx in range(max_len):
        ref_val = reference[idx] if idx < len(reference) else None
        cand_val = candidate[idx] if idx < len(candidate) else None
        if ref_val != cand_val:
            return idx
    return None
