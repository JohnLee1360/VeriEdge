from __future__ import annotations

from typing import Dict

import numpy as np


def clone_checkpoint_bundle(bundle: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        stage: {checkpoint: np.array(tensor, copy=True) for checkpoint, tensor in stage_map.items()}
        for stage, stage_map in bundle.items()
    }


def inject_tamper(
    bundle: Dict[str, Dict[str, np.ndarray]],
    checkpoint: str,
    strength: float,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    tampered = clone_checkpoint_bundle(bundle)
    rng = np.random.default_rng(seed)
    for stage, stage_map in tampered.items():
        if checkpoint not in stage_map:
            raise KeyError(f"checkpoint {checkpoint} missing from stage {stage}")
        noise = rng.normal(loc=0.0, scale=strength, size=stage_map[checkpoint].shape).astype(np.float32)
        stage_map[checkpoint] = stage_map[checkpoint].astype(np.float32) + noise
    return tampered


def inject_hetero_noise(
    bundle: Dict[str, Dict[str, np.ndarray]],
    noise_std: float,
    fp16_cast: bool,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    perturbed = clone_checkpoint_bundle(bundle)
    rng = np.random.default_rng(seed)

    for stage_map in perturbed.values():
        for checkpoint, tensor in stage_map.items():
            noise = rng.normal(loc=0.0, scale=noise_std, size=tensor.shape).astype(np.float32)
            updated = tensor.astype(np.float32) + noise
            if fp16_cast:
                updated = updated.astype(np.float16).astype(np.float32)
            stage_map[checkpoint] = updated

    return perturbed
