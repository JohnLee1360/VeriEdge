from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from attack import inject_hetero_noise, inject_tamper
from checkpoint_qwen import (
    active_stage_families,
    capture_qwen_checkpoints,
    checkpoint_order,
    checkpoint_provider_map,
    clone_checkpoint_bundle,
    ordered_stage_keys,
    load_prompt_records,
    stage_decode_step,
    stage_family,
)
from hash_chain import HashConfig, compute_hash_chain, delta_descriptor, first_mismatch_index, sample_descriptor


def _default_prompt_record(config: Dict[str, Any]) -> Dict[str, str]:
    if "prompt" in config.get("experiment", {}):
        return {
            "prompt_id": "legacy_prompt",
            "split": str(config["experiment"].get("default_split", "evaluation")),
            "text": str(config["experiment"]["prompt"]),
        }
    rows = load_prompt_records(config, split=str(config["experiment"].get("default_split", "evaluation")))
    if not rows:
        raise ValueError("no prompt found for default split")
    return rows[0]


def _hash_config(config: Dict[str, Any], verifier: str, hash_params: Optional[Dict[str, Any]]) -> HashConfig:
    tstc_cfg = dict(config.get("tstc", {}))
    params = dict(hash_params or {})
    prefill_cfg = dict(tstc_cfg.get("prefill", {}).get("default", {}))
    decode_cfg = dict(tstc_cfg.get("decode", {}).get("default", {}))

    if verifier == "thc":
        return HashConfig(mode="thc")

    return HashConfig(
        mode="tstc",
        seed_base=int(params.get("seed_base", tstc_cfg.get("seed_base", 2026))),
        delta_map=dict(params.get("delta_map", tstc_cfg.get("delta_map", {}))),
        prefill_token_samples=int(params.get("prefill_token_samples", prefill_cfg.get("token_samples", 4))),
        prefill_channel_samples=int(params.get("prefill_channel_samples", prefill_cfg.get("channel_samples", 16))),
        decode_channel_samples=int(params.get("decode_channel_samples", decode_cfg.get("channel_samples", 32))),
    )


def run_qwen_trial(
    config: Dict[str, Any],
    scenario: str,
    verifier: str,
    trial_index: int,
    prompt_record: Optional[Dict[str, str]] = None,
    hetero_profile: Optional[Dict[str, Any]] = None,
    hash_params: Optional[Dict[str, Any]] = None,
    captured_bundle: Optional[Dict[str, Dict[str, Any]]] = None,
    captured_metadata: Optional[List[Dict[str, Any]]] = None,
    captured_runtime: str = "",
) -> Dict[str, Any]:
    exp_cfg = dict(config["experiment"])
    tamper_cfg = dict(config["tamper"])
    verifier = verifier.lower()
    if verifier not in {"thc", "tstc"}:
        raise ValueError(f"unsupported verifier: {verifier}")

    seed = int(exp_cfg["seed"]) + trial_index
    prompt = dict(prompt_record or _default_prompt_record(config))
    if captured_bundle is None:
        checkpoint_bundle, metadata_rows, runtime = capture_qwen_checkpoints(config, prompt, seed)
    else:
        checkpoint_bundle = clone_checkpoint_bundle(captured_bundle)
        metadata_rows = [dict(row) for row in (captured_metadata or [])]
        runtime = captured_runtime or "capture_root"
    validator_bundle = clone_checkpoint_bundle(checkpoint_bundle)
    provider_bundle = clone_checkpoint_bundle(checkpoint_bundle)
    checkpoint_to_provider = checkpoint_provider_map(config)
    checkpoints = checkpoint_order(config)
    hash_cfg = _hash_config(config, verifier, hash_params)

    determinism_profile = "homogeneous"
    tamper_checkpoint = str(tamper_cfg.get("checkpoint", "C2"))
    expected_malicious = "none"

    if scenario == "tamper":
        provider_bundle = inject_tamper(
            provider_bundle,
            checkpoint=tamper_checkpoint,
            strength=float(tamper_cfg["strength"]),
            seed=seed + 3000,
        )
        expected_malicious = checkpoint_to_provider[tamper_checkpoint]
    elif scenario == "honest_hetero":
        if hetero_profile is None:
            raise ValueError("hetero_profile is required for honest_hetero scenario")
        determinism_profile = f"hetero_{hetero_profile['name']}"
        provider_bundle = inject_hetero_noise(
            provider_bundle,
            noise_std=float(hetero_profile["noise_std"]),
            fp16_cast=bool(hetero_profile.get("fp16_cast", False)),
            seed=seed + 4000,
        )
    elif scenario != "honest_homo":
        raise ValueError(f"unsupported scenario: {scenario}")

    active_stages = active_stage_families(config)
    stage_groups: Dict[str, List[str]] = {stage: [] for stage in active_stages}
    for stage_key in ordered_stage_keys(provider_bundle.keys()):
        family = stage_family(stage_key)
        if family in stage_groups:
            stage_groups[family].append(stage_key)

    stage_results: List[Dict[str, Any]] = []
    for stage_name in active_stages:
        stage_keys = stage_groups.get(stage_name, [])
        if not stage_keys:
            continue

        step_results: List[Dict[str, Any]] = []
        for stage_key in stage_keys:
            provider_chain = compute_hash_chain(provider_bundle[stage_key], checkpoints, stage_name, hash_cfg)
            validator_chain = compute_hash_chain(validator_bundle[stage_key], checkpoints, stage_name, hash_cfg)
            mismatch = first_mismatch_index(validator_chain, provider_chain)
            first_checkpoint = checkpoints[mismatch] if mismatch is not None else ""
            step_results.append(
                {
                    "stage_key": stage_key,
                    "decode_step": stage_decode_step(stage_key),
                    "first_mismatch_index": int(mismatch) if mismatch is not None else -1,
                    "first_mismatch_checkpoint": first_checkpoint,
                    "detected": mismatch is not None,
                }
            )

        first_detected = next((row for row in step_results if row["detected"]), None)
        detected = first_detected is not None
        first_checkpoint = str(first_detected["first_mismatch_checkpoint"]) if first_detected else ""
        first_mismatch_pos = int(first_detected["first_mismatch_index"]) if first_detected else -1
        first_decode_step = int(first_detected["decode_step"]) if first_detected else 0
        malicious_pred = checkpoint_to_provider[first_checkpoint] if first_checkpoint else "none"
        localization_correct = scenario == "tamper" and first_checkpoint == tamper_checkpoint and malicious_pred == expected_malicious
        false_positive = scenario in {"honest_homo", "honest_hetero"} and detected

        stage_results.append(
            {
                "model": f"qwen({runtime})",
                "prompt_id": str(prompt["prompt_id"]),
                "split": str(prompt["split"]),
                "scenario": scenario,
                "verifier": verifier,
                "trial_index": trial_index,
                "stage": stage_name,
                "shard_plan": list(exp_cfg["shard_plan"]),
                "checkpoint_order": list(checkpoints),
                "determinism_profile": determinism_profile,
                "hetero_level": str(hetero_profile["name"]) if hetero_profile is not None else "",
                "first_mismatch_index": first_mismatch_pos,
                "first_mismatch_checkpoint": first_checkpoint,
                "first_mismatch_decode_step": first_decode_step,
                "detected": bool(detected),
                "detected_step_count": int(sum(1 for row in step_results if row["detected"])),
                "decode_steps_total": int(len(stage_keys)) if stage_name == "decode" else 0,
                "malicious_provider_pred": malicious_pred,
                "expected_malicious_provider": expected_malicious,
                "localization_correct": bool(localization_correct),
                "false_positive": bool(false_positive),
                "step_results": step_results,
                "sampling_spec": json.dumps(sample_descriptor(hash_cfg), ensure_ascii=True, sort_keys=True) if verifier == "tstc" else "",
                "delta_used": json.dumps(delta_descriptor(hash_cfg), ensure_ascii=True, sort_keys=True) if verifier == "tstc" else "",
            }
        )

    return {
        "records": stage_results,
        "checkpoint_metadata": metadata_rows,
        "checkpoint_shapes": {
            str(meta.get("stage_key", meta["stage"])): {
                row["checkpoint"]: row["shape"]
                for row in metadata_rows
                if str(row.get("stage_key", row["stage"])) == str(meta.get("stage_key", meta["stage"]))
            }
            for meta in metadata_rows
        },
        "runtime": runtime,
    }
