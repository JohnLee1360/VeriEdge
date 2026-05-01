from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from attack import inject_hetero_noise, inject_tamper
from checkpoint_qwen import (
    capture_qwen_checkpoints,
    checkpoint_order,
    clone_checkpoint_bundle,
    load_capture_bundle_for_prompt,
    load_prompt_records,
    ordered_stage_keys,
    stage_family,
)
from hash_chain import HashConfig, compute_hash_chain, first_mismatch_index


SIZE_COLUMNS = [
    "prompt_id",
    "split",
    "trace_label",
    "scenario",
    "hetero_level",
    "verifier",
    "checkpoint_entries",
    "raw_capture_bytes",
    "capture_file_bytes",
    "metadata_bytes",
    "commitment_chain_bytes",
    "commitment_head_bytes",
    "verdict_bytes",
    "validator_storage_bytes_head_commitment",
    "validator_storage_bytes_full_chain",
]

LATENCY_COLUMNS = [
    "prompt_id",
    "split",
    "trace_label",
    "scenario",
    "hetero_level",
    "verifier",
    "capture_generation_ms",
    "capture_load_ms",
    "commitment_generation_ms",
    "replay_ms",
    "compare_ms",
    "verdict_emission_ms",
    "challenge_latency_ms",
    "detected",
    "first_mismatch_stage",
    "first_mismatch_checkpoint",
    "verdict",
]

STORAGE_COLUMNS = [
    "prompt_id",
    "split",
    "trace_label",
    "scenario",
    "hetero_level",
    "verifier",
    "component",
    "bytes",
]

SUMMARY_COLUMNS = [
    "trace_label",
    "scenario",
    "hetero_level",
    "verifier",
    "sample_count",
    "mean_capture_file_bytes",
    "mean_commitment_head_bytes",
    "mean_commitment_chain_bytes",
    "mean_validator_storage_head_bytes",
    "mean_replay_ms",
    "mean_challenge_latency_ms",
    "detection_rate",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E4 verifier overhead tables from capture roots")
    parser.add_argument("--config", required=True, help="Path to qwen config")
    parser.add_argument("--capture-root", default="", help="Capture root with captures/*.npz")
    parser.add_argument("--delta-map-file", default="", help="Optional calibrated delta_map.json")
    parser.add_argument("--split", default="evaluation", help="Prompt split filter")
    parser.add_argument("--prompt-id", action="append", default=[], help="Specific prompt ids to include")
    parser.add_argument("--limit-prompts", type=int, default=0, help="Optional prompt limit")
    parser.add_argument(
        "--scenarios",
        default="honest_homo,honest_hetero,tamper",
        help="Comma-separated scenarios",
    )
    parser.add_argument(
        "--verifiers",
        default="",
        help="Comma-separated verifiers; default uses config experiment.hash_modes",
    )
    parser.add_argument(
        "--hetero-profile",
        default="default",
        help="default | all | <profile_name> for honest_hetero",
    )
    parser.add_argument("--owner", default="shared", help="Owner suffix in output file names")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output dir; default is artifacts/E4/commits/logs/<date>_<owner>",
    )
    return parser.parse_args()


def _load_structured(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        return dict(yaml.safe_load(text))


def _apply_delta_map_file(config: Dict[str, Any], delta_map_file: str) -> Dict[str, Any]:
    if not str(delta_map_file).strip():
        return config
    payload = _load_structured(Path(delta_map_file).expanduser().resolve())
    updated = dict(config)
    tstc_cfg = dict(updated.get("tstc", {}))
    tstc_cfg["delta_map"] = dict(payload.get("delta_map", {}))
    updated["tstc"] = tstc_cfg
    return updated


def _verifiers(config: Dict[str, Any], raw: str) -> List[str]:
    if raw.strip():
        return [value.strip().lower() for value in raw.split(",") if value.strip()]
    return [str(value).strip().lower() for value in config["experiment"].get("hash_modes", ["thc", "tstc"])]


def _scenarios(raw: str) -> List[str]:
    allowed = {"honest_homo", "honest_hetero", "tamper"}
    values = [value.strip() for value in raw.split(",") if value.strip()]
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"unsupported scenarios: {invalid}")
    return values or ["honest_homo", "honest_hetero", "tamper"]


def _hash_config(config: Dict[str, Any], verifier: str) -> HashConfig:
    if verifier == "thc":
        return HashConfig(mode="thc")
    tstc_cfg = dict(config.get("tstc", {}))
    prefill_cfg = dict(tstc_cfg.get("prefill", {}).get("default", {}))
    decode_cfg = dict(tstc_cfg.get("decode", {}).get("default", {}))
    return HashConfig(
        mode="tstc",
        seed_base=int(tstc_cfg.get("seed_base", 2026)),
        delta_map=dict(tstc_cfg.get("delta_map", {})),
        prefill_token_samples=int(prefill_cfg.get("token_samples", 4)),
        prefill_channel_samples=int(prefill_cfg.get("channel_samples", 16)),
        decode_channel_samples=int(decode_cfg.get("channel_samples", 32)),
    )


def _resolve_hetero_profiles(config: Dict[str, Any], selector: str) -> List[Dict[str, Any]]:
    profiles = [dict(row) for row in config.get("determinism", {}).get("hetero_levels", [])]
    if not profiles:
        return []
    if selector == "all":
        return profiles
    if selector == "default":
        for row in profiles:
            if str(row.get("name", "")).strip().lower() == "mid":
                return [row]
        return [profiles[0]]
    for row in profiles:
        if str(row.get("name", "")).strip() == selector:
            return [row]
    raise ValueError(f"unknown hetero profile: {selector}")


def _trace_label(scenario: str) -> str:
    return {
        "honest_homo": "honest_trace",
        "honest_hetero": "challenged_honest_trace",
        "tamper": "tamper_trace",
    }[scenario]


def _prompt_index(config: Dict[str, Any], split: str) -> Dict[str, Dict[str, str]]:
    try:
        rows = load_prompt_records(config, split=split)
    except Exception:
        return {}
    return {str(row["prompt_id"]): dict(row) for row in rows}


def _select_prompt_records(
    config: Dict[str, Any],
    capture_root: Optional[Path],
    split: str,
    prompt_ids: Sequence[str],
    limit_prompts: int,
) -> List[Dict[str, str]]:
    if capture_root is not None:
        prompt_index = _prompt_index(config, split)
        capture_dir = capture_root / "captures"
        if not capture_dir.exists():
            raise ValueError(f"capture directory not found: {capture_dir}")
        selected_ids = sorted(path.stem for path in capture_dir.glob("*.npz"))
        if prompt_ids:
            wanted = {str(value) for value in prompt_ids}
            selected_ids = [value for value in selected_ids if value in wanted]
        rows: List[Dict[str, str]] = []
        for prompt_id in selected_ids:
            row = prompt_index.get(prompt_id, {"prompt_id": prompt_id, "split": split, "text": ""})
            rows.append({"prompt_id": str(row["prompt_id"]), "split": str(row["split"]), "text": str(row.get("text", ""))})
        if limit_prompts > 0:
            rows = rows[:limit_prompts]
        return rows

    rows = load_prompt_records(config, split=split)
    if prompt_ids:
        wanted = {str(value) for value in prompt_ids}
        rows = [row for row in rows if str(row["prompt_id"]) in wanted]
    if limit_prompts > 0:
        rows = rows[:limit_prompts]
    return rows


def _metadata_bytes(rows: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        total += len((json.dumps(row, ensure_ascii=True) + "\n").encode("utf-8"))
    return total


def _raw_capture_bytes(bundle: Mapping[str, Mapping[str, Any]]) -> int:
    total = 0
    for stage_map in bundle.values():
        for tensor in stage_map.values():
            total += int(getattr(tensor, "nbytes", 0))
    return total


def _checkpoint_entries(bundle: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(len(stage_map) for stage_map in bundle.values())


def _load_capture(
    config: Dict[str, Any],
    prompt_record: Dict[str, str],
    capture_root: Optional[Path],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], str, float, float, int]:
    if capture_root is not None:
        start = time.perf_counter()
        bundle, metadata_rows, runtime = load_capture_bundle_for_prompt(capture_root, str(prompt_record["prompt_id"]))
        load_ms = (time.perf_counter() - start) * 1000.0
        npz_path = capture_root / "captures" / f"{prompt_record['prompt_id']}.npz"
        capture_file_bytes = int(npz_path.stat().st_size)
        return bundle, metadata_rows, runtime, 0.0, load_ms, capture_file_bytes

    start = time.perf_counter()
    bundle, metadata_rows, runtime = capture_qwen_checkpoints(config, prompt_record, seed=int(config["experiment"]["seed"]))
    capture_generation_ms = (time.perf_counter() - start) * 1000.0
    return bundle, metadata_rows, runtime, capture_generation_ms, 0.0, _raw_capture_bytes(bundle)


def _apply_scenario(
    bundle: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    scenario: str,
    hetero_profile: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if scenario == "honest_homo":
        return clone_checkpoint_bundle(bundle)
    if scenario == "honest_hetero":
        if hetero_profile is None:
            raise ValueError("hetero_profile is required for honest_hetero")
        return inject_hetero_noise(
            bundle,
            noise_std=float(hetero_profile["noise_std"]),
            fp16_cast=bool(hetero_profile.get("fp16_cast", False)),
            seed=4007,
        )
    if scenario == "tamper":
        return inject_tamper(
            bundle,
            checkpoint=str(config["tamper"].get("checkpoint", "C2")),
            strength=float(config["tamper"].get("strength", 0.15)),
            seed=3007,
        )
    raise ValueError(f"unsupported scenario: {scenario}")


def _compute_stage_chains(
    bundle: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    hash_cfg: HashConfig,
) -> Dict[str, List[str]]:
    checkpoints = checkpoint_order(config)
    out: Dict[str, List[str]] = {}
    for stage_key in ordered_stage_keys(bundle.keys()):
        out[stage_key] = compute_hash_chain(
            bundle[stage_key],
            checkpoints,
            stage_family(stage_key),
            hash_cfg,
        )
    return out


def _chain_sizes(chains: Mapping[str, Sequence[str]]) -> Tuple[int, int]:
    total_chain_bytes = 0
    total_head_bytes = 0
    for chain in chains.values():
        if not chain:
            continue
        total_chain_bytes += sum(len(bytes.fromhex(value)) for value in chain)
        total_head_bytes += len(bytes.fromhex(chain[-1]))
    return total_chain_bytes, total_head_bytes


def _first_mismatch(
    reference: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
    config: Dict[str, Any],
) -> Tuple[bool, str, str]:
    checkpoints = checkpoint_order(config)
    for stage_key in ordered_stage_keys(reference.keys()):
        mismatch = first_mismatch_index(reference.get(stage_key, []), candidate.get(stage_key, []))
        if mismatch is None:
            continue
        checkpoint = checkpoints[mismatch] if mismatch < len(checkpoints) else ""
        return True, stage_family(stage_key), checkpoint
    return False, "", ""


def measure_trace_from_bundle(
    *,
    config: Dict[str, Any],
    prompt_record: Dict[str, str],
    bundle: Dict[str, Dict[str, Any]],
    metadata_rows: List[Dict[str, Any]],
    scenario: str,
    verifier: str,
    hetero_profile: Optional[Dict[str, Any]],
    capture_generation_ms: float,
    capture_load_ms: float,
    capture_file_bytes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    validator_bundle = clone_checkpoint_bundle(bundle)
    provider_bundle = _apply_scenario(clone_checkpoint_bundle(bundle), config, scenario, hetero_profile)
    hash_cfg = _hash_config(config, verifier)

    commit_start = time.perf_counter()
    provider_chains = _compute_stage_chains(provider_bundle, config, hash_cfg)
    commitment_generation_ms = (time.perf_counter() - commit_start) * 1000.0

    replay_start = time.perf_counter()
    validator_chains = _compute_stage_chains(validator_bundle, config, hash_cfg)
    replay_ms = (time.perf_counter() - replay_start) * 1000.0

    compare_start = time.perf_counter()
    detected, first_stage, first_checkpoint = _first_mismatch(validator_chains, provider_chains, config)
    compare_ms = (time.perf_counter() - compare_start) * 1000.0

    verdict = "challenge_upheld" if scenario == "tamper" and detected else "honest_execution"
    if scenario == "tamper" and not detected:
        verdict = "challenge_failed"
    if scenario != "tamper" and detected:
        verdict = "false_alarm"

    verdict_payload = {
        "prompt_id": str(prompt_record["prompt_id"]),
        "scenario": scenario,
        "trace_label": _trace_label(scenario),
        "verifier": verifier,
        "hetero_level": str((hetero_profile or {}).get("name", "")),
        "detected": bool(detected),
        "first_mismatch_stage": first_stage,
        "first_mismatch_checkpoint": first_checkpoint,
        "verdict": verdict,
    }

    verdict_start = time.perf_counter()
    verdict_bytes = len(json.dumps(verdict_payload, ensure_ascii=True, sort_keys=True).encode("utf-8"))
    verdict_emission_ms = (time.perf_counter() - verdict_start) * 1000.0

    commitment_chain_bytes, commitment_head_bytes = _chain_sizes(provider_chains)
    raw_capture_bytes = _raw_capture_bytes(bundle)
    metadata_bytes = _metadata_bytes(metadata_rows)
    storage_head_bytes = capture_file_bytes + metadata_bytes + commitment_head_bytes + verdict_bytes
    storage_full_chain_bytes = capture_file_bytes + metadata_bytes + commitment_chain_bytes + verdict_bytes

    common = {
        "prompt_id": str(prompt_record["prompt_id"]),
        "split": str(prompt_record.get("split", "")),
        "trace_label": _trace_label(scenario),
        "scenario": scenario,
        "hetero_level": str((hetero_profile or {}).get("name", "")),
        "verifier": verifier,
    }

    size_row = {
        **common,
        "checkpoint_entries": _checkpoint_entries(bundle),
        "raw_capture_bytes": raw_capture_bytes,
        "capture_file_bytes": capture_file_bytes,
        "metadata_bytes": metadata_bytes,
        "commitment_chain_bytes": commitment_chain_bytes,
        "commitment_head_bytes": commitment_head_bytes,
        "verdict_bytes": verdict_bytes,
        "validator_storage_bytes_head_commitment": storage_head_bytes,
        "validator_storage_bytes_full_chain": storage_full_chain_bytes,
    }

    latency_row = {
        **common,
        "capture_generation_ms": round(capture_generation_ms, 6),
        "capture_load_ms": round(capture_load_ms, 6),
        "commitment_generation_ms": round(commitment_generation_ms, 6),
        "replay_ms": round(replay_ms, 6),
        "compare_ms": round(compare_ms, 6),
        "verdict_emission_ms": round(verdict_emission_ms, 6),
        "challenge_latency_ms": round(capture_load_ms + replay_ms + compare_ms + verdict_emission_ms, 6),
        "detected": bool(detected),
        "first_mismatch_stage": first_stage,
        "first_mismatch_checkpoint": first_checkpoint,
        "verdict": verdict,
    }

    storage_rows = [
        {**common, "component": "capture_file", "bytes": capture_file_bytes},
        {**common, "component": "metadata", "bytes": metadata_bytes},
        {**common, "component": "commitment_head", "bytes": commitment_head_bytes},
        {**common, "component": "commitment_chain", "bytes": commitment_chain_bytes},
        {**common, "component": "verdict_json", "bytes": verdict_bytes},
        {**common, "component": "validator_storage_head_commitment", "bytes": storage_head_bytes},
        {**common, "component": "validator_storage_full_chain", "bytes": storage_full_chain_bytes},
    ]
    return size_row, latency_row, storage_rows


def collect_overhead_rows(
    *,
    config: Dict[str, Any],
    capture_root: Optional[Path],
    split: str,
    prompt_ids: Sequence[str],
    limit_prompts: int,
    scenarios: Sequence[str],
    verifiers: Sequence[str],
    hetero_profile_selector: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_records = _select_prompt_records(config, capture_root, split, prompt_ids, limit_prompts)
    if not prompt_records:
        raise ValueError("no prompts selected")

    hetero_profiles = _resolve_hetero_profiles(config, hetero_profile_selector)
    size_rows: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []
    storage_rows: List[Dict[str, Any]] = []

    for prompt_record in prompt_records:
        bundle, metadata_rows, _, capture_generation_ms, capture_load_ms, capture_file_bytes = _load_capture(
            config,
            prompt_record,
            capture_root,
        )
        for scenario in scenarios:
            scenario_profiles = [None]
            if scenario == "honest_hetero":
                scenario_profiles = hetero_profiles or [None]
            for hetero_profile in scenario_profiles:
                for verifier in verifiers:
                    size_row, latency_row, component_rows = measure_trace_from_bundle(
                        config=config,
                        prompt_record=prompt_record,
                        bundle=bundle,
                        metadata_rows=metadata_rows,
                        scenario=scenario,
                        verifier=verifier,
                        hetero_profile=hetero_profile,
                        capture_generation_ms=capture_generation_ms,
                        capture_load_ms=capture_load_ms,
                        capture_file_bytes=capture_file_bytes,
                    )
                    size_rows.append(size_row)
                    latency_rows.append(latency_row)
                    storage_rows.extend(component_rows)
    return size_rows, latency_rows, storage_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_summary(size_rows: List[Dict[str, Any]], latency_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latency_index = {
        (
            row["prompt_id"],
            row["scenario"],
            row["hetero_level"],
            row["verifier"],
        ): row
        for row in latency_rows
    }
    grouped: Dict[Tuple[str, str, str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for size_row in size_rows:
        key = (
            str(size_row["trace_label"]),
            str(size_row["scenario"]),
            str(size_row["hetero_level"]),
            str(size_row["verifier"]),
        )
        latency_row = latency_index[
            (
                str(size_row["prompt_id"]),
                str(size_row["scenario"]),
                str(size_row["hetero_level"]),
                str(size_row["verifier"]),
            )
        ]
        grouped[key].append((size_row, latency_row))

    summary_rows: List[Dict[str, Any]] = []
    for (trace_label, scenario, hetero_level, verifier), bucket in sorted(grouped.items()):
        size_bucket = [pair[0] for pair in bucket]
        latency_bucket = [pair[1] for pair in bucket]
        summary_rows.append(
            {
                "trace_label": trace_label,
                "scenario": scenario,
                "hetero_level": hetero_level,
                "verifier": verifier,
                "sample_count": len(bucket),
                "mean_capture_file_bytes": round(_mean([float(row["capture_file_bytes"]) for row in size_bucket]), 6),
                "mean_commitment_head_bytes": round(_mean([float(row["commitment_head_bytes"]) for row in size_bucket]), 6),
                "mean_commitment_chain_bytes": round(_mean([float(row["commitment_chain_bytes"]) for row in size_bucket]), 6),
                "mean_validator_storage_head_bytes": round(
                    _mean([float(row["validator_storage_bytes_head_commitment"]) for row in size_bucket]),
                    6,
                ),
                "mean_replay_ms": round(_mean([float(row["replay_ms"]) for row in latency_bucket]), 6),
                "mean_challenge_latency_ms": round(_mean([float(row["challenge_latency_ms"]) for row in latency_bucket]), 6),
                "detection_rate": round(
                    _mean([1.0 if row["detected"] else 0.0 for row in latency_bucket]),
                    6,
                ),
            }
        )
    return summary_rows


def main() -> None:
    args = _parse_args()
    config = _apply_delta_map_file(_load_structured(Path(args.config).expanduser().resolve()), args.delta_map_file)
    capture_root = Path(args.capture_root).expanduser().resolve() if str(args.capture_root).strip() else None
    scenarios = _scenarios(args.scenarios)
    verifiers = _verifiers(config, args.verifiers)

    size_rows, latency_rows, storage_rows = collect_overhead_rows(
        config=config,
        capture_root=capture_root,
        split=args.split,
        prompt_ids=args.prompt_id,
        limit_prompts=int(args.limit_prompts),
        scenarios=scenarios,
        verifiers=verifiers,
        hetero_profile_selector=str(args.hetero_profile),
    )
    summary_rows = _build_summary(size_rows, latency_rows)

    stamp = dt.datetime.now().strftime("%Y%m%d")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else Path("artifacts/E4/commits/logs") / f"{stamp}_{args.owner}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    size_path = output_dir / f"exp_e4_{stamp}_{args.owner}_size_breakdown.csv"
    latency_path = output_dir / f"exp_e4_{stamp}_{args.owner}_latency_breakdown.csv"
    storage_path = output_dir / f"exp_e4_{stamp}_{args.owner}_storage_breakdown.csv"
    summary_path = output_dir / f"exp_e4_{stamp}_{args.owner}_summary.csv"

    _write_csv(size_path, size_rows, SIZE_COLUMNS)
    _write_csv(latency_path, latency_rows, LATENCY_COLUMNS)
    _write_csv(storage_path, storage_rows, STORAGE_COLUMNS)
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)

    print(f"size breakdown   : {size_path}")
    print(f"latency breakdown: {latency_path}")
    print(f"storage breakdown: {storage_path}")
    print(f"summary          : {summary_path}")


if __name__ == "__main__":
    main()
