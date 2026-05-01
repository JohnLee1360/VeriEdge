from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from checkpoint_qwen import load_prompt_records
from checkpoint_qwen import load_capture_bundle_for_prompt
from checkpoint_qwen import active_stage_families
from metrics import aggregate_summary, write_jsonl, write_raw_json, write_summary_csv
from pipeline_qwen import run_qwen_trial
from plot import generate_figures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised Qwen THC/TSTC verification experiments")
    parser.add_argument("--config", required=True, help="Path to qwen config JSON/YAML-compatible file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["honest_homo", "honest_hetero", "tamper", "all"],
        help="Scenario selection",
    )
    parser.add_argument(
        "--split",
        default="evaluation",
        choices=["calibration", "evaluation"],
        help="Prompt split to use for this run",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=0,
        help="Optional limit on the number of prompts loaded from the selected split",
    )
    parser.add_argument(
        "--runs-per-mode",
        type=int,
        default=0,
        help="Override runs_per_mode from config; 0 means use config value",
    )
    parser.add_argument(
        "--calibrate-tstc",
        default="false",
        choices=["true", "false"],
        help="Whether to search TSTC sampling parameters before the final run",
    )
    parser.add_argument(
        "--capture-root",
        default="",
        help="Optional capture root produced by hetero_qwen_capture.py; when set, trials load prompts from this root",
    )
    parser.add_argument(
        "--delta-map-file",
        default="",
        help="Optional delta_map.json produced by T3 calibration; overrides tstc.delta_map in config",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional override for experiment.output_root",
    )
    parser.add_argument(
        "--paper-img-dir",
        default="",
        help="Optional override for experiment.paper_img_dir",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_delta_map_file(config: Dict[str, Any], delta_map_file: str) -> Dict[str, Any]:
    if not str(delta_map_file).strip():
        return config
    path = Path(str(delta_map_file)).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    delta_map = dict(payload.get("delta_map", {}))
    config = dict(config)
    tstc_cfg = dict(config.get("tstc", {}))
    tstc_cfg["delta_map"] = delta_map
    config["tstc"] = tstc_cfg
    return config


def _scenarios_from_mode(mode: str) -> List[str]:
    if mode == "all":
        return ["honest_homo", "honest_hetero", "tamper"]
    return [mode]


def _iter_hetero_profiles(config: Dict[str, Any], scenario: str) -> List[Optional[Dict[str, Any]]]:
    if scenario != "honest_hetero":
        return [None]
    levels = list(config.get("determinism", {}).get("hetero_levels", []))
    if not levels:
        raise ValueError("determinism.hetero_levels must be provided for honest_hetero scenario")
    return levels


def _load_prompts(config: Dict[str, Any], split: str, limit: int) -> List[Dict[str, str]]:
    rows = load_prompt_records(config, split=split)
    if limit > 0:
        rows = rows[: int(limit)]
    if not rows:
        raise ValueError(f"no prompts found for split={split}")
    return rows


def _run_records(
    config: Dict[str, Any],
    prompts: Iterable[Dict[str, str]],
    scenarios: Iterable[str],
    run_id: str,
    runs_per_mode: int,
    hash_modes: Iterable[str],
    tstc_override: Optional[Dict[str, Any]] = None,
    capture_root: Optional[Path] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    checkpoint_metadata: List[Dict[str, Any]] = []
    seen_meta: set[tuple[str, str, str]] = set()
    capture_cache: Dict[str, tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], str]] = {}

    for prompt in prompts:
        captured_bundle = None
        captured_metadata = None
        captured_runtime = ""
        if capture_root is not None:
            prompt_id = str(prompt["prompt_id"])
            cached = capture_cache.get(prompt_id)
            if cached is None:
                cached = load_capture_bundle_for_prompt(capture_root, prompt_id)
                capture_cache[prompt_id] = cached
            captured_bundle, captured_metadata, captured_runtime = cached
        for scenario in scenarios:
            for verifier in hash_modes:
                for hetero_profile in _iter_hetero_profiles(config, scenario):
                    for trial_index in range(runs_per_mode):
                        hash_params = tstc_override if verifier == "tstc" else None
                        out = run_qwen_trial(
                            config=config,
                            scenario=scenario,
                            verifier=verifier,
                            trial_index=trial_index,
                            prompt_record=prompt,
                            hetero_profile=hetero_profile,
                            hash_params=hash_params,
                            captured_bundle=captured_bundle,
                            captured_metadata=captured_metadata,
                            captured_runtime=captured_runtime,
                        )
                        for record in out["records"]:
                            record["run_id"] = run_id
                            records.append(record)
                        for row in out["checkpoint_metadata"]:
                            stage_key = str(row.get("stage_key", row["stage"]))
                            key = (row["prompt_id"], stage_key, row["checkpoint"])
                            if key in seen_meta:
                                continue
                            seen_meta.add(key)
                            checkpoint_metadata.append(row)
    return records, checkpoint_metadata


def _candidate_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tstc_cfg = dict(config.get("tstc", {}))
    active_stages = active_stage_families(config)
    prefill_grid = list(tstc_cfg.get("prefill", {}).get("grid", []))
    decode_grid = list(tstc_cfg.get("decode", {}).get("grid", []))
    if "prefill" in active_stages and not prefill_grid:
        raise ValueError("prefill TSTC grid must not be empty when prefill is active")
    if "decode" in active_stages and not decode_grid:
        raise ValueError("decode TSTC grid must not be empty when decode is active")
    if "prefill" not in active_stages:
        prefill_grid = [{"token_samples": int(tstc_cfg.get("prefill", {}).get("default", {}).get("token_samples", 4)), "channel_samples": int(tstc_cfg.get("prefill", {}).get("default", {}).get("channel_samples", 16))}]
    if "decode" not in active_stages:
        decode_grid = [int(tstc_cfg.get("decode", {}).get("default", {}).get("channel_samples", 32))]

    candidates: List[Dict[str, Any]] = []
    for prefill_spec in prefill_grid:
        for decode_channels in decode_grid:
            candidates.append(
                {
                    "seed_base": int(tstc_cfg.get("seed_base", 2026)),
                    "delta_map": dict(tstc_cfg.get("delta_map", {})),
                    "prefill_token_samples": int(prefill_spec["token_samples"]),
                    "prefill_channel_samples": int(prefill_spec["channel_samples"]),
                    "decode_channel_samples": int(decode_channels),
                }
            )
    return candidates


def _summarize_candidate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    active_stages = sorted({str(record["stage"]) for record in records})
    if not active_stages:
        raise ValueError("candidate summary requires at least one active stage")
    stage_metrics: Dict[str, Dict[str, float]] = {}
    for stage in active_stages:
        tamper = [r for r in records if r["scenario"] == "tamper" and r["stage"] == stage]
        hetero = [r for r in records if r["scenario"] == "honest_hetero" and r["stage"] == stage]
        if not tamper or not hetero:
            raise ValueError(f"candidate summary requires tamper and honest_hetero rows for stage={stage}")
        tpr = sum(1 for r in tamper if r.get("detected")) / len(tamper)
        loc = sum(1 for r in tamper if r.get("localization_correct")) / len(tamper)
        fpr = sum(1 for r in hetero if r.get("detected")) / len(hetero)
        stage_metrics[stage] = {"tpr": tpr, "localization_acc": loc, "fpr": fpr}

    prefill = stage_metrics.get("prefill", {"fpr": float("nan"), "tpr": float("nan"), "localization_acc": float("nan")})
    decode = stage_metrics.get("decode", {"fpr": float("nan"), "tpr": float("nan"), "localization_acc": float("nan")})
    return {
        "prefill_fpr": prefill["fpr"],
        "decode_fpr": decode["fpr"],
        "prefill_tpr": prefill["tpr"],
        "decode_tpr": decode["tpr"],
        "prefill_loc": prefill["localization_acc"],
        "decode_loc": decode["localization_acc"],
        "max_fpr": max(stage_metrics[stage]["fpr"] for stage in active_stages),
        "min_tpr": min(stage_metrics[stage]["tpr"] for stage in active_stages),
        "min_localization_acc": min(stage_metrics[stage]["localization_acc"] for stage in active_stages),
    }


def _pick_best_candidate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    qualified = [
        r
        for r in rows
        if float(r["min_tpr"]) >= 0.95 and float(r["min_localization_acc"]) >= 0.90 and float(r["max_fpr"]) <= 0.20
    ]
    pool = qualified if qualified else rows

    def _sort_key(r: Dict[str, Any]) -> tuple[float, float, float, int, int]:
        return (
            float(r["max_fpr"]),
            -float(r["min_tpr"]),
            -float(r["min_localization_acc"]),
            int(r["prefill_sample_count"]),
            int(r["decode_channel_samples"]),
        )

    selected_row = sorted(pool, key=_sort_key)[0]
    best = dict(selected_row)
    best["qualified"] = selected_row in qualified
    return best


def _write_candidate_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    cols = [
        "prefill_token_samples",
        "prefill_channel_samples",
        "prefill_sample_count",
        "decode_channel_samples",
        "prefill_tpr",
        "decode_tpr",
        "prefill_loc",
        "decode_loc",
        "prefill_fpr",
        "decode_fpr",
        "max_fpr",
        "min_tpr",
        "min_localization_acc",
        "qualified",
        "selected",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _calibrate_tstc(
    config: Dict[str, Any],
    prompts: List[Dict[str, str]],
    run_id: str,
    runs_per_mode: int,
    output_root: Path,
    capture_root: Optional[Path],
) -> tuple[Dict[str, Any], Path]:
    candidates = _candidate_grid(config)
    rows: List[Dict[str, Any]] = []

    for candidate in candidates:
        records, _ = _run_records(
            config=config,
            prompts=prompts,
            scenarios=["honest_homo", "honest_hetero", "tamper"],
            run_id=run_id,
            runs_per_mode=runs_per_mode,
            hash_modes=["tstc"],
            tstc_override=candidate,
            capture_root=capture_root,
        )
        metrics = _summarize_candidate(records)
        row = {
            **candidate,
            **{key: round(val, 6) for key, val in metrics.items()},
            "prefill_sample_count": int(candidate["prefill_token_samples"]) * int(candidate["prefill_channel_samples"]),
            "qualified": metrics["min_tpr"] >= 0.95 and metrics["min_localization_acc"] >= 0.90 and metrics["max_fpr"] <= 0.20,
            "selected": False,
        }
        rows.append(row)

    best = _pick_best_candidate(rows)
    for row in rows:
        row["selected"] = (
            int(row["prefill_token_samples"]) == int(best["prefill_token_samples"])
            and int(row["prefill_channel_samples"]) == int(best["prefill_channel_samples"])
            and int(row["decode_channel_samples"]) == int(best["decode_channel_samples"])
        )

    candidate_csv = output_root / "tstc_sampling_search.csv"
    _write_candidate_csv(rows, candidate_csv)

    selected = {
        "seed_base": int(best["seed_base"]),
        "delta_map": dict(best["delta_map"]),
        "prefill_token_samples": int(best["prefill_token_samples"]),
        "prefill_channel_samples": int(best["prefill_channel_samples"]),
        "decode_channel_samples": int(best["decode_channel_samples"]),
    }
    return selected, candidate_csv


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    config = _apply_delta_map_file(_load_config(config_path), args.delta_map_file)

    exp_cfg = dict(config.get("experiment", {}))
    if str(exp_cfg.get("model", "")).lower() != "qwen":
        raise ValueError("this runner only supports experiment.model == 'qwen'")

    prompts = _load_prompts(config, split=args.split, limit=int(args.limit_prompts))
    scenarios = _scenarios_from_mode(args.mode)
    runs_per_mode = int(args.runs_per_mode) if int(args.runs_per_mode) > 0 else int(exp_cfg.get("runs_per_mode", 10))
    hash_modes = [str(m).lower() for m in exp_cfg.get("hash_modes", ["thc", "tstc"])]
    hash_modes = [m for m in hash_modes if m in {"thc", "tstc"}]
    if not hash_modes:
        raise ValueError("experiment.hash_modes must include thc and/or tstc")

    calibrate = args.calibrate_tstc == "true"
    if calibrate and args.mode != "all":
        raise ValueError("--calibrate-tstc true requires --mode all")

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{run_id}_qwen_{args.mode}_{args.split}"
    output_root_base = Path(args.output_root) if args.output_root else Path(exp_cfg["output_root"])
    output_root = output_root_base / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    paper_img_dir = Path(args.paper_img_dir) if args.paper_img_dir else Path(exp_cfg["paper_img_dir"])

    selected_tstc: Optional[Dict[str, Any]] = None
    candidate_csv: Optional[Path] = None
    if calibrate:
        selected_tstc, candidate_csv = _calibrate_tstc(
            config=config,
            prompts=prompts,
            run_id=run_id,
            runs_per_mode=runs_per_mode,
            output_root=output_root,
            capture_root=Path(args.capture_root).expanduser().resolve() if args.capture_root else None,
        )

    if "tstc" in hash_modes and selected_tstc is None:
        tstc_cfg = dict(config.get("tstc", {}))
        selected_tstc = {
            "seed_base": int(tstc_cfg.get("seed_base", 2026)),
            "delta_map": dict(tstc_cfg.get("delta_map", {})),
            "prefill_token_samples": int(tstc_cfg.get("prefill", {}).get("default", {}).get("token_samples", 4)),
            "prefill_channel_samples": int(tstc_cfg.get("prefill", {}).get("default", {}).get("channel_samples", 16)),
            "decode_channel_samples": int(tstc_cfg.get("decode", {}).get("default", {}).get("channel_samples", 32)),
        }

    records, checkpoint_metadata = _run_records(
        config=config,
        prompts=prompts,
        scenarios=scenarios,
        run_id=run_id,
        runs_per_mode=runs_per_mode,
        hash_modes=hash_modes,
        tstc_override=selected_tstc,
        capture_root=Path(args.capture_root).expanduser().resolve() if args.capture_root else None,
    )

    raw_json = output_root / "raw_results.json"
    write_raw_json(records, raw_json)
    write_jsonl(checkpoint_metadata, output_root / "checkpoint_metadata.jsonl")

    summary_rows = aggregate_summary(records)
    summary_csv = output_root / "summary_metrics.csv"
    write_summary_csv(summary_rows, summary_csv)

    generate_figures(
        summary_csv=summary_csv,
        run_dir=output_root,
        paper_img_dir=paper_img_dir,
        candidate_csv=candidate_csv,
        file_suffix="qwen",
    )

    meta = {
        "run_id": run_id,
        "config": str(config_path),
        "mode": args.mode,
        "split": args.split,
        "prompt_count": len(prompts),
        "runs_per_mode": runs_per_mode,
        "calibrate_tstc": calibrate,
        "capture_root": str(Path(args.capture_root).expanduser().resolve()) if args.capture_root else "",
        "delta_map_file": str(Path(args.delta_map_file).expanduser().resolve()) if args.delta_map_file else "",
        "hash_modes": hash_modes,
        "selected_tstc": selected_tstc,
        "record_count": len(records),
        "candidate_csv": str(candidate_csv) if candidate_csv else "",
    }
    with (output_root / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    print(f"THC/TSTC run complete: {output_root}")


if __name__ == "__main__":
    main()
