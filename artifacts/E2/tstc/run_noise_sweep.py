from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import math
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
THC_SRC = THIS_DIR.parents[1] / "_shared" / "thc" / "src"
if str(THC_SRC) not in sys.path:
    sys.path.insert(0, str(THC_SRC))

from checkpoint_qwen import capture_qwen_checkpoints, clone_checkpoint_bundle, load_capture_bundle_for_prompt  # type: ignore
from hash_chain import HashConfig, compute_hash_chain, first_mismatch_index  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TSTC C2 Gaussian-noise sweep for the prefill stage")
    parser.add_argument("--sweep-config", default=str(THIS_DIR / "noise_sweep_config.json"))
    parser.add_argument("--capture-root", default="")
    parser.add_argument("--prompt-id", default="")
    parser.add_argument("--repetitions", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--use-mock-if-unavailable", default="false", choices=["true", "false"])
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sample_indices(total_size: int, sample_count: int, seed: int) -> np.ndarray:
    if total_size <= 0:
        return np.array([], dtype=np.int32)
    k = min(max(1, int(sample_count)), total_size)
    if k == total_size:
        return np.arange(total_size, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_size, size=k, replace=False).astype(np.int32))


def _prefill_sample_values(tensor: np.ndarray, token_samples: int, channel_samples: int, seed: int) -> np.ndarray:
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"prefill tensor must be rank-3, got shape {arr.shape}")
    batch, seq_len, hidden = arr.shape
    rows = arr.reshape(batch * seq_len, hidden)
    token_idx = _sample_indices(rows.shape[0], token_samples, seed)

    collected: List[np.ndarray] = []
    for offset, row_idx in enumerate(token_idx):
        channel_idx = _sample_indices(hidden, channel_samples, seed + 1000 + offset + int(row_idx))
        collected.append(np.asarray(rows[int(row_idx), channel_idx], dtype=np.float32).reshape(-1))
    if not collected:
        return np.array([], dtype=np.float32)
    return np.concatenate(collected, axis=0)


def _fraction_to_count(total_size: int, fraction: float, rounding: str) -> int:
    if total_size <= 0:
        return 1
    scaled = float(total_size) * float(fraction)
    if rounding == "floor":
        return max(1, int(math.floor(scaled)))
    if rounding == "round":
        return max(1, int(round(scaled)))
    return max(1, int(math.ceil(scaled)))


def _resolve_prefill_sampling(sweep_cfg: Dict[str, Any], tensor: np.ndarray) -> tuple[int, int]:
    prefill_cfg = dict(sweep_cfg["tstc"].get("prefill", {}))
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"prefill tensor must be rank-3, got shape {arr.shape}")

    _, seq_len, hidden = arr.shape
    rounding = str(prefill_cfg.get("rounding", "ceil")).strip().lower() or "ceil"
    token_fraction = float(prefill_cfg.get("token_fraction", 1.0 / 3.0))
    channel_fraction = float(prefill_cfg.get("channel_fraction", 1.0 / 8.0))
    token_samples = _fraction_to_count(seq_len, token_fraction, rounding)
    channel_samples = _fraction_to_count(hidden, channel_fraction, rounding)
    return token_samples, channel_samples


def _build_hash_config(sweep_cfg: Dict[str, Any]) -> HashConfig:
    tstc_cfg = dict(sweep_cfg["tstc"])
    return HashConfig(
        mode="tstc",
        seed_base=int(tstc_cfg.get("seed_base", 2026)),
        delta_map=dict(tstc_cfg.get("delta_map", {})),
        prefill_token_samples=1,
        prefill_channel_samples=1,
        decode_channel_samples=1,
    )


def _prepare_thc_config(base_cfg: Dict[str, Any], use_mock_if_unavailable: bool) -> Dict[str, Any]:
    config = json.loads(json.dumps(base_cfg))
    config.setdefault("experiment", {})
    config["experiment"]["active_stages"] = ["prefill"]

    qwen_cfg = dict(config.get("qwen", {}))
    has_mlx = importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_lm") is not None
    qwen_cfg["use_mock_if_unavailable"] = bool(use_mock_if_unavailable)
    qwen_cfg["allow_mock_fallback"] = bool(use_mock_if_unavailable)
    qwen_cfg["enable_torch_fallback"] = True
    if not has_mlx:
        qwen_cfg["enable_mlx"] = False
        qwen_cfg["model_id"] = str(qwen_cfg.get("hetero_model_id", "Qwen/Qwen3-0.6B"))
        qwen_cfg["local_files_only"] = bool(qwen_cfg.get("hetero_local_files_only", qwen_cfg.get("local_files_only", True)))
        if str(qwen_cfg.get("hetero_hf_endpoint", "")).strip():
            qwen_cfg["hf_endpoint"] = str(qwen_cfg["hetero_hf_endpoint"])
    config["qwen"] = qwen_cfg
    return config


def _prompt_record(sweep_cfg: Dict[str, Any]) -> Dict[str, str]:
    prompt_cfg = dict(sweep_cfg["prompt"])
    return {
        "prompt_id": str(prompt_cfg.get("prompt_id", "tstc_noise_prompt")),
        "split": str(prompt_cfg.get("split", "evaluation")),
        "text": str(prompt_cfg["text"]),
    }


def _load_reference_bundle(
    sweep_cfg: Dict[str, Any],
    thc_config: Dict[str, Any],
    capture_root: str,
    prompt_id: str,
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
    target_stage = str(sweep_cfg["target"].get("stage", "prefill"))
    prompt = _prompt_record(sweep_cfg)
    if capture_root:
        effective_prompt_id = prompt_id or prompt["prompt_id"]
        bundle, metadata_rows, runtime = load_capture_bundle_for_prompt(Path(capture_root).expanduser().resolve(), effective_prompt_id)
        if target_stage not in bundle:
            raise ValueError(f"stage {target_stage} not found in capture bundle")
        meta = {
            "runtime": runtime,
            "prompt_id": effective_prompt_id,
            "prompt_text": prompt["text"],
        }
        return bundle, meta

    bundle, _, runtime = capture_qwen_checkpoints(thc_config, prompt, seed=int(sweep_cfg.get("random_seed", 20260322)))
    meta = {
        "runtime": runtime,
        "prompt_id": prompt["prompt_id"],
        "prompt_text": prompt["text"],
    }
    return bundle, meta


def _checkpoint_order(thc_config: Dict[str, Any]) -> List[str]:
    return [str(shard["checkpoint"]) for shard in thc_config["experiment"]["shards"]]


def _effective_noise_scale(sample_std: float, fallback_scale: float) -> float:
    if np.isfinite(sample_std) and sample_std > 0.0:
        return float(sample_std)
    return float(fallback_scale)


def _inject_c2_noise(bundle: Dict[str, Dict[str, np.ndarray]], stage: str, checkpoint: str, raw_noise_std: float, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    perturbed = clone_checkpoint_bundle(bundle)
    tensor = np.asarray(perturbed[stage][checkpoint], dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=raw_noise_std, size=tensor.shape).astype(np.float32)
    perturbed[stage][checkpoint] = tensor + noise
    return perturbed


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def _write_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "noise_std",
        "effective_noise_std",
        "sigma_ref",
        "repetitions",
        "detected_count",
        "detection_rate",
        "c2_first_mismatch_count",
        "non_c2_first_mismatch_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_detection_counts(summary_rows: Sequence[Mapping[str, Any]], output_path: Path, paper_img_path: Path | None) -> None:
    labels = [f"{float(row['noise_std']):.0e}" if float(row["noise_std"]) > 0 else "0" for row in summary_rows]
    counts = [int(row["detected_count"]) for row in summary_rows]
    xs = np.arange(len(summary_rows))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#cbd5e1",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.plot(xs, counts, color="#0f766e", marker="o", linewidth=2.2, markersize=6)
    ax.fill_between(xs, counts, color="#99f6e4", alpha=0.22)
    for x, count in zip(xs, counts):
        ax.text(x, count + max(1, int(max(counts) * 0.015) if counts else 1), str(count), ha="center", va="bottom", fontsize=9, color="#115e59")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Noise Std (normalized s)")
    ax.set_ylabel("Detection Count")
    ax.set_title("TSTC Detection Count under C2 Gaussian Perturbation")
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.25, color="#475569")
    ax.set_ylim(0.0, max(counts + [1]) * 1.12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    if paper_img_path is not None:
        paper_img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_img_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    sweep_cfg = _load_json(Path(args.sweep_config).expanduser().resolve())
    thc_config_path = Path(str(sweep_cfg["thc_config"])).expanduser().resolve()
    thc_config = _prepare_thc_config(_load_json(thc_config_path), use_mock_if_unavailable=args.use_mock_if_unavailable == "true")

    target_cfg = dict(sweep_cfg["target"])
    target_stage = str(target_cfg.get("stage", "prefill"))
    target_checkpoint = str(target_cfg.get("checkpoint", "C2"))
    repetitions = int(args.repetitions) if int(args.repetitions) > 0 else int(sweep_cfg.get("repetitions_per_noise", 1000))
    hash_cfg = _build_hash_config(sweep_cfg)
    checkpoint_order = _checkpoint_order(thc_config)
    if target_checkpoint not in checkpoint_order:
        raise ValueError(f"checkpoint {target_checkpoint} not present in thc config")

    reference_bundle, reference_meta = _load_reference_bundle(sweep_cfg, thc_config, args.capture_root, args.prompt_id)
    if target_stage not in reference_bundle or target_checkpoint not in reference_bundle[target_stage]:
        raise ValueError(f"target {target_stage}/{target_checkpoint} missing from reference bundle")

    token_samples, channel_samples = _resolve_prefill_sampling(sweep_cfg, reference_bundle[target_stage][target_checkpoint])
    hash_cfg.prefill_token_samples = int(token_samples)
    hash_cfg.prefill_channel_samples = int(channel_samples)

    reference_chain = compute_hash_chain(reference_bundle[target_stage], checkpoint_order, target_stage, hash_cfg)
    c2_seed = int(hash_cfg.seed_base) + checkpoint_order.index(target_checkpoint)
    sample_values = _prefill_sample_values(
        reference_bundle[target_stage][target_checkpoint],
        token_samples=int(hash_cfg.prefill_token_samples),
        channel_samples=int(hash_cfg.prefill_channel_samples),
        seed=c2_seed,
    )
    sigma_ref = float(np.std(sample_values, dtype=np.float64))
    fallback_scale = float(dict(sweep_cfg.get("normalization", {})).get("fallback_scale", 1.0))
    scale = _effective_noise_scale(sigma_ref, fallback_scale)

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f") + "_tstc_c2_noise_sweep"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (Path(str(sweep_cfg["output_root"])).expanduser().resolve() / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    noise_values = [float(v) for v in sweep_cfg.get("noise_std_sweep", [])]
    random_seed = int(sweep_cfg.get("random_seed", 20260322))

    for noise_index, noise_std in enumerate(noise_values):
        detected_count = 0
        c2_first_mismatch_count = 0
        non_c2_first_mismatch_count = 0
        effective_noise_std = float(noise_std * scale)

        for repeat_idx in range(repetitions):
            seed = random_seed + noise_index * 100000 + repeat_idx
            perturbed_bundle = _inject_c2_noise(reference_bundle, target_stage, target_checkpoint, effective_noise_std, seed)
            candidate_chain = compute_hash_chain(perturbed_bundle[target_stage], checkpoint_order, target_stage, hash_cfg)
            mismatch_index = first_mismatch_index(reference_chain, candidate_chain)
            first_mismatch_checkpoint = checkpoint_order[mismatch_index] if mismatch_index is not None else ""
            detected = mismatch_index is not None

            if detected:
                detected_count += 1
                if first_mismatch_checkpoint == target_checkpoint:
                    c2_first_mismatch_count += 1
                else:
                    non_c2_first_mismatch_count += 1

            trial_rows.append(
                {
                    "noise_std": noise_std,
                    "effective_noise_std": effective_noise_std,
                    "sigma_ref": sigma_ref,
                    "repeat_index": repeat_idx,
                    "seed": seed,
                    "detected": bool(detected),
                    "first_mismatch_index": int(mismatch_index) if mismatch_index is not None else -1,
                    "first_mismatch_checkpoint": first_mismatch_checkpoint,
                    "stage": target_stage,
                    "checkpoint": target_checkpoint,
                }
            )

        summary_rows.append(
            {
                "noise_std": noise_std,
                "effective_noise_std": effective_noise_std,
                "sigma_ref": sigma_ref,
                "repetitions": repetitions,
                "detected_count": detected_count,
                "detection_rate": round(detected_count / repetitions, 6) if repetitions > 0 else 0.0,
                "c2_first_mismatch_count": c2_first_mismatch_count,
                "non_c2_first_mismatch_count": non_c2_first_mismatch_count,
            }
        )

    run_meta = {
        "name": str(sweep_cfg.get("name", "tstc_c2_noise_sweep")),
        "run_id": run_id,
        "reference_runtime": reference_meta["runtime"],
        "prompt_id": reference_meta["prompt_id"],
        "prompt_text": reference_meta["prompt_text"],
        "stage": target_stage,
        "checkpoint": target_checkpoint,
        "repetitions_per_noise": repetitions,
        "noise_std_sweep": noise_values,
        "sigma_ref": sigma_ref,
        "normalization_scale": scale,
        "tstc": {
            "seed_base": hash_cfg.seed_base,
            "delta_map": hash_cfg.delta_map,
            "prefill_token_samples": hash_cfg.prefill_token_samples,
            "prefill_channel_samples": hash_cfg.prefill_channel_samples,
            "prefill_token_fraction": float(dict(sweep_cfg["tstc"].get("prefill", {})).get("token_fraction", 1.0 / 3.0)),
            "prefill_channel_fraction": float(dict(sweep_cfg["tstc"].get("prefill", {})).get("channel_fraction", 1.0 / 8.0)),
            "rounding": str(dict(sweep_cfg["tstc"].get("prefill", {})).get("rounding", "ceil")),
        },
    }

    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=True)
    _write_jsonl(trial_rows, output_dir / "trial_results.jsonl")
    _write_summary_csv(summary_rows, output_dir / "summary.csv")

    paper_img_cfg = str(sweep_cfg.get("paper_img_path", "")).strip()
    paper_img_path = Path(paper_img_cfg).expanduser().resolve() if paper_img_cfg else None
    _plot_detection_counts(summary_rows, output_dir / "noise_sweep_detection_count.png", paper_img_path)

    print(f"Completed TSTC C2 noise sweep with {len(noise_values)} noise levels x {repetitions} repetitions")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
