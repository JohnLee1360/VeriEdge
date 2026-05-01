from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from checkpoint_qwen import stage_decode_step, stage_family


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate stage/checkpoint delta_map from honest checkpoint captures")
    parser.add_argument("--capture-roots", nargs="+", required=True, help="Capture directories produced by capture_qwen.py")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory")
    parser.add_argument("--percentile", type=float, default=99.0, help="Percentile used for delta calibration")
    return parser.parse_args()


def _load_npz_map(capture_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    capture_dir = capture_root / "captures"
    if not capture_dir.exists():
        raise ValueError(f"capture directory not found: {capture_dir}")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for npz_path in sorted(capture_dir.glob("*.npz")):
        prompt_id = npz_path.stem
        with np.load(npz_path) as data:
            out[prompt_id] = {key: data[key].astype(np.float32) for key in data.files}
    return out


def _iter_values(payloads: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(arr, dtype=np.float32).reshape(-1) for arr in payloads]
    if not arrays:
        return np.array([], dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def main() -> None:
    args = _parse_args()
    capture_roots = [Path(path) for path in args.capture_roots]
    machine_maps = {root.name: _load_npz_map(root) for root in capture_roots}
    common_prompts = set.intersection(*(set(prompt_map.keys()) for prompt_map in machine_maps.values()))
    if not common_prompts:
        raise ValueError("no common prompt ids across capture roots")

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/_shared/thc/output") / f"{timestamp}_delta_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], List[np.ndarray]] = {}
    step_grouped: Dict[Tuple[str, str, int, str], List[np.ndarray]] = {}

    for left_name, right_name in combinations(machine_maps.keys(), 2):
        left_prompts = machine_maps[left_name]
        right_prompts = machine_maps[right_name]
        for prompt_id in sorted(common_prompts):
            left_payload = left_prompts[prompt_id]
            right_payload = right_prompts[prompt_id]
            shared_keys = sorted(set(left_payload.keys()) & set(right_payload.keys()))
            for key in shared_keys:
                stage_key, checkpoint = key.split("__", 1)
                stage = stage_family(stage_key)
                decode_step = stage_decode_step(stage_key)
                diff = np.abs(left_payload[key].reshape(-1) - right_payload[key].reshape(-1)).astype(np.float32)
                grouped.setdefault((stage, checkpoint), []).append(diff)
                step_grouped.setdefault((stage, checkpoint, decode_step, stage_key), []).append(diff)
                raw_rows.append(
                    {
                        "machine_left": left_name,
                        "machine_right": right_name,
                        "prompt_id": prompt_id,
                        "stage": stage,
                        "stage_key": stage_key,
                        "decode_step": decode_step,
                        "checkpoint": checkpoint,
                        "count": int(diff.size),
                        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
                        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
                    }
                )

    with (output_dir / "delta_raw_records.json").open("w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2, ensure_ascii=True)

    summary_rows: List[Dict[str, object]] = []
    delta_map = {"prefill": {}, "decode": {}}
    for (stage, checkpoint), diffs in sorted(grouped.items()):
        values = _iter_values(diffs)
        percentile_value = float(np.percentile(values, args.percentile)) if values.size else 0.0
        summary_rows.append(
            {
                "stage": stage,
                "checkpoint": checkpoint,
                "count": int(values.size),
                "mean_abs_diff": float(np.mean(values)) if values.size else 0.0,
                "max_abs_diff": float(np.max(values)) if values.size else 0.0,
                "p95_abs_diff": float(np.percentile(values, 95.0)) if values.size else 0.0,
                "p99_abs_diff": float(np.percentile(values, 99.0)) if values.size else 0.0,
                "selected_percentile": float(args.percentile),
                "delta": percentile_value,
            }
        )
        delta_map[stage][checkpoint] = percentile_value

    step_summary_rows: List[Dict[str, object]] = []
    for (stage, checkpoint, decode_step, stage_key), diffs in sorted(
        step_grouped.items(),
        key=lambda item: (
            0 if item[0][0] == "prefill" else 1,
            item[0][2],
            item[0][1],
            item[0][3],
        ),
    ):
        values = _iter_values(diffs)
        step_summary_rows.append(
            {
                "stage": stage,
                "stage_key": stage_key,
                "decode_step": int(decode_step),
                "checkpoint": checkpoint,
                "count": int(values.size),
                "mean_abs_diff": float(np.mean(values)) if values.size else 0.0,
                "max_abs_diff": float(np.max(values)) if values.size else 0.0,
                "p95_abs_diff": float(np.percentile(values, 95.0)) if values.size else 0.0,
                "p99_abs_diff": float(np.percentile(values, 99.0)) if values.size else 0.0,
                "selected_percentile": float(args.percentile),
                "delta": float(np.percentile(values, args.percentile)) if values.size else 0.0,
            }
        )

    with (output_dir / "delta_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "checkpoint",
                "count",
                "mean_abs_diff",
                "max_abs_diff",
                "p95_abs_diff",
                "p99_abs_diff",
                "selected_percentile",
                "delta",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (output_dir / "delta_step_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "stage_key",
                "decode_step",
                "checkpoint",
                "count",
                "mean_abs_diff",
                "max_abs_diff",
                "p95_abs_diff",
                "p99_abs_diff",
                "selected_percentile",
                "delta",
            ],
        )
        writer.writeheader()
        for row in step_summary_rows:
            writer.writerow(row)

    with (output_dir / "delta_map.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "percentile": float(args.percentile),
                "delta_map": delta_map,
                "note": "Prototype-level empirical calibration under the provided capture roots.",
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    print(f"Wrote delta calibration outputs to {output_dir}")


if __name__ == "__main__":
    main()
