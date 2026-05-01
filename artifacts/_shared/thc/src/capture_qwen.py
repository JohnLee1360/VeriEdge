from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

from checkpoint_qwen import capture_qwen_checkpoints, load_prompt_records, write_capture_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Qwen shard-boundary checkpoints for THC/TSTC")
    parser.add_argument("--config", required=True, help="Path to qwen config JSON file")
    parser.add_argument("--split", default="evaluation", help="Prompt split to export")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of prompts to capture")
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory")
    parser.add_argument("--decode-steps", type=int, default=0, help="Optional override for decode probe steps")
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()
    config = _load_config(Path(args.config))
    if int(args.decode_steps) > 0:
        exp_cfg = dict(config.get("experiment", {}))
        probe_cfg = dict(exp_cfg.get("decode_probe", {}))
        probe_cfg["num_steps"] = int(args.decode_steps)
        exp_cfg["decode_probe"] = probe_cfg
        config["experiment"] = exp_cfg
    prompts = load_prompt_records(config, split=args.split)
    if args.limit > 0:
        prompts = prompts[: int(args.limit)]
    if not prompts:
        raise ValueError(f"no prompts found for split={args.split}")

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["experiment"]["output_root"]) / f"{timestamp}_capture_{args.split}"
    export_dir = output_dir / "captures"
    export_dir.mkdir(parents=True, exist_ok=True)

    seed_base = int(config["experiment"].get("seed", 7))
    summary: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        bundle, metadata_rows, runtime = capture_qwen_checkpoints(config, prompt, seed_base + index)
        npz_path, meta_path = write_capture_bundle(export_dir, prompt, bundle, metadata_rows)
        summary.append(
            {
                "prompt_id": prompt["prompt_id"],
                "split": prompt["split"],
                "runtime": runtime,
                "npz_path": str(npz_path),
                "metadata_path": str(meta_path),
            }
        )

    with (output_dir / "capture_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Captured {len(summary)} prompt(s) to {output_dir}")


if __name__ == "__main__":
    main()
