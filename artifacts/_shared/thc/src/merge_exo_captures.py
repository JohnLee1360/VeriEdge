from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_KEYS = {
    "prefill__C1",
    "prefill__C2",
    "prefill__C3",
    "decode__C1",
    "decode__C2",
    "decode__C3",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-node exo capture outputs into a calibration-ready capture root"
    )
    parser.add_argument("--node-roots", nargs="+", required=True, help="Per-node capture roots produced by exo_capture_runner.py")
    parser.add_argument("--output-dir", required=True, help="Merged capture root")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    node_roots = [Path(path).expanduser().resolve() for path in args.node_roots]
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    merged_capture_dir = output_dir / "captures"
    merged_capture_dir.mkdir(parents=True, exist_ok=True)

    prompt_payloads: dict[str, dict[str, np.ndarray]] = {}
    merged_meta: list[dict[str, Any]] = []

    for node_root in node_roots:
        capture_dir = node_root / "captures"
        if not capture_dir.exists():
            raise ValueError(f"capture directory not found: {capture_dir}")
        for npz_path in sorted(capture_dir.glob("*.npz")):
            prompt_id = npz_path.stem
            bucket = prompt_payloads.setdefault(prompt_id, {})
            with np.load(npz_path) as data:
                for key in data.files:
                    if key in bucket:
                        raise ValueError(f"duplicate payload for prompt={prompt_id} key={key}")
                    bucket[key] = data[key].astype(np.float32)

        metadata_path = node_root / "checkpoint_metadata.jsonl"
        if metadata_path.exists():
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if text:
                    merged_meta.append(json.loads(text))

    for prompt_id, payloads in prompt_payloads.items():
        missing = sorted(EXPECTED_KEYS - set(payloads.keys()))
        extra = sorted(set(payloads.keys()) - EXPECTED_KEYS)
        if missing or extra:
            raise ValueError(
                f"merged prompt={prompt_id} has invalid key set; missing={missing} extra={extra}"
            )
        np.savez_compressed(merged_capture_dir / f"{prompt_id}.npz", **payloads)

    with (output_dir / "checkpoint_metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in merged_meta:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "prompt_count": len(prompt_payloads),
        "node_roots": [str(path) for path in node_roots],
        "keys_per_prompt": {prompt_id: sorted(payloads.keys()) for prompt_id, payloads in prompt_payloads.items()},
    }
    with (output_dir / "merge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Merged exo captures into {output_dir}")


if __name__ == "__main__":
    main()
