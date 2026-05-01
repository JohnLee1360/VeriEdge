from __future__ import annotations

from pathlib import Path

import numpy as np

from checkpoint_qwen import write_capture_bundle
from overhead_report import collect_overhead_rows, measure_trace_from_bundle


def _config() -> dict:
    return {
        "experiment": {
            "seed": 7,
            "hash_modes": ["thc", "tstc"],
            "prompt_dataset": "unused.jsonl",
            "shard_plan": ["provider_a", "provider_b", "provider_c"],
            "shards": [
                {"provider": "provider_a", "checkpoint": "C1"},
                {"provider": "provider_b", "checkpoint": "C2"},
                {"provider": "provider_c", "checkpoint": "C3"},
            ],
            "active_stages": ["prefill"],
        },
        "tamper": {"checkpoint": "C2", "strength": 0.15},
        "determinism": {
            "hetero_levels": [
                {"name": "mid", "noise_std": 0.00001, "fp16_cast": False},
            ]
        },
        "tstc": {
            "seed_base": 2026,
            "delta_map": {"prefill": {"C1": 0.001, "C2": 0.001, "C3": 0.001}},
            "prefill": {"default": {"token_samples": 2, "channel_samples": 2}},
            "decode": {"default": {"channel_samples": 2}},
        },
    }


def _bundle() -> dict:
    return {
        "prefill": {
            "C1": np.ones((1, 2, 4), dtype=np.float32),
            "C2": np.full((1, 2, 4), 2.0, dtype=np.float32),
            "C3": np.full((1, 2, 4), 3.0, dtype=np.float32),
        }
    }


def _metadata(prompt_id: str) -> list[dict]:
    return [
        {
            "prompt_id": prompt_id,
            "split": "evaluation",
            "stage": "prefill",
            "stage_key": "prefill",
            "decode_step": 0,
            "checkpoint": checkpoint,
            "shape": [1, 2, 4],
            "provider_plan": ["provider_a", "provider_b", "provider_c"],
            "provider": provider,
            "runtime": "unit_test",
        }
        for checkpoint, provider in [("C1", "provider_a"), ("C2", "provider_b"), ("C3", "provider_c")]
    ]


def test_measure_trace_detects_tamper() -> None:
    size_row, latency_row, storage_rows = measure_trace_from_bundle(
        config=_config(),
        prompt_record={"prompt_id": "p1", "split": "evaluation", "text": "hello"},
        bundle=_bundle(),
        metadata_rows=_metadata("p1"),
        scenario="tamper",
        verifier="thc",
        hetero_profile=None,
        capture_generation_ms=0.0,
        capture_load_ms=1.0,
        capture_file_bytes=128,
    )
    assert size_row["commitment_head_bytes"] > 0
    assert latency_row["detected"] is True
    assert latency_row["first_mismatch_checkpoint"] == "C2"
    assert latency_row["challenge_latency_ms"] >= latency_row["replay_ms"]
    assert any(row["component"] == "validator_storage_full_chain" for row in storage_rows)


def test_collect_overhead_rows_from_capture_root(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture_root"
    capture_dir = capture_root / "captures"
    prompt_record = {"prompt_id": "p2", "split": "evaluation", "text": "world"}
    write_capture_bundle(capture_dir, prompt_record, _bundle(), _metadata("p2"))

    size_rows, latency_rows, storage_rows = collect_overhead_rows(
        config=_config(),
        capture_root=capture_root,
        split="evaluation",
        prompt_ids=["p2"],
        limit_prompts=0,
        scenarios=["honest_homo", "tamper"],
        verifiers=["thc"],
        hetero_profile_selector="default",
    )

    assert len(size_rows) == 2
    assert len(latency_rows) == 2
    assert len(storage_rows) == 14
    assert {row["scenario"] for row in size_rows} == {"honest_homo", "tamper"}
