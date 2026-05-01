from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "artifacts" / "_shared" / "thc" / "src"))

from checkpoint_qwen import capture_qwen_checkpoints
from pipeline_qwen import run_qwen_trial


def _base_config() -> dict:
    return {
        "experiment": {
            "model": "qwen",
            "seed": 7,
            "default_split": "evaluation",
            "prompt_dataset": str(ROOT / "artifacts" / "_shared" / "thc" / "data" / "qwen_prompt_splits.jsonl"),
            "shard_plan": ["provider_a", "provider_b", "provider_c"],
            "shards": [
                {"name": "Shard 1", "provider": "provider_a", "start_layer": 0, "end_layer": 1, "checkpoint": "C1"},
                {"name": "Shard 2", "provider": "provider_b", "start_layer": 2, "end_layer": 3, "checkpoint": "C2"},
                {"name": "Shard 3", "provider": "provider_c", "start_layer": 4, "end_layer": 5, "checkpoint": "C3"},
            ],
            "decode_probe": {
                "strategy": "repeat_last_prompt_token",
                "fallback_token_id": 0,
                "num_steps": 3,
            },
        },
        "qwen": {
            "model_id": "unused_in_mock",
            "enable_mlx": False,
            "allow_mock_fallback": True,
            "use_mock_if_unavailable": True,
            "enable_torch_fallback": False,
            "mock_hidden_dim": 32,
        },
        "determinism": {
            "hetero_levels": [
                {"name": "low", "noise_std": 0.000005, "fp16_cast": False},
            ]
        },
        "tamper": {
            "checkpoint": "C2",
            "strength": 0.2,
        },
        "tstc": {
            "seed_base": 2026,
            "delta_map": {
                "prefill": {"C1": 0.001, "C2": 0.001, "C3": 0.001},
                "decode": {"C1": 0.001, "C2": 0.001, "C3": 0.001},
            },
            "prefill": {
                "default": {"token_samples": 2, "channel_samples": 8},
            },
            "decode": {
                "default": {"channel_samples": 16},
            },
        },
    }


class TestPipelineQwen(unittest.TestCase):
    def test_capture_outputs_prefill_and_decode_shapes(self) -> None:
        cfg = _base_config()
        prompt = {"prompt_id": "unit_prompt", "split": "evaluation", "text": "hello world"}
        bundle, metadata_rows, runtime = capture_qwen_checkpoints(cfg, prompt, seed=7)

        self.assertEqual(runtime, "qwen_mock")
        self.assertEqual(sorted(bundle.keys()), ["decode_s01", "decode_s02", "decode_s03", "prefill"])
        self.assertEqual(bundle["decode_s01"]["C1"].shape[1], 1)
        self.assertEqual(bundle["prefill"]["C1"].ndim, 3)
        self.assertEqual(len(metadata_rows), 12)
        decode_rows = [row for row in metadata_rows if row["stage"] == "decode"]
        self.assertEqual({row["decode_step"] for row in decode_rows}, {1, 2, 3})

    def test_honest_homo_thc(self) -> None:
        cfg = _base_config()
        out = run_qwen_trial(cfg, scenario="honest_homo", verifier="thc", trial_index=0)
        stage_rows = out["records"]

        self.assertEqual({row["stage"] for row in stage_rows}, {"prefill", "decode"})
        self.assertTrue(all(not row["detected"] for row in stage_rows))
        self.assertTrue(all(row["first_mismatch_index"] == -1 for row in stage_rows))
        self.assertTrue(all(not row["false_positive"] for row in stage_rows))
        decode_row = next(row for row in stage_rows if row["stage"] == "decode")
        self.assertEqual(decode_row["decode_steps_total"], 3)
        self.assertEqual(decode_row["detected_step_count"], 0)
        self.assertEqual(len(out["checkpoint_metadata"]), 12)
        self.assertIn("decode_s03", out["checkpoint_shapes"])

    def test_honest_hetero_records_profile(self) -> None:
        cfg = _base_config()
        profile = cfg["determinism"]["hetero_levels"][0]
        out = run_qwen_trial(
            cfg,
            scenario="honest_hetero",
            verifier="thc",
            trial_index=0,
            hetero_profile=profile,
        )

        self.assertTrue(all(row["determinism_profile"].startswith("hetero_") for row in out["records"]))

    def test_tamper_localization_with_thc(self) -> None:
        cfg = _base_config()
        out = run_qwen_trial(cfg, scenario="tamper", verifier="thc", trial_index=0)
        stage_rows = out["records"]

        self.assertTrue(all(row["detected"] for row in stage_rows))
        self.assertTrue(all(row["first_mismatch_checkpoint"] == "C2" for row in stage_rows))
        self.assertTrue(all(row["localization_correct"] for row in stage_rows))
        decode_row = next(row for row in stage_rows if row["stage"] == "decode")
        self.assertGreaterEqual(int(decode_row["first_mismatch_decode_step"]), 1)

    def test_tstc_reproducible_given_same_trial_and_params(self) -> None:
        cfg = _base_config()
        params = {
            "seed_base": 2026,
            "delta_map": cfg["tstc"]["delta_map"],
            "prefill_token_samples": 2,
            "prefill_channel_samples": 8,
            "decode_channel_samples": 16,
        }

        out1 = run_qwen_trial(cfg, scenario="honest_homo", verifier="tstc", trial_index=1, hash_params=params)
        out2 = run_qwen_trial(cfg, scenario="honest_homo", verifier="tstc", trial_index=1, hash_params=params)

        rows1 = {(row["stage"], row["first_mismatch_checkpoint"], row["detected"]) for row in out1["records"]}
        rows2 = {(row["stage"], row["first_mismatch_checkpoint"], row["detected"]) for row in out2["records"]}
        self.assertEqual(rows1, rows2)

    def test_prefill_only_config_skips_decode_capture_and_records(self) -> None:
        cfg = _base_config()
        cfg["experiment"]["active_stages"] = ["prefill"]
        cfg["experiment"]["decode_probe"]["num_steps"] = 0
        prompt = {"prompt_id": "prefill_only", "split": "evaluation", "text": "hello world"}

        bundle, metadata_rows, runtime = capture_qwen_checkpoints(cfg, prompt, seed=7)
        self.assertEqual(runtime, "qwen_mock")
        self.assertEqual(sorted(bundle.keys()), ["prefill"])
        self.assertTrue(all(row["stage"] == "prefill" for row in metadata_rows))

        out = run_qwen_trial(cfg, scenario="honest_homo", verifier="thc", trial_index=0, prompt_record=prompt)
        self.assertEqual([row["stage"] for row in out["records"]], ["prefill"])


if __name__ == "__main__":
    unittest.main()
