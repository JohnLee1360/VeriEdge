from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "artifacts" / "_shared" / "thc" / "src"))

from hash_chain import HashConfig, compute_hash_chain, first_mismatch_index


def _tstc_cfg() -> HashConfig:
    return HashConfig(
        mode="tstc",
        seed_base=2026,
        delta_map={
            "prefill": {"C1": 1e-2, "C2": 1e-2, "C3": 1e-2},
            "decode": {"C1": 1e-2, "C2": 1e-2, "C3": 1e-2},
        },
        prefill_token_samples=2,
        prefill_channel_samples=2,
        decode_channel_samples=2,
    )


class TestHashChain(unittest.TestCase):
    def test_thc_chain_is_deterministic(self) -> None:
        checkpoints = {
            "C1": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            "C2": np.array([[[0.5, -0.2], [1.2, 2.2]]], dtype=np.float32),
        }
        cfg = HashConfig(mode="thc")
        chain_1 = compute_hash_chain(checkpoints, ["C1", "C2"], "prefill", cfg)
        chain_2 = compute_hash_chain(checkpoints, ["C1", "C2"], "prefill", cfg)

        self.assertEqual(chain_1, chain_2)
        self.assertIsNone(first_mismatch_index(chain_1, chain_2))

    def test_tstc_sampling_is_reproducible_for_prefill(self) -> None:
        checkpoints = {"C1": np.array([[[1.111111, 2.222222], [3.333333, 4.444444]]], dtype=np.float32)}
        cfg = _tstc_cfg()

        chain_1 = compute_hash_chain(checkpoints, ["C1"], "prefill", cfg)
        chain_2 = compute_hash_chain(checkpoints, ["C1"], "prefill", cfg)

        self.assertEqual(chain_1, chain_2)

    def test_tstc_can_absorb_micro_noise_with_delta(self) -> None:
        base = {
            "C1": np.array([[[1.10, 2.20, 3.30, 4.40]]], dtype=np.float32),
        }
        perturbed = {
            "C1": base["C1"] + np.array([[[0.0002, -0.0002, 0.0003, -0.0003]]], dtype=np.float32),
        }

        thc_cfg = HashConfig(mode="thc")
        tstc_cfg = _tstc_cfg()
        tstc_cfg.decode_channel_samples = 4

        thc_chain_base = compute_hash_chain(base, ["C1"], "decode", thc_cfg)
        thc_chain_perturbed = compute_hash_chain(perturbed, ["C1"], "decode", thc_cfg)
        tstc_chain_base = compute_hash_chain(base, ["C1"], "decode", tstc_cfg)
        tstc_chain_perturbed = compute_hash_chain(perturbed, ["C1"], "decode", tstc_cfg)

        self.assertNotEqual(thc_chain_base, thc_chain_perturbed)
        self.assertEqual(tstc_chain_base, tstc_chain_perturbed)

    def test_first_mismatch_index_detects_difference(self) -> None:
        reference = ["aa", "bb", "cc"]
        candidate = ["aa", "xx", "yy"]
        self.assertEqual(first_mismatch_index(reference, candidate), 1)


if __name__ == "__main__":
    unittest.main()
