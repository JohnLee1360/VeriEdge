from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "artifacts" / "_shared" / "thc" / "src" / "calibrate_delta.py"


def _write_capture_root(root: Path, *, offset: float = 0.0) -> None:
    capture_dir = root / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        capture_dir / "prompt_a.npz",
        prefill__C1=np.zeros((1, 4, 2), dtype=np.float32) + offset,
        prefill__C2=np.ones((1, 4, 2), dtype=np.float32),
        decode_s01__C1=np.zeros((1, 1, 2), dtype=np.float32),
        decode_s02__C1=np.zeros((1, 1, 2), dtype=np.float32) + offset,
    )


class CalibrateDeltaTests(unittest.TestCase):
    def test_calibrate_delta_emits_step_summary_and_nonzero_decode_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root_a = tmp / "run_a"
            root_b = tmp / "run_b"
            output_dir = tmp / "delta_out"
            _write_capture_root(root_a, offset=0.0)
            _write_capture_root(root_b, offset=0.01)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output-dir",
                    str(output_dir),
                    "--capture-roots",
                    str(root_a),
                    str(root_b),
                ],
                check=True,
                cwd=str(ROOT),
            )

            delta_map = json.loads((output_dir / "delta_map.json").read_text(encoding="utf-8"))
            self.assertGreater(float(delta_map["delta_map"]["prefill"]["C1"]), 0.0)
            self.assertGreater(float(delta_map["delta_map"]["decode"]["C1"]), 0.0)

            with (output_dir / "delta_step_summary.csv").open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            decode_rows = [row for row in rows if row["stage_key"] == "decode_s02" and row["checkpoint"] == "C1"]
            self.assertEqual(len(decode_rows), 1)
            self.assertGreater(float(decode_rows[0]["delta"]), 0.0)

    def test_identical_roots_produce_zero_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root_a = tmp / "run_a"
            root_b = tmp / "run_b"
            output_dir = tmp / "delta_out"
            _write_capture_root(root_a, offset=0.0)
            _write_capture_root(root_b, offset=0.0)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output-dir",
                    str(output_dir),
                    "--capture-roots",
                    str(root_a),
                    str(root_b),
                ],
                check=True,
                cwd=str(ROOT),
            )

            delta_map = json.loads((output_dir / "delta_map.json").read_text(encoding="utf-8"))
            self.assertEqual(float(delta_map["delta_map"]["prefill"]["C1"]), 0.0)
            self.assertEqual(float(delta_map["delta_map"]["decode"]["C1"]), 0.0)


if __name__ == "__main__":
    unittest.main()
