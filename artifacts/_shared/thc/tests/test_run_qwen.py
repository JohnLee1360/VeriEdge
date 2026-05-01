from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "artifacts" / "_shared" / "thc" / "src"))

matplotlib = types.ModuleType("matplotlib")
pyplot = types.ModuleType("matplotlib.pyplot")
matplotlib.pyplot = pyplot
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.pyplot", pyplot)

from run import _summarize_candidate


class TestRunQwen(unittest.TestCase):
    def test_summarize_candidate_prefill_only(self) -> None:
        records = [
            {"scenario": "tamper", "stage": "prefill", "detected": True, "localization_correct": True},
            {"scenario": "tamper", "stage": "prefill", "detected": True, "localization_correct": True},
            {"scenario": "honest_hetero", "stage": "prefill", "detected": False, "localization_correct": False},
            {"scenario": "honest_hetero", "stage": "prefill", "detected": True, "localization_correct": False},
        ]
        summary = _summarize_candidate(records)
        self.assertEqual(summary["prefill_fpr"], 0.5)
        self.assertEqual(summary["prefill_tpr"], 1.0)
        self.assertEqual(summary["prefill_loc"], 1.0)
        self.assertEqual(summary["max_fpr"], 0.5)
        self.assertEqual(summary["min_tpr"], 1.0)
        self.assertEqual(summary["min_localization_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
