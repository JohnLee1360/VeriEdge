from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import score  # noqa: E402


def test_extract_prediction_prefers_final_answer_line():
    text = "Reasoning\nFinal Answer: 1,234\n"
    pred, source = score.extract_prediction(text)
    assert pred == "1,234"
    assert source == "final_answer_line"


def test_extract_prediction_fallback():
    pred, source = score.extract_prediction("The answer is 77.")
    assert pred == "77"
    assert source == "fallback"


def test_normalize_numeric_answer():
    assert score.normalize_numeric_answer("1,230.00") == "1230"
    assert score.normalize_numeric_answer(" -0.500 ") == "-0.5"


def test_extract_gsm8k_gold_answer():
    assert score.extract_gsm8k_gold_answer("Some steps #### 42") == "42"


def test_is_exact_match():
    assert score.is_exact_match("12", "12") is True
    assert score.is_exact_match("12", "13") is False
