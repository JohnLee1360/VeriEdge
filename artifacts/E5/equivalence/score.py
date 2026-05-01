from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple


FINAL_ANSWER_RE = re.compile(
    r"(?im)^\s*Final\s+Answer\s*:\s*(.+?)\s*$"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_prediction(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str):
        return None, None
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip(), "final_answer_line"
    number_matches = NUMBER_RE.findall(text)
    if number_matches:
        return number_matches[-1].strip(), "fallback"
    stripped = text.strip()
    if stripped:
        return stripped, "fallback"
    return None, None


def normalize_numeric_answer(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = text.strip(" \t\r\n.,;:!?()[]{}<>`'\"")
    text = text.replace("$", "").replace("%", "")
    if not text:
        return None
    number_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    if number_match is None:
        fallback = NUMBER_RE.findall(text)
        if not fallback:
            return None
        text = fallback[-1].replace(",", "")
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return None
    normalized = format(decimal_value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized


def extract_gsm8k_gold_answer(answer_text: str) -> Optional[str]:
    if not isinstance(answer_text, str):
        return None
    marker = "####"
    if marker in answer_text:
        candidate = answer_text.split(marker)[-1].strip()
        normalized = normalize_numeric_answer(candidate)
        if normalized is not None:
            return normalized
    matches = NUMBER_RE.findall(answer_text)
    if not matches:
        return None
    return normalize_numeric_answer(matches[-1])


def is_exact_match(prediction: Optional[str], gold: Optional[str]) -> bool:
    if prediction is None or gold is None:
        return False
    return prediction == gold
