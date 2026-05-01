from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

PROMPT_RE = re.compile(r"^\s*\d+\.\s+(.*\S)\s*$")


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile_or_nan(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), p))


def dedupe_preserve(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        item = str(raw).strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_prompts_from_markdown(path: Path) -> List[Tuple[int, str]]:
    prompts: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = PROMPT_RE.match(line.rstrip("\n"))
            if not m:
                continue
            text = m.group(1).strip()
            if text:
                prompts.append((len(prompts) + 1, text))
    if not prompts:
        raise ValueError(f"No numbered prompts found in {path}")
    return prompts


def prompt_lookup(prompts: Sequence[Tuple[int, str]]) -> dict[int, str]:
    return {int(prompt_id): str(text) for prompt_id, text in prompts}


def select_prompts_by_ids(
    prompts: Sequence[Tuple[int, str]],
    *,
    prompt_ids: Sequence[int],
) -> List[Tuple[int, str]]:
    lookup = prompt_lookup(prompts)
    selected: List[Tuple[int, str]] = []
    missing: List[int] = []
    for raw_id in prompt_ids:
        prompt_id = int(raw_id)
        text = lookup.get(prompt_id)
        if text is None:
            missing.append(prompt_id)
            continue
        selected.append((prompt_id, text))
    if missing:
        raise ValueError(f"Missing prompt ids: {missing[:10]}")
    return selected


def expand_prompt_id_spec(prompt_spec: Any) -> List[int]:
    if isinstance(prompt_spec, list):
        return [int(item) for item in prompt_spec]
    if isinstance(prompt_spec, dict):
        if "prompt_ids" in prompt_spec:
            return [int(item) for item in prompt_spec["prompt_ids"]]
        if "prompt_id_range" in prompt_spec:
            raw_range = prompt_spec["prompt_id_range"]
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ValueError(f"Invalid prompt_id_range: {raw_range}")
            start = int(raw_range[0])
            end = int(raw_range[1])
            if start > end:
                raise ValueError(f"Invalid prompt_id_range start>end: {raw_range}")
            return list(range(start, end + 1))
    raise ValueError(f"Unsupported prompt spec: {prompt_spec}")


def select_task_prompts(
    prompts: List[Tuple[int, str]],
    *,
    question_count: int,
    seed: int,
) -> List[Tuple[int, str]]:
    if question_count <= 0:
        raise ValueError("question_count must be positive")
    if question_count > len(prompts):
        raise ValueError(
            f"question_count={question_count} exceeds available prompts={len(prompts)}"
        )
    rng = random.Random(seed)
    return rng.sample(prompts, k=question_count)


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
