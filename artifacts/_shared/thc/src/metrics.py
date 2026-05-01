from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SUMMARY_COLUMNS = [
    "scenario",
    "stage",
    "hetero_level",
    "model",
    "verifier",
    "tpr",
    "fpr",
    "localization_acc",
]


def write_raw_json(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, indent=2, ensure_ascii=True)


def write_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def aggregate_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    scenario_order = {"honest_homo": 0, "honest_hetero": 1, "tamper": 2}
    stage_order = {"prefill": 0, "decode": 1}

    def _group_key(record: Dict[str, Any], hetero_level: str) -> Tuple[str, str, str, str, str]:
        return (
            str(record["scenario"]),
            str(record["stage"]),
            hetero_level,
            str(record["model"]),
            str(record["verifier"]),
        )

    for record in records:
        scenario = str(record["scenario"])
        hetero_level = str(record.get("hetero_level", ""))
        grouped[_group_key(record, "")].append(record)
        if scenario == "honest_hetero" and hetero_level:
            grouped[_group_key(record, hetero_level)].append(record)

    rows: List[Dict[str, Any]] = []
    for (scenario, stage, hetero_level, model, verifier), bucket in sorted(
        grouped.items(),
        key=lambda kv: (
            scenario_order.get(kv[0][0], 99),
            stage_order.get(kv[0][1], 99),
            kv[0][2],
            kv[0][3],
            kv[0][4],
        ),
    ):
        n = len(bucket)
        if n <= 0:
            continue

        detected_rate = sum(1 for r in bucket if r.get("detected")) / n
        localization_acc = sum(1 for r in bucket if r.get("localization_correct")) / n

        row: Dict[str, Any] = {
            "scenario": scenario,
            "stage": stage,
            "hetero_level": hetero_level,
            "model": model,
            "verifier": verifier,
            "tpr": "",
            "fpr": "",
            "localization_acc": "",
        }

        if scenario == "tamper":
            row["tpr"] = round(detected_rate, 6)
            row["localization_acc"] = round(localization_acc, 6)
        else:
            row["fpr"] = round(detected_rate, 6)

        rows.append(row)

    return rows


def write_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})
