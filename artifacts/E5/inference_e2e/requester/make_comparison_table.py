#!/usr/bin/env python3
"""Generate paper-ready task-level EXO summary tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "instance_node_count",
    "network",
    "mean_task_latency_s_per_task",
    "mean_question_latency_s_per_q",
    "mean_download_s_per_task",
    "mean_ttft_p50_s",
    "mean_otps_p50_tok_s",
    "sum_question_success_count",
    "sum_question_fail_count",
    "completed_task_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build direct summary table from summary_by_cell.csv")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def _fmt(v: object, digits: int = 3) -> str:
    if pd.isna(v):
        return "NaN"
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.{digits}f}"
    except Exception:  # noqa: BLE001
        return str(v)


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| Node Count | Network | Task Latency (mean s/task) | Question Latency (mean s/q) | Download (mean s/task) | TTFT (mean p50 s/task) | OTPS (mean p50 tok/s/task) | Question Success (sum) | Question Fail (sum) | Completed Tasks |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["instance_node_count"])),
                    str(row["network"]),
                    _fmt(row["mean_task_latency_s_per_task"]),
                    _fmt(row["mean_question_latency_s_per_q"]),
                    _fmt(row["mean_download_s_per_task"]),
                    _fmt(row["mean_ttft_p50_s"]),
                    _fmt(row["mean_otps_p50_tok_s"]),
                    _fmt(row["sum_question_success_count"], 0),
                    _fmt(row["sum_question_fail_count"], 0),
                    _fmt(row["completed_task_count"], 0),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    present = [col for col in COLUMNS if col in df.columns]
    df = df[present].sort_values(["network", "instance_node_count"])
    output_csv = Path(args.output_csv) if args.output_csv else input_path.parent / "comparison_table.csv"
    output_md = Path(args.output_md) if args.output_md else input_path.parent / "comparison_table.md"
    df.to_csv(output_csv, index=False)
    output_md.write_text(build_markdown(df), encoding="utf-8")
    print(f"comparison csv: {output_csv}")
    print(f"comparison md : {output_md}")


if __name__ == "__main__":
    main()
