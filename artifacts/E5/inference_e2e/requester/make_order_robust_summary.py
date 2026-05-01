#!/usr/bin/env python3
"""Merge two batch runs with different execution orders and build median summary."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


SUMMARY_CANDIDATES: List[str] = ["summary_by_cell.csv"]
ORDER_MAP = {"LAN": 0, "WAN": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two batch summaries and output median-by-cell results."
    )
    parser.add_argument("--batch-a", required=True, help="Path to first batch dir")
    parser.add_argument("--batch-b", required=True, help="Path to second batch dir")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: sibling folder next to batch-a)",
    )
    return parser.parse_args()


def pick_summary_file(batch_dir: Path) -> Path:
    for name in SUMMARY_CANDIDATES:
        path = batch_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No summary file found in {batch_dir} (tried {SUMMARY_CANDIDATES})")


def main() -> None:
    args = parse_args()
    batch_a = Path(args.batch_a).resolve()
    batch_b = Path(args.batch_b).resolve()
    if not batch_a.is_dir():
        raise SystemExit(f"batch-a is not a directory: {batch_a}")
    if not batch_b.is_dir():
        raise SystemExit(f"batch-b is not a directory: {batch_b}")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = batch_a.parent / f"batch_median_{batch_a.name}__{batch_b.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_a = pick_summary_file(batch_a)
    summary_b = pick_summary_file(batch_b)

    df_a = pd.read_csv(summary_a)
    df_a.insert(0, "source_batch", batch_a.name)
    df_b = pd.read_csv(summary_b)
    df_b.insert(0, "source_batch", batch_b.name)
    combined = pd.concat([df_a, df_b], ignore_index=True)

    combined_path = output_dir / "summary_runs_combined.csv"
    combined.to_csv(combined_path, index=False)

    key_cols = ["network", "instance_node_count"]
    metric_cols = [
        "mean_task_latency_s_per_task",
        "mean_question_latency_s_per_q",
        "mean_download_s_per_task",
        "mean_ttft_p50_s",
        "mean_otps_p50_tok_s",
        "sum_question_success_count",
        "sum_question_fail_count",
        "completed_task_count",
    ]
    metric_cols = [c for c in metric_cols if c in combined.columns]

    median_df = combined.groupby(key_cols, as_index=False)[metric_cols].median()
    for int_col in ["sum_question_success_count", "sum_question_fail_count", "completed_task_count"]:
        if int_col in median_df.columns:
            median_df[int_col] = median_df[int_col].round().astype(int)

    median_df["network_order"] = median_df["network"].map(ORDER_MAP).fillna(999)
    median_df = median_df.sort_values(["network_order", "instance_node_count"]).drop(columns=["network_order"])

    median_path = output_dir / "summary_by_cell_median.csv"
    median_df.to_csv(median_path, index=False)

    table_script = Path(__file__).resolve().parent / "make_comparison_table.py"
    comp_csv = output_dir / "comparison_table_median.csv"
    comp_md = output_dir / "comparison_table_median.md"
    subprocess.check_call(
        [
            sys.executable,
            str(table_script),
            "--input",
            str(median_path),
            "--output-csv",
            str(comp_csv),
            "--output-md",
            str(comp_md),
        ]
    )

    print(f"combined summary: {combined_path}")
    print(f"median summary  : {median_path}")
    print(f"comparison csv  : {comp_csv}")
    print(f"comparison md   : {comp_md}")


if __name__ == "__main__":
    main()
