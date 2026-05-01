#!/usr/bin/env python3
"""Plot task-level benchmark curves from summary_by_cell.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_SPECS = [
    ("mean_task_latency_s_per_task", "Task Latency (mean s/task)", "mean_task_latency_s_per_task.png"),
    ("mean_question_latency_s_per_q", "Question Latency (mean s/q)", "mean_question_latency_s_per_q.png"),
    ("mean_download_s_per_task", "Download (mean s/task)", "mean_download_s_per_task.png"),
    ("mean_ttft_p50_s", "TTFT (mean p50 s/task)", "mean_ttft_p50_s.png"),
    ("mean_otps_p50_tok_s", "OTPS (mean p50 tok/s/task)", "mean_otps_p50_tok_s.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot task-level EXO summary curves")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for network in sorted(df["network"].unique()):
        sub = df[df["network"] == network].sort_values("instance_node_count")
        ax.plot(sub["instance_node_count"], sub[metric], marker="o", linewidth=2, label=network)
    ax.set_xlabel("Node Count")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Node Count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows found in {input_path}")
    for metric, ylabel, filename in METRIC_SPECS:
        if metric in df.columns:
            plot_metric(df, metric, ylabel, output_dir / filename)
    print(f"Plots generated under: {output_dir}")


if __name__ == "__main__":
    main()
