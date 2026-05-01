from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_plot_script_generates_files(tmp_path):
    summary = tmp_path / "summary_by_cell.csv"
    pd.DataFrame(
        [
            {
                "network": "LAN",
                "instance_node_count": 1,
                "mean_task_latency_s_per_task": 10.0,
                "mean_question_latency_s_per_q": 0.2,
                "mean_download_s_per_task": 1.0,
                "mean_ttft_p50_s": 0.5,
                "mean_otps_p50_tok_s": 12.0,
            },
            {
                "network": "WAN",
                "instance_node_count": 2,
                "mean_task_latency_s_per_task": 12.0,
                "mean_question_latency_s_per_q": 0.24,
                "mean_download_s_per_task": 2.0,
                "mean_ttft_p50_s": 0.8,
                "mean_otps_p50_tok_s": 11.0,
            },
        ]
    ).to_csv(summary, index=False)

    plot_script = Path(__file__).resolve().parents[1] / "plot.py"
    out_dir = tmp_path / "plots"

    subprocess.run([sys.executable, str(plot_script), "--input", str(summary), "--output-dir", str(out_dir)], check=True)

    assert (out_dir / "mean_task_latency_s_per_task.png").exists()
    assert (out_dir / "mean_question_latency_s_per_q.png").exists()
