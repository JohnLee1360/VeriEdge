from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "make_comparison_table.py"
    spec = importlib.util.spec_from_file_location("compare_table", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_markdown():
    mod = _load_module()
    df = pd.DataFrame(
        [
            {
                "instance_node_count": 1,
                "network": "LAN",
                "mean_task_latency_s_per_task": 10.0,
                "mean_question_latency_s_per_q": 0.2,
                "mean_download_s_per_task": 1.0,
                "mean_ttft_p50_s": 0.5,
                "mean_otps_p50_tok_s": 12.0,
                "sum_question_success_count": 250,
                "sum_question_fail_count": 0,
                "completed_task_count": 5,
            },
            {
                "instance_node_count": 2,
                "network": "WAN",
                "mean_task_latency_s_per_task": 12.0,
                "mean_question_latency_s_per_q": 0.24,
                "mean_download_s_per_task": 2.0,
                "mean_ttft_p50_s": 0.8,
                "mean_otps_p50_tok_s": 11.0,
                "sum_question_success_count": 245,
                "sum_question_fail_count": 5,
                "completed_task_count": 5,
            },
        ]
    )

    md = mod.build_markdown(df)
    assert "Question Latency (mean s/q)" in md
    assert "| 1 | LAN | 10.000 | 0.200 | 1.000 |" in md
    assert "| 2 | WAN | 12.000 | 0.240 | 2.000 |" in md
