#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from equivalence_common import SETTING_COLORS, SETTING_LABELS, SETTING_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate independent EXO equivalence results into polished comparison figures")
    parser.add_argument("--results-root", default="artifacts/E5/equivalence/output")
    parser.add_argument("--output-dir", default="", help="Directory to write figures and comparison summaries")
    parser.add_argument("--output", default="", help="Optional path for the composite summary PNG")
    return parser.parse_args()


def load_results_by_setting(results_root: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for setting in SETTING_ORDER:
        path = results_root / setting / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"missing results for {setting}: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if str(payload.get("setting", "")) != setting:
            raise ValueError(f"results file {path} does not match setting={setting}")
        out[setting] = payload
    return out


def validate_question_alignment(results_by_setting: Dict[str, Dict[str, Any]]) -> List[int]:
    reference_ids: List[int] = []
    for setting in SETTING_ORDER:
        questions = results_by_setting[setting].get("questions", [])
        sample_ids = [int(row["sample_id"]) for row in questions]
        if not reference_ids:
            reference_ids = sample_ids
            continue
        if sample_ids != reference_ids:
            raise ValueError(f"sample ordering mismatch for {setting}")
    return reference_ids


def accuracy_values(results_by_setting: Dict[str, Dict[str, Any]]) -> List[float]:
    return [float(results_by_setting[name]["summary"].get("accuracy") or 0.0) for name in SETTING_ORDER]


def question_result(question_row: Dict[str, Any]) -> Dict[str, Any]:
    result = question_row.get("result")
    return result if isinstance(result, dict) else {}


def pairwise_similarity_stats(results_by_setting: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for left in SETTING_ORDER:
        left_questions = results_by_setting[left]["questions"]
        for right in SETTING_ORDER:
            right_questions = results_by_setting[right]["questions"]
            total = len(left_questions)
            same_answer = 0
            same_correctness = 0
            disagreement_sample_ids: List[int] = []
            for left_row, right_row in zip(left_questions, right_questions):
                left_result = question_result(left_row)
                right_result = question_result(right_row)
                if left_result.get("normalized_answer") == right_result.get("normalized_answer"):
                    same_answer += 1
                if left_result.get("is_correct") == right_result.get("is_correct"):
                    same_correctness += 1
                if left_result.get("is_correct") != right_result.get("is_correct"):
                    disagreement_sample_ids.append(int(left_row["sample_id"]))
            stats[(left, right)] = {
                "answer_match_rate": (same_answer / total) if total else 0.0,
                "correctness_agreement_rate": (same_correctness / total) if total else 0.0,
                "correctness_disagreement_count": len(disagreement_sample_ids),
                "correctness_disagreement_sample_ids": disagreement_sample_ids,
            }
    return stats


def correctness_matrix(results_by_setting: Dict[str, Dict[str, Any]]) -> Tuple[List[List[int]], List[int]]:
    matrix: List[List[int]] = []
    disagreement_columns: List[int] = []
    sample_count = len(results_by_setting[SETTING_ORDER[0]]["questions"])
    for setting in SETTING_ORDER:
        row: List[int] = []
        for question_row in results_by_setting[setting]["questions"]:
            row.append(1 if question_result(question_row).get("is_correct") is True else 0)
        matrix.append(row)
    for idx in range(sample_count):
        values = {matrix[row_idx][idx] for row_idx in range(len(SETTING_ORDER))}
        if len(values) > 1:
            disagreement_columns.append(idx)
    return matrix, disagreement_columns


def disagreement_rows(results_by_setting: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sample_ids = validate_question_alignment(results_by_setting)
    for index, sample_id in enumerate(sample_ids):
        per_setting: Dict[str, Dict[str, Any]] = {}
        normalized_answers = set()
        correctness = set()
        gold_answer: Any = None
        question_text = ""
        for setting in SETTING_ORDER:
            question_row = results_by_setting[setting]["questions"][index]
            result = question_result(question_row)
            gold_answer = question_row.get("gold_final_answer")
            question_text = str(question_row.get("question", ""))
            normalized_answers.add(result.get("normalized_answer"))
            correctness.add(result.get("is_correct"))
            per_setting[setting] = {
                "normalized_answer": result.get("normalized_answer"),
                "predicted_answer": result.get("predicted_answer"),
                "is_correct": result.get("is_correct"),
                "raw_output": result.get("raw_output"),
            }
        if len(normalized_answers) > 1 or len(correctness) > 1:
            rows.append(
                {
                    "sample_id": sample_id,
                    "gold_answer": gold_answer,
                    "question": question_text,
                    "settings": per_setting,
                }
            )
    return rows


def comparison_summary_payload(
    results_by_setting: Dict[str, Dict[str, Any]],
    similarity_stats: Dict[Tuple[str, str], Dict[str, Any]],
    disagreement_details: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "settings": {},
        "pairwise": {},
        "multiway": {
            "sample_count": len(results_by_setting[SETTING_ORDER[0]]["questions"]),
            "disagreement_count": len(disagreement_details),
            "disagreement_sample_ids": [int(row["sample_id"]) for row in disagreement_details],
        },
    }
    for setting in SETTING_ORDER:
        payload = results_by_setting[setting]
        questions = payload["questions"]
        done = [row for row in questions if row.get("result") is not None]
        correct = sum(1 for row in done if question_result(row).get("is_correct") is True)
        summary["settings"][setting] = {
            "label": SETTING_LABELS[setting],
            "count": len(done),
            "correct": correct,
            "accuracy": payload["summary"].get("accuracy"),
            "instance_id": payload.get("instance_evidence", {}).get("instance_id"),
        }
    for left in SETTING_ORDER:
        for right in SETTING_ORDER:
            key = f"{left}_vs_{right}"
            summary["pairwise"][key] = similarity_stats[(left, right)]
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFBFA",
        }
    )
    return plt


def plot_accuracy_panel(ax, results_by_setting: Dict[str, Dict[str, Any]]) -> None:
    values = accuracy_values(results_by_setting)
    labels = [SETTING_LABELS[name] for name in SETTING_ORDER]
    colors = [SETTING_COLORS[name] for name in SETTING_ORDER]
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="#3B3B3B", linewidth=0.7)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("GSM8K Exact-Match Accuracy")
    ax.set_title("A. Functional Correctness by Device Count", loc="left", pad=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25, color="#7A7A7A")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.018,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
            fontweight="bold",
        )


def plot_similarity_panel(ax, similarity_stats: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    matrix: List[List[float]] = []
    for left in SETTING_ORDER:
        row: List[float] = []
        for right in SETTING_ORDER:
            row.append(float(similarity_stats[(left, right)]["answer_match_rate"]))
        matrix.append(row)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.95, vmax=1.0)
    ax.set_xticks(range(len(SETTING_ORDER)), [SETTING_LABELS[name] for name in SETTING_ORDER], rotation=18, ha="right")
    ax.set_yticks(range(len(SETTING_ORDER)), [SETTING_LABELS[name] for name in SETTING_ORDER])
    ax.set_title("B. Pairwise Similarity Across Settings", loc="left", pad=10, fontweight="bold")
    for left_idx, left in enumerate(SETTING_ORDER):
        for right_idx, right in enumerate(SETTING_ORDER):
            stat = similarity_stats[(left, right)]
            answer_rate = stat["answer_match_rate"]
            correctness_rate = stat["correctness_agreement_rate"]
            ax.text(
                right_idx,
                left_idx,
                f"ans {answer_rate:.2%}\ncorr {correctness_rate:.2%}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#0F172A",
                fontweight="bold" if left == right else None,
            )
    for spine in ax.spines.values():
        spine.set_color("#777777")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized Answer Match Rate")


def plot_correctness_panel(ax, results_by_setting: Dict[str, Dict[str, Any]]) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    matrix, disagreement_columns = correctness_matrix(results_by_setting)
    cmap = mcolors.ListedColormap(["#D96C6C", "#2A9D8F"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title("C. Per-Question Correctness Pattern", loc="left", pad=10, fontweight="bold")
    ax.set_yticks(range(len(SETTING_ORDER)), [SETTING_LABELS[name] for name in SETTING_ORDER])
    sample_count = len(matrix[0]) if matrix else 0
    xticks = list(range(0, sample_count, 10))
    if sample_count and (sample_count - 1) not in xticks:
        xticks.append(sample_count - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(tick) for tick in xticks])
    ax.set_xlabel("Sample ID")
    for column in disagreement_columns:
        ax.axvspan(column - 0.5, column + 0.5, facecolor="none", edgecolor="#111111", linewidth=1.4)
    for spine in ax.spines.values():
        spine.set_color("#777777")
    legend_items = [
        Patch(facecolor="#2A9D8F", edgecolor="none", label="Correct"),
        Patch(facecolor="#D96C6C", edgecolor="none", label="Incorrect"),
        Patch(facecolor="white", edgecolor="#111111", linewidth=1.2, label="Cross-setting disagreement"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)


def plot_accuracy_comparison(results_by_setting: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plot_accuracy_panel(ax, results_by_setting)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_similarity(similarity_stats: Dict[Tuple[str, str], Dict[str, Any]], output_path: Path) -> None:
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    plot_similarity_panel(ax, similarity_stats)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correctness_raster(results_by_setting: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    plot_correctness_panel(ax, results_by_setting)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_equivalence_summary(
    results_by_setting: Dict[str, Dict[str, Any]],
    similarity_stats: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: Path,
) -> None:
    plt = configure_matplotlib()
    fig = plt.figure(figsize=(12.6, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.36, wspace=0.28)
    ax_accuracy = fig.add_subplot(grid[0, 0])
    ax_similarity = fig.add_subplot(grid[0, 1])
    ax_correctness = fig.add_subplot(grid[1, :])

    plot_accuracy_panel(ax_accuracy, results_by_setting)
    plot_similarity_panel(ax_similarity, similarity_stats)
    plot_correctness_panel(ax_correctness, results_by_setting)

    fig.suptitle("EXO Functional Equivalence Across 1, 2, and 3 Devices", y=0.98, fontsize=17, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root
    summary_png = Path(args.output) if args.output else output_dir / "equivalence_summary.png"

    results_by_setting = load_results_by_setting(results_root)
    validate_question_alignment(results_by_setting)
    similarity_stats = pairwise_similarity_stats(results_by_setting)
    disagreement_details = disagreement_rows(results_by_setting)
    comparison_summary = comparison_summary_payload(results_by_setting, similarity_stats, disagreement_details)

    write_json(output_dir / "comparison_summary.json", comparison_summary)
    write_json(output_dir / "disagreements.json", disagreement_details)

    plot_accuracy_comparison(results_by_setting, output_dir / "accuracy_comparison.png")
    plot_pairwise_similarity(similarity_stats, output_dir / "pairwise_similarity.png")
    plot_correctness_raster(results_by_setting, output_dir / "correctness_raster.png")
    plot_equivalence_summary(results_by_setting, similarity_stats, summary_png)

    print(
        json.dumps(
            {
                "accuracy_plot": str(output_dir / "accuracy_comparison.png"),
                "similarity_plot": str(output_dir / "pairwise_similarity.png"),
                "correctness_plot": str(output_dir / "correctness_raster.png"),
                "summary_plot": str(summary_png),
                "comparison_summary": str(output_dir / "comparison_summary.json"),
                "disagreements": str(output_dir / "disagreements.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
