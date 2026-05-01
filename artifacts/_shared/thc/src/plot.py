from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_RCPARAMS = getattr(mpl, "rcParams", getattr(plt, "rcParams", None))
if _RCPARAMS is not None:
    _RCPARAMS.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.edgecolor": "#9ca3af",
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfd",
        }
    )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def _save(fig: plt.Figure, out1: Path, out2: Path) -> None:
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out1, dpi=220, bbox_inches="tight")
    fig.savefig(out2, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _display_name(verifier: str) -> str:
    return "THC" if verifier.lower() == "thc" else "TSTC"


def _overall_rows(rows: List[Dict[str, str]], scenario: str, stage: str) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("scenario") == scenario and row.get("stage") == stage and not str(row.get("hetero_level", "")).strip()
    ]


def _metric_by_verifier(rows: List[Dict[str, str]], metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        key = str(row.get("verifier", "")).lower()
        value = _to_float(row.get(metric, ""))
        if np.isnan(value):
            continue
        out[key] = value
    return out


def _available_stages(rows: List[Dict[str, str]]) -> List[str]:
    ordered = ["prefill", "decode"]
    present = {str(row.get("stage", "")).strip() for row in rows}
    return [stage for stage in ordered if stage in present]


def _overall_metric(rows: List[Dict[str, str]], scenario: str, verifier: str, metric: str, stage: str = "prefill") -> float:
    matched = [
        row
        for row in rows
        if row.get("scenario") == scenario
        and row.get("stage") == stage
        and not str(row.get("hetero_level", "")).strip()
        and str(row.get("verifier", "")).lower() == verifier
    ]
    return _metric_by_verifier(matched, metric).get(verifier, float("nan"))


def _soften_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.18, color="#475569")
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


def _soften_x_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.18, color="#475569")
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


def _plot_fpr_by_stage(rows: List[Dict[str, str]], run_dir: Path, paper_img_dir: Path, suffix: str) -> None:
    stages = _available_stages(rows)
    if not stages:
        return
    stage_titles = {"prefill": "Prefill", "decode": "Decode"}
    verifiers = ["thc", "tstc"]
    x = np.arange(len(stages))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    colors = {"thc": "#1f77b4", "tstc": "#ff7f0e"}
    for offset, verifier in [(-width / 2, "thc"), (width / 2, "tstc")]:
        values = []
        for stage in stages:
            stage_rows = _overall_rows(rows, scenario="honest_hetero", stage=stage)
            values.append(_metric_by_verifier(stage_rows, "fpr").get(verifier, float("nan")))
        bars = ax.bar(x + offset, values, width=width, label=_display_name(verifier), color=colors[verifier])
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([stage_titles[s] for s in stages])
    ax.set_ylabel("FPR")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Honest-Hetero False Positive Rate by Stage")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    _save(fig, run_dir / f"thc_tstc_fpr_by_stage{suffix}.png", paper_img_dir / f"thc_tstc_fpr_by_stage{suffix}.png")


def _plot_tpr_loc_by_stage(rows: List[Dict[str, str]], run_dir: Path, paper_img_dir: Path, suffix: str) -> None:
    stages = _available_stages(rows)
    if not stages:
        return
    stage_titles = {"prefill": "Prefill", "decode": "Decode"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    colors = {"thc": "#2ca02c", "tstc": "#9467bd"}

    for ax, metric, title in zip(axes, ["tpr", "localization_acc"], ["TPR", "Localization Accuracy"]):
        x = np.arange(len(stages))
        width = 0.32
        for offset, verifier in [(-width / 2, "thc"), (width / 2, "tstc")]:
            values = []
            for stage in stages:
                stage_rows = _overall_rows(rows, scenario="tamper", stage=stage)
                values.append(_metric_by_verifier(stage_rows, metric).get(verifier, float("nan")))
            bars = ax.bar(x + offset, values, width=width, label=_display_name(verifier), color=colors[verifier])
            for bar, val in zip(bars, values):
                if np.isnan(val):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([stage_titles[s] for s in stages])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Score")
    axes[1].legend(loc="lower right")
    _save(
        fig,
        run_dir / f"thc_tstc_tpr_loc_by_stage{suffix}.png",
        paper_img_dir / f"thc_tstc_tpr_loc_by_stage{suffix}.png",
    )


def _plot_hetero_fpr_breakdown(rows: List[Dict[str, str]], run_dir: Path, paper_img_dir: Path, suffix: str) -> None:
    levels = ["low", "mid", "high"]
    stages = _available_stages(rows)
    if not stages:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True, sharey=True)
    colors = {"thc": "#1f77b4", "tstc": "#ff7f0e"}
    if len(stages) == 1:
        axes = [axes[0]]

    for ax, stage in zip(axes, stages):
        x = np.arange(len(levels))
        width = 0.32
        for offset, verifier in [(-width / 2, "thc"), (width / 2, "tstc")]:
            values = []
            for level in levels:
                matched = [
                    row
                    for row in rows
                    if row.get("scenario") == "honest_hetero"
                    and row.get("stage") == stage
                    and row.get("hetero_level") == level
                    and str(row.get("verifier", "")).lower() == verifier
                ]
                values.append(_metric_by_verifier(matched, "fpr").get(verifier, float("nan")))
            ax.bar(x + offset, values, width=width, label=_display_name(verifier), color=colors[verifier])
        ax.set_xticks(x)
        ax.set_xticklabels([level.capitalize() for level in levels])
        ax.set_title(f"{stage.capitalize()} FPR by Heterogeneity")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("FPR")
    axes[0].set_ylim(0.0, 1.05)
    axes[-1].legend(loc="upper right")
    _save(
        fig,
        run_dir / f"thc_tstc_hetero_fpr_breakdown{suffix}.png",
        paper_img_dir / f"thc_tstc_hetero_fpr_breakdown{suffix}.png",
    )


def _plot_candidate_search(candidate_rows: List[Dict[str, str]], run_dir: Path, paper_img_dir: Path, suffix: str) -> None:
    if not candidate_rows:
        return

    labels = []
    values = []
    colors = []
    for row in candidate_rows:
        labels.append(f"P{row['prefill_sample_count']}/D{row['decode_channel_samples']}")
        values.append(_to_float(row.get("max_fpr", "")))
        colors.append("#2ca02c" if str(row.get("selected", "")).lower() == "true" else "#9ecae1")

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Max Stage FPR")
    ax.set_title("TSTC Sampling Search")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, val in zip(bars, values):
        if np.isnan(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    _save(
        fig,
        run_dir / f"thc_tstc_sampling_search{suffix}.png",
        paper_img_dir / f"thc_tstc_sampling_search{suffix}.png",
    )


def _plot_prefill_main_metrics(rows: List[Dict[str, str]], run_dir: Path, paper_img_dir: Path, suffix: str) -> None:
    if "prefill" not in _available_stages(rows):
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    colors = {"thc": "#6B7280", "tstc": "#0F766E"}
    width = 0.32

    fpr_categories = ["Honest-Homo", "Honest-Hetero"]
    fpr_scenarios = ["honest_homo", "honest_hetero"]
    thc_vals = [_overall_metric(rows, scenario, "thc", "fpr") for scenario in fpr_scenarios]
    tstc_vals = [_overall_metric(rows, scenario, "tstc", "fpr") for scenario in fpr_scenarios]
    x = np.arange(len(fpr_categories))
    thc_bars = axes[0].bar(x - width / 2, thc_vals, width=width, color=colors["thc"], edgecolor="white", linewidth=1.2, label="THC", zorder=3)
    tstc_bars = axes[0].bar(x + width / 2, tstc_vals, width=width, color=colors["tstc"], edgecolor="white", linewidth=1.2, label="TSTC", zorder=3)
    for bar, val in zip(thc_bars, thc_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=colors["thc"])
    for bar, val in zip(tstc_bars, tstc_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=colors["tstc"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fpr_categories)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("FPR")
    axes[0].set_title("False Positive Rate")
    _soften_axis(axes[0])

    tamper_metrics = ["TPR", "LocAcc"]
    metric_keys = ["tpr", "localization_acc"]
    thc_tamper = [_overall_metric(rows, "tamper", "thc", metric) for metric in metric_keys]
    tstc_tamper = [_overall_metric(rows, "tamper", "tstc", metric) for metric in metric_keys]
    x = np.arange(len(tamper_metrics))
    thc_bars = axes[1].bar(x - width / 2, thc_tamper, width=width, color=colors["thc"], edgecolor="white", linewidth=1.2, label="THC", zorder=3)
    tstc_bars = axes[1].bar(x + width / 2, tstc_tamper, width=width, color=colors["tstc"], edgecolor="white", linewidth=1.2, label="TSTC", zorder=3)
    for bar, val in zip(thc_bars, thc_tamper):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=colors["thc"])
    for bar, val in zip(tstc_bars, tstc_tamper):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=colors["tstc"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tamper_metrics)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_title("Tamper Detection Quality")
    _soften_axis(axes[1])
    axes[1].legend(loc="lower right")

    _save(
        fig,
        run_dir / f"thc_tstc_prefill_main{suffix}.png",
        paper_img_dir / f"thc_tstc_prefill_main{suffix}.png",
    )


def _plot_prefill_hetero_breakdown(rows: List[Dict[str, str]], run_dir: Path, paper_img_dir: Path, suffix: str) -> None:
    levels = ["low", "mid", "high"]
    if "prefill" not in _available_stages(rows):
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    x = np.arange(len(levels))
    colors = {"thc": "#6B7280", "tstc": "#0F766E"}
    width = 0.32
    thc_vals = []
    tstc_vals = []
    for level in levels:
        matched_thc = [
            row
            for row in rows
            if row.get("scenario") == "honest_hetero"
            and row.get("stage") == "prefill"
            and row.get("hetero_level") == level
            and str(row.get("verifier", "")).lower() == "thc"
        ]
        matched_tstc = [
            row
            for row in rows
            if row.get("scenario") == "honest_hetero"
            and row.get("stage") == "prefill"
            and row.get("hetero_level") == level
            and str(row.get("verifier", "")).lower() == "tstc"
        ]
        thc_vals.append(_metric_by_verifier(matched_thc, "fpr").get("thc", float("nan")))
        tstc_vals.append(_metric_by_verifier(matched_tstc, "fpr").get("tstc", float("nan")))

    thc_bars = ax.bar(x - width / 2, thc_vals, width=width, color=colors["thc"], edgecolor="white", linewidth=1.2, label="THC", zorder=3)
    tstc_bars = ax.bar(x + width / 2, tstc_vals, width=width, color=colors["tstc"], edgecolor="white", linewidth=1.2, label="TSTC", zorder=3)
    for bar, val in zip(thc_bars, thc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=colors["thc"])
    for bar, val in zip(tstc_bars, tstc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.03, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=colors["tstc"])
    ax.set_xticks(x)
    ax.set_xticklabels([level.capitalize() for level in levels])
    ax.set_ylabel("FPR")
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Honest-Hetero FPR by Heterogeneity Level")
    _soften_axis(ax)
    ax.legend(loc="upper right")

    _save(
        fig,
        run_dir / f"thc_tstc_prefill_hetero_breakdown{suffix}.png",
        paper_img_dir / f"thc_tstc_prefill_hetero_breakdown{suffix}.png",
    )


def generate_figures(
    summary_csv: Path,
    run_dir: Path,
    paper_img_dir: Path,
    candidate_csv: Optional[Path] = None,
    file_suffix: str = "",
) -> None:
    rows = _read_csv(summary_csv)
    suffix = f"_{file_suffix}" if file_suffix else ""
    _plot_prefill_main_metrics(rows, run_dir, paper_img_dir, suffix)
    _plot_prefill_hetero_breakdown(rows, run_dir, paper_img_dir, suffix)
    if "decode" in _available_stages(rows):
        _plot_fpr_by_stage(rows, run_dir, paper_img_dir, suffix)
        _plot_tpr_loc_by_stage(rows, run_dir, paper_img_dir, suffix)
        _plot_hetero_fpr_breakdown(rows, run_dir, paper_img_dir, suffix)
    if candidate_csv is not None and "decode" in _available_stages(rows):
        _plot_candidate_search(_read_csv(candidate_csv), run_dir, paper_img_dir, suffix)
