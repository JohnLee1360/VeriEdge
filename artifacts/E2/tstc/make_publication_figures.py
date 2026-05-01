from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create polished bilingual publication figures for a TSTC noise sweep.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_summary(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "noise_std": float(row["noise_std"]),
                    "detected_count": float(row["detected_count"]),
                    "detection_rate": float(row["detection_rate"]),
                    "repetitions": float(row["repetitions"]),
                }
            )
    return rows


def _font_prop(candidates: Sequence[str]) -> font_manager.FontProperties | None:
    for name in candidates:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            return font_manager.FontProperties(fname=path)
        except Exception:
            continue
    return None


def _sci_label(value: float, _pos: float | None = None) -> str:
    if value <= 0:
        return "0"
    exponent = int(np.floor(np.log10(value)))
    coeff = value / (10 ** exponent)
    coeff_rounded = int(round(coeff)) if np.isclose(coeff, round(coeff)) else coeff
    if coeff_rounded in {1, 2, 5}:
        return f"{coeff_rounded}e{exponent}"
    return f"{coeff:.1f}e{exponent}"


def _pct_label(value: float, _pos: float | None = None) -> str:
    return f"{int(round(value * 100))}%"


def _transition_band(rows: Sequence[Dict[str, float]]) -> Tuple[float, float] | None:
    positives = [row["noise_std"] for row in rows if row["detected_count"] > 0]
    partials = [row["noise_std"] for row in rows if 0 < row["detection_rate"] < 0.999]
    if partials:
        return min(partials), max(partials)
    if positives:
        return min(positives), max(positives)
    return None


def _apply_theme(base_font: font_manager.FontProperties | None) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#cbd5e1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#94a3b8",
            "grid.alpha": 0.22,
            "axes.titleweight": "semibold",
        }
    )
    if base_font is not None:
        plt.rcParams["font.family"] = base_font.get_name()


def _draw_figure(
    rows: Sequence[Dict[str, float]],
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    note: str,
    rate_ylabel: str,
    font_prop: font_manager.FontProperties | None,
) -> None:
    xs = np.asarray([row["noise_std"] for row in rows], dtype=np.float64)
    ys = np.asarray([row["detected_count"] for row in rows], dtype=np.float64)
    repetitions = int(rows[0]["repetitions"]) if rows else 0
    band = _transition_band(rows)

    _apply_theme(font_prop)

    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    ax.set_xscale("log")

    if band is not None and band[0] < band[1]:
        ax.axvspan(band[0], band[1], color="#fde68a", alpha=0.25, zorder=0)
        ax.text(
            np.sqrt(band[0] * band[1]),
            repetitions * 0.14,
            note,
            ha="center",
            va="center",
            fontsize=10,
            color="#92400e",
            fontproperties=font_prop,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fffbeb", "edgecolor": "#fcd34d", "alpha": 0.9},
        )

    ax.plot(xs, ys, color="#0f766e", linewidth=2.8, zorder=3)
    ax.scatter(xs, ys, s=44, color="#115e59", edgecolors="white", linewidths=1.0, zorder=4)
    ax.fill_between(xs, ys, color="#99f6e4", alpha=0.22, zorder=1)

    label_offsets = [1.06, 1.085, 1.11, 1.075]
    for idx, (x, y) in enumerate(zip(xs, ys)):
        if y in {0, repetitions}:
            continue
        multiplier = label_offsets[idx % len(label_offsets)]
        ax.text(
            x,
            min(repetitions * 1.01, y * multiplier + repetitions * 0.01),
            f"{int(y)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#134e4a",
            fontproperties=font_prop,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )

    ax.axhline(0, color="#cbd5e1", linewidth=1.0)
    ax.axhline(repetitions / 2.0, color="#e2e8f0", linewidth=1.0, linestyle=(0, (4, 4)))
    ax.axhline(repetitions, color="#cbd5e1", linewidth=1.0)

    ax.set_title(title, fontsize=15, pad=14, fontproperties=font_prop)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10, fontproperties=font_prop)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10, fontproperties=font_prop)
    ax.set_ylim(-repetitions * 0.02, repetitions * 1.04)
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.9)
    ax.grid(axis="x", which="major", linestyle=(0, (3, 3)), linewidth=0.7, alpha=0.12)
    ax.set_xticks(xs)
    ax.set_xticklabels([_sci_label(x) for x in xs], fontsize=10)
    ax.tick_params(axis="x", labelsize=10, rotation=28)

    rate_ax = ax.twinx()
    rate_ax.set_ylim(ax.get_ylim()[0] / repetitions, ax.get_ylim()[1] / repetitions)
    rate_ax.set_ylabel(rate_ylabel, fontsize=11, color="#475569", fontproperties=font_prop)
    rate_ax.tick_params(axis="y", colors="#64748b")
    rate_ax.yaxis.set_major_formatter(FuncFormatter(_pct_label))
    rate_ax.spines["top"].set_visible(False)
    rate_ax.spines["left"].set_visible(False)
    rate_ax.spines["right"].set_color("#cbd5e1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.summary_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = _read_summary(summary_path)

    english_font = _font_prop(["Times New Roman", "Georgia", "Helvetica"])
    chinese_font = _font_prop(["Songti SC", "STHeiti", "PingFang SC", "Arial Unicode MS"])

    en_path = output_dir / "noise_sweep_detection_count_publication_en.png"
    zh_path = output_dir / "noise_sweep_detection_count_publication_zh.png"

    _draw_figure(
        rows,
        en_path,
        title="TSTC Detection Response to Perturbations",
        xlabel="Noise Standard Deviation $s$",
        ylabel="Detection Count",
        note="sensitive region",
        rate_ylabel="Detection Rate",
        font_prop=english_font,
    )
    _draw_figure(
        rows,
        zh_path,
        title="TSTC对扰动的检测响应",
        xlabel="噪声标准差 $s$",
        ylabel="检测个数",
        note="敏感区间",
        rate_ylabel="检出比例",
        font_prop=chinese_font,
    )
    print(f"Wrote polished figures to {output_dir}")


if __name__ == "__main__":
    main()
