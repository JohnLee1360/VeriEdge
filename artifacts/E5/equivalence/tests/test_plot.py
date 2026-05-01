from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plot  # noqa: E402


def _write_result(path: Path, setting: str, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"setting": setting, "summary": {"accuracy": accuracy}}), encoding="utf-8")


def test_plot_accuracy_comparison(tmp_path):
    results_by_setting = {
        "1device": {"summary": {"accuracy": 0.51}},
        "2device": {"summary": {"accuracy": 0.49}},
        "3device": {"summary": {"accuracy": 0.50}},
    }
    out_path = tmp_path / "accuracy_comparison.png"
    plot.plot_accuracy_comparison(results_by_setting, out_path)
    assert out_path.exists()


def test_load_results_by_setting(tmp_path):
    _write_result(tmp_path / "1device" / "results.json", "1device", 0.4)
    _write_result(tmp_path / "2device" / "results.json", "2device", 0.5)
    _write_result(tmp_path / "3device" / "results.json", "3device", 0.6)
    loaded = plot.load_results_by_setting(tmp_path)
    assert loaded["2device"]["summary"]["accuracy"] == 0.5


def test_plot_cli(tmp_path, monkeypatch):
    _write_result(tmp_path / "1device" / "results.json", "1device", 0.4)
    _write_result(tmp_path / "2device" / "results.json", "2device", 0.5)
    _write_result(tmp_path / "3device" / "results.json", "3device", 0.6)
    monkeypatch.setattr(sys, "argv", ["plot.py", "--results-root", str(tmp_path)])
    plot.main()
    assert (tmp_path / "accuracy_comparison.png").exists()
