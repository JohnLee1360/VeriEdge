from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_TEMPLATE = REPO_ROOT / "artifacts/_shared/thc/config/qwen_drift_search.yaml"
DEFAULT_CLUSTER_TEMPLATE = REPO_ROOT / "artifacts/_shared/thc/config/hetero_qwen_cluster_drift_base.json"
DEFAULT_PYTHON_BIN = REPO_ROOT / ".venv/bin/python3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for a non-zero drift configuration for strict T3")
    parser.add_argument("--config-template", default=str(DEFAULT_CONFIG_TEMPLATE))
    parser.add_argument("--cluster-template", default=str(DEFAULT_CLUSTER_TEMPLATE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON_BIN))
    parser.add_argument("--execute", choices=["true", "false"], default="false")
    parser.add_argument("--calibration-runs", type=int, default=3)
    parser.add_argument("--runs-per-mode", type=int, default=10)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--fallback-decode-steps", type=int, default=16)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "artifacts/_shared/thc/output" / f"{stamp}_drift_search"


def _node_defaults(cluster_template: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(node["node_name"]): dict(node) for node in cluster_template["nodes"]}


def _node_payload(
    defaults: Dict[str, Dict[str, Any]],
    *,
    machine: str,
    checkpoint: str,
    start_layer: int,
    end_layer: int,
    torch_dtype: str,
    quantization: str,
    first_shard: bool,
    last_shard: bool,
) -> Dict[str, Any]:
    base = dict(defaults[machine])
    base["checkpoint"] = checkpoint
    base["start_layer"] = int(start_layer)
    base["end_layer"] = int(end_layer)
    base["torch_dtype"] = str(torch_dtype)
    base["quantization"] = str(quantization)
    base["first_shard"] = bool(first_shard)
    base["last_shard"] = bool(last_shard)
    return base


def _build_variants(cluster_template: Dict[str, Any], decode_steps: int) -> List[Dict[str, Any]]:
    defaults = _node_defaults(cluster_template)
    variants: List[Dict[str, Any]] = []

    linux_modes = [
        {"label": "fp16", "torch_dtype": "float16", "quantization": "none"},
        {"label": "bf16", "torch_dtype": "bfloat16", "quantization": "none"},
        {"label": "bnb8", "torch_dtype": "float16", "quantization": "bitsandbytes_8bit"},
    ]
    mac_c3_modes = [
        {"label": "fp32", "torch_dtype": "float32", "quantization": "none"},
        {"label": "fp16", "torch_dtype": "float16", "quantization": "none"},
    ]
    mac_c2_modes = [
        {"label": "fp32", "torch_dtype": "float32", "quantization": "none"},
        {"label": "fp16", "torch_dtype": "float16", "quantization": "none"},
    ]

    for linux_mode in linux_modes:
        for mac_mode in mac_c3_modes:
            label = f"d{decode_steps}_pA_c2_{linux_mode['label']}_c3_{mac_mode['label']}"
            variants.append(
                {
                    "label": label,
                    "decode_steps": decode_steps,
                    "placement": "linux_as_C2",
                    "cluster": {
                        "model_id": cluster_template["model_id"],
                        "nodes": [
                            _node_payload(
                                defaults,
                                machine="jlmini_3",
                                checkpoint="C1",
                                start_layer=0,
                                end_layer=7,
                                torch_dtype="float16",
                                quantization="metal_8bit",
                                first_shard=True,
                                last_shard=False,
                            ),
                            _node_payload(
                                defaults,
                                machine="linux124",
                                checkpoint="C2",
                                start_layer=8,
                                end_layer=15,
                                torch_dtype=linux_mode["torch_dtype"],
                                quantization=linux_mode["quantization"],
                                first_shard=False,
                                last_shard=False,
                            ),
                            _node_payload(
                                defaults,
                                machine="jlmini_2",
                                checkpoint="C3",
                                start_layer=16,
                                end_layer=23,
                                torch_dtype=mac_mode["torch_dtype"],
                                quantization=mac_mode["quantization"],
                                first_shard=False,
                                last_shard=True,
                            ),
                        ],
                    },
                }
            )

    for mac_mode in mac_c2_modes:
        for linux_mode in linux_modes:
            label = f"d{decode_steps}_pB_c2_{mac_mode['label']}_c3_{linux_mode['label']}"
            variants.append(
                {
                    "label": label,
                    "decode_steps": decode_steps,
                    "placement": "linux_as_C3",
                    "cluster": {
                        "model_id": cluster_template["model_id"],
                        "nodes": [
                            _node_payload(
                                defaults,
                                machine="jlmini_3",
                                checkpoint="C1",
                                start_layer=0,
                                end_layer=7,
                                torch_dtype="float16",
                                quantization="metal_8bit",
                                first_shard=True,
                                last_shard=False,
                            ),
                            _node_payload(
                                defaults,
                                machine="jlmini_2",
                                checkpoint="C2",
                                start_layer=8,
                                end_layer=15,
                                torch_dtype=mac_mode["torch_dtype"],
                                quantization=mac_mode["quantization"],
                                first_shard=False,
                                last_shard=False,
                            ),
                            _node_payload(
                                defaults,
                                machine="linux124",
                                checkpoint="C3",
                                start_layer=16,
                                end_layer=23,
                                torch_dtype=linux_mode["torch_dtype"],
                                quantization=linux_mode["quantization"],
                                first_shard=False,
                                last_shard=True,
                            ),
                        ],
                    },
                }
            )

    return variants


def _variant_config(config_template: Dict[str, Any], decode_steps: int) -> Dict[str, Any]:
    config = json.loads(json.dumps(config_template, ensure_ascii=True))
    exp_cfg = dict(config["experiment"])
    probe_cfg = dict(exp_cfg.get("decode_probe", {}))
    probe_cfg["num_steps"] = int(decode_steps)
    exp_cfg["decode_probe"] = probe_cfg
    config["experiment"] = exp_cfg
    return config


def _command_manifest(
    python_bin: str,
    cluster_file: Path,
    config_file: Path,
    output_dir: Path,
    runs_per_mode: int,
) -> Dict[str, str]:
    capture_calib = (
        f"{python_bin} {REPO_ROOT / 'artifacts/_shared/thc/src/hetero_qwen_capture.py'} "
        f"--config {config_file} --cluster-file {cluster_file} --split calibration --output-dir {output_dir / 'capture_calibration'}"
    )
    capture_eval = (
        f"{python_bin} {REPO_ROOT / 'artifacts/_shared/thc/src/hetero_qwen_capture.py'} "
        f"--config {config_file} --cluster-file {cluster_file} --split evaluation --output-dir {output_dir / 'capture_evaluation'}"
    )
    calibrate = (
        f"{python_bin} {REPO_ROOT / 'artifacts/_shared/thc/src/calibrate_delta.py'} "
        f"--output-dir {output_dir / 'delta_calibration'} --capture-roots <run_a> <run_b> <run_c>"
    )
    t5 = (
        f"{python_bin} {REPO_ROOT / 'artifacts/_shared/thc/src/run.py'} "
        f"--config {config_file} --mode all --split evaluation --runs-per-mode {runs_per_mode} "
        f"--calibrate-tstc true --capture-root {output_dir / 'capture_evaluation'} "
        f"--delta-map-file {output_dir / 'delta_calibration/delta_map.json'}"
    )
    return {
        "capture_calibration": capture_calib,
        "capture_evaluation": capture_eval,
        "calibrate_delta": calibrate,
        "run_t5": t5,
    }


def _run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True)


def _run_capture(python_bin: str, config_file: Path, cluster_file: Path, split: str, output_dir: Path) -> str:
    cmd = [
        python_bin,
        str(REPO_ROOT / "artifacts/_shared/thc/src/hetero_qwen_capture.py"),
        "--config",
        str(config_file),
        "--cluster-file",
        str(cluster_file),
        "--split",
        split,
        "--output-dir",
        str(output_dir),
    ]
    return _run_cmd(cmd, REPO_ROOT).stdout.strip()


def _run_calibration(python_bin: str, output_dir: Path, capture_roots: Iterable[Path]) -> Path:
    cmd = [
        python_bin,
        str(REPO_ROOT / "artifacts/_shared/thc/src/calibrate_delta.py"),
        "--output-dir",
        str(output_dir),
        "--capture-roots",
        *[str(path) for path in capture_roots],
    ]
    _run_cmd(cmd, REPO_ROOT)
    return output_dir / "delta_map.json"


def _run_t5(
    python_bin: str,
    config_file: Path,
    capture_root: Path,
    delta_map_file: Path,
    runs_per_mode: int,
) -> str:
    cmd = [
        python_bin,
        str(REPO_ROOT / "artifacts/_shared/thc/src/run.py"),
        "--config",
        str(config_file),
        "--mode",
        "all",
        "--split",
        "evaluation",
        "--runs-per-mode",
        str(runs_per_mode),
        "--calibrate-tstc",
        "true",
        "--capture-root",
        str(capture_root),
        "--delta-map-file",
        str(delta_map_file),
    ]
    completed = _run_cmd(cmd, REPO_ROOT)
    for line in completed.stdout.splitlines():
        if line.startswith("THC/TSTC run complete: "):
            return line.split("THC/TSTC run complete: ", 1)[1].strip()
    return completed.stdout.strip()


def _has_nonzero_delta(path: Path) -> Tuple[bool, float]:
    payload = _load_json(path)
    values = [
        float(value)
        for stage_map in payload.get("delta_map", {}).values()
        for value in dict(stage_map).values()
    ]
    max_value = max(values) if values else 0.0
    return max_value > 0.0, max_value


def _server_launch_commands(cluster_file: Path) -> Dict[str, str]:
    return {
        "jlmini_2": f"LOCAL_NODE=jlmini_2 CLUSTER_FILE={cluster_file} bash {REPO_ROOT / 'artifacts/_shared/thc/scripts/run_t3_hetero_server.sh'}",
        "jlmini_3": f"ssh Mac3 'cd ~/repo/paper/bc-ra-paper && LOCAL_NODE=jlmini_3 CLUSTER_FILE={cluster_file} bash artifacts/_shared/thc/scripts/run_t3_hetero_server.sh'",
        "linux124": f"ssh 3090 'cd ~/repo/paper/bc-ra-paper && LOCAL_NODE=linux124 CLUSTER_FILE={cluster_file} bash artifacts/_shared/thc/scripts/run_t3_hetero_server.sh'",
    }


def main() -> None:
    args = _parse_args()
    config_template = _load_json(Path(args.config_template).expanduser().resolve())
    cluster_template = _load_json(Path(args.cluster_template).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir()
    execute = args.execute == "true"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    chosen_variant: Dict[str, Any] | None = None
    round_decode_steps = [int(args.decode_steps)]
    if int(args.fallback_decode_steps) > int(args.decode_steps):
        round_decode_steps.append(int(args.fallback_decode_steps))

    for round_index, decode_steps in enumerate(round_decode_steps, start=1):
        variants = _build_variants(cluster_template, decode_steps)
        round_nonzero = False
        for variant_index, variant in enumerate(variants, start=1):
            variant_dir = output_dir / f"{round_index:02d}_{variant_index:02d}_{variant['label']}"
            config_file = variant_dir / "config.json"
            cluster_file = variant_dir / "cluster.json"
            _write_json(config_file, _variant_config(config_template, decode_steps))
            _write_json(cluster_file, variant["cluster"])

            row: Dict[str, Any] = {
                "round": round_index,
                "label": variant["label"],
                "decode_steps": decode_steps,
                "placement": variant["placement"],
                "config_file": str(config_file),
                "cluster_file": str(cluster_file),
                "server_launch": _server_launch_commands(cluster_file),
                "commands": _command_manifest(args.python_bin, cluster_file, config_file, variant_dir, int(args.runs_per_mode)),
                "status": "prepared",
            }

            if execute:
                capture_roots: List[Path] = []
                for run_index in range(1, int(args.calibration_runs) + 1):
                    capture_root = variant_dir / f"capture_calibration_run_{run_index}"
                    _run_capture(args.python_bin, config_file, cluster_file, "calibration", capture_root)
                    capture_roots.append(capture_root)
                delta_map_file = _run_calibration(args.python_bin, variant_dir / "delta_calibration", capture_roots)
                nonzero, max_delta = _has_nonzero_delta(delta_map_file)
                row["delta_map_file"] = str(delta_map_file)
                row["max_delta"] = max_delta
                row["status"] = "nonzero_delta" if nonzero else "zero_delta"

                if nonzero:
                    eval_root = variant_dir / "capture_evaluation"
                    _run_capture(args.python_bin, config_file, cluster_file, "evaluation", eval_root)
                    row["evaluation_capture_root"] = str(eval_root)
                    row["t5_run_dir"] = _run_t5(
                        args.python_bin,
                        config_file,
                        eval_root,
                        delta_map_file,
                        int(args.runs_per_mode),
                    )
                    chosen_variant = row
                    round_nonzero = True
                    manifest_rows.append(row)
                    break

            manifest_rows.append(row)

        if round_nonzero:
            break

    manifest = {
        "output_dir": str(output_dir),
        "config_template": str(Path(args.config_template).expanduser().resolve()),
        "cluster_template": str(Path(args.cluster_template).expanduser().resolve()),
        "execute": execute,
        "chosen_variant": chosen_variant,
        "variants": manifest_rows,
    }
    _write_json(output_dir / "drift_search_manifest.json", manifest)
    print(f"Drift search manifest written to {output_dir / 'drift_search_manifest.json'}")


if __name__ == "__main__":
    main()
