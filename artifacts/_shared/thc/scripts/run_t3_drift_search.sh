#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-$REPO_ROOT/artifacts/_shared/thc/config/qwen_drift_search.yaml}"
CLUSTER_TEMPLATE="${CLUSTER_TEMPLATE:-$REPO_ROOT/artifacts/_shared/thc/config/hetero_qwen_cluster_drift_base.json}"
EXECUTE="${EXECUTE:-false}"
CALIBRATION_RUNS="${CALIBRATION_RUNS:-3}"
RUNS_PER_MODE="${RUNS_PER_MODE:-10}"
DECODE_STEPS="${DECODE_STEPS:-8}"
FALLBACK_DECODE_STEPS="${FALLBACK_DECODE_STEPS:-16}"

thc_prepare_python_bin "$PYTHON_BIN"
thc_install_missing_modules "$PYTHON_BIN" \
  "numpy::numpy" \
  "transformers::transformers" \
  "safetensors::safetensors" \
  "accelerate::accelerate" \
  "sentencepiece::sentencepiece"

OUTPUT_DIR="${OUTPUT_DIR:-}"

ARGS=(
  "$PYTHON_BIN"
  "$REPO_ROOT/artifacts/_shared/thc/src/drift_search.py"
  --config-template "$CONFIG_TEMPLATE"
  --cluster-template "$CLUSTER_TEMPLATE"
  --python-bin "$PYTHON_BIN"
  --execute "$EXECUTE"
  --calibration-runs "$CALIBRATION_RUNS"
  --runs-per-mode "$RUNS_PER_MODE"
  --decode-steps "$DECODE_STEPS"
  --fallback-decode-steps "$FALLBACK_DECODE_STEPS"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

exec "${ARGS[@]}"
