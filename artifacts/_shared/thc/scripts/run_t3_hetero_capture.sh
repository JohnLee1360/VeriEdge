#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

thc_require_env CLUSTER_FILE

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/artifacts/_shared/thc/config/qwen.yaml}"
SPLIT="${SPLIT:-calibration}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-0}"

thc_require_env OUTPUT_DIR
thc_prepare_python_bin "$PYTHON_BIN"
thc_install_missing_modules "$PYTHON_BIN" \
  "numpy::numpy" \
  "transformers::transformers" \
  "safetensors::safetensors" \
  "accelerate::accelerate" \
  "sentencepiece::sentencepiece"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/hetero_qwen_capture.py" \
  --config "$CONFIG_PATH" \
  --cluster-file "$CLUSTER_FILE" \
  --split "$SPLIT" \
  --limit-prompts "$LIMIT_PROMPTS" \
  --output-dir "$OUTPUT_DIR"
