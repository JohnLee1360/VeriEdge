#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

: "${EXO_ROOT:=$REPO_ROOT/../third_party/exo}"
: "${MODEL_ID:=mlx-community/Qwen3-0.6B-8bit}"
: "${PROMPT_DATASET:=$REPO_ROOT/artifacts/_shared/thc/data/qwen_prompt_splits.jsonl}"
: "${SPLIT:=calibration}"
: "${LIMIT_PROMPTS:=0}"
: "${INSTANCE_ID:=thc-t3-capture}"
: "${SEED:=42}"

thc_require_env LOCAL_NODE
thc_require_env CLUSTER_FILE
thc_require_env OUTPUT_DIR

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"

thc_prepare_python_bin "$PYTHON_BIN"
thc_install_missing_modules "$PYTHON_BIN" \
  "numpy::numpy" \
  "mlx::mlx" \
  "mlx_lm::mlx-lm"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/exo_capture_runner.py" \
  --exo-root "$EXO_ROOT" \
  --cluster-file "$CLUSTER_FILE" \
  --local-node "$LOCAL_NODE" \
  --model-id "$MODEL_ID" \
  --prompt-dataset "$PROMPT_DATASET" \
  --split "$SPLIT" \
  --limit-prompts "$LIMIT_PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --instance-id "$INSTANCE_ID" \
  --seed "$SEED"
