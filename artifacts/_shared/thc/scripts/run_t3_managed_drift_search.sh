#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-$REPO_ROOT/artifacts/_shared/thc/config/qwen_drift_search.yaml}"
CLUSTER_TEMPLATE="${CLUSTER_TEMPLATE:-$REPO_ROOT/artifacts/_shared/thc/config/hetero_qwen_cluster_drift_base.json}"
MAC_ALIAS="${MAC_ALIAS:-Mac3}"
LINUX_ALIAS="${LINUX_ALIAS:-3090}"
MAC_REPO_ROOT="${MAC_REPO_ROOT:-/Users/jlmini_3/repo/paper/bc-ra-paper-exp_verification}"
LINUX_REPO_ROOT="${LINUX_REPO_ROOT:-/home/hzh/repo/paper/bc-ra-paper-exp_verification}"
MAC_PYTHON_BIN="${MAC_PYTHON_BIN:-/Users/jlmini_3/repo/paper/bc-ra-paper/.venv/bin/python3}"
LINUX_PYTHON_BIN="${LINUX_PYTHON_BIN:-/home/hzh/repo/paper/bc-ra-paper/.venv/bin/python3}"
EXECUTE="${EXECUTE:-true}"
RESUME="${RESUME:-true}"
CALIBRATION_RUNS="${CALIBRATION_RUNS:-3}"
RUNS_PER_MODE="${RUNS_PER_MODE:-10}"
DECODE_STEPS="${DECODE_STEPS:-8}"
FALLBACK_DECODE_STEPS="${FALLBACK_DECODE_STEPS:-16}"
LOCAL_PORT="${LOCAL_PORT:-18312}"
LINUX_TUNNEL_PORT="${LINUX_TUNNEL_PORT:-18311}"
PING_TIMEOUT_SECONDS="${PING_TIMEOUT_SECONDS:-120}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

thc_prepare_python_bin "$PYTHON_BIN"
thc_install_missing_modules "$PYTHON_BIN" \
  "numpy::numpy" \
  "transformers::transformers" \
  "safetensors::safetensors" \
  "accelerate::accelerate" \
  "sentencepiece::sentencepiece"

ARGS=(
  "$PYTHON_BIN"
  "$REPO_ROOT/artifacts/_shared/thc/src/managed_drift_search.py"
  --config-template "$CONFIG_TEMPLATE"
  --cluster-template "$CLUSTER_TEMPLATE"
  --python-bin "$PYTHON_BIN"
  --mac-alias "$MAC_ALIAS"
  --linux-alias "$LINUX_ALIAS"
  --mac-repo-root "$MAC_REPO_ROOT"
  --linux-repo-root "$LINUX_REPO_ROOT"
  --mac-python-bin "$MAC_PYTHON_BIN"
  --linux-python-bin "$LINUX_PYTHON_BIN"
  --execute "$EXECUTE"
  --resume "$RESUME"
  --calibration-runs "$CALIBRATION_RUNS"
  --runs-per-mode "$RUNS_PER_MODE"
  --decode-steps "$DECODE_STEPS"
  --fallback-decode-steps "$FALLBACK_DECODE_STEPS"
  --local-port "$LOCAL_PORT"
  --linux-tunnel-port "$LINUX_TUNNEL_PORT"
  --ping-timeout-seconds "$PING_TIMEOUT_SECONDS"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

exec "${ARGS[@]}"
