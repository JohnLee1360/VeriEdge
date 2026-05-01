#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

thc_require_env CLUSTER_FILE
thc_require_env LOCAL_NODE

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"

thc_prepare_python_bin "$PYTHON_BIN"
bash "$SCRIPT_DIR/check_t3_hetero_env.sh"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/hetero_qwen_server.py" \
  --cluster-file "$CLUSTER_FILE" \
  --local-node "$LOCAL_NODE"
