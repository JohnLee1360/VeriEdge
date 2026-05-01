#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"

thc_prepare_python_bin "$PYTHON_BIN"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/scripts/t3_hetero_cli.py" "$@"
