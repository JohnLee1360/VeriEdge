#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <output_dir> <node_root_1> <node_root_2> <node_root_3> [more_node_roots...]"
  exit 1
fi

OUTPUT_DIR="$1"
shift
PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"

thc_prepare_python_bin "$PYTHON_BIN"
thc_install_missing_modules "$PYTHON_BIN" "numpy::numpy"

"$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/merge_exo_captures.py" \
  --output-dir "$OUTPUT_DIR" \
  --node-roots "$@"
