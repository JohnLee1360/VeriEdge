#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <output_dir> <capture_root_1> <capture_root_2> [more_capture_roots...]"
  exit 1
fi

OUTPUT_DIR="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"
PERCENTILE="${PERCENTILE:-99.0}"

thc_prepare_python_bin "$PYTHON_BIN"
thc_install_missing_modules "$PYTHON_BIN" "numpy::numpy"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/calibrate_delta.py" \
  --output-dir "$OUTPUT_DIR" \
  --percentile "$PERCENTILE" \
  --capture-roots "$@"
