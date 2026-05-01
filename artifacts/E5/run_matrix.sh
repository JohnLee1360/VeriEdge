#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/E5/inference_e2e/requester/matrix_control.py" --help
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="${BATCH_DIR:-$REPO_ROOT/workspace/runs/E5/matrix_$STAMP}"
mkdir -p "$BATCH_DIR"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/E5/inference_e2e/requester/matrix_control.py" \
  --config "$REPO_ROOT/artifacts/E5/inference_e2e/requester/config.example.yaml" \
  --batch-dir "$BATCH_DIR" \
  "$@"
