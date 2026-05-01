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
  exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/scripts/t3_hetero_cli.py" capture --help
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/workspace/captures/E1/$STAMP}"
mkdir -p "$OUTPUT_DIR"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/scripts/t3_hetero_cli.py" capture \
  --cluster-file "$REPO_ROOT/artifacts/_shared/thc/config/hetero_qwen_cluster.json" \
  --config-path "$REPO_ROOT/artifacts/_shared/thc/config/qwen.yaml" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
