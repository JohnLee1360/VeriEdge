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
  exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/overhead_report.py" --help
fi

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/overhead_report.py" \
  --config "$REPO_ROOT/artifacts/_shared/thc/config/qwen.yaml" \
  "$@"
