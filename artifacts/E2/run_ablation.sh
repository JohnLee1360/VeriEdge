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
  exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/run.py" --help
fi

LOG_DIR="${LOG_DIR:-$REPO_ROOT/artifacts/E2/commits/logs}"
FIGURE_DIR="${FIGURE_DIR:-$REPO_ROOT/artifacts/E2/commits/figures}"
mkdir -p "$LOG_DIR" "$FIGURE_DIR"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/run.py" \
  --config "$REPO_ROOT/artifacts/_shared/thc/config/qwen.yaml" \
  --mode all \
  --split evaluation \
  --output-root "$LOG_DIR" \
  --paper-img-dir "$FIGURE_DIR" \
  "$@"
