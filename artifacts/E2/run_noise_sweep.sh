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
  exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/E2/tstc/run_noise_sweep.py" --help
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/E2/commits/logs/noise_sweep_$STAMP}"
mkdir -p "$OUTPUT_DIR"

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/E2/tstc/run_noise_sweep.py" \
  --sweep-config "$REPO_ROOT/artifacts/E2/tstc/noise_sweep_config.json" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
