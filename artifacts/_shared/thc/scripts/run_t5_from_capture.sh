#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/artifacts/_shared/thc/config/qwen.yaml}"
MODE="${MODE:-all}"
SPLIT="${SPLIT:-evaluation}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-0}"
RUNS_PER_MODE="${RUNS_PER_MODE:-10}"
CALIBRATE_TSTC="${CALIBRATE_TSTC:-false}"

if [[ -z "${CAPTURE_ROOT:-}" ]]; then
  echo "CAPTURE_ROOT is required"
  exit 1
fi

if [[ -z "${DELTA_MAP_FILE:-}" ]]; then
  echo "DELTA_MAP_FILE is required"
  exit 1
fi

exec "$PYTHON_BIN" "$REPO_ROOT/artifacts/_shared/thc/src/run.py" \
  --config "$CONFIG_PATH" \
  --mode "$MODE" \
  --split "$SPLIT" \
  --limit-prompts "$LIMIT_PROMPTS" \
  --runs-per-mode "$RUNS_PER_MODE" \
  --calibrate-tstc "$CALIBRATE_TSTC" \
  --capture-root "$CAPTURE_ROOT" \
  --delta-map-file "$DELTA_MAP_FILE"
