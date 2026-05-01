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

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -lt 1 ]]; then
  echo "usage: bash artifacts/E5/build_policy_table.sh <summary_by_cell.csv> [extra make_comparison_table args]"
  exit 0
fi

INPUT_PATH="$1"
shift

STAMP="${STAMP:-$(date +%Y%m%d)}"
OWNER="${OWNER:-shared}"
TABLE_DIR="${TABLE_DIR:-$REPO_ROOT/artifacts/E5/commits/tables}"
CONFIG_SOURCE="${CONFIG_SOURCE:-$REPO_ROOT/artifacts/E5/inference_e2e/requester/config.example.yaml}"
CSV_OUT="${CSV_OUT:-$TABLE_DIR/exp_e5_${STAMP}_${OWNER}_policy_compare.csv}"
MD_OUT="${MD_OUT:-$TABLE_DIR/exp_e5_${STAMP}_${OWNER}_policy_compare.md}"
CONFIG_OUT="${CONFIG_OUT:-$TABLE_DIR/exp_e5_${STAMP}_${OWNER}_policy_config.json}"
mkdir -p "$TABLE_DIR"

"$PYTHON_BIN" "$REPO_ROOT/artifacts/E5/inference_e2e/requester/make_comparison_table.py" \
  --input "$INPUT_PATH" \
  --output-csv "$CSV_OUT" \
  --output-md "$MD_OUT" \
  "$@"

"$PYTHON_BIN" - "$CONFIG_SOURCE" "$CONFIG_OUT" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
try:
    data = json.loads(text)
except json.JSONDecodeError:
    import yaml
    data = yaml.safe_load(text)
dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
PY
