#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

: "${EXO_ROOT:=$REPO_ROOT/../third_party/exo}"
: "${MODEL_DIR:=$HOME/.exo/models/mlx-community--Qwen3-0.6B-8bit}"

thc_require_env CLUSTER_FILE

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"

echo "[check] repo_root=$(thc_display_path "$REPO_ROOT")"
echo "[check] exo_root=$(thc_display_path "$EXO_ROOT")"
echo "[check] python_bin=$(thc_display_path "$PYTHON_BIN")"
echo "[check] model_dir=$(thc_display_path "$MODEL_DIR")"
echo "[check] cluster_file=$(thc_display_path "$CLUSTER_FILE")"

thc_require_dir "$REPO_ROOT"
thc_require_dir "$EXO_ROOT"
thc_prepare_python_bin "$PYTHON_BIN"
thc_require_dir "$MODEL_DIR"
thc_require_file "$CLUSTER_FILE"

thc_install_missing_modules "$PYTHON_BIN" \
  "numpy::numpy" \
  "mlx::mlx" \
  "mlx_lm::mlx-lm"

"$PYTHON_BIN" - "$CLUSTER_FILE" "$EXO_ROOT" <<'PY'
import importlib
import json
import sys
from pathlib import Path

cluster_file = Path(sys.argv[1])
data = json.loads(cluster_file.read_text(encoding="utf-8"))
if not isinstance(data, list) or len(data) != 3:
    raise SystemExit("[error] cluster file must be a JSON list of exactly 3 nodes")
names = [str(item["name"]) for item in data]
if len(names) != len(set(names)):
    raise SystemExit("[error] cluster node names must be unique")
mods = [
    "numpy",
    "mlx",
    "mlx_lm",
]
for name in mods:
    importlib.import_module(name)
exo_src = Path(sys.argv[2]).expanduser().resolve() / "src"
if not exo_src.exists():
    raise SystemExit(f"[error] exo src path not found: {exo_src}")
sys.path.insert(0, str(exo_src))
import exo  # noqa: F401
print(f"[check] cluster nodes={names}")
print("[check] python imports ok: numpy, mlx, mlx_lm, exo")
PY

echo "[ok] T3 environment check passed"
