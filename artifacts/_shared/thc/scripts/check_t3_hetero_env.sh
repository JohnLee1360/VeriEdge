#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

thc_require_env CLUSTER_FILE
thc_require_env LOCAL_NODE

PYTHON_BIN="${PYTHON_BIN:-$REPO_VENV_PYTHON}"

echo "[check] repo_root=$(thc_display_path "$REPO_ROOT")"
echo "[check] python_bin=$(thc_display_path "$PYTHON_BIN")"
echo "[check] cluster_file=$(thc_display_path "$CLUSTER_FILE")"
echo "[check] local_node=$LOCAL_NODE"

thc_prepare_python_bin "$PYTHON_BIN"
thc_require_file "$CLUSTER_FILE"

NODE_INFO_RAW="$("$PYTHON_BIN" - "$CLUSTER_FILE" "$LOCAL_NODE" <<'PY'
import json
import sys
from pathlib import Path

cluster_file = Path(sys.argv[1])
local_node = sys.argv[2]
data = json.loads(cluster_file.read_text(encoding="utf-8"))
nodes = data.get("nodes", [])
if not isinstance(nodes, list) or len(nodes) != 3:
    raise SystemExit("[error] cluster file must contain exactly 3 nodes")
names = [str(node["node_name"]) for node in nodes]
if local_node not in names:
    raise SystemExit(f"[error] local node {local_node!r} not found in cluster file")
node = next(node for node in nodes if str(node["node_name"]) == local_node)
print(str(node["checkpoint"]))
print(str(node["device"]).lower())
print(str(node["start_layer"]))
print(str(node["end_layer"]))
print(str(node.get("quantization", "none")).lower())
PY
)"

NODE_INFO=()
while IFS= read -r line; do
  NODE_INFO+=("$line")
done <<EOF
$NODE_INFO_RAW
EOF

CHECKPOINT="${NODE_INFO[0]}"
DEVICE="${NODE_INFO[1]}"
START_LAYER="${NODE_INFO[2]}"
END_LAYER="${NODE_INFO[3]}"
QUANTIZATION="${NODE_INFO[4]}"

MODULE_SPECS=(
  "numpy::numpy"
  "torch::${THC_TORCH_PIP_SPEC:-torch}"
  "transformers::transformers"
  "safetensors::safetensors"
  "accelerate::accelerate"
  "sentencepiece::sentencepiece"
)
if [[ "$QUANTIZATION" == "metal_8bit" ]]; then
  MODULE_SPECS+=("kernels::${THC_METAL_KERNELS_PIP_SPEC:-kernels}")
fi
if [[ "$QUANTIZATION" == "bitsandbytes_8bit" ]]; then
  MODULE_SPECS+=("bitsandbytes::${THC_BITSANDBYTES_PIP_SPEC:-bitsandbytes}")
fi

thc_install_missing_modules "$PYTHON_BIN" "${MODULE_SPECS[@]}"

echo "[check] node entry checkpoint=$CHECKPOINT device=$DEVICE layers=$START_LAYER..$END_LAYER"

"$PYTHON_BIN" - "$CLUSTER_FILE" "$LOCAL_NODE" <<'PY'
import importlib
import json
import sys
from pathlib import Path

import torch

cluster_file = Path(sys.argv[1])
local_node = sys.argv[2]
mods = ["numpy", "torch", "transformers", "safetensors", "accelerate", "sentencepiece"]
for name in mods:
    importlib.import_module(name)

data = json.loads(cluster_file.read_text(encoding="utf-8"))
nodes = data.get("nodes", [])
if not isinstance(nodes, list) or len(nodes) != 3:
    raise SystemExit("[error] cluster file must contain exactly 3 nodes")
names = [str(node["node_name"]) for node in nodes]
if local_node not in names:
    raise SystemExit(f"[error] local node {local_node!r} not found in cluster file")
node = next(node for node in nodes if str(node["node_name"]) == local_node)
device = str(node["device"]).lower()
quantization = str(node.get("quantization", "none")).lower()
cuda_ok = torch.cuda.is_available()
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
if device == "cuda" and not cuda_ok:
    raise SystemExit("[error] cluster expects cuda, but torch.cuda.is_available() is false")
if device == "mps" and not mps_ok:
    raise SystemExit("[error] cluster expects mps, but torch.backends.mps.is_available() is false")
if quantization == "metal_8bit":
    importlib.import_module("kernels")
if quantization == "bitsandbytes_8bit":
    importlib.import_module("bitsandbytes")
print(
    "[check] python imports ok:",
    {
        "torch": torch.__version__,
        "cuda": cuda_ok,
        "mps": mps_ok,
        "local_device": device,
        "quantization": quantization,
    },
)
PY

echo "[ok] heterogeneous T3 environment check passed"
