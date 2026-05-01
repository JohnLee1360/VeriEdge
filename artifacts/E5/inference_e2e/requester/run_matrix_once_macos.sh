#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/runner.py"
CHECKER="${SCRIPT_DIR}/check_exo_ready.py"
NETEM="${SCRIPT_DIR}/netem_macos.sh"
TABLE_SCRIPT="${SCRIPT_DIR}/make_comparison_table.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot.py"
DEFAULT_CONFIG="${SCRIPT_DIR}/config.yaml"

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi

  local candidates=(
    "/Users/johnlee/repo/paper/exp_exo/.venv/bin/python"
    "/Users/johnlee/repo/paper/bc-ra-paper/.venv/bin/python"
    "/Users/jlmini_2/repo/paper/exp_exo/.venv/bin/python"
    "/Users/jlmini_2/repo/paper/bc-ra-paper/.venv/bin/python"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  command -v python3
}

usage() {
  cat <<EOF
Usage:
  ${0} [--config <path-to-config.yaml>] [--batch-dir <existing-or-new-batch-dir>]

Runs the fixed 4-cell order:
  1) LAN + instance_node_count=1
  2) LAN + instance_node_count=2
  3) LAN + instance_node_count=3
  4) WAN + instance_node_count=3

Each cell is executed as:
  smoke task -> 5 main tasks
EOF
}

CONFIG_PATH="${DEFAULT_CONFIG}"
BATCH_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 1; }
      CONFIG_PATH="$2"
      shift 2
      ;;
    --batch-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --batch-dir" >&2; exit 1; }
      BATCH_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="$(resolve_python_bin)"

[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${RUNNER}" ]] || { echo "Runner not found: ${RUNNER}" >&2; exit 1; }
[[ -f "${CHECKER}" ]] || { echo "Checker not found: ${CHECKER}" >&2; exit 1; }
[[ -f "${NETEM}" ]] || { echo "Netem script not found: ${NETEM}" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ["cryptography", "httpx", "pandas", "yaml"]
optional = ["matplotlib"]
missing_required = [name for name in required if importlib.util.find_spec(name) is None]
missing_optional = [name for name in optional if importlib.util.find_spec(name) is None]

if missing_required:
    print(
        "Python environment is missing required modules for batch execution: "
        + ", ".join(missing_required),
        file=sys.stderr,
    )
    sys.exit(1)
if missing_optional:
    print(
        "Warning: Python environment is missing optional plotting modules: "
        + ", ".join(missing_optional),
        file=sys.stderr,
    )
PY

OUTPUT_ROOT_RAW="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("runtime", {}).get("output_root", "artifacts/E5/inference_e2e/requester/output"))
PY
)"

WAN_SPEC="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["network_profiles"]["WAN"]["target_spec"])
PY
)"

WAN_PORTS="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
from urllib.parse import urlparse
import sys
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ports = set()
ports.add(urlparse(cfg["endpoints"]["entry_url"]).port or 52415)
ports.add(urlparse(cfg["endpoints"]["requester_callback_url"]).port or 18081)
ports.add(urlparse(cfg["ipfs"]["gateway_url"]).port or 8080)
for provider in cfg.get("providers", []):
    launcher_port = urlparse(str(provider.get("launcher_url", ""))).port
    if launcher_port is not None:
        ports.add(launcher_port)
print(",".join(str(port) for port in sorted(ports)))
PY
)"

if [[ "${OUTPUT_ROOT_RAW}" = /* ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT_RAW}"
else
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT_RAW}"
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ -z "${BATCH_DIR}" ]]; then
  BATCH_ID="$(date +%Y%m%d_%H%M%S)"
  BATCH_DIR="${OUTPUT_ROOT}/batch_${BATCH_ID}_task_level_multi_task"
fi
mkdir -p "${BATCH_DIR}"
mkdir -p "${BATCH_DIR}/cells"

SUDO_KEEPALIVE_PID=""
SUDO_READY=0
cleanup() {
  set +e
  if [[ -n "${SUDO_KEEPALIVE_PID}" ]]; then
    kill "${SUDO_KEEPALIVE_PID}" >/dev/null 2>&1 || true
    wait "${SUDO_KEEPALIVE_PID}" >/dev/null 2>&1 || true
  fi
  sudo -n "${NETEM}" reset >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Batch dir: ${BATCH_DIR}"
echo "Config   : ${CONFIG_PATH}"
echo "Python   : ${PYTHON_BIN}"

"${PYTHON_BIN}" - "${BATCH_DIR}" "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import json
import sys

batch_dir = Path(sys.argv[1])
config_path = sys.argv[2]
cells = [
    {"cell_id": "lan_n1", "network": "LAN", "instance_node_count": 1},
    {"cell_id": "lan_n2", "network": "LAN", "instance_node_count": 2},
    {"cell_id": "lan_n3", "network": "LAN", "instance_node_count": 3},
    {"cell_id": "wan_n3", "network": "WAN", "instance_node_count": 3},
]
(batch_dir / "batch_manifest.json").write_text(json.dumps({
    "config_path": config_path,
    "cells": cells,
}, indent=2), encoding="utf-8")
(batch_dir / "batch_status.json").write_text(json.dumps({
    "status": "running",
    "current_cell_id": None,
    "completed_cells": [],
    "failed_cell_id": None,
}, indent=2), encoding="utf-8")
PY

update_batch_status() {
  local status="$1"
  local current_cell_id="$2"
  local failed_cell_id="$3"
  "${PYTHON_BIN}" - "${BATCH_DIR}/batch_status.json" "${status}" "${current_cell_id}" "${failed_cell_id}" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
status = sys.argv[2]
current_cell_id = sys.argv[3] if sys.argv[3] else None
failed_cell_id = sys.argv[4] if sys.argv[4] else None
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = status
payload["current_cell_id"] = current_cell_id
payload["failed_cell_id"] = failed_cell_id
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

mark_cell_completed() {
  local cell_id="$1"
  "${PYTHON_BIN}" - "${BATCH_DIR}/batch_status.json" "${cell_id}" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
cell_id = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
completed = list(payload.get("completed_cells", []))
if cell_id not in completed:
    completed.append(cell_id)
payload["completed_cells"] = completed
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

ensure_sudo_ready() {
  if [[ "${SUDO_READY}" == "1" ]]; then
    return 0
  fi
  echo "Requesting sudo for WAN shaping..."
  sudo -v
  ( while true; do sudo -n true; sleep 20; done ) &
  SUDO_KEEPALIVE_PID="$!"
  SUDO_READY=1
}

run_cell() {
  local network="$1"
  local instance_node_count="$2"
  local cell_id="$3"
  local cell_dir="${BATCH_DIR}/cells/${cell_id}"
  local log_file="${cell_dir}/runner.log"
  mkdir -p "${cell_dir}"

  if [[ -f "${cell_dir}/cell_status.json" ]]; then
    existing_status="$("${PYTHON_BIN}" - "${cell_dir}/cell_status.json" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload.get("status", ""))
PY
)"
    if [[ "${existing_status}" == "completed" && -f "${cell_dir}/summary_by_cell.csv" ]]; then
      echo "Skipping completed cell: ${cell_id}"
      mark_cell_completed "${cell_id}"
      return 0
    fi
  fi

  echo
  echo "========================================================================"
  echo "Running cell: ${cell_id}"
  echo "========================================================================"
  update_batch_status "running" "${cell_id}" ""

  if [[ "${network}" == "LAN" ]]; then
    if [[ "${SUDO_READY}" == "1" ]]; then
      sudo "${NETEM}" reset
    fi
  else
    ensure_sudo_ready
    sudo "${NETEM}" apply --ports "${WAN_PORTS}" --target-spec "${WAN_SPEC}"
  fi

  if ! "${PYTHON_BIN}" "${CHECKER}" --config "${CONFIG_PATH}" --instance-node-count "${instance_node_count}" \
    > "${cell_dir}/preflight_check.json"; then
    update_batch_status "failed" "${cell_id}" "${cell_id}"
    return 1
  fi

  if ! "${PYTHON_BIN}" "${RUNNER}" \
    --config "${CONFIG_PATH}" \
    --network-profile "${network}" \
    --instance-node-count "${instance_node_count}" \
    --output-dir "${cell_dir}" \
    --no-plot | tee "${log_file}"; then
    update_batch_status "failed" "${cell_id}" "${cell_id}"
    return 1
  fi

  mark_cell_completed "${cell_id}"
}

run_cell "LAN" "1" "lan_n1"
run_cell "LAN" "2" "lan_n2"
run_cell "LAN" "3" "lan_n3"
run_cell "WAN" "3" "wan_n3"
sudo "${NETEM}" reset

"${PYTHON_BIN}" - "${BATCH_DIR}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

batch_dir = Path(sys.argv[1])
frames = []
task_frames = []
for cell_summary in sorted(batch_dir.glob("cells/*/summary_by_cell.csv")):
    frames.append(pd.read_csv(cell_summary))
for task_summary in sorted(batch_dir.glob("cells/*/summary_by_task.csv")):
    task_frames.append(pd.read_csv(task_summary))
combined = pd.concat(frames, ignore_index=True)
combined = combined.sort_values(["network", "instance_node_count"])
combined.to_csv(batch_dir / "summary_by_cell.csv", index=False)
if task_frames:
    tasks = pd.concat(task_frames, ignore_index=True)
    tasks.to_csv(batch_dir / "summary_by_task.csv", index=False)
PY

"${PYTHON_BIN}" "${TABLE_SCRIPT}" --input "${BATCH_DIR}/summary_by_cell.csv" --output-csv "${BATCH_DIR}/comparison_table.csv" --output-md "${BATCH_DIR}/comparison_table.md"
"${PYTHON_BIN}" "${PLOT_SCRIPT}" --input "${BATCH_DIR}/summary_by_cell.csv" --output-dir "${BATCH_DIR}"

update_batch_status "completed" "" ""

echo "All done."
echo "Batch dir: ${BATCH_DIR}"
