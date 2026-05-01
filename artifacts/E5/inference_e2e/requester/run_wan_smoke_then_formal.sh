#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<EOF
Usage:
  ${0} [--config <config.yaml>] [--run-dir <output-dir>]

Runs:
  1. WAN smoke: 1 task, 2 questions
  2. WAN formal: normal WAN run

This script is intended to be launched locally on the requester host
because WAN shaping requires local sudo access.
EOF
}

CONFIG_PATH="${DEFAULT_CONFIG}"
RUN_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 1; }
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-dir" >&2; exit 1; }
      RUN_DIR="$2"
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

[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }

cd "${REPO_ROOT}"

RUNNER="${SCRIPT_DIR}/runner.py"
CHECKER="${SCRIPT_DIR}/check_exo_ready.py"
NETEM="${SCRIPT_DIR}/netem_macos.sh"
NETEM_CTL="${SCRIPT_DIR}/netem_root_ctl.py"
[[ -f "${RUNNER}" ]] || { echo "Runner not found: ${RUNNER}" >&2; exit 1; }
[[ -f "${CHECKER}" ]] || { echo "Checker not found: ${CHECKER}" >&2; exit 1; }
[[ -f "${NETEM}" ]] || { echo "Netem script not found: ${NETEM}" >&2; exit 1; }
[[ -f "${NETEM_CTL}" ]] || { echo "Netem helper client not found: ${NETEM_CTL}" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }

OUTPUT_ROOT_RAW="$("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("runtime", {}).get("output_root", "artifacts/E5/inference_e2e/output"))
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

if [[ "${OUTPUT_ROOT_RAW}" = /* ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT_RAW}"
else
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT_RAW}"
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${OUTPUT_ROOT}/wan_auto_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${RUN_DIR}"
SMOKE_DIR="${RUN_DIR}/smoke"
FORMAL_DIR="${RUN_DIR}/formal"
mkdir -p "${SMOKE_DIR}" "${FORMAL_DIR}"
STATUS_PATH="${RUN_DIR}/control_status.json"

write_status() {
  "${PYTHON_BIN}" - "${STATUS_PATH}" "$1" "$2" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "detail": sys.argv[3],
}
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

SMOKE_CONFIG="${RUN_DIR}/config_wan_smoke.yaml"
FORMAL_CONFIG="${RUN_DIR}/config_wan_formal.yaml"
SMOKE_SCHEDULE="${RUN_DIR}/task_schedule_smoke_2q.json"

CURRENT_INSTANCE_ID="$("${PYTHON_BIN}" - "${CONFIG_PATH}" "${SMOKE_CONFIG}" "${FORMAL_CONFIG}" "${SMOKE_SCHEDULE}" "${REPO_ROOT}" <<'PY'
from pathlib import Path
import copy
import json
import sys
import urllib.request
import yaml

src = Path(sys.argv[1])
smoke_dst = Path(sys.argv[2])
formal_dst = Path(sys.argv[3])
smoke_schedule_dst = Path(sys.argv[4])
repo_root = Path(sys.argv[5])

with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

config_base = src.parent
schedule_src = Path(str(cfg["task"]["schedule_path"]))
if not schedule_src.is_absolute():
    config_relative = (config_base / schedule_src).resolve()
    repo_relative = (repo_root / schedule_src).resolve()
    if config_relative.exists():
        schedule_src = config_relative
    else:
        schedule_src = repo_relative

entry_url = str(cfg["endpoints"]["entry_url"]).rstrip("/")
model_id = str(cfg["model"]["model_id"])
state = json.load(urllib.request.urlopen(f"{entry_url}/state", timeout=10))

instances = state.get("instances") or {}
matching_ids = []
for instance_id, payload in instances.items():
    ring = payload.get("MlxRingInstance") if isinstance(payload, dict) else None
    if not isinstance(ring, dict):
        continue
    shard_assignments = ring.get("shardAssignments") or {}
    if str(shard_assignments.get("modelId", "")) != model_id:
        continue
    runner_to_shard = shard_assignments.get("runnerToShard") or {}
    world_sizes = set()
    for shard_payload in runner_to_shard.values():
        meta = shard_payload.get("PipelineShardMetadata") if isinstance(shard_payload, dict) else None
        if isinstance(meta, dict):
            world_sizes.add(int(meta.get("worldSize", 0)))
    if world_sizes == {3} and len(runner_to_shard) == 3:
        matching_ids.append(str(instance_id))

if len(matching_ids) != 1:
    raise SystemExit(
        f"Expected exactly one active n=3 instance for {model_id}, got {matching_ids}"
    )

n3_instance_id = matching_ids[0]

smoke = copy.deepcopy(cfg)
smoke["task"]["question_count"] = 2
smoke["task"]["smoke_question_count"] = 2
smoke["task"]["tasks_per_cell"] = 1
smoke["matrix"]["network_profiles"] = ["WAN"]
smoke["matrix"]["instance_node_counts"] = [3]
smoke["matrix"]["instance_node_counts_by_network"] = {"WAN": [3]}
smoke.setdefault("external_exo", {})["expected_instance_ids_by_node_count"] = {"3": [n3_instance_id]}
smoke_schedule = copy.deepcopy(json.loads(schedule_src.read_text(encoding="utf-8")))
smoke_prompt_ids = list((smoke_schedule.get("smoke") or {}).get("prompt_ids") or [])
if len(smoke_prompt_ids) < 2:
    raise SystemExit(f"Smoke schedule needs at least 2 prompts, got {smoke_prompt_ids}")
smoke_schedule["smoke"] = {"prompt_ids": smoke_prompt_ids[:2]}
smoke["task"]["schedule_path"] = str(smoke_schedule_dst)

formal = copy.deepcopy(cfg)
formal["matrix"]["network_profiles"] = ["WAN"]
formal["matrix"]["instance_node_counts"] = [3]
formal["matrix"]["instance_node_counts_by_network"] = {"WAN": [3]}
formal.setdefault("external_exo", {})["expected_instance_ids_by_node_count"] = {"3": [n3_instance_id]}

with smoke_schedule_dst.open("w", encoding="utf-8") as f:
    json.dump(smoke_schedule, f, indent=2)
with smoke_dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(smoke, f, sort_keys=False, allow_unicode=True)
with formal_dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(formal, f, sort_keys=False, allow_unicode=True)

print(n3_instance_id)
PY
)"

cleanup() {
  if [[ -S "/tmp/bcra_netem_root.sock" ]]; then
    "${PYTHON_BIN}" "${NETEM_CTL}" reset >/dev/null 2>&1 || true
  else
    sudo "${NETEM}" reset >/dev/null 2>&1 || true
  fi
}

on_error() {
  local line_no="$1"
  write_status "failed" "Command failed at line ${line_no}"
}

trap cleanup EXIT INT TERM
trap 'on_error ${LINENO}' ERR

echo "Run dir: ${RUN_DIR}"
echo "Resolved n=3 instance: ${CURRENT_INSTANCE_ID}"
write_status "starting" "Applying WAN shaping"
if [[ -S "/tmp/bcra_netem_root.sock" ]]; then
  "${PYTHON_BIN}" "${NETEM_CTL}" apply --ports "52415,18080,18081,8080" --target-spec "${WAN_SPEC}"
else
  sudo "${NETEM}" apply --ports "52415,18080,18081,8080" --target-spec "${WAN_SPEC}"
fi

write_status "preflight_smoke" "Running black-box WAN smoke preflight"
"${PYTHON_BIN}" "${CHECKER}" --config "${SMOKE_CONFIG}" --instance-node-count 3 \
  > "${RUN_DIR}/preflight_smoke.json"

write_status "running_smoke" "Running WAN smoke (1 task, 2 questions)"
"${PYTHON_BIN}" "${RUNNER}" \
  --config "${SMOKE_CONFIG}" \
  --network-profile WAN \
  --instance-node-count 3 \
  --smoke \
  --output-dir "${SMOKE_DIR}" \
  --no-plot \
  > "${RUN_DIR}/smoke.log" 2>&1

write_status "preflight_formal" "Smoke passed; running WAN formal preflight"
"${PYTHON_BIN}" "${CHECKER}" --config "${FORMAL_CONFIG}" --instance-node-count 3 \
  > "${RUN_DIR}/preflight_formal.json"

write_status "running_formal" "Running WAN formal experiment"
"${PYTHON_BIN}" "${RUNNER}" \
  --config "${FORMAL_CONFIG}" \
  --network-profile WAN \
  --instance-node-count 3 \
  --output-dir "${FORMAL_DIR}" \
  --no-plot \
  > "${RUN_DIR}/formal.log" 2>&1

write_status "completed" "WAN smoke + formal completed"
echo "Completed: ${RUN_DIR}"
