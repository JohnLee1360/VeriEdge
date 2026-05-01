#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/provider.env"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
PID_FILE="${RUNTIME_DIR}/exo.pid"
LOG_FILE="${RUNTIME_DIR}/exo.log"
LAUNCHER_PID_FILE="${RUNTIME_DIR}/launcher.pid"
LAUNCHER_LOG_FILE="${RUNTIME_DIR}/launcher.log"

mkdir -p "${RUNTIME_DIR}"

usage() {
  cat <<EOF
Usage:
  ${0} start      # start exo process in background
  ${0} stop       # stop process by PID
  ${0} restart    # stop then start
  ${0} status     # print process + endpoint status
  ${0} health     # run health check only
  ${0} probe      # run one tiny inference probe
  ${0} launcher-start   # start task launcher in background
  ${0} launcher-stop    # stop task launcher
  ${0} launcher-status  # print task launcher status
  ${0} keys-gen         # generate provider RSA keypair
  ${0} freeze-verify [manifest-path]
                 # verify local exo tree matches freeze manifest
  ${0} lan-reset <lan-ip> [node-port]
                 # rewrite provider.env with LAN binding, hard reset cache, restart and check
  ${0} lan-reset-frozen <lan-ip> [node-port] [manifest-path]
                 # freeze-verify first, then lan-reset

Setup:
  cp ${SCRIPT_DIR}/provider.env.example ${SCRIPT_DIR}/provider.env
  edit ${SCRIPT_DIR}/provider.env (EXO_START_CMD is required)
EOF
}

load_env() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}. Copy provider.env.example first." >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a

  : "${EXO_ENDPOINT:?EXO_ENDPOINT is required in provider.env}"
}

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

endpoint_port() {
  local endpoint="${EXO_ENDPOINT:-}"
  local hostport
  hostport="${endpoint#*://}"
  hostport="${hostport%%/*}"

  if [[ "$hostport" =~ :([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

listener_pids_by_endpoint() {
  local port
  port="$(endpoint_port || true)"
  [[ -n "$port" ]] || return 0
  lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true
}

launcher_url() {
  local host="${LAUNCHER_HOST:-127.0.0.1}"
  local port
  port="$(launcher_port || true)"
  [[ -n "$port" ]] || return 1
  if [[ "${host}" == "0.0.0.0" || "${host}" == "::" ]]; then
    host="127.0.0.1"
  fi
  printf 'http://%s:%s\n' "${host}" "${port}"
}

kill_pid_tree() {
  local sig="$1"
  local pid="$2"
  local child

  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_pid_tree "$sig" "$child"
  done
  kill "-${sig}" "$pid" >/dev/null 2>&1 || true
}

is_ipv4() {
  local ip="$1"
  [[ "${ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
}

expand_home_path() {
  local p="$1"
  if [[ "${p}" == "~" ]]; then
    printf '%s\n' "${HOME}"
    return
  fi
  if [[ "${p}" == "~/"* ]]; then
    printf '%s\n' "${HOME}/${p#~/}"
    return
  fi
  printf '%s\n' "${p}"
}

upsert_env_line() {
  local key="$1"
  local value="$2"
  local tmp
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  awk -v key="${key}" -v value="${value}" '
    BEGIN { updated=0 }
    index($0, key "=") == 1 {
      print key "=" value
      updated=1
      next
    }
    { print }
    END {
      if (updated == 0) {
        print key "=" value
      }
    }
  ' "${ENV_FILE}" > "${tmp}"
  mv "${tmp}" "${ENV_FILE}"
}

wait_until() {
  local label="$1"
  local timeout_sec="$2"
  local poll_sec="$3"
  shift 3

  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if "$@"; then
      echo "${label}: ready"
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if (( elapsed >= timeout_sec )); then
      echo "${label}: timeout after ${timeout_sec}s" >&2
      return 1
    fi
    sleep "${poll_sec}"
  done
}

spawn_detached_shell() {
  local log_file="$1"
  local shell_command="$2"
  local python_bin
  python_bin="$(command -v python3 || true)"
  [[ -n "${python_bin}" ]] || {
    echo "python3 not found" >&2
    return 1
  }
  "${python_bin}" - "${log_file}" "${shell_command}" <<'PY'
import subprocess
import sys

log_path, shell_command = sys.argv[1], sys.argv[2]
with open(log_path, "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", shell_command],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
print(proc.pid)
PY
}

spawn_detached_argv() {
  local log_file="$1"
  shift
  local python_bin
  python_bin="$(command -v python3 || true)"
  [[ -n "${python_bin}" ]] || {
    echo "python3 not found" >&2
    return 1
  }
  "${python_bin}" - "${log_file}" "$@" <<'PY'
import subprocess
import sys

log_path = sys.argv[1]
argv = sys.argv[2:]
with open(log_path, "ab", buffering=0) as log_file:
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
print(proc.pid)
PY
}

exo_health_ready() {
  load_env
  local timeout="${HEALTH_TIMEOUT_SEC:-5}"
  python3 "${SCRIPT_DIR}/health_check.py" --endpoint "${EXO_ENDPOINT}" --timeout-sec "${timeout}" >/dev/null 2>&1
}

launcher_health_ready() {
  load_env
  local url
  url="$(launcher_url)" || return 1
  curl -fsS --max-time "${HEALTH_TIMEOUT_SEC:-5}" "${url%/}/health" >/dev/null 2>&1
}

provider_visible_in_state() {
  load_env
  : "${PROVIDER_IP:?PROVIDER_IP is required in provider.env}"
  local state_json
  state_json="$(curl -fsS --max-time "${HEALTH_TIMEOUT_SEC:-5}" "${EXO_ENDPOINT%/}/state" 2>/dev/null)" || return 1
  STATE_JSON="${state_json}" python3 - "${PROVIDER_IP}" <<'PY'
import json
import os
import sys

provider_ip = sys.argv[1]
try:
    state = json.loads(os.environ["STATE_JSON"])
except json.JSONDecodeError:
    raise SystemExit(1)

node_network = state.get("nodeNetwork")
if not isinstance(node_network, dict):
    raise SystemExit(1)

for info in node_network.values():
    if not isinstance(info, dict):
        continue
    for item in info.get("interfaces", []):
        if isinstance(item, dict) and item.get("ipAddress") == provider_ip:
            raise SystemExit(0)
raise SystemExit(1)
PY
}

cluster_node_count_ready() {
  load_env
  local expected="${CLUSTER_JOIN_EXPECTED_NODE_COUNT:-}"
  if [[ -z "${expected}" ]]; then
    return 0
  fi
  [[ "${expected}" =~ ^[0-9]+$ ]] || {
    echo "Invalid CLUSTER_JOIN_EXPECTED_NODE_COUNT=${expected}" >&2
    return 1
  }

  local state_json
  state_json="$(curl -fsS --max-time "${HEALTH_TIMEOUT_SEC:-5}" "${EXO_ENDPOINT%/}/state" 2>/dev/null)" || return 1
  STATE_JSON="${state_json}" python3 - "${expected}" <<'PY'
import json
import os
import sys

expected = int(sys.argv[1])
try:
    state = json.loads(os.environ["STATE_JSON"])
except json.JSONDecodeError:
    raise SystemExit(1)

node_network = state.get("nodeNetwork")
if not isinstance(node_network, dict):
    raise SystemExit(1)

raise SystemExit(0 if len(node_network) >= expected else 1)
PY
}

wait_for_exo_ready() {
  local startup_timeout="${STARTUP_TIMEOUT_SEC:-180}"
  local poll_sec="${STARTUP_POLL_SEC:-2}"
  wait_until "exo endpoint ${EXO_ENDPOINT}" "${startup_timeout}" "${poll_sec}" exo_health_ready
}

wait_for_launcher_ready() {
  local launcher_timeout="${LAUNCHER_STARTUP_TIMEOUT_SEC:-60}"
  local poll_sec="${LAUNCHER_STARTUP_POLL_SEC:-1}"
  wait_until "launcher $(launcher_url)" "${launcher_timeout}" "${poll_sec}" launcher_health_ready
}

wait_for_cluster_join() {
  local join_timeout="${CLUSTER_JOIN_TIMEOUT_SEC:-120}"
  local poll_sec="${CLUSTER_JOIN_POLL_SEC:-2}"

  wait_until "provider ${PROVIDER_IP} visible in /state" "${join_timeout}" "${poll_sec}" provider_visible_in_state
  if [[ -n "${CLUSTER_JOIN_EXPECTED_NODE_COUNT:-}" ]]; then
    wait_until \
      "cluster size >= ${CLUSTER_JOIN_EXPECTED_NODE_COUNT}" \
      "${join_timeout}" \
      "${poll_sec}" \
      cluster_node_count_ready
  fi
}

resolve_nix_bin() {
  local nix_bin
  nix_bin="$(command -v nix || true)"
  if [[ -z "${nix_bin}" && -x "/nix/var/nix/profiles/default/bin/nix" ]]; then
    nix_bin="/nix/var/nix/profiles/default/bin/nix"
  fi
  if [[ -z "${nix_bin}" ]]; then
    echo "nix not found; install Nix to run EXO via nix run ." >&2
    return 1
  fi
  printf '%s' "${nix_bin}"
}

build_default_start_cmd() {
  local exo_repo="${EXO_REPO_DIR:-~/repo/paper/third_party/exo}"
  local hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
  local exo_home="${EXO_HOME:-~/.cache/exo}"
  local exo_debug="${EXO_DEBUG:-9}"
  local exo_repo_abs
  exo_repo_abs="$(expand_home_path "${exo_repo}")"
  if [[ ! -f "${exo_repo_abs}/flake.nix" ]]; then
    echo "flake.nix not found in ${exo_repo_abs}; cannot derive nix start command" >&2
    return 1
  fi
  local nix_bin
  nix_bin="$(resolve_nix_bin)" || return 1
  local exo_repo_q hf_q exo_home_q debug_q nix_q
  exo_repo_q="$(printf '%q' "${exo_repo_abs}")"
  hf_q="$(printf '%q' "${hf_endpoint}")"
  exo_home_q="$(printf '%q' "${exo_home}")"
  debug_q="$(printf '%q' "${exo_debug}")"
  nix_q="$(printf '%q' "${nix_bin}")"
  printf 'cd %s && HF_ENDPOINT=%s EXO_HOME=%s DEBUG=%s %s run .' \
    "${exo_repo_q}" "${hf_q}" "${exo_home_q}" "${debug_q}" "${nix_q}"
}

get_effective_provider_ip() {
  if [[ -n "${PROVIDER_IP:-}" ]]; then
    printf '%s' "${PROVIDER_IP}"
    return
  fi
  local endpoint="${EXO_ENDPOINT:-}"
  local host
  host="${endpoint#*://}"
  host="${host%%/*}"
  host="${host%%:*}"
  host="${host#[}"
  host="${host%]}"
  if [[ -n "${host}" ]]; then
    printf '%s' "${host}"
  fi
}

cluster_ready_check() {
  local state_url
  state_url="${EXO_ENDPOINT%/}/state"
  local state_body
  if ! state_body="$(curl -sSf "${state_url}" 2>/dev/null)"; then
    return 1
  fi
  local state_file
  state_file="$(mktemp)"
  printf '%s' "${state_body}" >"${state_file}"
  local python_bin
  python_bin="$(command -v python3 || true)"
  if [[ -z "${python_bin}" ]]; then
    rm -f "${state_file}"
    return 1
  fi
  local provider_node_id="${PROVIDER_NODE_ID:-}"
  local provider_ip
  provider_ip="$(get_effective_provider_ip)"
  "${python_bin}" - "${state_file}" "${provider_node_id}" "${provider_ip}" <<'PY'
import json, sys

state_path = sys.argv[1]
target_node = sys.argv[2] or None
target_ip = sys.argv[3] or None

with open(state_path, encoding='utf-8') as fh:
    data = json.load(fh)

network = data.get('nodeNetwork') or {}
if target_node and target_node in network:
    sys.exit(0)
if target_ip:
    for info in network.values():
        for iface in info.get('interfaces') or []:
            if iface.get('ipAddress') == target_ip:
                sys.exit(0)
sys.exit(1)
PY
  local rc=$?
  rm -f "${state_file}"
  return ${rc}
}

wait_for_cluster_ready() {
  local timeout="${CLUSTER_READY_TIMEOUT_SEC:-120}"
  local interval="${CLUSTER_READY_POLL_INTERVAL_SEC:-3}"
  timeout="${timeout%.*}"
  interval="${interval%.*}"
  if [[ -z "${timeout}" || "${timeout}" -le 0 ]]; then
    timeout=120
  fi
  if [[ -z "${interval}" || "${interval}" -le 0 ]]; then
    interval=3
  fi
  local start_time
  start_time=${SECONDS}
  while true; do
    if cluster_ready_check; then
      local elapsed=$((SECONDS - start_time))
      echo "EXO node joined cluster after ${elapsed}s"
      return 0
    fi
    if (( SECONDS - start_time >= timeout )); then
      echo "Timed out waiting for EXO cluster readiness after ${timeout}s" >&2
      return 1
    fi
    sleep ${interval}
  done
}

cmd_start() {
  load_env
  : "${EXO_START_CMD:?EXO_START_CMD is required in provider.env}"

  if is_running; then
    echo "Already running. PID=$(cat "${PID_FILE}")"
    return 0
  fi

  local existing
  existing="$(listener_pids_by_endpoint)"
  if [[ -n "${existing}" ]]; then
    local first_pid
    first_pid="$(head -n1 <<<"${existing}")"
    echo "${first_pid}" >"${PID_FILE}"
    echo "Already running (detected by endpoint). PID=${first_pid}"
    return 0
  fi

  echo "Starting provider: ${PROVIDER_NODE_ID:-unknown}"
  echo "Endpoint: ${EXO_ENDPOINT}"
  echo "Log: ${LOG_FILE}"

  local pid
  pid="$(spawn_detached_shell "${LOG_FILE}" "${EXO_START_CMD}")" || {
    echo "Failed to launch provider process" >&2
    return 1
  }
  echo "${pid}" >"${PID_FILE}"

  sleep 1
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Started. PID=${pid}"
    wait_for_exo_ready
    wait_for_cluster_ready
  else
    echo "Process exited early. Check log: ${LOG_FILE}" >&2
    return 1
  fi
}

cmd_stop() {
  local have_env="0"
  if [[ -f "${ENV_FILE}" ]]; then
    load_env
    have_env="1"
  fi

  local targets
  targets=""

  if [[ -f "${PID_FILE}" ]]; then
    targets+="$(cat "${PID_FILE}")"$'\n'
  fi

  if [[ "$have_env" == "1" ]]; then
    targets+="$(listener_pids_by_endpoint)"$'\n'
  fi

  targets="$(printf '%s' "${targets}" | awk '/^[0-9]+$/' | sort -u)"
  if [[ -z "${targets}" ]]; then
    echo "Not running."
    rm -f "${PID_FILE}"
    return 0
  fi

  local pid
  for pid in ${targets}; do
    echo "Stopping PID=${pid}"
    kill_pid_tree TERM "${pid}"
  done

  for _ in {1..20}; do
    local alive="0"
    for pid in ${targets}; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        alive="1"
        break
      fi
    done
    if [[ "${alive}" == "0" ]]; then
      break
    fi
    sleep 0.2
  done

  for pid in ${targets}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Force killing PID=${pid}"
      kill_pid_tree KILL "${pid}"
    fi
  done

  if [[ "$have_env" == "1" ]]; then
    local leaked
    leaked="$(listener_pids_by_endpoint)"
    if [[ -n "${leaked}" ]]; then
      for pid in ${leaked}; do
        echo "Force killing leaked listener PID=${pid}"
        kill_pid_tree KILL "${pid}"
      done
    fi
  fi

  rm -f "${PID_FILE}"
  echo "Stopped."
}

cmd_status() {
  load_env
  if is_running; then
    echo "Process: running (PID=$(cat "${PID_FILE}"))"
  else
    local listeners
    listeners="$(listener_pids_by_endpoint)"
    if [[ -n "${listeners}" ]]; then
      local port
      port="$(endpoint_port || true)"
      echo "Process: running (detected by endpoint port ${port}; PID(s): $(xargs <<<"${listeners}"))"
    else
      echo "Process: not running"
    fi
  fi

  local timeout="${HEALTH_TIMEOUT_SEC:-5}"
  if python3 "${SCRIPT_DIR}/health_check.py" --endpoint "${EXO_ENDPOINT}" --timeout-sec "${timeout}" >/dev/null; then
    echo "Endpoint: healthy (${EXO_ENDPOINT})"
  else
    echo "Endpoint: unhealthy (${EXO_ENDPOINT})"
  fi
}

cmd_health() {
  load_env
  local timeout="${HEALTH_TIMEOUT_SEC:-5}"
  python3 "${SCRIPT_DIR}/health_check.py" --endpoint "${EXO_ENDPOINT}" --timeout-sec "${timeout}" --pretty
}

cmd_probe() {
  load_env
  local timeout="${HEALTH_TIMEOUT_SEC:-5}"
  python3 "${SCRIPT_DIR}/health_check.py" \
    --endpoint "${EXO_ENDPOINT}" \
    --timeout-sec "${timeout}" \
    --model "${MODEL_ID:-mlx-community/Qwen3-0.6B-8bit}" \
    --probe \
    --pretty
}

cmd_lan_reset() {
  local lan_ip="${1:-}"
  local node_port="${2:-50051}"
  if [[ -z "${lan_ip}" ]]; then
    echo "Missing lan-ip argument." >&2
    usage
    return 1
  fi
  if ! is_ipv4 "${lan_ip}"; then
    echo "Invalid lan-ip: ${lan_ip}" >&2
    return 1
  fi
  if ! [[ "${node_port}" =~ ^[0-9]+$ ]]; then
    echo "Invalid node-port: ${node_port}" >&2
    return 1
  fi

  load_env

  local hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
  local exo_home="${EXO_HOME:-~/.cache/exo-bcra}"
  local exo_repo="${EXO_REPO_DIR:-~/repo/paper/third_party/exo}"
  local exo_debug="${EXO_DEBUG:-9}"
  local exo_repo_abs nix_bin
  exo_repo_abs="$(expand_home_path "${exo_repo}")"
  nix_bin="$(command -v nix || true)"
  if [[ -z "${nix_bin}" && -x "/nix/var/nix/profiles/default/bin/nix" ]]; then
    nix_bin="/nix/var/nix/profiles/default/bin/nix"
  fi
  if [[ -z "${nix_bin}" ]]; then
    echo "nix not found. Install Nix first." >&2
    return 1
  fi
  if [[ ! -f "${exo_repo_abs}/flake.nix" ]]; then
    echo "Missing flake.nix in ${exo_repo_abs}" >&2
    return 1
  fi

  local start_cmd
  # Match the user's proven-good startup path: `cd <exo_repo> && nix run .`
  start_cmd="cd ${exo_repo_abs} && HF_ENDPOINT=${hf_endpoint} EXO_HOME=${exo_home} DEBUG=${exo_debug} ${nix_bin} run ."

  upsert_env_line "HF_ENDPOINT" "${hf_endpoint}"
  upsert_env_line "EXO_HOME" "${exo_home}"
  upsert_env_line "EXO_START_CMD" "\"${start_cmd}\""

  echo "Updated ${ENV_FILE}:"
  grep -E '^HF_ENDPOINT=|^EXO_HOME=|^EXO_START_CMD=' "${ENV_FILE}" || true

  local exo_home_abs
  exo_home_abs="$(expand_home_path "${exo_home}")"
  echo "Hard reset EXO_HOME: ${exo_home_abs}"

  cmd_stop || true
  cmd_launcher_stop || true
  pkill -9 -f '[e]xo' >/dev/null 2>&1 || true
  rm -rf "${exo_home_abs}"
  mkdir -p "${exo_home_abs}"

  cmd_start
  cmd_launcher_start
  wait_for_cluster_join
  cmd_status
  cmd_launcher_status

  local m_code s_code
  m_code="$(curl -s -o /dev/null -w "%{http_code}" "${EXO_ENDPOINT%/}/v1/models" || true)"
  s_code="$(curl -s -o /dev/null -w "%{http_code}" "${EXO_ENDPOINT%/}/state" || true)"
  echo "Local probe: /v1/models=${m_code} /state=${s_code}"
}

cmd_freeze_verify() {
  load_env
  local manifest_path="${1:-${FREEZE_MANIFEST_PATH:-${SCRIPT_DIR}/../freeze/exo_env_manifest.json}}"
  local verify_script="${FREEZE_VERIFY_SCRIPT:-${SCRIPT_DIR}/../requester/verify_exo_env.py}"
  local exo_repo="${EXO_REPO_DIR:-~/repo/paper/third_party/exo}"
  local python_bin=""

  manifest_path="$(expand_home_path "${manifest_path}")"
  verify_script="$(expand_home_path "${verify_script}")"
  exo_repo="$(expand_home_path "${exo_repo}")"

  if [[ ! -f "${verify_script}" ]]; then
    echo "Missing verify script: ${verify_script}" >&2
    return 1
  fi
  if [[ ! -f "${manifest_path}" ]]; then
    echo "Missing freeze manifest: ${manifest_path}" >&2
    return 1
  fi

  if [[ -x "${exo_repo}/.venv/bin/python" ]]; then
    python_bin="${exo_repo}/.venv/bin/python"
  else
    python_bin="$(command -v python3 || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    echo "python interpreter not found for freeze verification" >&2
    return 1
  fi

  "${python_bin}" "${verify_script}" \
    --manifest "${manifest_path}" \
    --exo-dir "${exo_repo}"
}

cmd_lan_reset_frozen() {
  local lan_ip="${1:-}"
  local node_port="${2:-50051}"
  local manifest_path="${3:-${FREEZE_MANIFEST_PATH:-${SCRIPT_DIR}/../freeze/exo_env_manifest.json}}"
  cmd_freeze_verify "${manifest_path}"
  cmd_lan_reset "${lan_ip}" "${node_port}"
}

launcher_port() {
  local port="${LAUNCHER_PORT:-18080}"
  [[ "$port" =~ ^[0-9]+$ ]] || return 0
  printf '%s\n' "$port"
}

launcher_listener_pids() {
  local port
  port="$(launcher_port || true)"
  [[ -n "$port" ]] || return 0
  lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true
}

cmd_launcher_start() {
  load_env
  : "${PROVIDER_PRIVATE_KEY_PATH:?PROVIDER_PRIVATE_KEY_PATH is required in provider.env}"
  : "${PROVIDER_IP:?PROVIDER_IP is required in provider.env}"

  if [[ -f "${LAUNCHER_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${LAUNCHER_PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Launcher already running. PID=${pid}"
      return 0
    fi
  fi

  local existing
  existing="$(launcher_listener_pids)"
  if [[ -n "${existing}" ]]; then
    local first_pid
    first_pid="$(head -n1 <<<"${existing}")"
    echo "${first_pid}" >"${LAUNCHER_PID_FILE}"
    echo "Launcher already running (detected by port). PID=${first_pid}"
    return 0
  fi

  local python_bin
  python_bin="$(command -v python3 || true)"
  [[ -n "${python_bin}" ]] || { echo "python3 not found" >&2; return 1; }

  echo "Starting launcher on ${PROVIDER_IP}:${LAUNCHER_PORT:-18080}"
  local pid
  pid="$(spawn_detached_argv "${LAUNCHER_LOG_FILE}" "${python_bin}" "${SCRIPT_DIR}/launcher.py")" || {
    echo "Failed to launch launcher process" >&2
    return 1
  }
  echo "${pid}" >"${LAUNCHER_PID_FILE}"
  sleep 1
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Launcher started. PID=${pid}"
    wait_for_launcher_ready
  else
    echo "Launcher exited early. Check log: ${LAUNCHER_LOG_FILE}" >&2
    return 1
  fi
}

cmd_launcher_stop() {
  local targets=""
  if [[ -f "${LAUNCHER_PID_FILE}" ]]; then
    targets+="$(cat "${LAUNCHER_PID_FILE}")"$'\n'
  fi
  targets+="$(launcher_listener_pids)"$'\n'
  targets="$(printf '%s' "${targets}" | awk '/^[0-9]+$/' | sort -u)"
  if [[ -z "${targets}" ]]; then
    echo "Launcher not running."
    rm -f "${LAUNCHER_PID_FILE}"
    return 0
  fi
  local pid
  for pid in ${targets}; do
    echo "Stopping launcher PID=${pid}"
    kill_pid_tree TERM "${pid}"
  done
  sleep 1
  for pid in ${targets}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill_pid_tree KILL "${pid}"
    fi
  done
  rm -f "${LAUNCHER_PID_FILE}"
  echo "Launcher stopped."
}

cmd_launcher_status() {
  load_env
  if [[ -f "${LAUNCHER_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${LAUNCHER_PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Launcher: running (PID=${pid})"
      return 0
    fi
  fi
  local listeners
  listeners="$(launcher_listener_pids)"
  if [[ -n "${listeners}" ]]; then
    echo "Launcher: running (detected by port ${LAUNCHER_PORT:-18080}; PID(s): $(xargs <<<"${listeners}"))"
  else
    echo "Launcher: not running"
  fi
}

cmd_keys_gen() {
  load_env
  : "${PROVIDER_PRIVATE_KEY_PATH:?PROVIDER_PRIVATE_KEY_PATH is required in provider.env}"
  : "${PROVIDER_PUBLIC_KEY_PATH:?PROVIDER_PUBLIC_KEY_PATH is required in provider.env}"
  python3 "${SCRIPT_DIR}/generate_provider_keys.py" \
    --private-key-path "${PROVIDER_PRIVATE_KEY_PATH}" \
    --public-key-path "${PROVIDER_PUBLIC_KEY_PATH}"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd="$1"
  shift

  case "${cmd}" in
    start)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_start
      ;;
    stop)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_stop
      ;;
    restart)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_stop || true
      cmd_start
      ;;
    status)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_status
      ;;
    health)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_health
      ;;
    probe)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_probe
      ;;
    launcher-start)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_launcher_start
      ;;
    launcher-stop)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_launcher_stop
      ;;
    launcher-status)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_launcher_status
      ;;
    keys-gen)
      [[ $# -eq 0 ]] || { usage; exit 1; }
      cmd_keys_gen
      ;;
    freeze-verify)
      [[ $# -le 1 ]] || { usage; exit 1; }
      cmd_freeze_verify "$@"
      ;;
    lan-reset)
      [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 1; }
      cmd_lan_reset "$@"
      ;;
    lan-reset-frozen)
      [[ $# -ge 1 && $# -le 3 ]] || { usage; exit 1; }
      cmd_lan_reset_frozen "$@"
      ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
