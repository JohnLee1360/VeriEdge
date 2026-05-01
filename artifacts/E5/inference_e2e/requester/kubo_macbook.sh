#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
PID_FILE="${RUNTIME_DIR}/kubo.pid"
LOG_FILE="${RUNTIME_DIR}/kubo.log"
IPFS_BIN="${IPFS_BIN:-ipfs}"
GATEWAY_URL="${KUBO_GATEWAY_URL:-http://127.0.0.1:8080}"

mkdir -p "${RUNTIME_DIR}"

usage() {
  cat <<EOF
用法:
  ${0} start
  ${0} status
  ${0} id
  ${0} check-gateway
  ${0} stop
EOF
}

require_ipfs() {
  command -v "${IPFS_BIN}" >/dev/null 2>&1 || {
    echo "找不到 ipfs 命令: ${IPFS_BIN}" >&2
    exit 1
  }
}

daemon_running() {
  "${IPFS_BIN}" id >/dev/null 2>&1
}

case "${1:-}" in
  start)
    require_ipfs
    if daemon_running; then
      echo "Kubo 已在运行"
      exit 0
    fi
    nohup "${IPFS_BIN}" daemon >"${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    for _ in {1..30}; do
      if daemon_running; then
        echo "Kubo 已启动"
        exit 0
      fi
      sleep 1
    done
    echo "Kubo 启动失败，请查看 ${LOG_FILE}" >&2
    exit 1
    ;;
  status)
    require_ipfs
    if daemon_running; then
      echo "running"
    else
      echo "stopped"
      exit 1
    fi
    ;;
  id)
    require_ipfs
    "${IPFS_BIN}" id
    ;;
  check-gateway)
    python3 - "${GATEWAY_URL}" <<'PY'
import sys
import urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        print(f"{url} {resp.status}")
except Exception as exc:  # noqa: BLE001
    print(f"{url} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
    ;;
  stop)
    if [[ -f "${PID_FILE}" ]]; then
      pid="$(cat "${PID_FILE}")"
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}"
        rm -f "${PID_FILE}"
        echo "Kubo 已停止"
        exit 0
      fi
      rm -f "${PID_FILE}"
    fi
    echo "未找到由本脚本启动的 Kubo 进程" >&2
    exit 1
    ;;
  -h|--help|"")
    usage
    [[ "${1:-}" == "" ]] && exit 1 || exit 0
    ;;
  *)
    echo "未知命令: ${1}" >&2
    usage
    exit 1
    ;;
esac
