#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONCE="${SCRIPT_DIR}/run_matrix_once_macos.sh"
MERGER="${SCRIPT_DIR}/make_order_robust_summary.py"
DEFAULT_CONFIG="${SCRIPT_DIR}/config.yaml"

usage() {
  cat <<EOF
Usage:
  ${0} [--config <path-to-config.yaml>]

Run both forward and reverse batches, then build order-robust median tables.
EOF
}

CONFIG_PATH="${DEFAULT_CONFIG}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 1; }
      CONFIG_PATH="$2"
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

[[ -f "${RUN_ONCE}" ]] || { echo "Missing script: ${RUN_ONCE}" >&2; exit 1; }
[[ -f "${MERGER}" ]] || { echo "Missing script: ${MERGER}" >&2; exit 1; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }

extract_batch_dir() {
  local log_file="$1"
  local batch_dir
  batch_dir="$(awk -F': ' '/^Batch summary dir: /{print $2}' "${log_file}" | tail -n 1)"
  if [[ -z "${batch_dir}" || ! -d "${batch_dir}" ]]; then
    echo "Failed to parse batch dir from ${log_file}" >&2
    exit 1
  fi
  printf '%s\n' "${batch_dir}"
}

run_one_order() {
  local order="$1"
  local log_file
  log_file="$(mktemp "/tmp/bcra_${order}_XXXX.log")"

  echo
  echo "============================================================"
  echo "Running order=${order}"
  echo "============================================================"
  bash "${RUN_ONCE}" --config "${CONFIG_PATH}" --order "${order}" | tee "${log_file}"

  local batch_dir
  batch_dir="$(extract_batch_dir "${log_file}")"
  rm -f "${log_file}"
  printf '%s\n' "${batch_dir}"
}

FORWARD_BATCH="$(run_one_order forward)"
REVERSE_BATCH="$(run_one_order reverse)"

echo
echo "============================================================"
echo "Building order-robust median summary"
echo "============================================================"
python3 "${MERGER}" --batch-a "${FORWARD_BATCH}" --batch-b "${REVERSE_BATCH}"

echo
echo "Done."
echo "forward batch : ${FORWARD_BATCH}"
echo "reverse batch : ${REVERSE_BATCH}"
