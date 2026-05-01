#!/usr/bin/env bash
set -euo pipefail

ANCHOR="com.apple/250.bcra_netem"
TMP_RULES="/tmp/bcra_netem.pf"

usage() {
  cat <<EOF
Usage:
  sudo $0 apply --targets ip1,ip2,ip3 [--ports 52415,18080,8080,18081] [--rtt-ms 50] [--bw-mbps 100] [--loss-pct 1]
  sudo $0 apply --target-spec ip:rtt:bw:loss,ip:rtt:bw:loss [--ports 52415,18080,8080,18081]
  sudo $0 status
  sudo $0 reset

Notes:
  - RTT is approximated as 2 x (rtt-ms/2) via bidirectional dummynet pipes.
  - --ports accepts a comma-separated list of service ports on either requester or provider.
  - --target-spec sets per-target shaping and overrides --targets/--rtt-ms/--bw-mbps/--loss-pct.
EOF
}

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "This command must run as root (use sudo)." >&2
    exit 1
  fi
}

build_pf_rules() {
  local targets_csv="$1"
  local ports_csv="$2"
  cat >"${TMP_RULES}" <<EOF
table <bcra_targets> persist { ${targets_csv} }
dummynet out quick proto tcp from any to <bcra_targets> port { ${ports_csv} } pipe 1
dummynet out quick proto tcp from any port { ${ports_csv} } to <bcra_targets> pipe 1
dummynet in quick proto tcp from <bcra_targets> port { ${ports_csv} } to any pipe 2
dummynet in quick proto tcp from <bcra_targets> to any port { ${ports_csv} } pipe 2
EOF
}

normalize_ports_csv() {
  local raw="$1"
  local ports=()
  local cleaned=()
  IFS=',' read -r -a ports <<<"${raw}"
  if [[ "${#ports[@]}" -eq 0 ]]; then
    echo "Empty --ports" >&2
    exit 1
  fi
  for item in "${ports[@]}"; do
    local port="${item//[[:space:]]/}"
    validate_integer "${port}" "port"
    cleaned+=("${port}")
  done
  local joined=""
  local first=1
  for port in "${cleaned[@]}"; do
    if [[ "${first}" -eq 1 ]]; then
      joined="${port}"
      first=0
    else
      joined="${joined},${port}"
    fi
  done
  printf '%s\n' "${joined}"
}

validate_number() {
  local value="$1"
  local name="$2"
  if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid ${name}: ${value}" >&2
    exit 1
  fi
}

validate_integer() {
  local value="$1"
  local name="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "Invalid ${name}: ${value}" >&2
    exit 1
  fi
}

apply_per_target_spec() {
  local target_spec="$1"
  local ports_csv="$2"
  local entries=()
  local idx=0

  IFS=',' read -r -a entries <<<"${target_spec}"
  if [[ "${#entries[@]}" -eq 0 ]]; then
    echo "--target-spec is empty" >&2
    exit 1
  fi

  : >"${TMP_RULES}"
  dnctl -q flush

  for raw_entry in "${entries[@]}"; do
    local entry="${raw_entry//[[:space:]]/}"
    local ip=""
    local rtt_ms=""
    local bw_mbps=""
    local loss_pct=""
    local extra=""
    IFS=':' read -r ip rtt_ms bw_mbps loss_pct extra <<<"${entry}"

    if [[ -n "${extra}" || -z "${ip}" || -z "${rtt_ms}" || -z "${bw_mbps}" || -z "${loss_pct}" ]]; then
      echo "Invalid --target-spec entry: ${raw_entry}" >&2
      echo "Expected format: ip:rtt_ms:bw_mbps:loss_pct" >&2
      exit 1
    fi

    validate_integer "${rtt_ms}" "rtt_ms(${ip})"
    validate_number "${bw_mbps}" "bw_mbps(${ip})"
    validate_number "${loss_pct}" "loss_pct(${ip})"

    local one_way_ms=$((rtt_ms / 2))
    local plr
    plr=$(awk -v x="${loss_pct}" 'BEGIN { printf "%.6f", x / 100.0 }')

    idx=$((idx + 1))
    local pipe_out=$((idx * 2 - 1))
    local pipe_in=$((idx * 2))

    dnctl pipe "${pipe_out}" config delay "${one_way_ms}"ms bw "${bw_mbps}"Mbit/s plr "${plr}"
    dnctl pipe "${pipe_in}" config delay "${one_way_ms}"ms bw "${bw_mbps}"Mbit/s plr "${plr}"

    cat >>"${TMP_RULES}" <<EOF
dummynet out quick proto tcp from any to ${ip} port { ${ports_csv} } pipe ${pipe_out}
dummynet out quick proto tcp from any port { ${ports_csv} } to ${ip} pipe ${pipe_out}
dummynet in quick proto tcp from ${ip} port { ${ports_csv} } to any pipe ${pipe_in}
dummynet in quick proto tcp from ${ip} to any port { ${ports_csv} } pipe ${pipe_in}
EOF

    echo "  - ${ip}: RTT=${rtt_ms}ms BW=${bw_mbps}Mbps LOSS=${loss_pct}% pipe_out=${pipe_out} pipe_in=${pipe_in}"
  done

  pfctl -E >/dev/null 2>&1 || true
  pfctl -a "${ANCHOR}" -f "${TMP_RULES}"
}

cmd_apply() {
  need_root
  local targets=""
  local target_spec=""
  local ports="52415"
  local rtt_ms="50"
  local bw_mbps="100"
  local loss_pct="1"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --targets)
        targets="$2"
        shift 2
        ;;
      --target-spec)
        target_spec="$2"
        shift 2
        ;;
      --ports)
        ports="$2"
        shift 2
        ;;
      --rtt-ms)
        rtt_ms="$2"
        shift 2
        ;;
      --bw-mbps)
        bw_mbps="$2"
        shift 2
        ;;
      --loss-pct)
        loss_pct="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  local ports_csv
  ports_csv="$(normalize_ports_csv "${ports}")"

  if [[ -n "${target_spec}" ]]; then
    echo "Applying WAN emulation (per-target spec) on ports=${ports_csv}"
    apply_per_target_spec "${target_spec}" "${ports_csv}"
    echo "Applied. Use '$0 status' to inspect, '$0 reset' to clear."
    return
  fi

  if [[ -z "${targets}" ]]; then
    echo "Either --targets or --target-spec is required" >&2
    usage
    exit 1
  fi

  validate_integer "${rtt_ms}" "rtt_ms"
  validate_number "${bw_mbps}" "bw_mbps"
  validate_number "${loss_pct}" "loss_pct"
  local one_way_ms=$((rtt_ms / 2))
  local plr
  plr=$(awk -v x="${loss_pct}" 'BEGIN { printf "%.6f", x / 100.0 }')

  echo "Applying WAN emulation: RTT=${rtt_ms}ms BW=${bw_mbps}Mbps LOSS=${loss_pct}% PORTS=${ports_csv}"
  echo "Targets: ${targets}"

  build_pf_rules "${targets}" "${ports_csv}"

  dnctl -q flush
  dnctl pipe 1 config delay "${one_way_ms}"ms bw "${bw_mbps}"Mbit/s plr "${plr}"
  dnctl pipe 2 config delay "${one_way_ms}"ms bw "${bw_mbps}"Mbit/s plr "${plr}"

  pfctl -E >/dev/null 2>&1 || true
  pfctl -a "${ANCHOR}" -f "${TMP_RULES}"

  echo "Applied. Use '$0 status' to inspect, '$0 reset' to clear."
}

cmd_status() {
  need_root
  echo "== PF anchor rules (${ANCHOR}) =="
  pfctl -a "${ANCHOR}" -s rules || true
  echo
  echo "== dummynet rules =="
  dnctl list || true
}

cmd_reset() {
  need_root
  echo "Resetting WAN emulation rules..."
  pfctl -a "${ANCHOR}" -F all || true
  dnctl -q flush
  rm -f "${TMP_RULES}"
  echo "Reset complete."
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd="$1"
  shift

  case "${cmd}" in
    apply)
      cmd_apply "$@"
      ;;
    status)
      cmd_status
      ;;
    reset)
      cmd_reset
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
