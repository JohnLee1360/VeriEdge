#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROVIDER_NODE_ID="provider-1"
EXO_ENDPOINT="http://127.0.0.1:52415"
MODEL_ID="mlx-community/Qwen3-0.6B-8bit"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
EXO_DIR="~/repo/paper/third_party/exo"
EXO_START_CMD=""
EXO_HOME="${EXO_HOME:-~/.cache/exo-bcra}"
EXO_DEBUG="${EXO_DEBUG:-9}"
HEALTH_TIMEOUT_SEC="5"
PIP_INDEX_URL="${BC_RA_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
NPM_REGISTRY="${BC_RA_NPM_REGISTRY:-https://registry.npmmirror.com}"
RUSTUP_DIST_SERVER="${BC_RA_RUSTUP_DIST_SERVER:-https://rsproxy.cn}"
RUSTUP_UPDATE_ROOT="${BC_RA_RUSTUP_UPDATE_ROOT:-https://rsproxy.cn/rustup}"
HOMEBREW_BOTTLE_DOMAIN="${BC_RA_HOMEBREW_BOTTLE_DOMAIN:-https://mirrors.ustc.edu.cn/homebrew-bottles}"
HOMEBREW_BREW_GIT_REMOTE="${BC_RA_HOMEBREW_BREW_GIT_REMOTE:-https://mirrors.ustc.edu.cn/brew.git}"
HOMEBREW_CORE_GIT_REMOTE="${BC_RA_HOMEBREW_CORE_GIT_REMOTE:-https://mirrors.ustc.edu.cn/homebrew-core.git}"

SKIP_BREW="0"
SKIP_RUST="0"
SKIP_XCODE_CHECK="0"
SKIP_DASHBOARD="0"
SKIP_START="0"
SKIP_PROBE="0"
FORCE_WRITE_ENV="0"

usage() {
  cat <<EOF
One-click provider bootstrap for nix-based EXO environments.

Usage:
  $0 [options]

Options:
  --node-id <id>              Provider node id (default: provider-1)
  --endpoint <url>            Local provider endpoint (default: http://127.0.0.1:52415)
  --model-id <id>             Model id for probe/config
  --hf-endpoint <url>         Hugging Face endpoint (default: https://hf-mirror.com)
  --exo-dir <path>            Existing exo repo path (default: ~/repo/paper/third_party/exo)
  --exo-start-cmd <cmd>       Override EXO_START_CMD in provider.env
  --health-timeout-sec <n>    Health/probe timeout seconds (default: 5)

  --skip-brew                 Do not auto-install brew/homebrew packages
  --skip-rust                 Do not auto-install rustup/nightly toolchain
  --skip-xcode-check          Skip Xcode CLI Tools check on macOS
  --skip-dashboard-build      Skip 'npm install && npm run build' under exo/dashboard
  --skip-start                Do not start provider process at the end
  --skip-probe                Skip tiny inference probe at the end
  --force                     Overwrite existing provider.env without prompt
  -h, --help                  Show this help

What this script does:
  1) Check exo repo exists and is nix-based (flake.nix + flake.lock)
  2) Check/install host-side dependencies from exo README (brew packages + rust nightly)
  3) Build exo dashboard
  4) Install provider Python requirements
  5) Generate artifacts/E5/inference_e2e/provider/provider.env with EXO_START_CMD='nix run .'
  6) Start provider and run health/probe checks (unless skipped)
EOF
}

log() {
  printf '[bootstrap] %s\n' "$*"
}

die() {
  printf '[bootstrap][error] %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  local c="$1"
  command -v "$c" >/dev/null 2>&1 || die "Missing command: $c"
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

setup_hf_endpoint() {
  local zshrc="${HOME}/.zshrc"
  local export_line="export HF_ENDPOINT=${HF_ENDPOINT}"

  export HF_ENDPOINT
  if ! grep -qxF "$export_line" "$zshrc" 2>/dev/null; then
    printf '\n%s\n' "$export_line" >> "$zshrc"
  fi

  log "HF_ENDPOINT=${HF_ENDPOINT}"
}

ensure_macos() {
  local os
  os="$(uname -s)"
  [[ "$os" == "Darwin" ]] || die "This bootstrap script targets macOS (Darwin). Current OS: ${os}"
}

load_homebrew_shellenv_if_present() {
  if [[ -x "/opt/homebrew/bin/brew" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x "/usr/local/bin/brew" ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

load_cargo_env_if_present() {
  if [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env"
  fi
}

ensure_xcode_cli_tools() {
  [[ "$SKIP_XCODE_CHECK" == "1" ]] && {
    log "Skip Xcode CLI Tools check (--skip-xcode-check)"
    return
  }

  if xcode-select -p >/dev/null 2>&1; then
    log "Found Xcode CLI Tools"
    return
  fi

  log "Xcode CLI Tools not found. Triggering installer..."
  xcode-select --install >/dev/null 2>&1 || true
  die "Please finish Xcode Command Line Tools installation, then re-run this script."
}

install_homebrew_if_missing() {
  if command -v brew >/dev/null 2>&1; then
    log "Found: brew"
    return
  fi

  [[ "$SKIP_BREW" == "1" ]] && die "brew is missing and --skip-brew is set"
  need_cmd curl
  log "Installing Homebrew..."
  export HOMEBREW_BOTTLE_DOMAIN
  export HOMEBREW_BREW_GIT_REMOTE
  export HOMEBREW_CORE_GIT_REMOTE
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  load_homebrew_shellenv_if_present
  need_cmd brew
}

install_with_brew_if_missing() {
  local cmd="$1"
  local pkg="$2"

  if command -v "$cmd" >/dev/null 2>&1; then
    log "Found: $cmd"
    return
  fi

  [[ "$SKIP_BREW" == "1" ]] && die "Missing $cmd and --skip-brew is set"
  install_homebrew_if_missing
  log "Installing via brew: $pkg"
  export HOMEBREW_BOTTLE_DOMAIN
  export HOMEBREW_BREW_GIT_REMOTE
  export HOMEBREW_CORE_GIT_REMOTE
  brew install "$pkg"
}

ensure_rust_nightly() {
  [[ "$SKIP_RUST" == "1" ]] && {
    log "Skip rustup/nightly install (--skip-rust)"
    return
  }

  if ! command -v rustup >/dev/null 2>&1; then
    need_cmd curl
    log "Installing rustup..."
    export RUSTUP_DIST_SERVER
    export RUSTUP_UPDATE_ROOT
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    load_cargo_env_if_present
  fi

  need_cmd rustup
  log "Installing/updating Rust nightly toolchain..."
  export RUSTUP_DIST_SERVER
  export RUSTUP_UPDATE_ROOT
  rustup toolchain install nightly
  load_cargo_env_if_present
  need_cmd cargo
  need_cmd rustc
}

resolve_nix_bin() {
  local nix_bin
  nix_bin="$(command -v nix || true)"
  if [[ -z "${nix_bin}" && -x "/nix/var/nix/profiles/default/bin/nix" ]]; then
    nix_bin="/nix/var/nix/profiles/default/bin/nix"
  fi
  [[ -n "${nix_bin}" ]] || die "nix not found; this provider bootstrap only supports 'nix run .'"
  printf '%s\n' "${nix_bin}"
}

escape_for_env_double_quotes() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '%s' "$s"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --node-id)
        PROVIDER_NODE_ID="$2"
        shift 2
        ;;
      --endpoint)
        EXO_ENDPOINT="$2"
        shift 2
        ;;
      --model-id)
        MODEL_ID="$2"
        shift 2
        ;;
      --hf-endpoint)
        HF_ENDPOINT="$2"
        shift 2
        ;;
      --exo-dir)
        EXO_DIR="$2"
        shift 2
        ;;
      --exo-start-cmd)
        EXO_START_CMD="$2"
        shift 2
        ;;
      --health-timeout-sec)
        HEALTH_TIMEOUT_SEC="$2"
        shift 2
        ;;
      --skip-brew)
        SKIP_BREW="1"
        shift
        ;;
      --skip-rust)
        SKIP_RUST="1"
        shift
        ;;
      --skip-xcode-check)
        SKIP_XCODE_CHECK="1"
        shift
        ;;
      --skip-dashboard-build)
        SKIP_DASHBOARD="1"
        shift
        ;;
      --skip-start)
        SKIP_START="1"
        shift
        ;;
      --skip-probe)
        SKIP_PROBE="1"
        shift
        ;;
      --force)
        FORCE_WRITE_ENV="1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

default_start_cmd() {
  local exo_dir_abs
  local nix_bin
  exo_dir_abs="$(expand_home_path "${EXO_DIR}")"
  nix_bin="$(resolve_nix_bin)"
  printf 'cd %q && HF_ENDPOINT=%q EXO_HOME=%q DEBUG=%q %q run .' \
    "${exo_dir_abs}" "${HF_ENDPOINT}" "${EXO_HOME}" "${EXO_DEBUG}" "${nix_bin}"
}

write_provider_env() {
  local env_file="${SCRIPT_DIR}/provider.env"

  if [[ -f "$env_file" ]]; then
    if [[ "$FORCE_WRITE_ENV" == "1" ]]; then
      cp "$env_file" "${env_file}.bak.$(date +%Y%m%d_%H%M%S)"
    else
      die "${env_file} already exists. Re-run with --force to overwrite."
    fi
  fi

  local resolved_start_cmd="$EXO_START_CMD"
  if [[ -z "$resolved_start_cmd" ]]; then
    resolved_start_cmd="$(default_start_cmd)"
  fi

  local safe_start_cmd
  safe_start_cmd="$(escape_for_env_double_quotes "$resolved_start_cmd")"

  cat > "$env_file" <<EOF
# Generated by bootstrap_provider.sh
PROVIDER_NODE_ID=${PROVIDER_NODE_ID}
EXO_ENDPOINT=${EXO_ENDPOINT}
MODEL_ID=${MODEL_ID}
HF_ENDPOINT=${HF_ENDPOINT}
EXO_REPO_DIR=${EXO_DIR}
EXO_HOME=${EXO_HOME}
EXO_DEBUG=${EXO_DEBUG}
EXO_START_CMD="${safe_start_cmd}"
HEALTH_TIMEOUT_SEC=${HEALTH_TIMEOUT_SEC}
EOF

  log "Generated ${env_file}"
}

build_dashboard() {
  [[ "$SKIP_DASHBOARD" == "1" ]] && {
    log "Skip dashboard build (--skip-dashboard-build)"
    return
  }

  local exo_dir_abs="${EXO_DIR}"
  local dash_dir
  exo_dir_abs="$(expand_home_path "${exo_dir_abs}")"
  dash_dir="${exo_dir_abs}/dashboard"
  [[ -d "$dash_dir" ]] || die "Missing dashboard dir: ${dash_dir}"

  need_cmd npm
  log "Building exo dashboard: ${dash_dir}"
  (
    cd "$dash_dir"
    npm install --registry "$NPM_REGISTRY"
    npm run build
  )
}

install_provider_python_reqs() {
  need_cmd python3
  local req="${SCRIPT_DIR}/requirements.txt"
  log "Installing provider Python deps"
  log "Using pip index: ${PIP_INDEX_URL}"
  python3 -m pip install -i "$PIP_INDEX_URL" -r "$req"
}

run_post_checks() {
  local manage="${SCRIPT_DIR}/manage_provider.sh"

  [[ -x "$manage" ]] || chmod +x "$manage"

  if [[ "$SKIP_START" == "1" ]]; then
    log "Skip provider start (--skip-start)"
    "$manage" status || true
    return
  fi

  "$manage" start
  "$manage" status
  "$manage" health

  if [[ "$SKIP_PROBE" != "1" ]]; then
    "$manage" probe
  else
    log "Skip tiny inference probe (--skip-probe)"
  fi
}

main() {
  parse_args "$@"

  ensure_macos
  setup_hf_endpoint
  ensure_xcode_cli_tools

  EXO_DIR="$(expand_home_path "${EXO_DIR}")"
  [[ -d "${EXO_DIR}" ]] || die "EXO_DIR not found: ${EXO_DIR}. Please prepare the EXO repo first."
  [[ -f "${EXO_DIR}/flake.nix" ]] || die "Missing flake.nix under ${EXO_DIR}"
  [[ -f "${EXO_DIR}/flake.lock" ]] || die "Missing flake.lock under ${EXO_DIR}"
  resolve_nix_bin >/dev/null

  if [[ "$SKIP_BREW" != "1" ]]; then
    install_homebrew_if_missing
    load_homebrew_shellenv_if_present
    install_with_brew_if_missing node node
    install_with_brew_if_missing macmon macmon
  else
    log "Skip brew package install (--skip-brew)"
  fi

  ensure_rust_nightly
  load_homebrew_shellenv_if_present
  load_cargo_env_if_present

  need_cmd python3
  need_cmd node
  need_cmd npm

  build_dashboard
  install_provider_python_reqs
  write_provider_env
  run_post_checks

  log "Done. Provider bootstrap completed."
  log "Default EXO start path is now nix-based: 'nix run .'"
  log "If needed, inspect logs: ${SCRIPT_DIR}/runtime/exo.log"
}

main "$@"
