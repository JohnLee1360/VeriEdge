#!/usr/bin/env bash

THC_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THC_SCRIPTS_DIR/../../.." && pwd)"
REPO_VENV_DIR="${THC_REPO_VENV_DIR:-$REPO_ROOT/.venv}"
REPO_VENV_PYTHON="$REPO_VENV_DIR/bin/python3"
BOOTSTRAP_PYTHON_BIN="${THC_BOOTSTRAP_PYTHON_BIN:-python3}"
THC_PIP_INDEX_URL="${THC_PIP_INDEX_URL:-${BC_RA_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}}"

thc_die() {
  echo "[error] $*" >&2
  exit 1
}

thc_require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || thc_die "$name is required"
}

thc_require_file() {
  local path="$1"
  [[ -f "$path" ]] || thc_die "file not found: $path"
}

thc_require_dir() {
  local path="$1"
  [[ -d "$path" ]] || thc_die "directory not found: $path"
}

thc_require_python_bin() {
  local python_bin="$1"
  if command -v "$python_bin" >/dev/null 2>&1; then
    return
  fi
  [[ -x "$python_bin" ]] || thc_die "python binary not found: $python_bin"
}

thc_display_path() {
  local path="$1"
  case "$path" in
    "$HOME"/*)
      printf '~%s\n' "${path#$HOME}"
      ;;
    *)
      printf '%s\n' "$path"
      ;;
  esac
}

thc_python_version() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

thc_python_meets_minimum() {
  local python_bin="$1"
  local min_major="$2"
  local min_minor="$3"
  "$python_bin" - "$min_major" "$min_minor" <<'PY' >/dev/null 2>&1
import sys

min_major = int(sys.argv[1])
min_minor = int(sys.argv[2])
raise SystemExit(0 if sys.version_info >= (min_major, min_minor) else 1)
PY
}

thc_ensure_repo_venv() {
  if [[ -x "$REPO_VENV_PYTHON" ]] && thc_python_meets_minimum "$REPO_VENV_PYTHON" 3 10; then
    return
  fi

  thc_require_python_bin "$BOOTSTRAP_PYTHON_BIN"
  thc_python_meets_minimum "$BOOTSTRAP_PYTHON_BIN" 3 10 || \
    thc_die "bootstrap python must be >= 3.10, got $(thc_python_version "$BOOTSTRAP_PYTHON_BIN")"

  if [[ -x "$REPO_VENV_PYTHON" ]] && ! thc_python_meets_minimum "$REPO_VENV_PYTHON" 3 10; then
    echo "[setup] recreating repo virtualenv at $(thc_display_path "$REPO_VENV_DIR") because Python >= 3.10 is required"
    rm -rf "$REPO_VENV_DIR"
  fi

  if [[ -x "$REPO_VENV_PYTHON" ]]; then
    return
  fi
  echo "[setup] creating repo virtualenv at $(thc_display_path "$REPO_VENV_DIR")"
  "$BOOTSTRAP_PYTHON_BIN" -m venv "$REPO_VENV_DIR"
}

thc_prepare_python_bin() {
  local python_bin="$1"
  if [[ "$python_bin" == "$REPO_VENV_PYTHON" || "$python_bin" == "$REPO_VENV_DIR/bin/python" ]]; then
    thc_ensure_repo_venv
  fi
  thc_require_python_bin "$python_bin"
}

thc_check_module_import() {
  local python_bin="$1"
  local module="$2"
  "$python_bin" - "$module" <<'PY'
import importlib
import sys

module = sys.argv[1]
try:
    importlib.import_module(module)
except ModuleNotFoundError as exc:
    if exc.name == module:
        raise SystemExit(1)
    raise SystemExit(2)
except Exception:
    raise SystemExit(2)
raise SystemExit(0)
PY
}

thc_install_missing_modules() {
  local python_bin="$1"
  shift

  local pair
  local module
  local spec
  local missing_specs=()
  local repair_specs=()
  local status

  for pair in "$@"; do
    module="${pair%%::*}"
    spec="${pair#*::}"
    set +e
    thc_check_module_import "$python_bin" "$module" >/dev/null 2>&1
    status=$?
    set -e
    if [[ "$status" -eq 1 ]]; then
      missing_specs+=("$spec")
    elif [[ "$status" -eq 2 ]]; then
      repair_specs+=("$spec")
    fi
  done

  if [[ ${#missing_specs[@]} -gt 0 ]]; then
    echo "[setup] installing missing packages into $(thc_display_path "$python_bin"): ${missing_specs[*]}"
    echo "[setup] pip index: $THC_PIP_INDEX_URL"
    "$python_bin" -m pip install -i "$THC_PIP_INDEX_URL" "${missing_specs[@]}"
  fi

  if [[ ${#repair_specs[@]} -gt 0 ]]; then
    echo "[setup] reinstalling broken packages into $(thc_display_path "$python_bin"): ${repair_specs[*]}"
    echo "[setup] pip index: $THC_PIP_INDEX_URL"
    "$python_bin" -m pip install --upgrade --force-reinstall -i "$THC_PIP_INDEX_URL" "${repair_specs[@]}"
  fi
}
