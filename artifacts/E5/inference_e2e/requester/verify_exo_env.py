#!/usr/bin/env python3
"""Verify that local or remote EXO trees match the experiment freeze manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(exo_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(exo_dir), *args],
        text=True,
    ).strip()


def collect_local(exo_dir: Path) -> Dict[str, Any]:
    pyproject = tomllib.loads((exo_dir / "pyproject.toml").read_text(encoding="utf-8"))
    flake_lock = json.loads((exo_dir / "flake.lock").read_text(encoding="utf-8"))
    nixpkgs = flake_lock.get("nodes", {}).get("nixpkgs", {}).get("locked", {})
    return {
        "branch": git_output(exo_dir, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git_output(exo_dir, "rev-parse", "HEAD"),
        "flake_lock_sha256": sha256_hex(exo_dir / "flake.lock"),
        "uv_lock_sha256": sha256_hex(exo_dir / "uv.lock"),
        "python_version_sha256": sha256_hex(exo_dir / ".python-version"),
        "pyproject_sha256": sha256_hex(exo_dir / "pyproject.toml"),
        "python_version_file": (exo_dir / ".python-version").read_text(encoding="utf-8").strip(),
        "requires_python": pyproject["project"]["requires-python"],
        "uv_required_version": pyproject.get("tool", {}).get("uv", {}).get("required-version"),
        "package_version": pyproject["project"]["version"],
        "nixpkgs_rev": nixpkgs.get("rev"),
    }


def collect_remote(ssh_target: str, exo_dir: str) -> Dict[str, Any]:
    remote_script = f"""
set -euo pipefail
cd {json.dumps(exo_dir)}
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import subprocess
try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None
root = Path('.').resolve()
def sh(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
pyproject = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8')) if tomllib is not None else {{}}
flake_lock = json.loads((root / 'flake.lock').read_text(encoding='utf-8'))
nixpkgs = flake_lock.get('nodes', {{}}).get('nixpkgs', {{}}).get('locked', {{}})
out = {{
    'branch': subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip(),
    'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    'flake_lock_sha256': sh(root / 'flake.lock'),
    'uv_lock_sha256': sh(root / 'uv.lock'),
    'python_version_sha256': sh(root / '.python-version'),
    'pyproject_sha256': sh(root / 'pyproject.toml'),
    'python_version_file': (root / '.python-version').read_text(encoding='utf-8').strip(),
    'requires_python': pyproject.get('project', {{}}).get('requires-python'),
    'uv_required_version': pyproject.get('tool', {{}}).get('uv', {{}}).get('required-version'),
    'package_version': pyproject.get('project', {{}}).get('version'),
    'nixpkgs_rev': nixpkgs.get('rev'),
}}
print(json.dumps(out))
PY
"""
    output = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh_target, remote_script],
        text=True,
        timeout=30,
    ).strip()
    return json.loads(output)


def comparison_mappings(expected: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    required = {
        "commit": expected["git"]["commit"],
        "flake_lock_sha256": expected["files"]["flake_lock_sha256"],
    }
    diagnostics = {
        "branch": expected["git"]["branch"],
        "uv_lock_sha256": expected["files"]["uv_lock_sha256"],
        "python_version_sha256": expected["files"]["python_version_sha256"],
        "pyproject_sha256": expected["files"]["pyproject_sha256"],
        "python_version_file": expected["python"]["python_version_file"],
        "requires_python": expected["python"]["requires_python"],
        "uv_required_version": expected["python"]["uv_required_version"],
        "package_version": expected["project"]["package_version"],
        "nixpkgs_rev": expected["nix"]["nixpkgs_rev"],
    }
    return required, diagnostics


def compare(mapping: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []
    for key, expected_value in mapping.items():
        if actual.get(key) != expected_value:
            mismatches.append(f"{key}: expected={expected_value!r} actual={actual.get(key)!r}")
    return mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify EXO freeze manifest against local/remote trees")
    parser.add_argument("--manifest", default="artifacts/E5/inference_e2e/freeze/exo_env_manifest.json")
    parser.add_argument("--exo-dir", default="~/repo/paper/third_party/exo")
    parser.add_argument("--ssh-target", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = manifest["external_exo"]
    required_mapping, diagnostic_mapping = comparison_mappings(expected)
    report: Dict[str, Any] = {
        "required_checks": sorted(required_mapping.keys()),
        "diagnostic_checks": sorted(diagnostic_mapping.keys()),
        "targets": [],
    }

    local_actual = collect_local(Path(args.exo_dir).expanduser().resolve())
    local_required_mismatches = compare(required_mapping, local_actual)
    local_diagnostic_mismatches = compare(diagnostic_mapping, local_actual)
    report["targets"].append(
        {
            "target": "local",
            "ok": local_required_mismatches == [],
            "required_mismatches": local_required_mismatches,
            "diagnostic_mismatches": local_diagnostic_mismatches,
            "actual": local_actual,
        }
    )
    for ssh_target in args.ssh_target:
        actual = collect_remote(ssh_target, str(Path(args.exo_dir).expanduser()))
        required_mismatches = compare(required_mapping, actual)
        diagnostic_mismatches = compare(diagnostic_mapping, actual)
        report["targets"].append(
            {
                "target": ssh_target,
                "ok": required_mismatches == [],
                "required_mismatches": required_mismatches,
                "diagnostic_mismatches": diagnostic_mismatches,
                "actual": actual,
            }
        )

    report["ok"] = all(target["ok"] for target in report["targets"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
