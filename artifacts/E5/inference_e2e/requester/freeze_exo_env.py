#!/usr/bin/env python3
"""Capture an experiment-side snapshot of the external EXO source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(exo_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(exo_dir), *args],
        text=True,
    ).strip()


def build_manifest(exo_dir: Path) -> Dict[str, Any]:
    pyproject = tomllib.loads((exo_dir / "pyproject.toml").read_text(encoding="utf-8"))
    flake_lock = json.loads((exo_dir / "flake.lock").read_text(encoding="utf-8"))
    nixpkgs = flake_lock.get("nodes", {}).get("nixpkgs", {}).get("locked", {})
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "external_exo": {
            "source_dir": str(exo_dir),
            "git": {
                "branch": git_output(exo_dir, "rev-parse", "--abbrev-ref", "HEAD"),
                "commit": git_output(exo_dir, "rev-parse", "HEAD"),
            },
            "files": {
                "flake_lock_sha256": sha256_hex(exo_dir / "flake.lock"),
                "uv_lock_sha256": sha256_hex(exo_dir / "uv.lock"),
                "python_version_sha256": sha256_hex(exo_dir / ".python-version"),
                "pyproject_sha256": sha256_hex(exo_dir / "pyproject.toml"),
            },
            "python": {
                "python_version_file": (exo_dir / ".python-version").read_text(encoding="utf-8").strip(),
                "requires_python": pyproject["project"]["requires-python"],
                "uv_required_version": pyproject.get("tool", {}).get("uv", {}).get("required-version"),
            },
            "project": {
                "package_version": pyproject["project"]["version"],
            },
            "nix": {
                "nixpkgs_rev": nixpkgs.get("rev"),
                "nixpkgs_last_modified": nixpkgs.get("lastModified"),
                "nixpkgs_nar_hash": nixpkgs.get("narHash"),
            },
        },
        "applies_to": {
            "requester": {"ip": "192.168.31.189"},
            "providers": [
                {"node_id": "jlmini_1", "ip": "192.168.31.52"},
                {"node_id": "jlmini_2", "ip": "192.168.31.159"},
                {"node_id": "jlmini_3", "ip": "192.168.31.83"},
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze external EXO/Nix/Python versions for the experiment")
    parser.add_argument("--exo-dir", default="~/repo/paper/third_party/exo")
    parser.add_argument(
        "--output",
        default="artifacts/E5/inference_e2e/freeze/exo_env_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exo_dir = Path(args.exo_dir).expanduser().resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(exo_dir)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
