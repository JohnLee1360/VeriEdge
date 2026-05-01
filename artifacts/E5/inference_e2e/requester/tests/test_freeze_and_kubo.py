from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUESTER_DIR = ROOT / "requester"


def test_freeze_manifest_matches_repo_snapshot():
    manifest_path = ROOT / "freeze" / "exo_env_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["external_exo"]["git"]["commit"] == "ea18a625813d36069956ba742e8f519eabee05b2"
    assert manifest["external_exo"]["nix"]["nixpkgs_rev"] == "ffbc9f8cbaacfb331b6017d5a5abb21a492c9a38"


def test_verify_exo_env_local():
    script = REQUESTER_DIR / "verify_exo_env.py"
    manifest = ROOT / "freeze" / "exo_env_manifest.json"
    exo_dir = Path("/Users/jlmini_2/repo/paper/third_party/exo")
    proc = subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest), "--exo-dir", str(exo_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True


def test_kubo_script_status_with_fake_ipfs(tmp_path: Path):
    ipfs_bin = tmp_path / "ipfs"
    ipfs_bin.write_text("#!/usr/bin/env bash\nif [[ \"$1\" == \"id\" ]]; then\n  echo '{\"ID\":\"fake\"}'\n  exit 0\nfi\nexit 1\n", encoding="utf-8")
    ipfs_bin.chmod(ipfs_bin.stat().st_mode | stat.S_IEXEC)
    script = REQUESTER_DIR / "kubo_macbook.sh"
    env = dict(os.environ)
    env["IPFS_BIN"] = str(ipfs_bin)
    proc = subprocess.run(
        ["bash", str(script), "status"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0
    assert "running" in proc.stdout
