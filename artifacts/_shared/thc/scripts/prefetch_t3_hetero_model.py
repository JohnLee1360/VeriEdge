#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


def _default_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").resolve()


def _repo_cache_dir(snapshot_path: Path) -> Path:
    parts = snapshot_path.parts
    if "snapshots" not in parts:
        raise ValueError(f"unexpected snapshot path layout: {snapshot_path}")
    idx = parts.index("snapshots")
    return Path(*parts[:idx])


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def _tar_tree(src: Path, tar_output: Path) -> None:
    tar_output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_output, "w:gz") as tf:
        tf.add(src, arcname=src.name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch THC heterogeneous Torch model via domestic HF mirror and optionally package cache for Linux transfer"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hf-endpoint", default=os.environ.get("BC_RA_HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--cache-dir", default=str(_default_cache_dir()))
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--tar-output", default="")
    parser.add_argument("--local-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    os.environ["HF_ENDPOINT"] = str(args.hf_endpoint).strip()
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=str(args.model_id),
            cache_dir=str(cache_dir),
            resume_download=True,
        )
    ).resolve()
    repo_cache_dir = _repo_cache_dir(snapshot_path)

    export_dir = Path(args.export_dir).expanduser().resolve() if args.export_dir else None
    if export_dir is not None:
        _copy_tree(repo_cache_dir, export_dir)

    if args.local_dir:
        local_dir = Path(args.local_dir).expanduser().resolve()
        _copy_tree(snapshot_path, local_dir)

    tar_output = Path(args.tar_output).expanduser().resolve() if args.tar_output else None
    if tar_output is not None:
        source_dir = export_dir if export_dir is not None else repo_cache_dir
        _tar_tree(source_dir, tar_output)

    payload = {
        "hf_endpoint": os.environ["HF_ENDPOINT"],
        "model_id": str(args.model_id),
        "cache_dir": str(cache_dir),
        "repo_cache_dir": str(repo_cache_dir),
        "snapshot_path": str(snapshot_path),
        "export_dir": str(export_dir) if export_dir is not None else "",
        "local_dir": str(Path(args.local_dir).expanduser().resolve()) if args.local_dir else "",
        "tar_output": str(tar_output) if tar_output is not None else "",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
