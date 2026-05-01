#!/usr/bin/env python3
"""Generate provider RSA keypair for task-key decryption."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from crypto_utils import generate_rsa_keypair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate provider RSA keypair")
    parser.add_argument("--private-key-path", required=True)
    parser.add_argument("--public-key-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    private_key_path = Path(args.private_key_path)
    public_key_path = Path(args.public_key_path)
    generate_rsa_keypair(private_key_path, public_key_path)
    print(f"private key: {private_key_path}")
    print(f"public key : {public_key_path}")


if __name__ == "__main__":
    main()
