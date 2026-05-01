#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path


DEFAULT_SOCKET = "/tmp/bcra_netem_root.sock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client for macOS WAN root helper")
    parser.add_argument("--socket-path", default=DEFAULT_SOCKET)
    sub = parser.add_subparsers(dest="cmd", required=True)

    apply_cmd = sub.add_parser("apply")
    apply_cmd.add_argument("--ports", required=True)
    apply_cmd.add_argument("--target-spec", required=True)

    sub.add_parser("reset")
    sub.add_parser("status")
    sub.add_parser("ping")
    return parser.parse_args()


def send(socket_path: str, payload: dict[str, object]) -> dict[str, object]:
    sock_path = Path(socket_path)
    if not sock_path.exists():
        raise SystemExit(f"Helper socket not found: {socket_path}")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(str(sock_path))
        client.sendall(json.dumps(payload).encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            data = client.recv(65536)
            if not data:
                break
            chunks.append(data)
    finally:
        client.close()
    if not chunks:
        raise SystemExit("Empty response from helper")
    response = json.loads(b"".join(chunks).decode("utf-8"))
    return response


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {"command": args.cmd}
    if args.cmd == "apply":
        payload["options"] = {"ports": args.ports, "target_spec": args.target_spec}
    response = send(args.socket_path, payload)
    print(json.dumps(response, indent=2))
    if not bool(response.get("ok")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
