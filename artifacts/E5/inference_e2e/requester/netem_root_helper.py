#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pwd
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SOCKET = "/tmp/bcra_netem_root.sock"
DEFAULT_PID = "/tmp/bcra_netem_root.pid"
DEFAULT_LOG = "/tmp/bcra_netem_root.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Root helper for macOS WAN netem control")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--socket-path", default=DEFAULT_SOCKET)
        p.add_argument("--pid-path", default=DEFAULT_PID)
        p.add_argument("--log-path", default=DEFAULT_LOG)
        p.add_argument(
            "--netem-script",
            default=str(Path(__file__).resolve().with_name("netem_macos.sh")),
        )
        p.add_argument("--allow-user", default=os.environ.get("SUDO_USER", "") or os.environ.get("USER", ""))

    start = sub.add_parser("start", help="Start detached root helper")
    add_common(start)

    serve = sub.add_parser("serve", help="Run helper in foreground")
    add_common(serve)

    stop = sub.add_parser("stop", help="Stop detached root helper")
    add_common(stop)

    status = sub.add_parser("status", help="Check detached root helper")
    add_common(status)
    return parser.parse_args()


def read_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ensure_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit("This helper must be managed as root.")


def chown_for_user(path: Path, user: str) -> None:
    if not user:
        return
    try:
        pw = pwd.getpwnam(user)
    except KeyError as exc:
        raise SystemExit(f"Unknown allow-user: {user}") from exc
    os.chown(path, pw.pw_uid, pw.pw_gid)


def start_detached(args: argparse.Namespace) -> None:
    ensure_root()
    pid_path = Path(args.pid_path)
    pid = read_pid(pid_path)
    if pid_running(pid):
        print(json.dumps({"ok": True, "status": "already_running", "pid": pid}))
        return

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as logf:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "serve",
                "--socket-path",
                args.socket_path,
                "--pid-path",
                args.pid_path,
                "--log-path",
                args.log_path,
                "--netem-script",
                args.netem_script,
                "--allow-user",
                args.allow_user,
            ],
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    for _ in range(50):
        if Path(args.socket_path).exists():
            break
        time.sleep(0.1)
    print(json.dumps({"ok": True, "status": "started", "pid": proc.pid, "socket_path": args.socket_path}))


def stop_detached(args: argparse.Namespace) -> None:
    ensure_root()
    pid_path = Path(args.pid_path)
    pid = read_pid(pid_path)
    if not pid_running(pid):
        print(json.dumps({"ok": True, "status": "not_running"}))
        return
    assert pid is not None
    os.kill(pid, 15)
    for _ in range(50):
        if not pid_running(pid):
            break
        time.sleep(0.1)
    if pid_running(pid):
        os.kill(pid, 9)
    print(json.dumps({"ok": True, "status": "stopped", "pid": pid}))


def helper_status(args: argparse.Namespace) -> None:
    pid = read_pid(Path(args.pid_path))
    payload = {
        "ok": True,
        "running": pid_running(pid),
        "pid": pid,
        "socket_exists": Path(args.socket_path).exists(),
        "socket_path": args.socket_path,
    }
    print(json.dumps(payload))


def run_netem_command(netem_script: str, command: str, options: dict[str, Any]) -> dict[str, Any]:
    argv = [netem_script, command]
    if command == "apply":
        ports = str(options.get("ports", "")).strip()
        target_spec = str(options.get("target_spec", "")).strip()
        if not ports or not target_spec:
            raise ValueError("apply requires ports and target_spec")
        argv.extend(["--ports", ports, "--target-spec", target_spec])
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "argv": argv,
    }


def serve(args: argparse.Namespace) -> None:
    ensure_root()
    socket_path = Path(args.socket_path)
    pid_path = Path(args.pid_path)
    log_path = Path(args.log_path)
    netem_script = str(Path(args.netem_script).resolve())
    allow_user = args.allow_user

    if socket_path.exists():
        socket_path.unlink()
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    if allow_user:
        chown_for_user(pid_path, allow_user)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
    if allow_user:
        chown_for_user(socket_path, allow_user)
    server.listen(8)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    raw = conn.recv(65536)
                    if not raw:
                        continue
                    payload = json.loads(raw.decode("utf-8"))
                    command = str(payload.get("command", "")).strip()
                    options = payload.get("options", {})
                    if command not in {"apply", "reset", "status", "ping"}:
                        raise ValueError(f"unsupported command: {command}")
                    if command == "ping":
                        result = {"ok": True, "pong": True, "pid": os.getpid()}
                    elif command == "status":
                        result = run_netem_command(netem_script, "status", {})
                    elif command == "reset":
                        result = run_netem_command(netem_script, "reset", {})
                    else:
                        result = run_netem_command(netem_script, "apply", dict(options))
                except Exception as exc:  # noqa: BLE001
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                conn.sendall(json.dumps(result).encode("utf-8"))
    finally:
        try:
            server.close()
        finally:
            if socket_path.exists():
                socket_path.unlink()
            if pid_path.exists():
                pid_path.unlink()
            log_path.touch(exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.cmd == "start":
        start_detached(args)
    elif args.cmd == "serve":
        serve(args)
    elif args.cmd == "stop":
        stop_detached(args)
    elif args.cmd == "status":
        helper_status(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
