from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAC_ALIAS = "Mac3"
DEFAULT_LINUX_ALIAS = "3090"
DEFAULT_MAC_HOST = "192.168.31.51"
DEFAULT_MAC_REPO_ROOT = "/Users/jlmini_3/repo/paper/bc-ra-paper-exp_verification"
DEFAULT_LINUX_REPO_ROOT = "/home/hzh/repo/paper/bc-ra-paper-exp_verification"
DEFAULT_MAC_PYTHON_BIN = "/Users/jlmini_3/repo/paper/bc-ra-paper/.venv/bin/python3"
DEFAULT_LINUX_PYTHON_BIN = "/home/hzh/repo/paper/bc-ra-paper/.venv/bin/python3"


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _run_local(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, text=True, capture_output=True)


def _run_remote(alias: str, command: str) -> subprocess.CompletedProcess[str]:
    return _run_local(["ssh", alias, command])


def _resolve_ssh_host(alias: str, fallback: str) -> str:
    try:
        completed = _run_local(["ssh", "-G", alias])
    except subprocess.CalledProcessError:
        return str(fallback)
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("hostname "):
            return stripped.split(None, 1)[1].strip()
    return str(fallback)


def _post_ping(url: str) -> Dict[str, Any]:
    request = Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_ping(url: str, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            payload = _post_ping(url)
            if bool(payload.get("ok", False)):
                return
            last_error = json.dumps(payload, ensure_ascii=True)
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"timed out waiting for ping {url}: {last_error}")


class T3ServerSupervisor:
    def __init__(
        self,
        *,
        local_python_bin: str,
        mac_alias: str = DEFAULT_MAC_ALIAS,
        linux_alias: str = DEFAULT_LINUX_ALIAS,
        mac_repo_root: str = DEFAULT_MAC_REPO_ROOT,
        linux_repo_root: str = DEFAULT_LINUX_REPO_ROOT,
        mac_python_bin: str = DEFAULT_MAC_PYTHON_BIN,
        linux_python_bin: str = DEFAULT_LINUX_PYTHON_BIN,
        mac_host: str = "",
        local_port: int = 18312,
        linux_tunnel_port: int = 18311,
        ping_timeout_seconds: float = 120.0,
    ) -> None:
        self.local_python_bin = str(local_python_bin)
        self.mac_alias = str(mac_alias)
        self.linux_alias = str(linux_alias)
        self.mac_repo_root = str(mac_repo_root)
        self.linux_repo_root = str(linux_repo_root)
        self.mac_python_bin = str(mac_python_bin)
        self.linux_python_bin = str(linux_python_bin)
        self.mac_host = str(mac_host).strip() or _resolve_ssh_host(self.mac_alias, DEFAULT_MAC_HOST)
        self.local_port = int(local_port)
        self.linux_tunnel_port = int(linux_tunnel_port)
        self.ping_timeout_seconds = float(ping_timeout_seconds)
        self._local_proc: subprocess.Popen[bytes] | None = None
        self._mac_proc: subprocess.Popen[bytes] | None = None
        self._linux_proc: subprocess.Popen[bytes] | None = None
        self._tunnel_proc: subprocess.Popen[bytes] | None = None

    def _ensure_remote_dir(self, alias: str, remote_dir: str) -> None:
        _run_remote(alias, f"mkdir -p {_quote(remote_dir)}")

    def _scp_file(self, local_path: Path, alias: str, remote_path: str) -> None:
        _run_local(["scp", str(local_path), f"{alias}:{remote_path}"])

    def _stop_remote_server(self, alias: str, local_node: str) -> None:
        pattern = f"hetero_qwen_server.py .*--local-node {local_node}"
        subprocess.run(["ssh", alias, f"pkill -f {_quote(pattern)} || true"], cwd=str(REPO_ROOT), check=False)

    def _stop_local_server(self, local_node: str) -> None:
        pattern = f"hetero_qwen_server.py .*--local-node {local_node}"
        subprocess.run(["bash", "-lc", f"pkill -f {_quote(pattern)} || true"], cwd=str(REPO_ROOT), check=False)

    def _terminate_proc(self, proc: subprocess.Popen[bytes] | None) -> None:
        if proc is None:
            return
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)

    def down(self) -> None:
        self._terminate_proc(self._mac_proc)
        self._terminate_proc(self._linux_proc)
        self._terminate_proc(self._local_proc)
        self._terminate_proc(self._tunnel_proc)
        self._mac_proc = None
        self._linux_proc = None
        self._local_proc = None
        self._tunnel_proc = None
        self._stop_local_server("jlmini_2")
        self._stop_remote_server(self.mac_alias, "jlmini_3")
        self._stop_remote_server(self.linux_alias, "linux124")

    def _ensure_tunnel(self) -> None:
        if self._tunnel_proc is not None and self._tunnel_proc.poll() is None:
            return
        self._tunnel_proc = subprocess.Popen(
            ["ssh", "-N", "-L", f"{self.linux_tunnel_port}:127.0.0.1:8311", self.linux_alias],
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _start_local_server(self, cluster_file: Path, log_file: Path) -> subprocess.Popen[bytes]:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stream = log_file.open("ab", buffering=0)
        command = (
            f"cd {_quote(REPO_ROOT)} && "
            f"env LOCAL_NODE=jlmini_2 CLUSTER_FILE={_quote(cluster_file)} PYTHON_BIN={_quote(self.local_python_bin)} "
            f"bash {_quote(REPO_ROOT / 'artifacts/_shared/thc/scripts/run_t3_hetero_server.sh')}"
        )
        return subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _start_remote_server(
        self,
        *,
        alias: str,
        remote_repo_root: str,
        local_node: str,
        cluster_file: str,
        python_bin: str,
        log_file: str,
    ) -> subprocess.Popen[bytes]:
        command = (
            f"cd {_quote(remote_repo_root)} && "
            f"mkdir -p {_quote(str(Path(log_file).parent))} && "
            f"env LOCAL_NODE={_quote(local_node)} CLUSTER_FILE={_quote(cluster_file)} PYTHON_BIN={_quote(python_bin)} "
            f"bash {_quote(str(Path(remote_repo_root) / 'artifacts/_shared/thc/scripts/run_t3_hetero_server.sh'))} "
            f"> {_quote(log_file)} 2>&1"
        )
        return subprocess.Popen(
            ["ssh", alias, command],
            cwd=str(REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def up(
        self,
        *,
        local_cluster_file: Path,
        remote_cluster_file_local: Path,
        remote_cluster_path_mac: str,
        remote_cluster_path_linux: str,
        local_log: Path,
        mac_log: str,
        linux_log: str,
    ) -> Dict[str, str]:
        self._ensure_remote_dir(self.mac_alias, str(Path(remote_cluster_path_mac).parent))
        self._ensure_remote_dir(self.linux_alias, str(Path(remote_cluster_path_linux).parent))
        self._scp_file(remote_cluster_file_local, self.mac_alias, remote_cluster_path_mac)
        self._scp_file(remote_cluster_file_local, self.linux_alias, remote_cluster_path_linux)

        self.down()
        self._ensure_tunnel()
        self._local_proc = self._start_local_server(local_cluster_file, local_log)
        self._mac_proc = self._start_remote_server(
            alias=self.mac_alias,
            remote_repo_root=self.mac_repo_root,
            local_node="jlmini_3",
            cluster_file=remote_cluster_path_mac,
            python_bin=self.mac_python_bin,
            log_file=mac_log,
        )
        self._linux_proc = self._start_remote_server(
            alias=self.linux_alias,
            remote_repo_root=self.linux_repo_root,
            local_node="linux124",
            cluster_file=remote_cluster_path_linux,
            python_bin=self.linux_python_bin,
            log_file=linux_log,
        )
        self.wait_ready()
        return {
            "jlmini_2": str(self._local_proc.pid if self._local_proc else ""),
            "jlmini_3": str(self._mac_proc.pid if self._mac_proc else ""),
            "linux124": str(self._linux_proc.pid if self._linux_proc else ""),
            "tunnel": str(self._tunnel_proc.pid if self._tunnel_proc else ""),
        }

    def wait_ready(self) -> None:
        _wait_for_ping(f"http://127.0.0.1:{self.local_port}/ping", timeout_seconds=self.ping_timeout_seconds)
        _wait_for_ping(f"http://{self.mac_host}:8311/ping", timeout_seconds=self.ping_timeout_seconds)
        _wait_for_ping(f"http://127.0.0.1:{self.linux_tunnel_port}/ping", timeout_seconds=self.ping_timeout_seconds)

    def status(self) -> Dict[str, Any]:
        return {
            "jlmini_2": _post_ping(f"http://127.0.0.1:{self.local_port}/ping"),
            "jlmini_3": _post_ping(f"http://{self.mac_host}:8311/ping"),
            "linux124": _post_ping(f"http://127.0.0.1:{self.linux_tunnel_port}/ping"),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="strict T3 server supervisor status helper")
    parser.add_argument("--local-port", type=int, default=18312)
    parser.add_argument("--linux-tunnel-port", type=int, default=18311)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    supervisor = T3ServerSupervisor(
        local_python_bin=str(REPO_ROOT / ".venv/bin/python3"),
        local_port=int(args.local_port),
        linux_tunnel_port=int(args.linux_tunnel_port),
    )
    print(json.dumps(supervisor.status(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
