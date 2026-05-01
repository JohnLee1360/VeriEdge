#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "artifacts/_shared/thc/scripts"
DEFAULT_CLUSTER_FILE = REPO_ROOT / "artifacts/_shared/thc/config/hetero_qwen_cluster.json"
DEFAULT_CONFIG_PATH = REPO_ROOT / "artifacts/_shared/thc/config/qwen.yaml"
DEFAULT_PYTHON_BIN = REPO_ROOT / ".venv/bin/python3"


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _display_path(value: str | Path) -> str:
    path = _resolve_path(value)
    home = Path.home().resolve()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)


def _timestamped_tmp(prefix: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/thc_t3/{prefix}_{timestamp}")


def _run_script(script_name: str, *, env: dict[str, str] | None = None, args: Sequence[str] = ()) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    cmd = ["bash", str(SCRIPTS_DIR / script_name), *args]
    subprocess.run(cmd, check=True, env=merged_env)


def _detect_ipv4_candidates() -> List[str]:
    values: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            values.add(str(sock.getsockname()[0]))
    except OSError:
        pass
    try:
        hostname = socket.gethostname()
        for entry in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            value = str(entry[4][0])
            if not value.startswith("127."):
                values.add(value)
    except OSError:
        pass
    return sorted(values)


def _print_guide() -> None:
    print("strict T3 cluster 信息说明")
    print()
    print("真实 cluster 文件里，大多数实验字段已经固定。正常情况下你只需要确认三台机器的 LAN IP。")
    print("通常需要你自己确认/修改的字段:")
    print("- nodes[].host: 三台机器的真实局域网 IP")
    print("- nodes[].port: 默认 8311；只有端口冲突时才改")
    print("- local_files_only: 如果三台机器都已预下载模型，可改为 true")
    print()
    print("通常不需要改的字段:")
    print("- node_name: jlmini_2 / jlmini_3 / linux124")
    print("- checkpoint: C1 / C2 / C3")
    print("- layer ranges: 0..7 / 8..15 / 16..23")
    print("- device: mps / cuda / mps")
    print("- quantization: metal_8bit / bitsandbytes_8bit / none")
    print("- model_id: Qwen/Qwen3-0.6B")
    print()
    print("查看本机 LAN IP 的常用命令:")
    print("- macOS: ipconfig getifaddr en0")
    print("- macOS: ifconfig | grep 'inet '")
    print("- Linux: hostname -I")
    print("- Linux: ip -4 addr")
    print()
    values = _detect_ipv4_candidates()
    if values:
        print(f"当前机器探测到的 IPv4 候选: {', '.join(values)}")
        print()
    print("默认 cluster 文件:")
    print(_display_path(DEFAULT_CLUSTER_FILE))


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw or default


def _prompt_path(label: str, default: Path) -> Path:
    return _resolve_path(_prompt(label, str(default)))


def _prompt_cluster_runtime(cluster_default: Path, hostname_default: str, python_default: str) -> tuple[Path, str, str]:
    cluster_file = _prompt_path("cluster 文件路径", cluster_default)
    local_node = _prompt("本机节点名", hostname_default)
    python_bin = _prompt("Python 可执行文件", python_default)
    return cluster_file, local_node, python_bin


def _command_check(cluster_file: Path, local_node: str, python_bin: str) -> None:
    _run_script(
        "check_t3_hetero_env.sh",
        env={
            "CLUSTER_FILE": str(cluster_file),
            "LOCAL_NODE": local_node,
            "PYTHON_BIN": python_bin,
        },
    )


def _command_serve(cluster_file: Path, local_node: str, python_bin: str) -> None:
    _run_script(
        "run_t3_hetero_server.sh",
        env={
            "CLUSTER_FILE": str(cluster_file),
            "LOCAL_NODE": local_node,
            "PYTHON_BIN": python_bin,
        },
    )


def _command_capture(
    cluster_file: Path,
    output_dir: Path,
    config_path: Path,
    split: str,
    limit_prompts: int,
    python_bin: str,
) -> None:
    _run_script(
        "run_t3_hetero_capture.sh",
        env={
            "CLUSTER_FILE": str(cluster_file),
            "OUTPUT_DIR": str(output_dir),
            "CONFIG_PATH": str(config_path),
            "SPLIT": split,
            "LIMIT_PROMPTS": str(int(limit_prompts)),
            "PYTHON_BIN": python_bin,
        },
    )


def _command_calibrate(output_dir: Path, capture_roots: Iterable[Path], percentile: float, python_bin: str) -> None:
    _run_script(
        "run_t3_delta_calibration.sh",
        env={
            "PERCENTILE": str(percentile),
            "PYTHON_BIN": python_bin,
        },
        args=[str(output_dir), *[str(path) for path in capture_roots]],
    )


def _default_capture_output() -> Path:
    return _timestamped_tmp("hetero_run")


def _default_delta_output() -> Path:
    return _timestamped_tmp("delta")


def _parse_capture_roots(raw: str) -> list[Path]:
    return [_resolve_path(value.strip()) for value in raw.split(",") if value.strip()]


def _menu() -> None:
    cluster_default = DEFAULT_CLUSTER_FILE
    config_default = DEFAULT_CONFIG_PATH
    python_default = str(DEFAULT_PYTHON_BIN)
    hostname_default = socket.gethostname()

    while True:
        print()
        print("strict T3 交互式启动器")
        print(f"仓库根目录: {_display_path(REPO_ROOT)}")
        print("1. 说明 cluster 文件需要填什么")
        print("2. 本机环境检查")
        print("3. 启动本机 shard server")
        print("4. 在协调机运行 capture")
        print("5. 运行 delta 校准")
        print("0. 退出")
        choice = input("请选择操作 [0-5]: ").strip()

        if choice == "0":
            return
        if choice == "1":
            _print_guide()
            continue
        if choice == "2":
            _command_check(*_prompt_cluster_runtime(cluster_default, hostname_default, python_default))
            continue
        if choice == "3":
            cluster_file, local_node, python_bin = _prompt_cluster_runtime(cluster_default, hostname_default, python_default)
            print("即将进入长期运行的 shard server。用 Ctrl+C 停止。")
            _command_serve(cluster_file, local_node, python_bin)
            continue
        if choice == "4":
            cluster_file = _prompt_path("cluster 文件路径", cluster_default)
            output_dir = _prompt_path("capture 输出目录", _default_capture_output())
            config_path = _prompt_path("qwen.yaml 路径", config_default)
            split = _prompt("split", "calibration")
            limit_prompts = int(_prompt("limit_prompts", "0"))
            python_bin = _prompt("Python 可执行文件", python_default)
            _command_capture(cluster_file, output_dir, config_path, split, limit_prompts, python_bin)
            continue
        if choice == "5":
            output_dir = _prompt_path("delta 输出目录", _default_delta_output())
            capture_roots = _parse_capture_roots(_prompt("capture roots，用逗号分隔", ""))
            if len(capture_roots) < 2:
                print("至少需要提供两个 capture root")
                continue
            percentile = float(_prompt("percentile", "99.0"))
            python_bin = _prompt("Python 可执行文件", python_default)
            _command_calibrate(output_dir, capture_roots, percentile, python_bin)
            continue
        print("无效选择，请重新输入。")


def _add_cluster_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster-file", default=str(DEFAULT_CLUSTER_FILE))


def _add_python_bin_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON_BIN))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="strict T3 统一启动入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("guide", help="解释真实 cluster 文件和需要填写的字段")
    subparsers.add_parser("menu", help="进入交互式菜单")
    subparsers.add_parser("detect-ip", help="显示当前机器可能的 IPv4 地址")

    check = subparsers.add_parser("check", help="执行本机环境检查")
    _add_cluster_file_arg(check)
    check.add_argument("--local-node", required=True)
    _add_python_bin_arg(check)

    serve = subparsers.add_parser("serve", help="启动本机 shard server")
    _add_cluster_file_arg(serve)
    serve.add_argument("--local-node", required=True)
    _add_python_bin_arg(serve)

    capture = subparsers.add_parser("capture", help="在协调机运行异构 capture")
    _add_cluster_file_arg(capture)
    capture.add_argument("--output-dir", default=str(_default_capture_output()))
    capture.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    capture.add_argument("--split", default="calibration", choices=["calibration", "evaluation"])
    capture.add_argument("--limit-prompts", type=int, default=0)
    _add_python_bin_arg(capture)

    calibrate = subparsers.add_parser("calibrate", help="从多个 capture roots 生成 delta_map")
    calibrate.add_argument("--output-dir", default=str(_default_delta_output()))
    calibrate.add_argument("--percentile", type=float, default=99.0)
    _add_python_bin_arg(calibrate)
    calibrate.add_argument("capture_roots", nargs="+")

    return parser.parse_args()


def _handle_detect_ip(_: argparse.Namespace) -> None:
    values = _detect_ipv4_candidates()
    if values:
        print("\n".join(values))
    else:
        print("未探测到可用 IPv4，请手动运行系统命令查看。")


def _handle_check(args: argparse.Namespace) -> None:
    _command_check(_resolve_path(args.cluster_file), str(args.local_node), str(args.python_bin))


def _handle_serve(args: argparse.Namespace) -> None:
    _command_serve(_resolve_path(args.cluster_file), str(args.local_node), str(args.python_bin))


def _handle_capture(args: argparse.Namespace) -> None:
    _command_capture(
        _resolve_path(args.cluster_file),
        _resolve_path(args.output_dir),
        _resolve_path(args.config_path),
        str(args.split),
        int(args.limit_prompts),
        str(args.python_bin),
    )


def _handle_calibrate(args: argparse.Namespace) -> None:
    _command_calibrate(
        _resolve_path(args.output_dir),
        [_resolve_path(value) for value in args.capture_roots],
        float(args.percentile),
        str(args.python_bin),
    )


def main() -> None:
    args = _parse_args()
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "guide": lambda _: _print_guide(),
        "menu": lambda _: _menu(),
        "detect-ip": _handle_detect_ip,
        "check": _handle_check,
        "serve": _handle_serve,
        "capture": _handle_capture,
        "calibrate": _handle_calibrate,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
