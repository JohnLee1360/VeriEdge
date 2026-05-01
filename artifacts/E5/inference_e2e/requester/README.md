# Requester 侧：黑盒 EXO 任务级实验编排

本目录只负责 requester 侧逻辑，默认把 EXO 当作外部系统处理。

当前实验固定为：

- 一个 task 含 `20` 个文本问题
- task 内严格串行执行
- 当前实验矩阵为：`LAN` 跑 `instance_node_count ∈ {1,2,3}`，`WAN` 只跑 `instance_node_count = 3`
- requester 将加密任务包上传到本地 Kubo
- requester 根据 `/state` 识别 first-shard provider
- first-shard provider 下载、解密并执行 task，然后通过 callback 回传结果

黑盒边界：

- 不默认调用 `place_instance`
- 不默认删除旧 instance
- 不默认尝试修复 EXO
- 仅校验当前外部 EXO 是否满足本次 cell 的运行条件

## 1. 安装依赖

```bash
cd ~/repo/paper/bc-ra-paper
export BC_RA_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install -i "$BC_RA_PIP_INDEX_URL" -r artifacts/E5/inference_e2e/requester/requirements.txt
```

## 2. 准备配置

```bash
cp artifacts/E5/inference_e2e/requester/config.example.yaml artifacts/E5/inference_e2e/requester/config.yaml
```

重点字段：

- `external_exo.mode`：默认 `external`
- `external_exo.source_dir`：本地 EXO 源码目录
- `external_exo.freeze_manifest_path`：版本快照清单路径
- `endpoints.entry_url`：EXO cluster entry
- `endpoints.requester_callback_url`：provider 回传结果的 callback 地址
- `providers[]`：provider 的 `node_id/ip/exo_port/launcher_url/public_key_path`
- `ipfs.api_url`：本地 Kubo API
- `ipfs.gateway_url`：provider 可访问的 Kubo gateway
- `task.prompt_file`：task 输入题库
- `task.question_count`：默认 `20`
- `matrix.network_profiles`：默认 `["LAN", "WAN"]`
- `matrix.instance_node_counts`：默认 `[1, 2, 3]`
- `matrix.instance_node_counts_by_network`：可按网络分别限制 node count；当前默认 `LAN=[1,2,3]`, `WAN=[3]`

默认设备：

- requester：`192.168.31.189`
- providers：`192.168.31.52`、`192.168.31.159`、`192.168.31.83`

## 3. 版本快照与一致性校验

生成实验侧 EXO 版本快照：

```bash
python3 artifacts/E5/inference_e2e/requester/freeze_exo_env.py \
  --exo-dir ~/repo/paper/third_party/exo \
  --output artifacts/E5/inference_e2e/freeze/exo_env_manifest.json
```

校验本地或远端 EXO 是否与快照一致：

```bash
python3 artifacts/E5/inference_e2e/requester/verify_exo_env.py \
  --manifest artifacts/E5/inference_e2e/freeze/exo_env_manifest.json \
  --exo-dir ~/repo/paper/third_party/exo
```

如需校验远端机器，可追加：

```bash
python3 artifacts/E5/inference_e2e/requester/verify_exo_env.py \
  --manifest artifacts/E5/inference_e2e/freeze/exo_env_manifest.json \
  --exo-dir ~/repo/paper/third_party/exo \
  --ssh-target jlmini_1 \
  --ssh-target jlmini_3
```

脚本只读校验，不自动升级或修复版本。

当前硬门槛默认只检查：

- EXO 仓库 `commit`
- `flake.lock` 的 SHA256

其余字段会作为诊断信息输出，但不作为失败条件。

## 4. Kubo 管理

MacBook 本地 Kubo 通过仓库脚本管理：

```bash
bash artifacts/E5/inference_e2e/requester/kubo_macbook.sh start
bash artifacts/E5/inference_e2e/requester/kubo_macbook.sh status
bash artifacts/E5/inference_e2e/requester/kubo_macbook.sh id
bash artifacts/E5/inference_e2e/requester/kubo_macbook.sh check-gateway
```

停止脚本拉起的 Kubo：

```bash
bash artifacts/E5/inference_e2e/requester/kubo_macbook.sh stop
```

## 5. 黑盒就绪检查

只做环境检查，不发任务：

```bash
python3 artifacts/E5/inference_e2e/requester/check_exo_ready.py \
  --config artifacts/E5/inference_e2e/requester/config.yaml \
  --instance-node-count 1
```

检查项包括：

- entry 与 provider 的 `/v1/models`
- provider launcher 的 `/health`
- requester callback 端口
- Kubo API 与 gateway
- `/state` 中目标模型的 active instance 数量
- first-shard provider 是否可解析
- provider IP 是否出现在 `/state`

## 6. 运行实验

单个 cell 的黑盒 smoke run：

```bash
python3 artifacts/E5/inference_e2e/requester/runner.py \
  --config artifacts/E5/inference_e2e/requester/config.yaml \
  --network-profile LAN \
  --instance-node-count 1 \
  --smoke
```

只做 preflight 与状态快照：

```bash
python3 artifacts/E5/inference_e2e/requester/runner.py \
  --config artifacts/E5/inference_e2e/requester/config.yaml \
  --network-profile LAN \
  --instance-node-count 1 \
  --check-only
```

运行完整当前矩阵：

```bash
python3 artifacts/E5/inference_e2e/requester/runner.py \
  --config artifacts/E5/inference_e2e/requester/config.yaml
```

使用单次 sudo 的 macOS 批处理入口：

```bash
bash artifacts/E5/inference_e2e/requester/run_matrix_once_macos.sh \
  --config artifacts/E5/inference_e2e/requester/config.yaml
```

该脚本只负责：

- LAN/WAN 切换
- 调用黑盒 preflight
- 运行矩阵
- 汇总与绘图

不会修改 EXO instance。

如果你希望用低 token 的 control-plane 方式运行长实验，优先把长流程放进单个后台进程：

```bash
nohup python3 artifacts/E5/inference_e2e/requester/matrix_control.py \
  --config artifacts/E5/inference_e2e/requester/config.yaml \
  --batch-dir artifacts/E5/inference_e2e/requester/output/control_lan_run \
  --network-profile LAN \
  --instance-node-count 1 \
  --instance-node-count 2 \
  --instance-node-count 3 \
  --provision-instances \
  --no-plot \
  > artifacts/E5/inference_e2e/requester/output/control_lan_run/launch.log 2>&1 &
```

这个控制脚本会把聊天外的真实长流程收口到一个后台 runner，并主要依赖这些文件做低开销监控：

- `batch_manifest.json`
- `batch_status.json`
- `cells/<cell_id>/control_status.json`
- `cells/<cell_id>/instance_control.json`
- `cells/<cell_id>/summary_by_cell.csv`
- `summary_by_cell.csv`

如果 WAN 需要在每个 cell 前先清缓存、重构建或重启 EXO/provider，可以把命令写进 `config.yaml`：

```yaml
control_plane:
  before_network_commands:
    WAN:
      - "<your cache reset command>"
      - "<your rebuild / restart command>"
```

这些 hook 会在 `matrix_control.py` 里于 `WAN shaping` 和 `instance create` 之前执行，输出写入 `cells/<cell_id>/control_hooks.log`。

## 7. 输出文件

每个 cell 目录包含：

- `cell_status.json`
- `state_snapshot.json`
- `state_snapshot_after_smoke.json`
- `smoke_summary.json`
- `summary_by_task.csv`
- `summary_by_cell.csv`
- `tasks/smoke_01/*`
- `tasks/main_01/*` 到 `tasks/main_05/*`

每次 run 顶层输出包含：

- `health_checks.csv`
- `aux_service_checks.csv`
- `summary_by_cell.csv`
- `summary_by_task.csv`
- `run_config_snapshot.yaml`
- `comparison_table.csv` / `comparison_table.md`
- 绘图产物

`summary_by_cell.csv` 主字段：

- `instance_node_count`
- `network`
- `mean_task_latency_s_per_task`
- `mean_question_latency_s_per_q`
- `mean_download_s_per_task`
- `mean_ttft_p50_s`
- `mean_otps_p50_tok_s`
- `sum_question_success_count`
- `sum_question_fail_count`
- `completed_task_count`
