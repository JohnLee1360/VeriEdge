# THC/TSTC Qwen 实验说明

这个目录用于运行 BC-RA 修订后的争议升级验证实验，当前实验模型为 Qwen3 0.6B。

## 实验范围

- 对比 `thc`（baseline）与 `tstc`（带容差的 sampled chain）
- 验证单元：`Shard k`
- checkpoint：`C1`、`C2`、`C3`
- 阶段：`prefill`、`decode`
- 场景：`honest_homo`、`honest_hetero`、`tamper`
- 核心指标：`TPR`、`FPR`、`Localization Accuracy`

## 环境

默认工作目录建议统一为：

```bash
~/repo/paper/bc-ra-paper
```

THC 相关脚本现在默认使用仓库虚拟环境：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3
```

如果 `.venv` 不存在，脚本会自动创建；如果 Python 版本过低，脚本会自动重建。

自动装包默认使用清华 PyPI 镜像：

```bash
https://pypi.tuna.tsinghua.edu.cn/simple
```

如果某台机器需要改成别的镜像，可以先导出：

```bash
export BC_RA_PIP_INDEX_URL=https://<your-mirror>/simple
export THC_PIP_INDEX_URL=https://<your-mirror>/simple
```

如果只想维护一套全仓库通用配置，优先导出 `BC_RA_PIP_INDEX_URL` 即可；`THC_PIP_INDEX_URL` 只在你想单独覆盖 THC 时再设置。

旧的单机 MLX 路径仍然依赖 `exo` 源码目录：

```bash
~/repo/paper/third_party/exo
```

## 运行入口

旧的单机 MLX 模型路径：

```bash
~/.exo/models/mlx-community--Qwen3-0.6B-8bit
```

旧的单机 checkpoint 导出演示：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/src/capture_qwen.py \
  --config artifacts/_shared/thc/config/qwen.yaml \
  --split evaluation \
  --limit 1
```

从 honest captures 生成 delta：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/src/calibrate_delta.py \
  --capture-roots /path/to/m4_a_capture /path/to/m4_b_capture /path/to/intel_capture
```

采样搜索预运行：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/src/run.py \
  --config artifacts/_shared/thc/config/qwen.yaml \
  --mode all \
  --split evaluation \
  --runs-per-mode 3 \
  --calibrate-tstc true
```

正式运行：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/src/run.py \
  --config artifacts/_shared/thc/config/qwen.yaml \
  --mode all \
  --split evaluation \
  --runs-per-mode 10 \
  --calibrate-tstc false
```

## 测试

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 -m unittest discover -s artifacts/_shared/thc/tests -q
```

## 输出内容

每次运行会写出：

- `artifacts/_shared/thc/output/<run_id>/raw_results.json`
- `artifacts/_shared/thc/output/<run_id>/checkpoint_metadata.jsonl`
- `artifacts/_shared/thc/output/<run_id>/summary_metrics.csv`
- `artifacts/_shared/thc/output/<run_id>/run_meta.json`
- `artifacts/_shared/thc/output/<run_id>/thc_tstc_fpr_by_stage_*.png`
- `artifacts/_shared/thc/output/<run_id>/thc_tstc_tpr_loc_by_stage_*.png`
- `artifacts/_shared/thc/output/<run_id>/thc_tstc_hetero_fpr_breakdown_*.png`
- 启用 sampling search 时，还会输出 `artifacts/_shared/thc/output/<run_id>/tstc_sampling_search.csv`

面向论文的图也会同步到：

- `paper/v2/img/thc_tstc_*.png`

## Paper1 协作版新增：E4 overhead 汇总

当前协作仓库新增了一个面向论文 1 `E4` 的统计入口：

```bash
python3 artifacts/_shared/thc/src/overhead_report.py \
  --config artifacts/_shared/thc/config/qwen.yaml \
  --capture-root /path/to/capture_root \
  --owner johnlee
```

它会输出：

- `exp_e4_<date>_<owner>_size_breakdown.csv`
- `exp_e4_<date>_<owner>_latency_breakdown.csv`
- `exp_e4_<date>_<owner>_storage_breakdown.csv`
- `exp_e4_<date>_<owner>_summary.csv`

默认输出目录：

- `artifacts/E4/commits/logs/<date>_<owner>/`

## strict T3：异构 Torch 分片路径

`revision.md` 当前固定的 strict T3 prototype stack 是：

- `Provider A`: Mac mini M4
- `Provider B`: Mac mini M4
- `Provider C`: Linux 3090

这条路径不再基于 `exo` 的 MLX distributed runtime，而是改成：

- 两台 M4 使用 `torch+mps`
- Linux 3090 使用 `torch+cuda`
- 三个 shard server 之间通过 HTTP 传输张量

当前默认 placement 为：

- `jlmini_3 -> C1 -> first_shard`
- `linux124 -> C2`
- `jlmini_2 -> C3 -> last_shard`

完整分步执行手册见：

- `artifacts/_shared/thc/T3_hetero_runbook.md`

这样做的目的，是在保留真实异构硬件执行的同时，让三台机器都运行同一个模型家族和同一套执行框架。

这里的 “cluster file” 指的是你实际启动 `jlmini_2`、`jlmini_3`、`linux124` 三台机器时使用的那份 JSON 配置文件。它不是示例模板，而是 strict T3 的真实启动参数，里面必须写入真实节点名、真实 LAN IP、真实 layer 范围、device 和量化参数。

### 模型

strict T3 使用：

```bash
Qwen/Qwen3-0.6B
```

它和旧的 MLX 路径分开：

```bash
~/.exo/models/mlx-community--Qwen3-0.6B-8bit
```

也就是说，MLX 模型目录仍然可以服务旧的单机路径，但已经不是 strict T3 的后端了。

因此，按当前 strict T3 实现，不能直接把 `mlx-community--Qwen3-0.6B-8bit` 当成三台机器共同使用的真实加载目标。

如果希望当前异构 runner 尽量贴近 `Qwen3-0.6B-8bit` 的目标，推荐的稳定参数是：

- `jlmini_3 (C1)`：
  - `device = mps`
  - `quantization = metal_8bit`
  - `quantization_bits = 8`
  - `quantization_group_size = 64`
- `jlmini_2 (C3)`：
  - `device = mps`
  - `quantization = none`
  - `torch_dtype = float16`
- `linux124`：
  - `device = cuda`
  - `quantization = bitsandbytes_8bit`
  - `quantization_bits = 8`

也就是说，strict T3 继续用统一基础模型 `Qwen/Qwen3-0.6B`。其中 `jlmini_3` 和 `linux124` 仍尽量保持 8bit，`jlmini_2` 的 `C3` 改成 `float16`，因为当前 MPS/Metal 8bit kernel 在非首 shard 接收 hidden states 时无法稳定运行。

这组参数的定位是“8bit 近似替代配置”，不是“完全复用同一个 MLX 8bit checkpoint”。

### Cluster 配置

直接使用仓库里的默认 cluster 文件：

```bash
artifacts/_shared/thc/config/hetero_qwen_cluster.json
```

这份 cluster file 包含：

- 节点名
- host / port
- shard checkpoint（`C1/C2/C3`）
- layer 范围
- device（`mps` 或 `cuda`）
- dtype
- quantization 参数

如果你想用统一入口，直接进入交互式菜单：

```bash
bash artifacts/_shared/thc/scripts/run_t3_hetero_console.sh menu
```

### 每台机器先做环境检查

建议三台机器都先：

```bash
cd ~/repo/paper/bc-ra-paper
```

在每台机器上执行：

```bash
LOCAL_NODE=jlmini_2 \
CLUSTER_FILE=/path/to/hetero_qwen_cluster.json \
bash artifacts/_shared/thc/scripts/check_t3_hetero_env.sh
```

也可以改用统一入口：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py check \
  --cluster-file /path/to/hetero_qwen_cluster.json \
  --local-node jlmini_2
```

它会检查：

- cluster file 里是否包含本机节点
- PyTorch / Transformers / NumPy 等依赖是否已安装
- cluster file 里要求的本地设备是否真的可用

如果缺包，脚本会自动安装到 `~/repo/paper/bc-ra-paper/.venv`；默认使用清华镜像源。

### 启动 shard server

每台机器各启动一个 shard server：

```bash
LOCAL_NODE=jlmini_2 \
CLUSTER_FILE=/path/to/hetero_qwen_cluster.json \
bash artifacts/_shared/thc/scripts/run_t3_hetero_server.sh
```

或者：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py serve \
  --cluster-file /path/to/hetero_qwen_cluster.json \
  --local-node jlmini_2
```

另外两台机器分别把 `LOCAL_NODE` 换成 `jlmini_3` 和 `linux124`。

### 运行异构 capture

三个 server 都启动后，在其中一台协调机上执行：

```bash
CLUSTER_FILE=/path/to/hetero_qwen_cluster.json \
OUTPUT_DIR=/tmp/thc_t3/hetero_run_a \
SPLIT=calibration \
bash artifacts/_shared/thc/scripts/run_t3_hetero_capture.sh
```

或者：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py capture \
  --cluster-file /path/to/hetero_qwen_cluster.json \
  --output-dir /tmp/thc_t3/hetero_run_a \
  --split calibration
```

输出结果会写成标准 capture root：

- `captures/<prompt_id>.npz`
- `checkpoint_metadata.jsonl`
- `capture_summary.json`

这些输出可以直接喂给：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/src/calibrate_delta.py --capture-roots /path/to/run_a /path/to/run_b
```

也可以使用封装脚本：

```bash
bash artifacts/_shared/thc/scripts/run_t3_delta_calibration.sh \
  /tmp/thc_t3/delta_calibration \
  /tmp/thc_t3/hetero_run_a \
  /tmp/thc_t3/hetero_run_b
```

或者：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py calibrate \
  --output-dir /tmp/thc_t3/delta_calibration \
  /tmp/thc_t3/hetero_run_a \
  /tmp/thc_t3/hetero_run_b
```

### 备注

- strict T3 现在要求真实的跨 backend shard 执行，不再接受单机 logical slicing
- 如果你想研究 placement sensitivity，可以准备多份 cluster file，让 Linux 3090 分别承担不同 shard，然后分别产出 capture roots 再去做 calibration
- 旧的 `exo_capture_runner.py` 仍可保留给 MLX-only 的遗留实验，但它已经不是 strict T3 路径
