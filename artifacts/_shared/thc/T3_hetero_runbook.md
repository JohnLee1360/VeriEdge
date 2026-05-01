# T3 异构校准执行手册

这份手册对应 `revision.md` 中定义的 strict T3 路径：

- `Provider A`: Mac mini M4
- `Provider B`: Mac mini M4
- `Provider C`: Linux 服务器 + RTX 3090

执行形态固定为：

- 两台 M4 走 `torch+mps`
- Linux 3090 走 `torch+cuda`
- 三个 shard server 之间通过 HTTP 传递张量

这是一条真实的异构 shard 执行链路。strict T3 不再使用旧的 MLX-only 路径。

## 1. 机器角色

建议统一命名为：

- `jlmini_2`
- `jlmini_3`
- `linux124`

当前默认 placement 为：

- `jlmini_3 -> C1 -> first_shard`
- `linux124 -> C2`
- `jlmini_2 -> C3 -> last_shard`

建议把 `jlmini_2` 作为协调机：

- 用来发起 capture
- 用来执行 calibration
- 同时也可以承担本地 shard server

## 2. 需要的文件

主要文件：

- `artifacts/_shared/thc/config/qwen.yaml`
- `artifacts/_shared/thc/config/hetero_qwen_cluster.json`
- `artifacts/_shared/thc/scripts/t3_hetero_cli.py`
- `artifacts/_shared/thc/scripts/run_t3_hetero_console.sh`
- `artifacts/_shared/thc/scripts/check_t3_hetero_env.sh`
- `artifacts/_shared/thc/scripts/run_t3_hetero_server.sh`
- `artifacts/_shared/thc/scripts/run_t3_hetero_capture.sh`
- `artifacts/_shared/thc/scripts/run_t3_delta_calibration.sh`

建议输出目录：

- capture 输出：`/tmp/thc_t3/<run_name>`
- delta 校准输出：`/tmp/thc_t3/delta_<date_or_tag>`

## 3. 三台机器上的仓库准备

三台机器都克隆同一份仓库。

尽量保持相同的相对目录结构，这样三台机器的命令可以完全一致。

你当前的推荐目录结构是：

- 实验与论文仓库：`~/repo/paper/bc-ra-paper`
- exo 仓库：`~/repo/paper/third_party/exo`

后续命令默认都以 `~/repo/paper/bc-ra-paper` 为当前工作目录。

仓库里已经直接提供默认的真实 cluster 文件：

```bash
artifacts/_shared/thc/config/hetero_qwen_cluster.json
```

其中默认 IP 为：

- `jlmini_2` 的 IP：`192.168.31.159`
- `jlmini_3` 的 IP：`192.168.31.51`
- `linux124` 的 IP：`172.31.100.124`

如果这些值不变，直接使用这份文件即可，不需要额外生成。

除非你明确要做 placement variation，否则不要改默认顺序：`jlmini_3 -> C1`、`linux124 -> C2`、`jlmini_2 -> C3`。

这里说的“真实 cluster 文件”，指的是你最终实际拿来启动三台机器实验的那份 JSON 配置文件，而不是仓库里的 example 模板。它就是 strict T3 的真实启动参数文件。默认情况下，你主要只需要改三台机器的 `host` IP；其他 shard/checkpoint/device/quantization 字段都应保持模板默认。它至少要包含：

- 真实节点名：`jlmini_2`、`jlmini_3`、`linux124`
- 三台机器的真实 LAN IP
- 每台机器对应的 checkpoint、layer 范围、device
- 当前实验实际使用的量化参数

## 4. Python 环境准备

建议三台机器都先：

```bash
cd ~/repo/paper/bc-ra-paper
```

当前这套 THC 脚本默认统一使用：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3
```

如果 `.venv` 不存在，脚本会自动创建；如果虚拟环境里的 Python 版本低于 3.10，脚本会自动重建。

每台机器都需要安装这些 Python 包：

- `numpy`
- `transformers`
- `safetensors`
- `accelerate`
- `sentencepiece`
- `torch`

注意：

- 两台 M4 必须安装支持 `mps` 的 PyTorch
- 两台 M4 如果使用 `metal_8bit`，还需要额外安装 `kernels`
- Linux 3090 必须安装带 CUDA 支持的 PyTorch，并且要和服务器上的 NVIDIA driver / CUDA 环境匹配
- Linux 3090 如果使用 `bitsandbytes_8bit`，还需要额外安装 `bitsandbytes`

不要把 Mac 的安装命令直接照搬到 Linux 上。

对于 `artifacts/_shared/thc/scripts` 目录下的脚本，你通常不需要手动 `pip install`。环境检查或启动脚本会在缺包时自动安装依赖到 `~/repo/paper/bc-ra-paper/.venv`。

默认镜像源为清华 PyPI：

```bash
https://pypi.tuna.tsinghua.edu.cn/simple
```

如果 Linux 机器需要改成别的国内镜像，可以先执行：

```bash
export BC_RA_PIP_INDEX_URL=https://<your-mirror>/simple
export THC_PIP_INDEX_URL=https://<your-mirror>/simple
```

如果你确实要手动安装，建议显式使用镜像源，例如：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy transformers safetensors accelerate sentencepiece
```

Mac M4 若启用 `metal_8bit`，额外安装：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kernels
```

Linux 3090 若启用 `bitsandbytes_8bit`，额外安装：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple bitsandbytes
```

## 5. 三台机器都提前下载模型

当前 strict T3 已实现代码路径使用的基础模型标识是：

```bash
Qwen/Qwen3-0.6B
```

也就是说，当前这版异构 runner 通过 `transformers` 从 Hugging Face 模型标识加载基础模型，而不是直接读取本地 MLX 模型目录。

你当前约定的本地 MLX 模型路径是：

- `jlmini_2` / `jlmini_3`：`~/.exo/models/mlx-community--Qwen3-0.6B-8bit`
- `linux124`：`~/repo/third_party/mlx-community--Qwen3-0.6B-8bit`

这两类路径目前主要用于：

- 旧的单机 MLX 路径
- 后续如果我们决定把 strict T3 重构成统一 MLX 后端时的本地模型布局

如果继续沿用当前已经实现的异构 runner，就不要把这两个 MLX 本地路径当成 strict T3 的实际加载入口。

原因很简单：

- `mlx-community--Qwen3-0.6B-8bit` 是 MLX 生态下的量化模型目录
- 当前 strict T3 的异构 runner 通过 `transformers` 加载统一基础模型，再按设备分别套用量化配置
- 因此，当前 strict T3 不能把 `mlx-community--Qwen3-0.6B-8bit` 当成三台机器共同使用的直接模型加载目标

如果你想在当前 runner 路线上尽量保持“Qwen3-0.6B-8bit”的目标，推荐的稳定替代配置是：

- 基础模型仍然使用：`Qwen/Qwen3-0.6B`
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

这不是“完全同一个 MLX 8bit checkpoint”，而是“在当前跨平台 runner 上，用同一个基础模型 + 平台对应的官方 8bit 量化加载方式”来尽量贴近你的 8bit 实验目标。

当前推荐替代参数可理解为：

- 统一基础模型参数：`Qwen/Qwen3-0.6B`
- `jlmini_3` 侧的 8bit 近似参数：
  - `quantization = metal_8bit`
  - `quantization_bits = 8`
  - `quantization_group_size = 64`
- `jlmini_2` 侧保留：
  - `quantization = none`
  - `torch_dtype = float16`
- Linux 3090 侧的 8bit 近似参数：
  - `quantization = bitsandbytes_8bit`
  - `quantization_bits = 8`

这组参数的目的，是让三台机器尽量贴近“Qwen3-0.6B-8bit”的实验意图，同时保留当前 strict T3 所需的跨平台真实 shard execution。这里 `jlmini_2` 的 `C3` 明确保留为 `float16`，因为当前 MPS/Metal 8bit kernel 在非首 shard 接收 hidden states 时会报缺失 kernel，不能稳定运行。

建议在每台机器上先做一次预下载：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
print("prefetch complete")
PY
```

这个步骤强烈建议先做。这样可以避免第一次启动 shard server 时边下模型边报错，也更容易定位真正的问题。

## 6. 每台机器先做环境检查

每台机器都先执行：

```bash
LOCAL_NODE=jlmini_2 \
CLUSTER_FILE=artifacts/_shared/thc/config/hetero_qwen_cluster.json \
bash artifacts/_shared/thc/scripts/check_t3_hetero_env.sh
```

如果你不想手敲环境变量，也可以执行：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py check \
  --cluster-file artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node jlmini_2
```

把 `LOCAL_NODE` 换成各自机器对应的值：

- 第一台 Mac：`jlmini_2`
- 第二台 Mac：`jlmini_3`
- Linux 服务器：`linux124`

这个检查会验证：

- 本机是否在 cluster file 中
- Python 包是否能正常导入
- 配置要求的本地设备是否可用

三台机器都通过之后，再继续下一步。

## 7. 启动三个 shard server

每台机器开一个终端，分别执行：

### 在 `jlmini_2`

```bash
LOCAL_NODE=jlmini_2 \
CLUSTER_FILE=artifacts/_shared/thc/config/hetero_qwen_cluster.json \
bash artifacts/_shared/thc/scripts/run_t3_hetero_server.sh
```

或者：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py serve \
  --cluster-file artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node jlmini_2
```

### 在 `jlmini_3`

```bash
LOCAL_NODE=jlmini_3 \
CLUSTER_FILE=artifacts/_shared/thc/config/hetero_qwen_cluster.json \
bash artifacts/_shared/thc/scripts/run_t3_hetero_server.sh
```

### 在 `linux124`

```bash
LOCAL_NODE=linux124 \
CLUSTER_FILE=artifacts/_shared/thc/config/hetero_qwen_cluster.json \
bash artifacts/_shared/thc/scripts/run_t3_hetero_server.sh
```

正常情况下，每个 server 会输出：

- 自己的 `node_name`
- 自己负责的 `checkpoint`
- 自己使用的 `device`
- 自己当前使用的 `quantization`

然后进程会持续监听 HTTP 请求。

capture 没跑完之前，不要关闭这三个终端。

## 8. 先跑一个 capture root

在协调机 `jlmini_2` 上执行：

```bash
CLUSTER_FILE=artifacts/_shared/thc/config/hetero_qwen_cluster.json \
OUTPUT_DIR=/tmp/thc_t3/hetero_run_a \
SPLIT=calibration \
LIMIT_PROMPTS=0 \
bash artifacts/_shared/thc/scripts/run_t3_hetero_capture.sh
```

或者：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py capture \
  --cluster-file artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --output-dir /tmp/thc_t3/hetero_run_a \
  --split calibration \
  --limit-prompts 0
```

说明：

- `OUTPUT_DIR` 必须是一个新的空目录
- T3 必须使用 `SPLIT=calibration`
- `LIMIT_PROMPTS=0` 表示跑完整个 calibration split

预期输出：

- `captures/<prompt_id>.npz`
- `checkpoint_metadata.jsonl`
- `capture_summary.json`

## 9. 生成多个 honest capture roots

至少要产出多个 capture roots，后面的 `calibrate_delta.py` 才有可比较的样本。

建议最少跑三次：

- `hetero_run_a`
- `hetero_run_b`
- `hetero_run_c`

第二次示例：

```bash
CLUSTER_FILE=artifacts/_shared/thc/config/hetero_qwen_cluster.json \
OUTPUT_DIR=/tmp/thc_t3/hetero_run_b \
SPLIT=calibration \
bash artifacts/_shared/thc/scripts/run_t3_hetero_capture.sh
```

第三次示例：

```bash
CLUSTER_FILE=artifacts/_shared/thc/config/hetero_qwen_cluster.json \
OUTPUT_DIR=/tmp/thc_t3/hetero_run_c \
SPLIT=calibration \
bash artifacts/_shared/thc/scripts/run_t3_hetero_capture.sh
```

## 10. 可选：做 placement sensitivity

如果你想看 `Δ_k` 是否会受到 “linux124 负责哪个 shard” 的影响，可以另外准备几份 cluster file，例如：

- `hetero_qwen_cluster_linux_c1.json`
- `hetero_qwen_cluster_linux_c2.json`
- `hetero_qwen_cluster_linux_c3.json`

每一份都要：

- 保持还是这三台机器
- 一致地修改 checkpoint / layer 分配
- 重新跑完整 capture 流程

不同 placement 的输出目录一定要分开，不能混写。

## 11. 执行 delta 校准

拿到多个 capture roots 之后，在协调机上运行：

```bash
bash artifacts/_shared/thc/scripts/run_t3_delta_calibration.sh \
  /tmp/thc_t3/delta_calibration_main \
  /tmp/thc_t3/hetero_run_a \
  /tmp/thc_t3/hetero_run_b \
  /tmp/thc_t3/hetero_run_c
```

或者：

```bash
~/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/_shared/thc/scripts/t3_hetero_cli.py calibrate \
  --output-dir /tmp/thc_t3/delta_calibration_main \
  /tmp/thc_t3/hetero_run_a \
  /tmp/thc_t3/hetero_run_b \
  /tmp/thc_t3/hetero_run_c
```

预期输出：

- `delta_raw_records.json`
- `delta_summary.csv`
- `delta_map.json`

后续接入 T5 最关键的文件是：

- `delta_map.json`

## 12. 完成后带回主会话的内容

T3 跑完之后，回到主会话时请带上：

- 实际使用的 cluster file
- 你跑出来的 capture roots 列表
- 最终的 `delta_map.json`
- 如果中途出错，对应的错误日志

拿到这些结果之后，主会话才能继续做：

- `T5`：接入真实 `delta_map` 跑正式 evaluation
- `T6`：回填论文中的表、图、结论

## 13. 建议目录命名

建议用：

- `/tmp/thc_t3/hetero_run_a`
- `/tmp/thc_t3/hetero_run_b`
- `/tmp/thc_t3/hetero_run_c`
- `/tmp/thc_t3/delta_calibration_main`

如果要做 placement variation，可以用：

- `/tmp/thc_t3/linux_c1_run_a`
- `/tmp/thc_t3/linux_c2_run_a`
- `/tmp/thc_t3/linux_c3_run_a`

## 14. 常见问题

### 14.1 Mac 上 `mps` 不可用

常见原因：

- 装错了 PyTorch
- 用错了 Python 环境
- 当前 shell 不是你以为的那个环境

处理方式：

- 先重新跑 `check_t3_hetero_env.sh`
- 确认 Python 环境切换正确后再开 server

### 14.2 Linux 上 `cuda` 不可用

常见原因：

- 装错了 torch wheel
- NVIDIA driver / CUDA 不匹配
- 当前 shell 没有用到你配置好的 Python

处理方式：

- 先把 CUDA 版 PyTorch 修好
- 再重新跑环境检查

### 14.3 capture 过程中出现 connection refused

常见原因：

- 三个 shard server 里有一个没启动
- cluster file 的 host / port 写错了
- 局域网连通性有问题

处理方式：

- 确认三个 server 终端都还活着
- 确认 cluster file 里的 IP 和 port
- 确认三台机器在同一网络下能互通

### 14.4 第一次运行特别慢

常见原因：

- server 启动时还在下载模型

处理方式：

- 先做模型预下载，再启动 server

### 14.5 输出目录报错

原因：

- capture / calibration 脚本明确要求输出目录为空，防止结果混写

处理方式：

- 每次运行都换一个新的输出目录

### 14.6 calibration split 和 evaluation split 混用

规则固定为：

- `T3` 只允许用 `calibration`
- `T5` 只允许用 `evaluation`

不要混用。
