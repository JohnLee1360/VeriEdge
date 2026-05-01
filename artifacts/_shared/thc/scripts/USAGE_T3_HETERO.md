# T3 异构脚本使用顺序

这份文档只讲 `artifacts/_shared/thc/scripts` 目录下这些 strict T3 脚本该怎么按顺序使用。

当前默认 placement 固定为：

- `jlmini_3 -> C1 -> first_shard`
- `linux124 -> C2`
- `jlmini_2 -> C3 -> last_shard`

当前默认 IP 固定为：

- `jlmini_2 -> 192.168.31.159`
- `jlmini_3 -> 192.168.31.51`
- `linux124 -> 172.31.100.124`

如果这三个 IP 不变，通常不需要手动改 cluster 文件内容。

## 1. 直接使用仓库里的 cluster 文件

默认工作目录按下面这个相对家目录路径理解：

```bash
~/repo/paper/bc-ra-paper
```

直接使用：

```bash
~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json
```

建议先看一眼内容：

```bash
sed -n '1,220p' ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json
```

## 2. 把同一份 cluster 文件同步到三台机器

三台机器都需要使用同一份 `artifacts/_shared/thc/config/hetero_qwen_cluster.json`。

当前仓库默认的稳定映射是：

- `jlmini_3 -> C1 -> mps + metal_8bit`
- `linux124 -> C2 -> cuda + bitsandbytes_8bit`
- `jlmini_2 -> C3 -> mps + float16 (quantization=none)`

这里故意没有让 `jlmini_2` 的 `C3` 继续使用 `metal_8bit`。当前 MPS/Metal 路径在“非首 shard 接收 hidden states”时会触发缺失 kernel，实测会报 `Kernel not found: affine_qmm_t_float_gs_64_b_8_alN_true_batch_1`。

只要保证内容一致，放置路径也可以变化，但后续命令要对应修改。

## 3. 每台机器先做环境检查

### 在 `jlmini_3`

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py check \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node jlmini_3
```

### 在 `linux124`

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py check \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node linux124
```

### 在 `jlmini_2`

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py check \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node jlmini_2
```

三台机器都通过后，再继续下一步。

## 4. 每台机器启动 shard server

每台机器各开一个终端，不要提前关闭。

### 在 `jlmini_3`

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py serve \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node jlmini_3
```

### 在 `linux124`

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py serve \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node linux124
```

### 在 `jlmini_2`

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py serve \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --local-node jlmini_2
```

正常情况下，三个终端会分别显示自己负责的 `checkpoint`、`device` 和 `quantization`。

## 5. 在协调机跑 capture

建议把 `jlmini_2` 继续作为协调机。

先跑一次：

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py capture \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --output-dir /tmp/thc_t3/hetero_run_a \
  --split calibration \
  --limit-prompts 0
```

然后再跑两次：

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py capture \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --output-dir /tmp/thc_t3/hetero_run_b \
  --split calibration \
  --limit-prompts 0
```

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py capture \
  --cluster-file ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/config/hetero_qwen_cluster.json \
  --output-dir /tmp/thc_t3/hetero_run_c \
  --split calibration \
  --limit-prompts 0
```

## 6. 生成 delta_map

在协调机 `jlmini_2` 上执行：

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py calibrate \
  --output-dir /tmp/thc_t3/delta_calibration_main \
  /tmp/thc_t3/hetero_run_a \
  /tmp/thc_t3/hetero_run_b \
  /tmp/thc_t3/hetero_run_c
```

核心产物是：

- `/tmp/thc_t3/delta_calibration_main/delta_map.json`

## 7. 如果想看帮助

交互式菜单：

```bash
bash ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/run_t3_hetero_console.sh menu
```

cluster 字段说明：

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py guide
```

探测当前机器 IP：

```bash
python3 ~/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/t3_hetero_cli.py detect-ip
```

## 8. 注意事项

- 三台机器必须能互相访问 cluster 文件里写的 `host:port`
- `linux124` 的 IP 现在是 `172.31.100.124`，请确认两台 Mac 能直接访问它的 `8311` 端口
- 如果 Linux 开了防火墙，需要放行对应端口
- `capture` 和 `calibrate` 的输出目录必须是新目录，不能复用旧目录

## 9. T3 完成后如何进入 T5

当你已经拿到：

- evaluation split 的真实异构 capture root
- T3 生成的 `delta_map.json`

可以直接运行：

```bash
CAPTURE_ROOT=/tmp/thc_t3/hetero_eval_run_a \
DELTA_MAP_FILE=/tmp/thc_t3/delta_calibration_main/delta_map.json \
RUNS_PER_MODE=10 \
bash /Users/jlmini_2/repo/paper/bc-ra-paper/artifacts/_shared/thc/scripts/run_t5_from_capture.sh
```

这个脚本会强制 `run.py` 从真实异构 capture root 读取 checkpoint bundle，并显式接入 T3 的 `delta_map.json`。
- 自动装包默认走清华 PyPI 镜像：`https://pypi.tuna.tsinghua.edu.cn/simple`
- 如果某台机器需要改镜像源，可以先导出 `THC_PIP_INDEX_URL=<你的镜像地址>` 再执行脚本
