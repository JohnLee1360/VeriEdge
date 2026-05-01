# TSTC `C2` Noise Sweep

这个目录用于运行一个更聚焦的补充实验：

- 目标：验证 TSTC 对 `prefill/C2` honest numeric perturbation 的响应能力
- 模型：`Qwen3-0.6B`
- 环境：`1 x Mac mini`
- 切分：`3 shards`
- verifier：`TSTC`
- 重复次数：每个噪声点默认 `3000` 次

## 文件说明

- `TSTC.md`：实验设计与论文口径
- `noise_sweep_config.json`：默认 sweep 配置
- `run_noise_sweep.py`：运行脚本

## 运行前提

建议在仓库根目录执行：

```bash
cd /Users/jlmini_2/repo/paper/bc-ra-paper
```

默认 Python：

```bash
/Users/jlmini_2/repo/paper/bc-ra-paper/.venv/bin/python3
```

## 运行模式

脚本支持两种 clean reference 来源：

1. 直接本地 capture
2. 从已有 `capture_root` 读取

如果你已经用 `artifacts/_shared/thc` 产出了某个 prompt 的 clean capture bundle，优先复用 `--capture-root`。如果没有，脚本会按配置直接抓一次本地 clean checkpoint，然后在内存中重复注入噪声。

## 默认实验命令

完整 sweep：

```bash
/Users/jlmini_2/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/E2/tstc/run_noise_sweep.py \
  --sweep-config artifacts/E2/tstc/noise_sweep_config.json
```

如果已有真实 capture root：

```bash
/Users/jlmini_2/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/E2/tstc/run_noise_sweep.py \
  --sweep-config artifacts/E2/tstc/noise_sweep_config.json \
  --capture-root /path/to/capture_root \
  --prompt-id <prompt_id>
```

## Smoke Run

先做一个小规模检查：

```bash
/Users/jlmini_2/repo/paper/bc-ra-paper/.venv/bin/python3 artifacts/E2/tstc/run_noise_sweep.py \
  --sweep-config artifacts/E2/tstc/noise_sweep_config.json \
  --repetitions 5 \
  --use-mock-if-unavailable true
```

这个命令用于确认：

- 脚本能抓到 clean reference
- `C2` 噪声注入逻辑正常
- summary / trial / figure 都能产出

## 主要参数

- `--sweep-config`：实验配置文件
- `--capture-root`：已有 capture root，可选
- `--prompt-id`：配合 `capture_root` 使用
- `--repetitions`：覆盖配置里的重复次数
- `--output-dir`：指定输出目录
- `--use-mock-if-unavailable`：本地无模型时是否允许 mock fallback

## 输出内容

默认输出到：

```bash
artifacts/E2/tstc/output/<run_id>/
```

其中包括：

- `run_meta.json`
- `trial_results.jsonl`
- `summary.csv`
- `noise_sweep_detection_count.png`

论文图默认只保存在当前 run 目录内，不再同步写到 `paper/v2/img/`。

## 输出字段

`summary.csv` 至少包含：

- `noise_std`
- `effective_noise_std`
- `sigma_ref`
- `repetitions`
- `detected_count`
- `detection_rate`
- `c2_first_mismatch_count`

## 工程约束

- 本实验只看 `prefill`
- 只扰动 `C2`
- 默认只跑 `TSTC`
- 默认采样规模是动态比例采样
- `token_samples = ceil(L / 8)`
- `channel_samples = ceil(H / 8)`
- 抽样方式是带 seed 的均匀无放回抽样

如果后续要扩展为：

- 比较 `THC`
- 改 `C1/C3`
- 多 prompt

建议另开独立脚本，不要直接把这次补充实验的输出语义混在一起。
