# T1 设计冻结说明

这份文档用于冻结本轮 THC/TSTC 修订实验的对象模型，避免后续实现再次漂移。

## 验证单元

- 验证单元：`Shard k`
- 模型：`Qwen3-0.6B`
- 分片方案：`3 shards / 3 providers`

## Checkpoint 定义

verifier 只作用在 shard 边界 checkpoint 上：

- `C1`：`Shard 1` 输出给 `Shard 2` 之前
- `C2`：`Shard 2` 输出给 `Shard 3` 之前
- `C3`：`Shard 3` 的 final hidden output

默认 final checkpoint 取 final hidden states，不把 logits 放进主 verifier 路径。

## 阶段定义

- `prefill`：张量 shape 为 `[B, L, H]`
- `decode`：张量 shape 为 `[B, 1, H]`

后续所有输出、日志、结果行、论文表述都统一使用：

- `stage in {prefill, decode}`
- `checkpoint in {C1, C2, C3}`

## Prompt Split

prompt 数据集必须固定拆成两部分：

- `calibration`
- `evaluation`

`delta` 校准只允许读取 `calibration` split。正式 verifier 评估只允许读取 `evaluation` split。

## 配置接口

配置层至少需要暴露：

- `experiment.prompt_dataset`
- `experiment.shards`
- `qwen.model_id`
- `tstc.prefill`
- `tstc.decode`
- `tstc.delta_map`
- `calibration.percentile`

## 命名规则

- 如果必须使用平铺 schema，阶段相关字段只允许使用 `prefill_*` 和 `decode_*`
- 更推荐统一的标准结果字段：
  - `prompt_id`
  - `split`
  - `stage`
  - `checkpoint`
  - `shape`
  - `hetero_level`
  - `sampling_spec`
  - `delta_used`

## 运行时默认口径

- 旧的单机 MLX 路径：`~/.exo/models/mlx-community--Qwen3-0.6B-8bit`
- strict T3/T5 的异构路径：`Qwen/Qwen3-0.6B`
  - Mac M4 上使用 `torch+mps`
  - `jlmini_3 (C1)`：`metal_8bit`, `bits=8`, `group_size=64`
  - `jlmini_2 (C3)`：`quantization=none`, `torch_dtype=float16`
  - Linux 3090 上使用 `torch+cuda`
  - Linux 3090 上的 8bit 近似参数：`bitsandbytes_8bit`, `bits=8`
