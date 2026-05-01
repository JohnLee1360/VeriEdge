# E1: Heterogeneous Checkpoint Capture

## Goal

E1 回答的问题是：VeriEdge 的 verifier 能否在真实 `3 providers / 3 shards` 异构执行路径上获得可复用的 shard boundary checkpoint。

本实验产出 `C1/C2/C3` checkpoint capture 与 pairwise/summary 结果，服务后续 detection、calibration 和 overhead 分析。

## Design

- 模型：`Qwen/Qwen3-0.6B`
- 执行路径：strict T3，三台设备真实 shard execution，不做单机逻辑切片替代
- providers：两台 Mac M4 与一台 Linux 3090
- checkpoints：`C1`、`C2`、`C3`
- splits：`calibration` 只用于 delta 校准，`evaluation` 只用于正式评估
- 共享实现：`artifacts/_shared/thc/`

## Run

```bash
bash artifacts/E1/run_capture.sh --help
bash artifacts/E1/run_capture.sh --split calibration
```

默认重型 capture root：

```text
workspace/captures/E1/
```

## Outputs

- 轻量过程结果：`artifacts/E1/results/`
- 重型 capture、中间 JSON、`.npz/.npy`：留在 `workspace/captures/E1/`
- 可交付结构化结果：`artifacts/E1/commits/logs/`
- 主稿候选表：`artifacts/E1/commits/tables/`
- 候选图：`artifacts/E1/commits/figures/`
- 运行说明：`artifacts/E1/commits/notes/`

建议命名：

```text
exp_e1_<yyyymmdd>_<owner>_summary.csv
exp_e1_<yyyymmdd>_<owner>_pairwise_details.csv
```

## Do Not

- 不把 `.npz/.npy` capture bundle 直接提交到 `artifacts/E1/commits/logs/`
- 不混用 `calibration` 与 `evaluation`
- 不静默修改 cluster file、shard plan 或 prompt split
- 不复制 `artifacts/_shared/thc/` 代码

## Acceptance

一次 E1 交付至少包含：结构化日志、可解释的 summary 或 pairwise 表、`notes/` 中的一页运行记录。记录里必须写清 split、cluster file、capture root 和是否可被 E2/E4 复用。
