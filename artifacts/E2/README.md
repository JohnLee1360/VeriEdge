# E2: Verification Detection Quality

## Goal

E2 回答的问题是：THC/TSTC verifier 在 honest heterogeneity、tamper 和 numeric perturbation 下的检测质量是否支持 VeriEdge 的 verification-aware orchestration 主张。

核心指标包括 `TPR`、`FPR`、`Localization Accuracy` 和 runtime。

## Design

E2 由两条路径组成：

- 主 ablation：复用 `artifacts/_shared/thc/`，比较 THC/TSTC 在不同场景和 checkpoint 上的表现。
- `prefill/C2` noise sweep：使用 `artifacts/E2/tstc/`，观察 TSTC 对 honest numeric perturbation 强度的响应。

默认对象：

- 模型：`Qwen3-0.6B`
- 验证单元：`Shard k`
- checkpoints：`C1`、`C2`、`C3`
- stages：`prefill`、`decode`
- split：正式评估只用 `evaluation`

## Run

```bash
bash artifacts/E2/run_ablation.sh --help
bash artifacts/E2/run_noise_sweep.sh --help
```

常规入口：

```bash
bash artifacts/E2/run_ablation.sh
bash artifacts/E2/run_noise_sweep.sh
```

## Outputs

- 轻量过程结果：`artifacts/E2/results/`
- ablation 原始结构化结果：`artifacts/E2/commits/logs/`
- noise sweep run：`artifacts/E2/commits/logs/noise_sweep_<stamp>/`
- 主稿候选表：`artifacts/E2/commits/tables/`
- 候选图：`artifacts/E2/commits/figures/`
- 运行说明：`artifacts/E2/commits/notes/`

建议命名：

```text
exp_e2_<yyyymmdd>_<owner>_samplesweep.csv
exp_e2_<yyyymmdd>_<owner>_tolerancesweep.csv
exp_e2_<yyyymmdd>_<owner>_runtime.csv
```

## Do Not

- 不只报 FPR，必须同时检查 TPR、localization 和 runtime
- 不把单个 sweep 当成完整 E2 结论
- 不把运行目录、清洗表和主稿候选图混成一层
- 不把 `calibration` 数据用于正式 verifier 评估

## Acceptance

一次 E2 交付至少包含：原始结构化结果、清洗后的候选表或图、`notes/` 中的一页说明。说明里必须写清场景、split、repetitions、指标口径和当前结果是否支持主稿叙事。
