# E5: Verification-Aware Orchestration

## Goal

E5 回答的问题是：在黑盒 EXO 任务级链路中，verification-aware orchestration 的端到端体验和调度指标是否可接受。

它关注 task latency、download time、throughput、TTFT、OTPS、success、challenge 与 verifier workload，不讨论 pricing 或 welfare。

## Design

E5 保留黑盒 EXO 路线：

- requester 构造任务、加密、上传到本地 Kubo/IPFS
- provider launcher 接收 first-shard task
- EXO instance 负责模型执行
- requester callback 汇总结果

实验矩阵：

- `LAN`: `instance_node_count in {1, 2, 3}`
- `WAN`: `instance_node_count = 3`

功能正确性 sanity check 放在 `artifacts/E5/equivalence/`；端到端 orchestration 主线放在 `artifacts/E5/inference_e2e/`。

## Run

```bash
bash artifacts/E5/run_matrix.sh --help
bash artifacts/E5/build_policy_table.sh
```

常规入口：

```bash
bash artifacts/E5/run_matrix.sh
bash artifacts/E5/build_policy_table.sh /path/to/summary_by_cell.csv
```

默认 batch run：

```text
workspace/runs/E5/
```

## Outputs

- 轻量过程结果：`artifacts/E5/results/`
- 重型 batch run：`workspace/runs/E5/`
- cell summary、task summary、运行日志：整理到 `artifacts/E5/commits/logs/`
- policy compare 表：`artifacts/E5/commits/tables/`
- 候选图：`artifacts/E5/commits/figures/`
- 运行说明：`artifacts/E5/commits/notes/`

建议命名：

```text
exp_e5_<yyyymmdd>_<owner>_policy_compare.csv
exp_e5_<yyyymmdd>_<owner>_policy_config.json
```

## Do Not

- 不只报 latency，必须同时看 success、challenge、verifier workload 和吞吐类指标
- 不把 E5 扩展成 pricing、welfare 或资源市场实验
- 不把整个 batch run 目录当正式交付物
- 不修改 EXO 本体来“修”本实验结果；本实验默认 EXO 是黑盒外部依赖

## Acceptance

一次 E5 交付至少包含：`summary_by_cell.csv` 或等价结构化日志、policy compare 表、`notes/` 中的一页运行记录。记录里必须写清 LAN/WAN 条件、目标 instance node count、EXO freeze 状态和失败 cell。
