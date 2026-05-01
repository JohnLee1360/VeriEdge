# E4: Verification Overhead

## Goal

E4 回答的问题是：VeriEdge verifier 在存储、传输和计算/延迟上引入多少额外成本，这些成本是否仍在可接受范围内。

本实验不证明检测效果；检测质量由 E2 支撑。E4 只负责 overhead。

## Design

E4 复用 `artifacts/_shared/thc/` 的 capture metadata、hash/checkpoint 结构和配置，生成：

- size breakdown
- latency breakdown
- storage breakdown
- summary table

输入通常来自 E1/E2 的 capture root 或共享配置。

## Run

```bash
bash artifacts/E4/run_overhead.sh --help
```

示例：

```bash
bash artifacts/E4/run_overhead.sh --capture-root /path/to/capture_root --owner johnlee
```

默认正式日志位置：

```text
artifacts/E4/commits/logs/<yyyymmdd>_<owner>/
```

## Outputs

轻量过程结果可放 `artifacts/E4/results/`。正式输出包括：

```text
exp_e4_<yyyymmdd>_<owner>_size_breakdown.csv
exp_e4_<yyyymmdd>_<owner>_latency_breakdown.csv
exp_e4_<yyyymmdd>_<owner>_storage_breakdown.csv
exp_e4_<yyyymmdd>_<owner>_summary.csv
```

主稿候选表放 `artifacts/E4/commits/tables/`，运行说明放 `artifacts/E4/commits/notes/`。

## Do Not

- 不只写“大概多少”
- 不提交没有单位的 overhead 表
- 不把估算值写成实测值
- 不把 E4 写成 detection accuracy 实验

## Acceptance

一次 E4 交付至少包含：带单位的结构化 CSV、主稿候选表或摘要表、`notes/` 中的一页说明。说明里必须区分 measured、derived 和 estimated，并写清 capture root 或配置来源。
