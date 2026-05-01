# Wenxin Onboarding

这份文档只帮助你快速接手仓库；实验细节以 `artifacts/Ex/README.md` 为准。

## Read Order

1. `README.md`
2. `AGENTS.md`
3. 本文档
4. 你负责实验的入口：
   - `artifacts/E1/README.md`
   - `artifacts/E2/README.md`
   - `artifacts/E5/README.md`

## Mental Model

- `artifacts/` 回答“怎么跑、代码在哪、实验方案是什么”。
- `artifacts/Ex/commits/` 回答“哪些东西可以交给主稿或老师看”。
- `workspace/` 回答“本机重型运行产物临时放哪里”。
- `docs/` 只放必要背景和协作说明。

## Ownership

- John：优先维护 E4 与 `artifacts/_shared/thc/`。
- Wenxin：优先推进 E1、E2、E5，并复用共享实现。
- 谁改共享实现，谁同步更新受影响实验 README，并在对应 `artifacts/Ex/commits/notes/` 写清影响。

## What To Commit

可以提交：

- `artifacts/` 下需要共享的脚本、配置、测试和实验说明。
- `artifacts/Ex/commits/logs|tables|figures|notes/` 下可用于主稿或汇报的交付物。
- 与改动同步的 `README.md`、`AGENTS.md` 或 `docs/`。

不要提交：

- `workspace/` 下的大型临时运行目录。
- 模型权重、密钥、私钥、SSH 凭据、本机配置。
- 没有结构化结果支撑的截图。

## Naming

正式结果建议统一：

```text
exp_e<id>_<yyyymmdd>_<owner>_<suffix>
```

例子：

```text
exp_e2_20260501_wenxin_runtime.csv
exp_e5_20260501_wenxin_policy_compare.csv
```

## Before Sharing Results

检查四件事：

1. 结果是否放在正确实验的 `artifacts/Ex/commits/`。
2. `logs/` 是否有结构化原始结果。
3. `tables/` 或 `figures/` 是否是清洗后的候选交付物。
4. `notes/` 是否写清输入、命令、输出、结论和剩余风险。
