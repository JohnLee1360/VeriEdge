# VeriEdge Paper1 Collaboration Repo

这个仓库只做一件事：把 VeriEdge Paper1 的四组实验代码、运行入口和正式交付物放在协作者与 AI 都容易接手的位置。

核心主张是 `verification-aware orchestration`。不要把实验扩回资源市场、pricing 或 welfare 叙事。

## Where To Start

Wenxin 第一次接手时按这个顺序读：

1. `README.md`
2. `AGENTS.md`
3. `docs/wenxin_onboarding.md`
4. 自己负责实验的 `artifacts/Ex/README.md`

## Repo Layout

```text
AGENTS.md
README.md
docs/
artifacts/
  E1/
    commits/
  E2/
    commits/
  E4/
    commits/
  E5/
    commits/
  _shared/
workspace/
```

- `docs/`：必要背景、协作说明和模板。
- `artifacts/`：实验入口、实验方案、源代码和轻量过程结果。
- `artifacts/_shared/thc/`：E1/E2/E4 共用 verifier 实现。
- `artifacts/Ex/commits/`：每个实验自己的正式交付物目录。
- `workspace/`：本地重型运行产物，不作为正式交付目录。

## Experiment Entrypoints

- `E1`：`artifacts/E1/README.md`，真实异构 shard execution 与 checkpoint capture。
- `E2`：`artifacts/E2/README.md`，THC/TSTC detection quality、ablation 和 noise sweep。
- `E4`：`artifacts/E4/README.md`，verification overhead。
- `E5`：`artifacts/E5/README.md`，黑盒 EXO 端到端 orchestration。

## Result Placement

每个正式交付目录固定为：

```text
artifacts/Ex/commits/
  logs/
  tables/
  figures/
  notes/
```

- `logs/`：结构化原始结果。
- `tables/`：清洗后的主稿候选表。
- `figures/`：候选图。
- `notes/`：一页说明、运行记录或周报。

如果文件还不能直接交给老师或回写主稿，默认先放 `workspace/`。

## Minimal Flow

```bash
git checkout -b feat/e2-sample-sweep
bash artifacts/E2/run_ablation.sh --help
```

运行后把可协作结果整理到 `artifacts/Ex/commits/`，并在 `notes/` 写清：做了什么、输入是什么、输出在哪里、结果是否可进主稿、还缺什么。
