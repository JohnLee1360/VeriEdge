# VeriEdge Paper1 Agent Protocol

本仓库只服务 VeriEdge Paper1 的四组核心实验：`E1`、`E2`、`E4`、`E5`。所有代码、文档和结果都要回到系统稿主张：`verification-aware orchestration`。

## Directory Contract

- `docs/`：给 Wenxin 和协作者看的必要背景、交接说明和模板；不做论文知识库。
- `artifacts/E1|E2|E4|E5/`：每个实验的目的、方案、运行入口、轻量过程结果和实验内代码。
- `artifacts/_shared/`：跨实验共享实现。`artifacts/_shared/thc/` 是 E1/E2/E4 的唯一 verifier/capture/calibration/overhead 实现，禁止复制到实验目录。
- `artifacts/E1|E2|E4|E5/commits/`：可用于主稿、汇报或交给老师看的正式交付物，只放 `logs/`、`tables/`、`figures/`、`notes/`。
- `workspace/`：本地重型运行产物、capture root、batch run 和临时缓存；默认不作为协作交付目录。

## Canonical Commands

```bash
bash artifacts/E1/run_capture.sh --help
bash artifacts/E2/run_ablation.sh --help
bash artifacts/E2/run_noise_sweep.sh --help
bash artifacts/E4/run_overhead.sh --help
bash artifacts/E5/run_matrix.sh --help
bash artifacts/E5/build_policy_table.sh
```

```bash
python3 -m unittest discover -s artifacts/_shared/thc/tests -q
python3 -m pytest artifacts/E5/equivalence/tests -q
python3 -m pytest artifacts/E5/inference_e2e/requester/tests -q
python3 -m pytest artifacts/E5/inference_e2e/provider/tests -q
```

## Forbidden Zones

- 不提交密钥、token、私钥、SSH 凭据、本机配置、模型权重、`.DS_Store`、大体积 capture 或 batch run。
- 不混用 `calibration` 与 `evaluation` 数据；目录、文件名、notes 必须写清 split。
- 不复制 `artifacts/_shared/thc/` 到各实验目录；共享逻辑只允许单一实现。
- 不把 E5 带回 pricing、welfare 或资源市场叙事；E5 只沿着 verification-aware orchestration 和黑盒 EXO 任务级链路推进。
- 不只提交截图；结构化 `csv/json/jsonl/md` 结果优先。
- 不把估算值写成实测值，不提交没有单位的 overhead 表。

## Acceptance Standard

每次改动至少满足：

- 能说明本次改动服务哪个实验或共享模块。
- 能指出运行入口、输入、输出位置和验证命令。
- 正式实验交付必须包含 `logs/`、`tables/` 或 `figures/`，以及 `notes/` 中的一页说明或运行记录。
- 修改共享代码时，同步更新受影响实验的 `artifacts/Ex/README.md` 或 `docs/wenxin_onboarding.md`。
- 改完代码主动跑可承受的验证；如果没跑，必须说明原因。

## Rule Update

同一类错误第二次出现时，先更新 `AGENTS.md` 或对应实验 README，把规则写清楚，再继续实践。规则比临时口头约定可靠。
