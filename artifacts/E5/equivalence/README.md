# EXO 功能等价性实验工程说明

本目录现在不再使用统一的 `run.py` 总控脚本，而是拆成三个独立实验入口：

- `run_1device.py`：只让 `Mac mini2` 进入 `equiv-1device`
- `run_2device.py`：只让 `Mac mini1 + Mac mini3` 进入 `equiv-2device`
- `run_3device.py`：使用 `equiv-3device`

其中 `run_1device.py` 与 `run_2device.py` 设计为可并发执行，因为它们从 EXO 的发现层开始就使用不同的 `EXO_LIBP2P_NAMESPACE`，不是在同一个 cluster 里抢 instance。

## 目录结构

- `config.example.yaml`：三组实验的统一配置模板
- `equivalence_common.py`：纯工具函数，负责采样、结果格式、状态解析、评分辅助
- `run_1device.py`：1-device 集群编排与实验运行
- `run_2device.py`：2-device 集群编排、代码同步、freeze 校验与实验运行
- `run_3device.py`：3-device 校验 / 可选编排与实验运行
- `score.py`：答案提取与规范化
- `plot.py`：聚合三份独立 `results.json` 并生成总图
- `tests/`：回归测试

## 运行方式

先做单组检查：

```bash
python3 artifacts/E5/equivalence/run_1device.py --config artifacts/E5/equivalence/config.example.yaml --check-only
python3 artifacts/E5/equivalence/run_2device.py --config artifacts/E5/equivalence/config.example.yaml --check-only
python3 artifacts/E5/equivalence/run_3device.py --config artifacts/E5/equivalence/config.example.yaml --check-only
```

正式运行：

```bash
python3 artifacts/E5/equivalence/run_3device.py --config artifacts/E5/equivalence/config.example.yaml --resume
```

随后可并发运行：

```bash
python3 artifacts/E5/equivalence/run_1device.py --config artifacts/E5/equivalence/config.example.yaml --resume
python3 artifacts/E5/equivalence/run_2device.py --config artifacts/E5/equivalence/config.example.yaml --resume
```

如果你希望实验结束后保留 instance 或 cluster：

```bash
python3 artifacts/E5/equivalence/run_2device.py \
  --config artifacts/E5/equivalence/config.example.yaml \
  --resume \
  --keep-instance \
  --keep-cluster
```

`run_2device.py` 默认会同步 `artifacts/E5/equivalence/` 到远端实验仓库；如需跳过：

```bash
python3 artifacts/E5/equivalence/run_2device.py \
  --config artifacts/E5/equivalence/config.example.yaml \
  --skip-sync
```

## 输出文件

每个脚本各自产出自己的结果：

- `artifacts/E5/equivalence/output/1device/results.json`
- `artifacts/E5/equivalence/output/2device/results.json`
- `artifacts/E5/equivalence/output/3device/results.json`

三份结果写完后，再聚合出总图：

```bash
python3 artifacts/E5/equivalence/plot.py \
  --results-root artifacts/E5/equivalence/output
```

输出：

- `artifacts/E5/equivalence/output/accuracy_comparison.png`

## 关键设计点

- `1-device` 与 `2-device` 用不同的 `EXO_LIBP2P_NAMESPACE`
- 每组还使用独立的 `EXO_HOME`，避免 node identity / event log / cache 串扰
- 每题完成后立即落盘，支持 `--resume`
- 不依赖黑盒自动挑 instance，而是先校验 cluster IP 集合，再用 `/instance/previews` 选择目标 placement
