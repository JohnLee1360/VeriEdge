# EXO 功能等价性实验设置

## 1. 科学目标

本实验用于验证：当同一个模型通过 EXO 部署在不同设备数上时，其功能正确性是否保持稳定。

更具体地说，我们比较 `1-device exo`、`2-device exo`、`3-device exo` 三种部署下，模型在 GSM8K 测试集上的 exact-match accuracy。

## 2. 实验边界

本实验只讨论功能正确性，不扩展到以下内容：

- 性能
- 吞吐
- 通信开销
- 数值一致性
- 系统级稳定性指标

因此，本实验的核心输出只有三组 accuracy、逐题预测结果，以及一张 accuracy 对比图。

## 3. 实验设置

比较三组部署：

- `1-device exo`
- `2-device exo`
- `3-device exo`

当前实例绑定如下：

- `1-device`：`c2c93e74-f223-4115-a7e2-1af9e733f37a`
- `2-device`：`7f48e1b7-8736-4b31-a420-843e55e1e84e`
- `3-device`：`a3ba1902-6b77-46c4-9212-c3d3a72c424b`

当前已验证的 first-shard 信息如下：

- `1-device` first-shard：`192.168.31.159`
- `2-device` first-shard：`192.168.31.135`
- `3-device` first-shard：`192.168.31.135`

模型固定为：

- `mlx-community/Qwen3-0.6B-8bit`

## 4. 数据集与采样协议

数据集使用：

- `GSM8K`
- `test split`

抽样协议如下：

- 固定随机种子
- 从 GSM8K test split 中随机抽样 `100` 题
- 三组实验使用完全相同的题目集合

同一批样本会被落盘保存，后续重跑或补跑时继续复用，避免跨组题目漂移。

## 5. Prompt 与解码协议

每道题使用简单 zero-shot prompt。

模型输出的最后一行必须满足：

```text
Final Answer: <answer>
```

固定解码参数为：

- `temperature=0`
- `top_p=1`
- `max_tokens=512`

为了避免不同 setting 使用不同请求模式带来额外变量，三组实验采用同一种 OpenAI 兼容推理接口与同一套解码参数。

## 6. 评分协议

对每道题，优先从模型输出中提取：

```text
Final Answer: ...
```

若该行缺失，则允许从全文做受限 fallback 提取，但会在结果文件中记录提取来源。

答案处理流程如下：

1. 提取预测答案
2. 做基础规范化
3. 从 GSM8K 标准答案中提取最终数值答案
4. 使用 exact match 计算是否正确

最终汇总三个指标：

- `accuracy_1device`
- `accuracy_2device`
- `accuracy_3device`

## 7. 运行协议

正式运行顺序固定为：

1. 先做全局 preflight
2. 并发运行 `1-device` 与 `2-device`
3. 等前两组完成后，再顺序运行 `3-device`

工程层面的并发协议如下：

- `1-device` 请求固定打到 `192.168.31.159`
- `2-device` 请求固定打到 `192.168.31.135`
- 两组共用同一份 100 题样本
- 每组内部单线程串行完成本组 100 题
- 并发阶段开始前记录一次 `/state` 摘要
- 并发阶段结束后再记录一次 `/state` 摘要
- `3-device` 阶段运行前后也分别记录 `/state` 摘要

每组开始前都需要完成 preflight，至少检查：

- `/state` 可访问
- 目标 instance id 存在
- 目标 instance 的 node count 正确
- first-shard IP 与预期一致
- 模型 id 为 `mlx-community/Qwen3-0.6B-8bit`

## 8. 结果解释约束

本实验保留黑盒 EXO 路线。

由于 EXO 公共 API 不支持在请求层显式绑定 `instance_id`，因此：

- `1-device` 与 `2-device` 的并发阶段采用 `best-effort` 黑盒并发路由
- 结果中必须记录运行前后 `/state` 证据
- 不应把并发阶段表述为“绝对强隔离的 instance routing”

换言之，本文可以报告：

- 在当前黑盒 EXO 部署与观测条件下，三组设置的功能正确率表现
- 并发阶段的 instance 映射证据

但不应报告：

- 请求在协议层被严格绑定到指定 instance
- 并发阶段不存在任何路由歧义可能性

`3-device` 由于在前两组之后顺序运行，其可解释性高于并发阶段，可在结果讨论中单独说明。

## 9. 输出文件

本实验保存以下核心产物：

- `results.json`
- `accuracy_comparison.png`

其中 `results.json` 需要包含：

- 每道题在三组设置下的输出
- 每道题的预测答案
- 每道题的正确性标签
- 三组 accuracy 汇总
- 并发与顺序阶段的 routing evidence

## 10. 图表要求

本实验只生成一张图：

- `accuracy_comparison.png`

图类型为柱状图：

- x 轴：`1-device exo / 2-device exo / 3-device exo`
- y 轴：`GSM8K Accuracy`

风格要求：

- 简洁
- 现代
- 学术化
- 高分辨率
- 柱顶显示数值
- 留白、字体、网格线、配色协调

该图应适合直接放入论文初稿，而不需要额外重画。
