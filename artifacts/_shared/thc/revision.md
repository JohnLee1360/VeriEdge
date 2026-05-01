# 修订实验说明

**先定“对象和接口”，再做“实现和校准”，最后做“评估和回填”**。这样前面的定义不稳，就不会污染后面的实验和写作。

## 当前 prototype stack 约束

本轮修订实验的真实异构执行环境固定为：

- `Provider A`: Mac mini M4
- `Provider B`: Mac mini M4
- `Provider C`: Linux server with Nvidia RTX 3090

三台设备位于**同一局域网 / 同一 Wi-Fi 环境**下，实验目标是做 **real 3-provider heterogeneous shard execution**，而不是单机 logical shard slicing。

这带来一个必须明确写死的实现约束：

- **不能再假设 3 台机器都走同一种 MLX distributed runtime**
- Mac M4 与 Linux 3090 属于**不同 backend 栈**
- 后续实现必须围绕“跨 backend 的 shard handoff / checkpoint capture / 数值校准”来设计
- 若要尽量贴近 `Qwen3-0.6B-8bit` 的实验目标，应采用“统一基础模型 `Qwen/Qwen3-0.6B` + 平台对应量化/精度加载参数”的替代方案，而不是假设三台机器共享同一个 MLX 8bit checkpoint
- 当前稳定配置固定为：`jlmini_3(C1)=mps+metal_8bit`，`linux124(C2)=cuda+bitsandbytes_8bit`，`jlmini_2(C3)=mps+float16`。`jlmini_2` 不使用 `metal_8bit`，因为当前 MPS/Metal 路径在非首 shard 接收 hidden states 时会触发缺失 kernel

因此，凡是只适用于 `MLX-only` 三机 pipeline 的方案，都**不满足**本版修订目标。

**总共6 个任务**。整体关系是：

**T1 → T2 → {T3, T4} → T5 → T6**

其中：

- **T1、T2、T5、T6 必须串行**
- **T3 和 T4 可以并行**
- 论文回填 **T6** 一定放最后

---

## 总体串并行关系

### 串行主链

1. **T1 机制与张量定义收束**
2. **T2 checkpoint 提取与阶段拆分实现**
3. **T3 Δ 校准实现**
4. **T4 采样与评估实现**
5. **T5 跑实验、选参数、出结果**
6. **T6 回填论文**

### 并行关系

- **T3 和 T4** 可以并行，因为：
  - T3 关注 honest heterogeneity calibration
  - T4 关注 sampling / metric / evaluation pipeline
- 但它们都要建立在 **T2 已经把 checkpoint 抽出来** 的前提下

---

# 任务拆分

## T1：先把机制定义彻底定死

这是最上游任务，也是最关键的任务。
目的就是让 codex 先别写太多代码，而是先把“到底测什么”完全固定下来。

### 任务目标

把 TSTC 的实验对象统一为：

- 验证单元：`Shard k`
- 模型场景：Qwen3-0.6B，3 shards，3 providers
- prototype deployment：`2 x Mac mini M4 + 1 x Linux 3090`
- checkpoint：
  - `C1`: Shard 1 输出给 Shard 2 之前
  - `C2`: Shard 2 输出给 Shard 3 之前
  - `C3`: Shard 3 的 final hidden output
- 阶段区分：
  - prefill：`[B, L, H]`
  - decode：`[B, 1, H]`

### 要求 codex 产出

1. 一份简短设计说明
2. 明确的数据结构/配置草案
3. checkpoint 命名规范
4. prefill / decode 的张量 shape 说明
5. 跨 backend shard handoff 的接口约束说明

### 依赖关系

- 无前置依赖
- 后面所有任务都依赖它

---

## T2：实现 checkpoint 抽取与阶段拆分

这个任务的本质是：把系统里真正能拿来做 TSTC 的张量先抽出来。

### 任务目标

在代码里实现：

- 3 个 shard 的切分
- 每个 shard 的输出 checkpoint 抽取
- 区分 prefill 与 decode 两种运行路径
- 默认 final checkpoint 取 final hidden，不取 logits
- 能适配 `Mac(M4/MLX) -> Mac(M4/MLX) -> Linux(3090/CUDA)` 的真实执行链路
- 明确 checkpoint / handoff tensor 的统一序列化格式、dtype 与 metadata

### 要求 codex 产出

1. 能输出 `C1 / C2 / C3`
2. 能分别保存 prefill 和 decode 的 checkpoint
3. 给出每个 checkpoint 的 shape 日志
4. 提供最小可运行 demo
5. 明确说明 demo 是否已经跨 backend 跑通，而不是只在单 backend 上 mock 出 shard

### 依赖关系

- 依赖 T1
- 是 T3 和 T4 的共同前置

---

## T3：实现 Δ 的异构校准流程

这个任务只关注一件事：
**在 honest heterogeneous execution 下，怎么测数值差，并把它变成 `Δ_k`。**

### 任务目标

基于3台机器（`2 x Mac mini M4 + 1 x Linux 3090`）做 calibration：

- 对同一模型、同一 shard plan、同一 prompt 集合**calibration split** 
- 在**真实 3-provider heterogeneous shard execution** 中收集 `C1 / C2 / C3` 的 honest-honest 数值差
- 分 prefill / decode 统计
- 输出每个 checkpoint 的 `Δ_k`

### 要求 codex 产出

1. calibration 脚本
2. 差值统计输出
3. 每个 checkpoint、每个阶段的 percentile 结果
4. 推荐的 `Δ_k` 配置文件
5. 明确记录每次 calibration run 的 provider placement / backend / device metadata

### 额外要求

- calibration 必须来自真实 `M4 + M4 + Linux3090` 协作执行
- 不允许用单机 full-model slicing 替代真实异构执行
- 如果需要多种 provider placement（例如 Linux 3090 分别承担 `Shard 1/2/3`）来估计位置敏感性，应在 T3 中单独记录并比较
- 由于 3 台设备运行在同一局域网 / Wi-Fi，网络时延波动可以记录，但 `Δ_k` 的对象仍然是**张量数值差**，不是 latency threshold

### 建议规则

- 默认用高分位点，比如 99th percentile
- 明确写成 prototype-level calibration，不做普适化解释
- `Δ_k` 的表述应绑定到当前 prototype stack：`2 x Mac mini M4 + 1 x Linux 3090`

### 请注意

- `Δ` 只允许在 **calibration split** 上估计；
- 不得使用后续正式评估所用的 prompts / runs / seeds；
- `Δ_k` 应被表述为 **under our prototype stack 的经验校准结果**，而不是普适阈值。
- 若不同 shard placement 的差值分布明显不同，应允许 `Δ_k` 进一步带上 placement/backend 条件，而不是强行压成单一全局阈值。

### 依赖关系

- 依赖 T2
- 可与 T4 并行

---

## T4：实现采样、摘要与评估管线

这个任务负责把 TSTC 本体搭起来。

### 任务目标

实现两套采样逻辑：

- **decode**：channel random sampling
- **prefill**：token-channel stratified random sampling

并实现：

- tolerance-aware quantization
- shard summary / chain commitment
- mismatch detection
- `TPR / LocAcc / FPR` 统计
- 面向真实异构 backend 输出的统一验证接口

### 要求 codex 产出

1. sampling 模块
2. quantization 模块
3. summary / verification 模块
4. evaluation 脚本
5. 参数搜索接口

### 搜索空间建议

围绕这些量级：

- `sample_count ∈ {16, 32, 48, 64, 96, 128}`

prefill 以 `(token_samples, channel_samples)` 组合表示，decode 直接用 channel count。

### 依赖关系

- 依赖 T2
- 可与 T3 并行
- T5 依赖它

---

## T5：正式跑实验并选参数

这是把 T3 和 T4 合起来，真正出结果的阶段。

### 任务目标

用：

- T3 给出的 `Δ_k`
- T4 的 sampling / verification pipeline
- 同一 prompt 集合**evaluation split**（T5 不得复用参与 `Δ_k` 估计的同一批 prompts / runs / seeds）

去跑完整实验，并输出：

- prefill / decode 分开的结果
- hetero 分档结果
- 最终选中的参数
- 且结果明确对应 `2 x Mac mini M4 + 1 x Linux 3090` 这一 prototype stack

### 要求 codex 产出

1. 结果表
2. 参数选择结果
3. 分阶段指标
4. 分 hetero level 的 FPR
5. 结果解释草稿

### 当前叙事要求

- `FPR <= 0.2` 可以作为工作筛选门槛
- 但结果解释不能写成强保证
- 结论要收束为：
  - 比 strict THC 误报更低
  - 能做 shard-level suspicion localization
  - 是 tolerance-aware verifier，不是 strict correctness proof
  - 结论只对当前 `M4 + M4 + Linux3090` prototype stack 成立，不外推到所有异构部署

### 依赖关系

- 依赖 T3 和 T4
- 是 T6 的前置

---

## T6：最后再回填论文

这个任务一定最后做。
不要在 T5 结果没稳定之前就改论文正文。

### 任务目标

根据最终实验结果，回填三部分：

1. 机制描述
2. 实验设置
3. 结果与局限性

### 要求 codex 产出

1. `Shard k` 版机制段落
2. 实验设置段落
3. 结果分析段落
4. limitation / scope 段落

### 写作约束

要明确写出：

- 3-shard / 3-provider 场景
- checkpoint 位于 shard 边界和 final hidden output
- prefill / decode 分开
- `Δ_k` 来自异构实测校准
- prototype stack 是 `2 x Mac mini M4 + 1 x Linux 3090`
- 实验建立在跨 backend 的真实 shard execution 上，而不是单机逻辑切片
- 不做 deployment-level strong claim

### 依赖关系

- 依赖 T5
- 必须最后执行

---

# 最推荐的执行顺序

你可以按下面这个顺序一条条发给 codex：

### 第一轮

发 **T1**
先让它把定义定死，不要急着写大段代码。

### 第二轮

发 **T2**
让它先把 checkpoint 抓出来，并打印 shape。

### 第三轮

把任务分成两条支线：

- 一条发 **T3**
- 一条发 **T4**

这两条可以分别开两个 codex 任务做。

### 第四轮

等 T3、T4 都完成后，发 **T5**
让它整合起来跑正式实验。

### 第五轮

最后发 **T6**
回填论文。

---

# 你可以直接这样理解依赖图

**Phase A：定义**

- T1

**Phase B：基础实现**

- T2

**Phase C：两条并行支线**

- T3：校准 Δ
- T4：采样与评估

**Phase D：整合出结果**

- T5

**Phase E：写论文**

- T6

---

# 建议

要求必要时输出：

- 修改了什么
- 新增了哪些文件
- 依赖什么前置结果
- 下一步建议做什么

这样我才能很好地控制上下文，不会让你越写越散。
