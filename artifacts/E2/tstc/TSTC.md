# TSTC 补充实验设计：`C2` 数值扰动鲁棒性

## 1. 实验目标

本补充实验的问题是：

> 当异构执行中的诚实 provider 仅在 `C2` checkpoint 上承受小幅数值扰动时，TSTC 能否保持对该类 honest numeric perturbation 的可检测响应，并给出随扰动强度变化的检测数量曲线。

这里我们关注的是 **TSTC 对数值扰动强度的响应特性**，而不是再次复现完整的 `THC vs. TSTC` 主实验矩阵。

## 2. 实验环境

本实验固定为单机原型环境：

- 机器：`1 x Mac mini`
- 模型：`Qwen3-0.6B`
- 模型切分：固定 `3 shards`
- checkpoint：固定 `C1 / C2 / C3`

对应切分含义：

- `C1`：`Shard 1` 输出
- `C2`：`Shard 2` 输出
- `C3`：`Shard 3` 输出

默认实现上，我们仍以 `Qwen3-0.6B` 的 layer boundary capture 作为 clean reference，并将其视为单机三分片执行的 checkpoint 边界张量。

## 3. 验证对象

本实验只看：

- 阶段：`prefill`
- 被扰动 checkpoint：`C2`
- verifier：`TSTC`


## 4. 主变量

### 4.1 扰动定义

对 `prefill/C2` 注入零均值高斯噪声：

$$
\epsilon_k \sim \mathcal{N}(0, s^2),
$$

其中 `s` 是噪声标准差。

本实验中的 `s` 采用 **归一化坐标** 定义。具体地：

1. 先按 TSTC 当前固定的结构化采样策略，从真实 `C2` 提取采样子向量
   $$
   u_{C2} = S_{\Omega_{C2}}(C2).
   $$
2. 计算该采样子向量的经验标准差
   $$
   \sigma_{\text{ref}} = \operatorname{Std}(u_{C2}).
   $$
3. 在归一化空间中注入
   $$
   \hat{\epsilon} \sim \mathcal{N}(0, s^2),
   $$
   再映射回原张量尺度：
   $$
   \epsilon = \sigma_{\text{ref}} \cdot \hat{\epsilon}.
   $$

因此，脚本实际写入 `C2` 张量的 raw-space 噪声标准差为：

$$
\sigma_{\text{raw}} = s \cdot \sigma_{\text{ref}}.
$$

这个定义与“先用真实 checkpoint 的结构化采样子向量统计值做归一化，再系统性变化噪声”的实验意图一致。

### 4.2 扰动语义

这里的噪声用于模拟：

- `honest numeric perturbation`

它表示诚实执行中由数值表示、设备内核、执行路径或张量舍入差异引入的小幅非恶意偏移。

## 5. 固定条件

本补充实验固定：

- prompt：固定 `1` 条简单 prompt
- 重复次数：每个 `s` 运行 `3000` 次
- model split：固定 `3 shards`
- stage：固定 `prefill`
- 扰动位置：固定 `C2`
- verifier：固定 `TSTC`

默认 prompt 在工程配置中写为：

```text
A rectangular garden is surrounded by a uniform-width concrete walkway that runs along all four sides of the garden. The total area covered by the garden and the walkway together is 336 square meters, and the area of the garden alone is 180 square meters. The length of the garden is 3 meters longer than twice its width.
Answer the following questions step by step:

1. Let $$x$$ represent the width of the garden in meters, and $$w$$ represent the uniform width of the concrete walkway in meters. Write an algebraic expression for the length of the garden in terms of $$x$$, and write two separate equations: one for the area of the garden alone, and one for the total area of the garden plus the walkway.

2. Calculate the exact length and width of the garden (in meters), showing all your simplification and solving steps for the quadratic equation you will form.

3. Find the uniform width of the concrete walkway, rounding your final answer to the nearest tenth of a meter if necessary. Explain why you reject any negative solution for the width of the walkway in the context of this real-world problem.

4. A local gardener suggests replacing half of the walkway with flower beds, while keeping the other half of the walkway at its original uniform width. Calculate the new total area of the remaining walkway and the garden, assuming the outer dimensions of the entire area (garden + modified walkway) stay the same.

Notes for Solving: All measurements are in whole meters for the garden dimensions, and the walkway width will be a positive rational number. You must define all variables clearly, show full algebraic manipulation, and check that your solutions satisfy all original conditions in the problem.
```

如果后续需要与主文完全统一，可以只替换 prompt 文本，不改变 sweep 逻辑。

## 6. 默认 TSTC 参数

本补充实验默认采用动态比例采样的 prefill-only TSTC 设置：

- token sampling：从 prompt token positions 中随机无放回抽取 `ceil(L / 8)` 个
- channel sampling：对每个已选 token，从 hidden channels 中随机无放回抽取 `ceil(H / 8)` 个
- 随机方式：基于 public seed 的均匀采样
- checkpoint order：`C1 -> C2 -> C3`
- delta map：

```json
{
  "prefill": {
    "C1": 0.0022,
    "C2": 0.00525,
    "C3": 0.02
  }
}
```

这里 `L` 是 prefill 序列长度，`H` 是 hidden size。若 `L` 或 `H` 不能被 `8` 整除，则向上取整，以保证采样覆盖不低于目标比例。

token 与 channel 的采样都保留 seed，并采用 uniform sampling without replacement，因此实验在相同 seed 下可复现。

这意味着该补充实验回答的是：

> 在当前论文所采用的比例式 prefill-focused TSTC 参数下，`C2` 的 honest numeric perturbation 扫值会如何改变检测数量。

## 7. 建议噪声扫值

建议 sweep：

- `1e-10`
- `2e-10`
- `5e-10`
- `1e-9`
- `2e-9`
- `5e-9`
- `1e-8`
- `2e-8`
- `5e-8`
- `1e-7`
- `2e-7`
- `5e-7`

## 8. 输出与交付

实验输出应至少包括：

- trial-level 明细
- 每个 `s` 的汇总统计
- 论文图

论文图要求：

- 横坐标：噪声强度 / 噪声标准差 `s`
- 纵坐标：检测个数

推荐同时保留：

- `detection_rate`
- `first_mismatch_checkpoint` 分布
- `effective_noise_std = s * sigma_ref`

这些字段不一定进入主文，但有助于解释曲线转折点。

## 9. 结果解读边界

### Confirmed result

当前文档只定义实验协议，不声明新的实验结果。

### Tentative interpretation

如果检测数量随 `s` 单调上升，这说明当前 TSTC 设置对 `C2` 数值扰动强度具有可观测响应。

### Open question

- 检测数量曲线是否存在明显阈值区间
- 在 `C2` 上出现首个失配时，是否几乎总是先定位到 `C2`
- prompt 更换后曲线是否稳定
