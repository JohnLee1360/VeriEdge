# THC/TSTC（Shard-k 版本）：Prefill-Focused Verification 实验说明

## 1. 实验定位与目标

本轮正式口径统一收敛为 **prefill-focused verification**。

系统设定保持不变：

- Qwen3-0.6B 按 3 个 shard 切分；
- 由 3 个 provider 协作执行；
- 验证单元统一为 `Shard k`（`Shard 1 / 2 / 3`）；
- checkpoint 固定为 `C1 / C2 / C3`。

当前主问题不再是“完整生成过程的逐阶段验证”，而是：

> 当 provider 声称自己按指定 shard plan 诚实完成一次请求的 prefill 计算时，基于 shard-boundary checkpoint 的 THC/TSTC 能否在 heterogeneous execution 下保持低误报，并在 tamper 场景下完成检测与定位。

## 2. 机制定义（统一为 Shard k）

后续机制与实验叙事统一为：

> 对每个 `Shard k` 的对外交付张量构造摘要，验证时重算并比较，首个失配的 shard 用于定位可疑责任区段。

记每个 shard 边界张量为 \(T_k\)，其中 \(k \in \{1,2,3\}\)。

### 2.1 THC（deterministic baseline）

$$
h_1 = H(T_1),\quad h_k = H\!\left(h_{k-1} \parallel H(T_k)\right),\; k=2,3.
$$

### 2.2 TSTC（tolerance-aware sampled chain）

对每个 `Shard k`：

1. 公开可复现采样：

   $$
   u_k = S_{\Omega_k}(T_k).
   $$

2. 容差量化：

   $$
   \tilde{u}_k = Q_{\Delta_k}(u_k), \qquad
   Q_{\Delta_k}(x) = \operatorname{round}\!\left(\frac{x}{\Delta_k}\right).
   $$

   即先按 checkpoint-specific 容差 \(\Delta_k\) 做缩放，再四舍五入到整数桶；当 \(\Delta_k = 0\) 时，退化为 strict 比较。

3. shard 摘要：

   $$
   s_k = H\!\left(\Omega_k \parallel \tilde{u}_k\right).
   $$
4. 链式承诺：

$$
c_1 = H(s_1),\quad c_k = H\!\left(c_{k-1} \parallel s_k\right),\; k=2,3.
$$

验证时比较重算链与承诺链，首个失配 shard 作为可疑责任区段定位结果。

## 3. Checkpoint 设计

仅保留 3 个 checkpoint：

1. `C1`：`Shard 1` 输出（发送给 `Shard 2` 前）
2. `C2`：`Shard 2` 输出（发送给 `Shard 3` 前）
3. `C3`：`Shard 3` 最终输出（默认 **final hidden state**）

默认不把 logits 作为主 checkpoint。

## 4. Prefill 作为唯一正式验证阶段

当前正式实验只验证 prefill 阶段。

设 batch size 为 \(B\)、prompt 长度为 \(L\)、hidden size 为 \(H\)：

- \(C1 \in \mathbb{R}^{B \times L \times H}\)
- \(C2 \in \mathbb{R}^{B \times L \times H}\)
- \(C3 \in \mathbb{R}^{B \times L \times H}\)

正式实验报告对象仅限于 prefill。

## 5. 采样策略

prefill 形状为 `B × L × H`，不能直接展平后随意抽样；正式方案固定为：

- `token-channel stratified random sampling with public seed`

流程：

1. 先采样少量 token 位置；
2. 再对每个 token 采样少量 hidden 维度；
3. 总采样数：

$$
\texttt{sample\_count} = \texttt{token\_samples} \times \texttt{channel\_samples}.
$$

## 6. Delta 口径

### 6.1 经验事实

严格的真实 T3 calibration 在当前 `2 x Mac mini M4 + 1 x Linux 3090` prototype 上曾多次得到 `delta_map = 0`，说明在当时比较的真实 capture roots 上，没有观测到可利用的非零 drift。

### 6.2 当前主文采用的 prefill delta

为了回答“prefill-focused verification 是否可行”这一更聚焦的问题，当前 follow-up 实验采用了一版 **prefill-only tuned delta map**：

```json
{
  "prefill": {
    "C1": 0.0022,
    "C2": 0.00525,
    "C3": 0.02
  }
}
```

文件见 [prefill_only_target_delta_map.json](/Users/jlmini_2/.codex/worktrees/de95/bc-ra-paper/artifacts/_shared/thc/output/20260312_manual_delta_followup_01_03/runtime/prefill_only_target_delta_map.json)。

这里必须明确：

- 这是一版 **prototype-level tuned tolerance**；
- 它支撑的是当前 evaluation object 下的经验结果；
- 不能表述为 deployment-independent guarantee。

## 7. 指标与正式验收口径

正式实验保留三类场景：

- `honest_homo`
- `honest_hetero`
- `tamper`

核心指标保留：

- `FPR`
- `TPR`
- `LocAcc`

额外要求：

- `honest_hetero` 继续按 `low / mid / high` 报告 FPR；
- 当前关注的是 `honest_hetero` 下误报是否相对 strict THC 显著下降；
- 结果应作为 prototype evidence 解读，而不是最终工程保证。

## 8. 当前主结果

当前主结果来自：

- capture root: [20260312_manual_delta_followup_01_03/capture_evaluation](/Users/jlmini_2/.codex/worktrees/de95/bc-ra-paper/artifacts/_shared/thc/output/20260312_manual_delta_followup_01_03/capture_evaluation)
- summary: [summary_metrics.csv](/Users/jlmini_2/.codex/worktrees/de95/bc-ra-paper/artifacts/_shared/thc/output/20260312_102858_225617_qwen_all_evaluation/summary_metrics.csv)

实验设置：

- split: `evaluation`
- prompts: `300`
- `runs_per_mode = 10`
- active stage: `prefill`
- stable placement: `jlmini_3 -> C1`, `linux124 -> C2`, `jlmini_2 -> C3`
- runtime profile: `C1=metal_8bit`, `C2=bf16`, `C3=float32`

正式结果如下：

- `honest_homo / prefill`
  - `THC FPR = 0.0`
  - `TSTC FPR = 0.0`
- `honest_hetero / prefill`
  - `THC FPR = 1.0`
  - `TSTC FPR = 0.170111`
- `tamper / prefill`
  - `THC TPR = 1.0`, `LocAcc = 1.0`
  - `TSTC TPR = 1.0`, `LocAcc = 1.0`

按 heterogeneity level 分解后：

- `low = 0.151333`
- `mid = 0.175333`
- `high = 0.183667`

这组结果表明，在当前 evaluation object 上，prefill-focused TSTC 相对 strict THC 显著降低了 `honest_hetero` 误报。

## 9. 论文回填叙事约束

### 9.1 允许的主结论

当前最强但仍然稳妥的口径是：

- THC 在 heterogeneous execution 下会对 honest prefill 产生严重误报；
- 在当前 prototype evaluation object 上，prefill-focused TSTC 能显著降低误报；
- 这种误报下降不以牺牲 tamper detection 或 shard-level localization 为代价。

### 9.2 必须保留的 caveat

仍需明确：

- 当前 `honest_hetero` 评估对象仍然是 capture bundle 上的 synthetic heterogeneity 注入，而不是真实 paired honest captures；
- 因此该结果应解释为 **prototype-level evidence for prefill-focused verification**；
- 不应扩写成“已证明跨设备部署下的完整生成验证”。

### 9.3 不再使用的旧口径

后续不再使用以下叙事：

- “多阶段并列 summary 是当前论文主口径”

这些说法属于旧版 strict T3 路径，不再作为主文结论。

## 10. 仓库执行链路口径

当前仓库虽仍保留更一般的 capture / run 能力，但论文主文口径固定为：

- active stage 仅 `prefill`
- checkpoint 仅 `C1 / C2 / C3`
- TSTC 仅报告 prefill token-channel sampling
- summary 仅以 prefill 的 `TPR / FPR / LocAcc` 为准

因此，后续图表、正文、表格和 discussion 都应围绕 **prefill-focused verification** 回填。
