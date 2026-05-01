# EXO 实现评估实验方案（黑盒 EXO 版本）

## 1. 科学问题

本实验在假设 requester 与 provider 已完成任务匹配的前提下，评估两个问题：

1. 在 `LAN` 与 `WAN` 条件下，`BC-RA + EXO` 的端到端任务级体验是否可接受；
2. 在相同问题集上，单个 `EXO instance` 覆盖节点数从 `1`、`2`、`3` 变化时，各项端到端指标如何变化，其中当前 `WAN` 只测 `instance_node_count = 3`。

## 2. 实验边界

本实验采用 `黑盒 EXO` 路线。

- 三台 Mac mini 上的 EXO 由外部线程预先启动并保持运行。
- 本实验代码不负责创建、删除、升级或修复 EXO。
- requester 仅通过 `/state` 观察当前外部 EXO 状态，并在满足条件时发任务。

EXO 源码位于：

- `~/repo/paper/third_party/exo`

为保证可复现性，实验目录内固定一份外部 EXO 的版本快照清单：

- `artifacts/E5/inference_e2e/freeze/exo_env_manifest.json`

## 3. 实验部署

- requester：MacBook，`192.168.31.189`
- providers：
  - `jlmini_1 = 192.168.31.52`
  - `jlmini_2 = 192.168.31.159`
  - `jlmini_3 = 192.168.31.83`
- 模型：`mlx-community/Qwen3-0.6B-8bit`
- 一个 task 固定包含 `20` 个文本问题
- task 数据包经 AES-GCM 加密后上传到 requester 本地 Kubo/IPFS

## 4. instance 的实验含义

在 EXO 中，`instance` 表示一次可运行的模型部署单元。

本实验固定只向一个目标 `instance` 发任务，不讨论多 instance 并行吞吐。

本实验中变化的是 `instance_node_count`，含义是：该单个 instance 的 shard 被放置到多少台 provider 节点上。

- `instance_node_count = 1`：单 instance 只覆盖 1 台 provider
- `instance_node_count = 2`：单 instance 跨 2 台 provider
- `instance_node_count = 3`：单 instance 跨 3 台 provider

不同 `instance_node_count` 条件下，均使用同一批问题集重复测试；每次正式运行前，需要先清空与本轮任务相关的缓存，再开始下一轮测量。

## 5. 实验矩阵

当前实验采用不对称矩阵设计：

- `LAN`: `instance_node_count ∈ {1, 2, 3}`
- `WAN`: `instance_node_count = 3`

共 4 个 cell：

- `n=1 / LAN`
- `n=2 / LAN`
- `n=3 / LAN`
- `n=3 / WAN`

## 6. 网络环境

- `LAN`：同一 Wi-Fi 局域网
- `WAN`：在 LAN 基础上，对 requester 与实验链路涉及的 provider / launcher / gateway / callback 相关端口施加 RTT / 带宽 / 丢包约束，模拟广域网条件

实现方式：

- `pf + dummynet(dnctl)` on macOS
- 脚本：`artifacts/E5/inference_e2e/requester/netem_macos.sh`

当前 `WAN` shaping 的实际配置为：

- 作用位置：仅在 MacBook requester 侧施加规则，不直接修改 provider 机器本身的网络栈
- 作用协议：仅匹配实验相关 `TCP` 流量
- 作用端口：
  - `52415`：EXO entry / inference API
  - `18080`：provider launcher
  - `8080`：IPFS gateway
  - `18081`：requester callback
- 实现方式：
  - 用 `pf` anchor 匹配目标 IP 和端口
  - 用 `dnctl pipe` 对每个目标 IP 分别施加入站 / 出站 pipe
  - 配置中的 `RTT` 被拆成双向各一半的 `delay`

当前 `target_spec` 为：

- `192.168.31.52`：`RTT=22ms`, `BW=120Mbps`, `LOSS=0.5%`
- `192.168.31.159`：`RTT=45ms`, `BW=60Mbps`, `LOSS=1.5%`
- `192.168.31.83`：`RTT=80ms`, `BW=40Mbps`, `LOSS=2.0%`

因此，`WAN` 实验模拟的是 MacBook requester 到三台 provider 的实验相关 TCP 链路具有不同质量的广域网条件，而不是整机统一降级。

## 7. 任务链路

1. requester 构造一个包含 20 个问题的文本 task
2. requester 生成 task manifest
3. requester 生成随机对称密钥并对任务包做 AES-GCM 加密
4. requester 将密文任务包上传到本地 Kubo，得到 CID
5. requester 从 EXO `/state` 中筛选与当前 `instance_node_count` 匹配的目标 instance，并识别其 first-shard provider
6. requester 将 `{task_id, cid, encrypted_task_key, metadata}` 发给 first-shard provider launcher
7. first-shard provider 使用本地私钥解密 task key，再解密任务包
8. provider 串行执行 20 个问题
9. provider 通过 callback 把结果回传给 requester

## 8. 指标定义

- `Task Latency (s)`：从 provider 开始下载任务包到 requester 收到完整任务结果的总时间
- `Download (s)`：provider 从 IPFS 下载并完成解密任务包的时间
- `Task Throughput (task/s)`：`1 / Task Latency`
- `TTFT (s)`：20 个问题的单题 TTFT 中位数
- `OTPS (tok/s)`：20 个问题的单题 OTPS 中位数
- `question_success_count`
- `question_fail_count`

诊断性辅助指标：

- `question_throughput_qps = 20 / Task Latency`

## 9. 运行前检查

每次正式实验前，应完成：

1. Kubo 已启动且 gateway 可达
2. provider launcher 已启动
3. EXO 版本与 `freeze` 清单一致
4. 当前 `/state` 中存在与待测 `instance_node_count` 匹配的目标 instance，且其 instance id 与配置一致
5. first-shard provider 可解析
6. 本轮正式运行前，相关缓存已清空

## 10. 输出文件

每个 cell 产出：

- `cell_status.json`
- `state_snapshot.json`
- `task_manifest.json`
- `dispatch_record.json`
- `launcher_ack.json`
- `task_result.json`
- `question_results.jsonl`

每个 run 产出：

- `health_checks.csv`
- `aux_service_checks.csv`
- `summary_by_cell.csv`
- `task_runs.jsonl`
- `run_config_snapshot.yaml`
- `comparison_table.csv`
- `comparison_table.md`

## 11. 主表模板

| Node Count | Network | Task Latency (s) | Download (s) | Task Throughput (task/s) | TTFT (s) | OTPS (tok/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | LAN |  |  |  |  |  |
| 2 | LAN |  |  |  |  |  |
| 3 | LAN |  |  |  |  |  |
| 3 | WAN |  |  |  |  |  |

## 12. 预期结论

- 用户体验层面：WAN 会明显恶化时延类指标，但个人使用场景下仍可能可接受。
- 放置层面：`instance_node_count` 增加时，`Task Latency`、`Download`、`TTFT` 和 `OTPS` 的变化趋势将帮助判断跨节点放置带来的额外通信与调度开销。
