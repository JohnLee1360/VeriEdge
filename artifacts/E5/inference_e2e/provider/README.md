# Provider 侧：launcher 与辅助脚本

本目录放 provider 侧可执行脚本。EXO 本体视为外部应用，源码位于 `~/repo/paper/third_party/exo`，不由本目录负责升级或切换版本。当前默认假设 EXO 运行环境已经由 Nix 准备好，provider 启动命令统一使用 `nix run .`。

## 目录内容

- `provider.env.example`：provider 环境变量模板
- `bootstrap_provider.sh`：provider 辅助初始化脚本
- `manage_provider.sh`：launcher 与检查脚本包装器
- `generate_provider_keys.py`：生成本地 RSA 密钥对
- `launcher.py`：first-shard provider launcher
- `health_check.py`：provider 健康检查

## 1. 准备 provider 环境

安装 provider 侧 Python 依赖：

```bash
cd ~/repo/paper/bc-ra-paper
export BC_RA_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install -i "$BC_RA_PIP_INDEX_URL" -r artifacts/E5/inference_e2e/provider/requirements.txt
```

复制配置模板：

```bash
cp artifacts/E5/inference_e2e/provider/provider.env.example artifacts/E5/inference_e2e/provider/provider.env
```

关键字段：

- `PROVIDER_NODE_ID`
- `PROVIDER_IP`
- `EXO_ENDPOINT`
- `MODEL_ID`
- `EXO_START_CMD`
- `PROVIDER_PRIVATE_KEY_PATH`
- `PROVIDER_PUBLIC_KEY_PATH`
- `LAUNCHER_PORT`

## 2. 生成密钥

每台 provider 只需生成一次：

```bash
artifacts/E5/inference_e2e/provider/manage_provider.sh keys-gen
```

requester 会读取各 provider 的公钥来加密 task key；私钥始终保留在 provider 本地。

## 3. 启动 EXO provider

先启动本机 EXO：

```bash
artifacts/E5/inference_e2e/provider/manage_provider.sh start
artifacts/E5/inference_e2e/provider/manage_provider.sh status
```

`start` 会先等待 endpoint 健康，再等待本机出现在 `/state`；如果在 `provider.env` 里配置了 `CLUSTER_JOIN_EXPECTED_NODE_COUNT`，还会继续等到集群节点数达到目标值后才返回。

## 4. 启动 launcher

```bash
artifacts/E5/inference_e2e/provider/manage_provider.sh launcher-start
artifacts/E5/inference_e2e/provider/manage_provider.sh launcher-status
```

停止：

```bash
artifacts/E5/inference_e2e/provider/manage_provider.sh launcher-stop
```

launcher 暴露两个接口：

- `GET /health`
- `POST /launch-task`

## 5. provider 在实验中的职责

provider 不构造 task，也不上传 IPFS。

实验链路是：

1. requester 构造并加密 task
2. requester 上传到本地 Kubo，得到 CID
3. requester 根据 `/state` 识别 first-shard provider
4. requester 向 first-shard provider 发送 `{CID + encrypted task key + metadata}`
5. first-shard provider 用本地私钥解密 task key，再解密任务包
6. provider 串行执行 50 个问题
7. provider 通过 callback 把结果回传给 requester

## 6. 说明

- provider launcher 只接受“当前机器正好是目标 instance 的 first-shard”这一类任务。
- requester 在正式实验前会先等待 EXO `/state` 连续多次满足目标实例布局，再开始 smoke/main task。
- 本目录脚本不默认启动或变更 EXO instance。
