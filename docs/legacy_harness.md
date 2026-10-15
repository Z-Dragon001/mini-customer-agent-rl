# 旧版 Mini Harness（归档）

以下保留原文；其中多卡 TRL 命令不适用于当前单卡 ART 主流程，请按根目录 README 操作。

# Mini Customer Agent RL

一个刻意做小的客服工具 Agent 项目：只保留查订单、改地址、申请退款、转人工四个工具，先跑通评测与 trace，再根据高频错误选择 SFT 或 GRPO。基础 Harness 只依赖 Python 标准库。

## 这个项目解决什么

同一套环境和测试集可以评测：

1. 未训练的小模型；
2. SFT/GRPO 后的小模型；
3. 通用强模型。

每次运行保存完整消息、工具参数、工具结果、状态快照和失败标签，并汇总：任务成功率、非法操作率、平均工具调用次数、单任务成本、平均/P50/P95 完成耗时。

失败被归为五类：

| 标签 | 含义 |
|---|---|
| `tool_selection_error` | 选错工具，或不必要地转人工 |
| `parameter_error` | 参数缺失、格式错误、订单号错误或违反操作条件 |
| `unhandled_exception` | 模型/工具异常，或重复同一失败调用而未恢复 |
| `premature_termination` | 目标未完成就结束或达到步数上限 |
| `insufficient_information` | 未取得确认等必要信息就执行 |

## 目录

```text
data/tasks/                 固定的 dev/test 任务与单轮 RL 训练任务
configs/                    三类模型的 OpenAI-compatible 配置示例
src/mini_customer_agent/    环境、Harness、模型客户端、评测、错误归因
src/.../training/           SFT 数据构建、LoRA SFT、可选 GRPO
tests/                      无外部依赖的单元与端到端测试
docs/experiment_plan.md     推荐实验顺序和判因方式
```

## 1. 先验证 Harness

推荐使用 uv（Linux / Bash）：

```bash
# 按 uv.lock 安装 API、训练和开发依赖，并自动创建 .venv
uv sync --all-extras --locked
uv run --all-extras --locked python -m unittest discover -s tests -v
uv run --all-extras --locked customer-agent evaluate --provider scripted
```

只做 API 评测和开发时，可使用 `uv sync --extra api --extra dev --locked`；
基础 Harness 无第三方运行依赖，可仅运行 `uv sync --locked`。
后续使用 `uv run --all-extras --locked` 作为命令前缀即可使用完整环境，无需手动激活。
例如四卡 SFT 的命令以 `uv run --all-extras --locked torchrun --standalone --nproc_per_node=4` 开头。
依赖安装不包含模型权重；真实模型评测还需要可访问的模型服务，训练需要模型和训练数据。

PowerShell：

```powershell
cd S:\Workfile\VScode\mini-customer-agent-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m unittest discover -s tests -v
python -m mini_customer_agent evaluate --provider scripted
```

`scripted_oracle` 只是验证环境是否可完成的确定性策略，不是实验 baseline。输出会写入 `traces/<时间>_scripted_oracle/`。

## 2. 接入一个真实模型

Harness 接受任意 OpenAI-compatible Chat Completions 服务，包括本地 vLLM/SGLang 和外部模型 API：

```powershell
pip install -e ".[api]"
$env:LOCAL_LLM_API_KEY="EMPTY"
python -m mini_customer_agent evaluate `
  --provider openai `
  --name small_base `
  --model Qwen/Qwen2.5-7B-Instruct `
  --base-url http://127.0.0.1:8000/v1
```

价格参数以“美元/百万 token”为单位；本地模型保持 0 即可：

```powershell
python -m mini_customer_agent evaluate `
  --provider openai --name strong_general --model YOUR_MODEL `
  --base-url https://YOUR_ENDPOINT/v1 `
  --api-key-env STRONG_MODEL_API_KEY `
  --input-price 0.50 --output-price 1.50
```

不要把 API Key 写进 JSON。`configs/models.example.json` 只保存环境变量名。

## 3. 同测试集比较三个模型

复制并修改配置：

```powershell
Copy-Item configs\models.example.json configs\models.json
python -m mini_customer_agent compare --config configs\models.json
```

三个 endpoint 分别指向未训练小模型、加载 LoRA 后的小模型和通用强模型。结果写入 `traces/comparison.json`，每个模型还有独立的 `traces.jsonl` 和 `metrics.json`。

## 4. 看错误，再决定是否训练

不要先设定“必须做 RL”。建议先为每个模型采样 3 次以上，按 `primary_failure` 计数并人工抽查 trace。如果反复出现的是：

- 参数和值映射错误：先用纠正后的成功轨迹做 SFT；
- 明知工具结果仍无法恢复、或长链决策不稳定：再考虑环境 RL；
- 主要是知识缺失或工具返回信息不足：先改工具/数据，不应让训练背锅；
- 强模型也失败：优先检查任务、政策和 verifier。

本项目推荐第一个训练目标选“参数填错”，因为标签清晰、可用少量数据验证、也容易解释能力增益。

## 5. 构造 SFT 数据并训练

先从强模型或人工修正后的成功 trace 构造 conversational tool-calling 数据：

```powershell
python -m mini_customer_agent build-sft `
  --traces traces\RUN_NAME\traces.jsonl `
  --output data\sft\train.jsonl
pip install -e ".[train]"
```

Linux 四卡训练示例：

```bash
torchrun --standalone --nproc_per_node=4 \
  -m mini_customer_agent.training.train_sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset data/sft/train.jsonl \
  --output-dir outputs/customer-agent-sft
```

这是 LoRA SFT，默认 batch size 很保守，适合先用几十条样本做 smoke test，再扩到数百条人工验收轨迹。`--assistant-only-loss` 只有在模型 chat template 支持 assistant token mask 时再开启。

## 6. 可选：在真实工具状态上做 GRPO

`rl_train.jsonl` 中都是单轮即可执行的任务，因为训练环境不会伪造用户后续回复。奖励直接来自执行后的环境状态：

- 正确达到任务目标：`1.0`，多余工具调用每次扣 `0.05`；
- 非法操作：`-1.0`；
- 普通工具失败：`-0.5`；
- 未完成且未违规：`0.0`。

```bash
torchrun --standalone --nproc_per_node=4 \
  -m mini_customer_agent.training.train_grpo \
  --model outputs/customer-agent-sft \
  --tasks data/tasks/rl_train.jsonl \
  --output-dir outputs/customer-agent-grpo \
  --max-steps 200
```

GRPO 工具环境依赖当前 TRL 的 `environment_factory` 能力和 `transformers>=5.2.0`，属于可选实验路径。先锁定可工作的依赖版本并运行 5～10 step smoke test；不要在 baseline 尚未稳定时直接长训。实现依据是 [TRL Agent Training 官方文档](https://huggingface.co/docs/trl/grpo_trainer)；SFT 数据格式依据 [TRL SFTTrainer 官方文档](https://huggingface.co/docs/trl/sft_trainer)。

## 环境规则

- 会话预先绑定 `customer_id`，无额外登录工具；任何跨客户访问都会被拒绝并计作非法操作。
- 改地址仅允许 `pending` 订单。
- 退款仅允许 `delivered`、`refundable=true` 且未申请过退款的订单。
- 改地址和退款必须看到用户明确确认。
- 成功由最终状态 verifier 判定，不要求模型复现唯一标准工具链。

设计思路参考了 [τ³-bench / tau2-bench](https://github.com/sierra-research/tau2-bench) 的状态化客服环境，但没有引入其完整任务规模或依赖。

## SSL `UNEXPECTED_EOF_WHILE_READING`

若接外部 API 时遇到该错误，先按顺序排查：

1. 本地 vLLM/SGLang 通常是 `http://127.0.0.1:端口/v1`，不要误写成 `https://`；
2. 用同一环境执行 `python -c "import ssl; print(ssl.OPENSSL_VERSION)"`；
3. 检查代理是否对 HTTPS 做了中间人解密，并更新系统 CA / `certifi`；
4. 确认 Debian 主机时间正确、域名与 SNI 一致；
5. 用 `curl -v https://目标域名/v1/models` 区分是 Python 客户端还是代理/服务端提前断开。

客户端已设置 60 秒超时和 2 次重试，但 TLS 握手阶段 EOF 通常需要修复 URL、代理或证书链，单纯增加重试次数不是根治。

