# 本机部署与操作

本机配置：`configs/retail.json`（不提交Git）；密钥：根目录`.env`（权限600）。可提交的`.env.example`只保留空密钥模板。

## 当前连接

| 项目 | 当前设置 |
|---|---|
| 云端教师 | qwen3.7-flash，流式调用，默认关闭思考 |
| 模拟用户/裁判 | 同一模型的独立对话，非流式调用 |
| 原始学生权重 | `/home/zxl/models/Qwen2.5-3B-Instruct`，两片safetensors |
| 实际加载目录 | 配置中`art.base_model`指向`.cache/local-models/`下的模型副本 |
| 模型副本内容 | 原始权重的符号链接、兼容JSON字符串工具参数的tokenizer模板、来源清单 |
| 使用GPU | 默认GPU 0；入口仅允许一张卡 |
| 上下文 | 16384 tokens；推理端与训练端一致 |
| 推理资源 | `gpu_memory_utilization=0.55`、最多2个序列、eager模式 |
| 训练设置 | 4-bit学生加载、LoRA rank16、SFT batch1；未开展正式训练 |

原始权重不复制、不修改。建立模型副本是为了兼容OpenAI工具参数的JSON字符串格式，避免训练时被模板二次编码成带转义引号的字符串。

恢复模型副本：

```bash
uv run --extra retail --locked retail-agent prepare-model \
  --model /home/zxl/models/Qwen2.5-3B-Instruct
```

将返回路径填入`configs/retail.json`的`art.base_model`即可。

## 已完成的API与数据联调

教师API连接检查通过。3个训练任务采集完成：任务43、78成功，任务82达到步数上限且出现工具错误。两个成功任务包含订单/身份查询、地址修改、商品修改、取消订单等操作；任务43也实际经过自然语言裁判。

原始记录：`traces/retail/20261003T112750Z_teacher_train_565e04e2/`。

自动筛选输出：`data/sft/setup_teacher_train.jsonl`及对应`.manifest.json`；保留2条、排除1条失败。两条样本经本地tokenizer实测超过8K，因此本机上下文调整为16K。长度检查已修正为统计`input_ids`，不会把分词结果字典的键数误当token数。

这些数据用于环境与训练接口联调；2条示范不足以支持有效微调的结论，尚未标注为人工审核通过。

## 已完成的本地学生联调

2026-10-03，使用本地3B模型、GPU 0和16K上下文，完整执行了开发任务103。运行正常退出并释放显存；记录在`traces/retail/20261003T115626Z_art_evaluate_219b3e04/`。任务耗时约73秒（不含模型启动），5次工具调用均报错，终止原因`too_many_errors`，任务未完成。进程运行期间观测到显存约14GB；这不是SFT/GRPO反向训练峰值。

轨迹显示学生未经询问就编造查询邮箱；随后模拟用户也编造了邮编和订单号。该记录用于确认端到端接口可运行，不作为正式能力评测结论。正式对照前应抽查模拟用户是否遵循任务中的已知信息，必要时加强用户提示或更换更可靠的模拟用户模型，并对所有阶段固定相同设置。

以上为部署联调阶段的历史记录。后续已实际执行SFT与GRPO反向更新；最新正式评测、训练曲线及显存记录见[实验报告](../experiments/retail-3b/report.md)和[过程日志](../experiments/retail-3b/journal.md)，不要把联调单任务结果混入正式对照。

## 常用入口

所有命令都从项目根目录运行，或直接使用脚本的绝对路径。脚本自动定位项目目录。

```bash
# 检查官方环境与数据（不调用模型）
bash scripts/retail.sh inspect

# 单次云端API请求
bash scripts/retail.sh probe

# 当前学生检查点评测；尚未训练时即为Base，默认只测1个开发任务
bash scripts/retail.sh baseline

# 扩大教师采样：59个训练任务，每个3次（会产生API费用）
bash scripts/retail.sh collect --trials 3

# 使用新采集目录生成正式候选数据，不覆盖已有文件
bash scripts/retail.sh build-sft \
  --traces traces/retail/新采集目录/traces.jsonl \
  --output data/sft/train_v1.jsonl

# 抽样审核数据后，SFT与RL分阶段执行
bash scripts/retail.sh sft --dataset data/sft/train_v1.jsonl --epochs 1
bash scripts/retail.sh evaluate --split dev --limit 15
bash scripts/retail.sh grpo --steps 5 --limit 3
bash scripts/retail.sh evaluate --split dev --limit 15
```

默认使用GPU 0；若它已被其他任务占用，请先指定另一张可用卡，例如`CUDA_VISIBLE_DEVICES=3 bash scripts/retail.sh baseline`。脚本不会停止其他人的进程。ART阶段运行时临时创建推理服务，结束时释放；未配置常驻公网服务。

## 依赖与可复现性

根目录`.venv`用于API和数据；`runtimes/art/.venv`用于GPU。训练栈固定ART0.5.16、vLLM0.15.1、PyTorch2.9.1+cu128、Transformers4.57.3、Unsloth2025.12.9、OpenAI SDK 2.54.0。已通过依赖一致性检查。

旧ART0.5.18在真实启动时触发Megatron/Bridge依赖缺失，故已替换并修改对应训练API调用。历史`.cache/art-vllm/`不参与当前运行。

OpenAI SDK 3.x切换了HTTP客户端，无法兼容ART0.5.16传入的超时对象，因此固定2.x。运行脚本让本地服务绕过网络代理；ART服务仅绑定127.0.0.1。运行结束时会回收本进程启动的推理子进程，避免残留占用显存。

```bash
uv sync --extra retail --extra dev --locked
uv sync --project runtimes/art --locked
uv run --project runtimes/art --locked retail-agent prepare-inference
```

同一`art.name`表示同一条SFT→RL实验链；新对照实验请换名称。当前`agent`字段用于自行启动的外部学生服务；脚本中的学生评测使用`art`字段，由ART自动加载本地模型，无需另开8000端口的服务。
