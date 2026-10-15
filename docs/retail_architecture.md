# 智能客服 ART 主流程说明

## GRPO 改进实验（2026-10-13）

新实验保留原始 SFT/GRPO 结果，从同一 SFT 权重建立独立模型分支。协议位于
`experiments/retail-grpo-repair/protocol.md`，运行入口为
`scripts/run_retail_grpo_repair.py`，实际阶段与输出目录登记在对应的 `runs.json`。

- `retail/rewards.py` 单独计算官方结果和过程惩罚，输出奖励分解；成功轨迹奖励始终高于失败轨迹。
- `art.kl_beta` 和 `art.kl_reference` 控制对冻结 SFT 适配器的 KL 约束；参考不是禁用 LoRA 后的 Base 模型。
- `art.eval_every` 在固定采样组边界评测全部开发任务，开发轨迹不进入训练；测试只在候选冻结后运行。
- `retail/optimizer_state.py` 与 `art_service.py` 保存、恢复优化器和本地随机状态。原始 SFT 权重分支首次 RL 更新允许全新优化器，后续恢复要求模型与优化器步数一致。
- `scripts/report_retail_grpo_repair.py` 从原始记录生成实际结果；不完整覆盖显示为中间结果，训练奖励不作为测试成功率。

三个奖励/约束对照分别为结果奖励、结果奖励＋KL、结果奖励＋KL＋有界过程惩罚；另设数值一致性修复臂，在最后一种配置上统一BF16训练/采样基座、使用FP32概率归一化，并将ART打包位置编号传入模型。`retail/art_numerics.py`负责位置传递和只读固定批次重放，`retail/logprobs.py`记录采样/训练概率偏差。详细设置、选择规则和实际结果均以该实验目录为准。

测试前追加业务进度奖励组：`retail/business_progress.py`在训练轨迹结束后，离线重放数据库状态并检查目标操作的参数与执行结果，为部分完成提供有界奖励。任务目标只供奖励计算，不进入模型对话；开发与测试仍使用官方成功率。`retail/worker.py`仅在train且显式启用该奖励时调用它。

后续掩码修复组F：锁定版Unsloth的外层模型入口会覆盖ART提供的分支注意力掩码，单独修正位置编号不能阻止同一打包序列中的轨迹互相看到内容。`art.packed_attention`显式启用decoder入口掩码保留及xFormers张量维度适配，默认关闭以保留原对照。`scripts/probe_retail_attention_gpu.py`使用随机小Qwen2验证实际GPU分支隔离与梯度；正式3B训练前另做固定批次只读重放。F运行入口为`run_retail_attention_repair.py`，单独冻结到`selection-attention.json`。

## 业务边界

主题仍是电商客服。相比原 Mini Harness 的四个自定义工具，新主流程直接使用固定版本 tau-bench retail 的16个工具，包括身份查找、用户/订单/商品查询、地址/商品/付款方式修改、取消、退换货和转人工。工具与业务政策来自上游，不能继续用旧环境的四个函数名和规则描述新版实现。

数据库是每次任务独立加载的模拟状态。用户模拟器根据任务场景扮演客户；客服学生模型按照政策决定追问、调用工具或答复。工具结果返回后，学生继续决策，直到用户结束、转人工或达到步数上限。标准动作仅供官方评测器重放和对照，不能塞进学生提示词。

## 三种模型角色与裁判

| 角色 | 输入 | 输出 | 是否更新参数 |
|---|---|---|---|
| 学生 | 政策、工具Schema、当前对话和工具反馈 | 回复或工具名称/参数 | SFT/GRPO时更新LoRA |
| 教师 | 与学生相同的可见业务信息 | 多轮示范轨迹 | 否 |
| 用户模拟器 | 任务场景、已发生的用户侧对话 | 客户提问、补充信息、确认、结束 | 否 |
| 自然语言裁判 | 官方断言及交互记录 | 官方断言检查结果 | 否 |

教师用于采集SFT，用户模拟器用于采集、RL与评测。默认只有学生在本地单卡运行；其他角色调用API。只需SFT训练时，不会调用这些外部角色。模型效果比较必须固定用户/裁判版本及参数。

## 数据如何流转

1. `data.py` 从官方固定commit下载源码、任务、数据库、政策、用户提示和MIT许可证，校验文件并生成清单。
2. `dataset.py` 保留官方40个测试任务，从官方74个训练任务中留出15个开发任务，剩余59个训练任务。开发集用于选方案；测试集用于配置冻结后的对照。
3. `collect` 只遍历训练ID。`rollout.py` 将教师API接入官方编排器，保存对话、工具Schema、每次工具反馈、状态评测、终止原因、模型token用量与耗时。
4. `build-sft` 接受成功且无工具错误/截断的教师轨迹；校验来源、工具消息配对和JSON参数、去重。输出`messages + tools + metadata`及数据哈希清单。自动筛选不等于人工审核，也不等于全政策验证。
5. `training.py` 对SFT数据再次检查训练ID、版本、哈希和上下文长度，交给ART的SFT接口。学生学习整条对话中的客服回复和工具调用，用户/系统/工具文本不作为目标回复。
6. RL按同一训练任务采样多条学生轨迹，每条用新的环境状态。任务种子在组内相同，学生使用随机采样；这降低无关变化，但不保证远端用户模型完全确定。
7. 官方评测器将实际状态和任务标准比较；需要自然语言断言的任务另调用裁判。奖励回传ART。当前默认使用官方任务奖励，不额外混入未经验证的“违规惩罚”或调用次数惩罚。
8. ART以组内相对优势、PPO裁剪目标更新LoRA。组内奖励相同没有区分信号，代码跳过更新并记录；不能把采样组数当作梯度更新数。
9. SFT与RL检查点保存在`.art/`，在同一留出任务和评测协议下测试。最终演示服务加载基础模型+LoRA，接工具环境；生产订单API、认证、发布服务不在本次实现内。

## 文件对应职责

| 核心文件 | 职责与阅读重点 |
|---|---|
| `src/mini_customer_agent/retail/cli.py` | `inspect / prepare-inference / collect / build-sft / evaluate / train`入口 |
| `retail/data.py` | 固定版本下载、Git blob校验、SHA-256清单；标准库即可运行 |
| `retail/config.py` | `.env`、模型端点、路径、配置校验和软件/代码来源记录 |
| `retail/dataset.py` | 固定数据划分、训练数据污染检查、SFT格式与筛选 |
| `retail/rollout.py` | 核心业务闭环：模型→工具/用户→反馈→模型→官方评分；保留ART原始Choice/logprobs |
| `retail/training.py` | 单GPU校验、SFT长度检查、LoRA配置、ART SFT/GRPO与检查点评测 |
| `retail/reporting.py` | 每次实验单独建目录、逐条写trace、汇总指标 |
| `configs/retail.example.json` | 3B模型、单卡参数和四类API角色配置模板 |
| `runtimes/art/pyproject.toml`、`uv.lock` | GPU训练专用依赖，与根目录历史TRL依赖隔离 |
| `tests/test_retail_data.py` | 划分隔离、消息配对、数据筛选和奖励错误边界 |
| `tests/test_retail_integration.py` | 固定脚本执行真实官方工具并评分；仅验证适配器，不是模型baseline |

目录层级的完整说明和可运行命令在根目录README。旧版代码保留在`src/mini_customer_agent/`原位置与`training/`下，其自定义任务位于`data/tasks/`。新版实验只使用`retail-agent`入口，不能混合两个环境的成功率。

## 单卡运行与依赖

训练入口要求显式设置一个`CUDA_VISIBLE_DEVICES`，进程中只能看到一张CUDA卡。示例采用3B、4-bit加载、LoRA rank16、单条SFT batch和串行rollout。ART与vLLM安装在独立的训练虚拟环境内，使用同一张可见卡。

实际启动验证发现ART0.5.18的单卡路径会强制导入Megatron和Bridge依赖，其中其声明的Bridge预发布版本无法从PyPI取得。因此训练栈调整到ART0.5.16、vLLM0.15.1、PyTorch2.9.1、Transformers4.57.3，并重新锁定依赖；没有忽略依赖检查。GRPO调用同步改为该版本的`ppo=True`接口。历史`.cache/art-vllm`缓存不再用于当前启动。

首次真实教师采集已完成3个任务，其中2个成功并通过自动筛选。它们不是正式评测结果。两条样本的上下文超过8K，本机配置改为16K；后续GPU验证状态以本地部署说明为准。遇到超长SFT样本会停止，不静默裁剪工具结果。若少数长任务RL溢出，先查看trace与实际token长度，不直接报告其为任务推理失败。

## 如何解读输出

- `run.json`：配置、tau版本、软件版本、代码提交和工作区是否有修改。
- `traces.jsonl`：逐任务对话与工具信息、官方评分、结束原因；每次实验独立目录。
- `metrics.json`：所有尝试与成功完成运行两种分母的成功率、工具错误、调用数、耗时。训练运行保存最近一个split的汇总，不把开发和训练样本混在一个成功率中。
- `arguments.json`：ART运行阶段、任务上限、组数、数据哈希、GPU型号等。
- `evaluation.json`：`sft-grpo`组合命令进入RL之前保存的SFT开发集结果；独立评测也保存此文件。
- `updates.jsonl`：每个RL采样组的奖励、有无梯度更新及检查点步数。

网络/裁判故障单列，有限重试后仍失败才中止，不当作奖励0训练。模型生成非法JSON参数或空消息时单列`invalid_model_output`，耗尽上下文时单列`context_length_exceeded`，均记失败0分；这是适配器的失败处理，不冒充一次成功完成的官方评分。未实现全面政策合规检测，不能把工具错误率叫“违规率”。尚未汇总全部API角色的真实费用。

## 继续完善的顺序

先配置API，采集3个任务核对完整trace；随后采集59个训练任务、抽样审核并构建SFT。完成单卡Base评测和小规模SFT验证后，开展完整SFT，再检查RL是否存在组内奖励差异。最后形成Base/SFT/SFT+GRPO的统一对照表，并保留不同随机种子的原始记录。现在不能在简历中填写不存在的性能提升。
