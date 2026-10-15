#!/usr/bin/env bash
# 可复现实验阶段清单；按阶段执行，不要重复覆盖已完成的SFT实验。
# 根目录.env提供API密钥。实际目录与完成状态见runs.json和journal.md。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export RETAIL_CONFIG=configs/retail.experiment.json
export CUDA_VISIBLE_DEVICES=0

# 1. 教师采集；中断后追加 --resume <runs.json中的teacher目录>
# uv run --extra retail --locked retail-agent --config "$RETAIL_CONFIG" collect --trials 2 --workers 3

# 2. Base开发/测试集评测
# bash scripts/retail.sh baseline --split all --limit 40 --workers 4

# 3. 生成候选数据并整条排除超长对话
# uv run --project runtimes/art --locked python scripts/prepare_retail_training.py

# uv run --extra retail --locked python scripts/review_retail_training.py

# 4. SFT与同协议评测
# bash scripts/retail.sh sft --dataset data/sft/experiment_train_reviewed.jsonl --epochs 2
# bash scripts/retail.sh evaluate --split all --limit 40 --workers 4

# 5. GRPO与同协议评测
# bash scripts/retail.sh grpo --steps 30 --target-updates 10 --workers 2 --task-ids experiments/retail-3b/rl_tasks.json
# bash scripts/retail.sh evaluate --split all --limit 40 --workers 4

# 随时从落盘证据更新进度/结果表
python scripts/retail_report.py

# 全部阶段完成后核验记录；绘图使用系统已安装的 matplotlib 3.10.3。
python scripts/audit_retail_experiment.py
python scripts/plot_retail_experiment.py
