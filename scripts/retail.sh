#!/usr/bin/env bash
# 从任意目录运行；本机配置在 configs/retail.json，密钥在 .env。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

stage="${1:-help}"
if (($#)); then shift; fi
config_path="${RETAIL_CONFIG:-configs/retail.json}"

case "$stage" in
  inspect|probe|collect|build-sft)
    exec uv run --extra retail --locked retail-agent --config "$config_path" "$stage" "$@"
    ;;
  baseline|evaluate|sft|grpo|sft-grpo)
    # 本次部署选择GPU 0；可以显式覆盖为另一张空闲卡，但不能传多卡列表。
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    if [[ "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == -1 ]]; then
      echo "请通过 CUDA_VISIBLE_DEVICES 选择一张可用GPU。" >&2
      exit 2
    fi
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export LITELLM_LOCAL_MODEL_COST_MAP=True
    # 首次加载适配器可能超过ART默认30秒；这里只调整本地服务就绪等待。
    export ART_SERVER_TIMEOUT="${ART_SERVER_TIMEOUT:-180}"
    # 本地ART服务直接连接；外部API仍可使用原有代理。
    export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,0.0.0.0"
    export no_proxy="$NO_PROXY"
    if [[ "$stage" == baseline ]]; then
      stage=evaluate
      if (($# == 0)); then set -- --split dev --limit 1; fi
    fi
    exec uv run --project runtimes/art --locked retail-agent --config "$config_path" train --stage "$stage" "$@"
    ;;
  *)
    cat <<'USAGE'
用法：bash scripts/retail.sh <阶段> [参数]
  inspect                         检查官方环境与数据
  probe                           测试教师API（单次请求）
  collect --limit 3 --trials 1     采集3个训练任务的教师示范
  build-sft --traces ... --output ...   筛选并生成SFT数据
  baseline                        评测当前检查点的1个开发任务（未训练时为Base）
  sft --dataset ... --epochs 1     执行SFT
  grpo --steps 5 --limit 3         执行小规模RL
  evaluate --split dev --limit 15  评测当前检查点

默认只使用GPU 0，并从本地读取学生权重。
可用 RETAIL_CONFIG 指定其他配置，用 CUDA_VISIBLE_DEVICES 指定另一张空闲卡。
每个阶段完成后释放模型进程；本脚本不启动常驻公网服务。
USAGE
    ;;
esac
