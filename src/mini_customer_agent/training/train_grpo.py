from __future__ import annotations

import argparse
from functools import partial

from .grpo_environment import CustomerServiceGRPOEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental LoRA GRPO in the tool environment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", default="outputs/customer-agent-grpo")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    try:
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("缺少训练依赖，请运行：pip install -e '.[train]'") from exc

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_completion_length=1024,
        max_tool_calling_iterations=4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=50,
        report_to="none",
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    factory = partial(CustomerServiceGRPOEnvironment, args.tasks)
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        environment_factory=factory,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

