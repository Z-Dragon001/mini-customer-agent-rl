from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .evaluation import aggregate_metrics, evaluate, write_run
from .model_clients import OpenAICompatibleClient, ScriptedPolicyClient
from .task_data import load_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = PROJECT_ROOT / "data" / "tasks" / "test.jsonl"


def _client_from_args(args: argparse.Namespace):
    if args.provider == "scripted":
        return ScriptedPolicyClient()
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None
    return OpenAICompatibleClient(
        name=args.name or args.model,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        input_price_per_million=args.input_price,
        output_price_per_million=args.output_price,
    )


def _run_evaluate(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    client = _client_from_args(args)
    traces, metrics = evaluate(client, tasks)
    run_dir = write_run(args.output_dir, client.name, traces, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"trace_dir={run_dir.resolve()}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        configs: list[dict[str, Any]] = json.load(handle)
    rows = []
    for config in configs:
        provider = config.get("provider", "openai_compatible")
        if provider == "scripted":
            client = ScriptedPolicyClient()
        else:
            client = OpenAICompatibleClient(
                name=config["name"],
                model=config["model"],
                base_url=config.get("base_url"),
                api_key=os.getenv(config.get("api_key_env", "OPENAI_API_KEY")),
                input_price_per_million=config.get("input_price_per_million", 0.0),
                output_price_per_million=config.get("output_price_per_million", 0.0),
            )
        traces, metrics = evaluate(client, tasks)
        run_dir = write_run(args.output_dir, client.name, traces, metrics)
        rows.append({"model": client.name, **metrics, "trace_dir": str(run_dir.resolve())})
    report_path = Path(args.output_dir) / "comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"comparison={report_path.resolve()}")
    return 0


def _run_summarize(args: argparse.Namespace) -> int:
    traces = []
    with Path(args.traces).open("r", encoding="utf-8") as handle:
        traces.extend(json.loads(line) for line in handle if line.strip())
    print(json.dumps(aggregate_metrics(traces), ensure_ascii=False, indent=2))
    return 0


def _run_build_sft(args: argparse.Namespace) -> int:
    from .training.build_sft import build_sft_dataset

    count = build_sft_dataset(args.traces, args.output, only_success=not args.include_failures)
    print(f"samples={count}\noutput={Path(args.output).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最小客服工具 Agent 评测与训练项目")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="运行同一套 Harness 评测")
    evaluate_parser.add_argument("--provider", choices=["scripted", "openai"], default="scripted")
    evaluate_parser.add_argument("--name")
    evaluate_parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "unused"))
    evaluate_parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    evaluate_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    evaluate_parser.add_argument("--input-price", type=float, default=0.0)
    evaluate_parser.add_argument("--output-price", type=float, default=0.0)
    evaluate_parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    evaluate_parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "traces")
    evaluate_parser.set_defaults(func=_run_evaluate)

    compare_parser = subparsers.add_parser("compare", help="对多个模型使用同一测试集")
    compare_parser.add_argument("--config", type=Path, required=True)
    compare_parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    compare_parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "traces")
    compare_parser.set_defaults(func=_run_compare)

    summarize_parser = subparsers.add_parser("summarize", help="重新汇总已有 trace")
    summarize_parser.add_argument("--traces", type=Path, required=True)
    summarize_parser.set_defaults(func=_run_summarize)

    sft_parser = subparsers.add_parser("build-sft", help="从成功 trace 构建 SFT 数据")
    sft_parser.add_argument("--traces", type=Path, required=True)
    sft_parser.add_argument("--output", type=Path, required=True)
    sft_parser.add_argument("--include-failures", action="store_true")
    sft_parser.set_defaults(func=_run_build_sft)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))

