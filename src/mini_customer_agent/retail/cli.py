from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import configure_tau, endpoint_kwargs, generation_kwargs, load_config, project_path
from .data import DATA, ROOT, REVISION, verify
from .dataset import build_sft, create_splits, load_splits


def inspect_data(args) -> None:
    manifest = verify()
    splits = create_splits() if not (DATA / "splits.json").exists() else load_splits(DATA / "splits.json")
    config = load_config(args.config)
    configure_tau(config)
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.user.user_simulator import UserSimulator
    environment = get_environment()
    tasks = get_tasks()
    UserSimulator(instructions="Offline environment inspection", llm="offline").get_init_state()
    print(json.dumps({"revision": manifest["revision"], "tasks": len(tasks),
                      "splits": {k: len(splits[k]) for k in ("train", "dev", "test")},
                      "tools": [t.name for t in environment.get_tools()],
                      "data_verified": True}, ensure_ascii=False, indent=2))


def evaluate(args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from .reporting import append_trace, start_run, write_summary
    from .worker import episode_job
    config = load_config(args.config)
    verify()
    splits = load_splits(project_path(config["splits"]))
    source = "teacher" if args.command == "collect" else "agent"
    split = "train" if source == "teacher" else args.split
    ids = splits[split][:args.limit] if args.limit else splits[split]
    for role in (source, "user", "judge"):
        endpoint_kwargs(config[role])
    rows = []
    if args.resume:
        folder = args.resume
        if json.loads((folder / "run.json").read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Resume config differs from the original run")
        rows = [json.loads(line) for line in (folder / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    else:
        folder = start_run(config, f"{source}_{split}")
    done = {(r["task_id"], r["seed"]) for r in rows if not r.get("error")}
    jobs = [{"config": config, "task_id": task_id, "split": split, "source": source,
             "seed": config["seed"] + trial, "tasks_sha256": splits["tasks_sha256"]}
            for trial in range(args.trials) for task_id in ids
            if (task_id, config["seed"] + trial) not in done]
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(episode_job, job) for job in jobs]
        for future in as_completed(futures):
            row, _ = future.result()
            append_trace(folder, row)
            rows.append(row)
            errors += bool(row.get("error"))
            write_summary(folder, rows)
            print(f"task={row['task_id']} seed={row['seed']} reward={row.get('official_reward')} error={row.get('error')}", flush=True)
    print(json.dumps(write_summary(folder, rows), ensure_ascii=False, indent=2))
    print(f"output={folder}", flush=True)
    if errors:
        raise RuntimeError(f"{errors} episodes had infrastructure/simulator errors; resume the same folder")


def build(args) -> None:
    config = load_config(args.config)
    splits = load_splits(project_path(config["splits"]))
    print(json.dumps(build_sft(args.traces, args.output, splits), indent=2))


def probe(args) -> None:
    from openai import OpenAI
    from .api import complete_chat
    spec = load_config(args.config)["teacher"]
    options = generation_kwargs(spec)
    if args.thinking is not None:
        options.setdefault("extra_body", {})["enable_thinking"] = args.thinking
    options["stream"] = True
    options["stream_options"] = {"include_usage": True}
    if args.max_tokens:
        options["max_tokens"] = args.max_tokens
    section = None

    def show(field, text):
        nonlocal section
        if field == "reasoning_content" and not args.show_reasoning:
            return
        if section != field:
            print("\n" + {"reasoning_content": "思考过程", "content": "回复", "refusal": "拒绝信息"}[field] + "：", flush=True)
            section = field
        print(text, end="", flush=True)

    with OpenAI(**endpoint_kwargs(spec)) as client:
        result = complete_chat(client.chat.completions.create, model=spec["model"],
                               messages=[{"role": "user", "content": args.prompt}], on_delta=show, **options)
    print(f"\nfinish_reason={result.choices[0].finish_reason}")
    if result.usage:
        print(json.dumps(result.usage.model_dump(exclude_none=True), ensure_ascii=False))
    if result.choices[0].finish_reason == "length":
        raise RuntimeError("Response was truncated; increase --max-tokens before assessing the reply")


def train(args) -> None:
    import asyncio
    from .training import run_training
    asyncio.run(run_training(args))


def prepare_inference(args) -> None:
    from importlib.metadata import version
    print(json.dumps({name: version(name) for name in ("openpipe-art", "vllm", "torch", "transformers")}, indent=2))


def prepare_model(args) -> None:
    from .local_model import prepare_local_model
    print(prepare_local_model(args.model))


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ServiceAgent-RL: pinned tau-bench retail + ART")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retail.example.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Verify downloads and load official environment without API calls").set_defaults(func=inspect_data)
    sub.add_parser("prepare-inference", help="Check inference dependencies installed by runtimes/art uv sync").set_defaults(func=prepare_inference)
    p = sub.add_parser("prepare-model", help="Create a linked local model view with a compatible tool template")
    p.add_argument("--model", type=Path, required=True)
    p.set_defaults(func=prepare_model)
    p = sub.add_parser("probe", help="Make one streamed teacher API request; no data collection or GPU training")
    p.add_argument("--prompt", default="你是谁？请简短回答。")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--show-reasoning", action="store_true", help="Print provider-returned reasoning_content separately")
    p.add_argument("--max-tokens", type=positive)
    p.set_defaults(func=probe)
    for name in ("evaluate", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=positive)
        p.add_argument("--trials", type=positive, default=1)
        p.add_argument("--workers", type=positive, default=1)
        p.add_argument("--resume", type=Path)
        if name == "evaluate":
            p.add_argument("--split", choices=["dev", "test"], default="dev")
        p.set_defaults(func=evaluate)
    p = sub.add_parser("build-sft")
    p.add_argument("--traces", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=build)
    p = sub.add_parser("train", help="Use runtimes/art environment and one explicitly selected GPU")
    p.add_argument("--stage", choices=["sft", "grpo", "sft-grpo", "evaluate"], required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--steps", type=positive, default=5)
    p.add_argument("--epochs", type=positive, default=1)
    p.add_argument("--target-updates", type=positive)
    p.add_argument("--task-ids", type=Path, help="JSON list of training-only curriculum task IDs")
    p.add_argument("--limit", type=positive, default=3)
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--workers", type=positive, default=2)
    p.add_argument("--resume", type=Path)
    p.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
