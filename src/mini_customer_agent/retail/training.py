"""Single-GPU ART 0.5.16 runner; synchronous tau episodes run off the event loop."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import random
import traceback
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from dataclasses import asdict
import math

from .config import configure_tau, endpoint_kwargs, load_config, project_path
from .data import REVISION, sha256, verify
from .dataset import load_splits, validate_messages
from .reporting import append_trace, start_run, write_summary
from .rollout import run_episode, training_reward
from .rewards import reward_breakdown, group_is_informative


def require_single_gpu() -> None:
    selected = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not selected or "," in selected or selected == "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one available GPU before starting ART")


def validate_sft_file(path: Path, splits: dict) -> int:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if (manifest["data_sha256"] != sha256(path) or manifest["tau_revision"] != REVISION
            or manifest["tasks_sha256"] != splits["tasks_sha256"]):
        raise ValueError("SFT data or provenance changed; rebuild with build-sft")
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row["metadata"]
        if (meta["task_id"] not in splits["train"] or meta["tau_revision"] != REVISION
                or meta["tasks_sha256"] != splits["tasks_sha256"]):
            raise ValueError("SFT data contains non-training or mismatched tasks")
        validate_messages(row["messages"])
        count += 1
    if not count or count != manifest["accepted"]:
        raise ValueError("Empty or inconsistent SFT dataset")
    return count


def model_config(spec: dict) -> dict:
    return {
        "init_args": {"max_seq_length": spec["max_seq_length"], "load_in_4bit": spec.get('load_in_4bit', True),
                      "use_gradient_checkpointing": "unsloth",
                      "gpu_memory_utilization": spec["gpu_memory_utilization"]},
        "peft_args": {"r": 16, "lora_alpha": 32, "lora_dropout": 0,
                      "use_gradient_checkpointing": "unsloth"},
        "engine_args": {"max_model_len": spec["max_seq_length"],
                        "gpu_memory_utilization": spec["gpu_memory_utilization"],
                        "max_num_seqs": 2, "enforce_eager": spec.get("enforce_eager", True)},
        "trainer_args": {"per_device_train_batch_size": 1, "seed": 42,
                         "gradient_checkpointing": True,
                         "gradient_checkpointing_kwargs": {"use_reentrant": True}},
    }


def check_sft_lengths(path: Path, spec: dict) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["base_model"])
    oversized = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = tokenizer.apply_chat_template(row["messages"], tools=row["tools"],
                                                tokenize=True, add_generation_prompt=False)
        if hasattr(tokens, "keys"):
            tokens = tokens["input_ids"]
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        if len(tokens) > spec["max_seq_length"]:
            oversized.append((index, len(tokens)))
    if oversized:
        raise ValueError(f"SFT samples exceed max_seq_length; ART would drop them. Increase context or prepare a reviewed subset: {oversized[:10]}")


async def run_training(args) -> None:
    require_single_gpu()
    config = load_config(args.config)
    configure_tau(config)
    verify()
    splits = load_splits(project_path(config["splits"]))
    spec = config["art"]
    if spec["rollouts_per_task"] < 2:
        raise ValueError("GRPO requires at least two trajectories per task")
    if args.stage in {"sft", "sft-grpo"}:
        if args.dataset is None:
            raise ValueError("SFT requires --dataset from build-sft")
        validate_sft_file(args.dataset, splits)
    if args.stage != "sft":
        endpoint_kwargs(config["user"])
        endpoint_kwargs(config["judge"])
    # Import GPU packages only after input validation and explicit GPU selection.
    import importlib.metadata
    if importlib.metadata.version("openpipe-art") != "0.5.16":
        raise RuntimeError("Use the locked runtimes/art environment (ART 0.5.16)")
    import art
    from .art_service import RetailLocalBackend
    from art.utils.sft import train_sft_from_file
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ART requires exactly one visible CUDA GPU")
    if args.stage in {"sft", "sft-grpo"}:
        check_sft_lengths(args.dataset, spec)
    os.environ.setdefault("WANDB_MODE", "offline")
    if getattr(args, "resume", None):
        folder = args.resume
        if json.loads((folder / "run.json").read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Resume config differs from the original run")
        from .config import provenance
        from datetime import datetime, timezone
        with (folder / 'resume_provenance.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'resumed_at': datetime.now(timezone.utc).isoformat(),
                                     **provenance(config)}, ensure_ascii=False) + '\n')
    else:
        folder = start_run(config, f"art_{args.stage}")
    (folder / "arguments.json").write_text(json.dumps({
        "stage": args.stage, "steps": args.steps, "epochs": args.epochs, "limit": args.limit,
        "workers": getattr(args, "workers", 2),
        "split": args.split, "dataset_sha256": sha256(args.dataset) if args.dataset else None,
        "task_ids_sha256": sha256(args.task_ids) if getattr(args, "task_ids", None) else None,
        "target_updates": getattr(args, "target_updates", None),
        "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }, indent=2) + "\n", encoding="utf-8")
    (folder / "model_config.json").write_text(json.dumps(model_config(spec), indent=2) + "\n", encoding="utf-8")
    backend = RetailLocalBackend(path=str(project_path(spec["path"])),
                                 persist_optimizer=spec.get('persist_optimizer', False),
                                 stable_logprobs=spec.get('stable_logprobs', False),
                                 packed_positions=spec.get('packed_positions', False),
                                 packed_attention=spec.get('packed_attention', False))
    model = art.TrainableModel(name=spec["name"], project=spec["project"], base_model=spec["base_model"],
                               _internal_config=model_config(spec))
    marker = project_path(spec["path"]) / "retail-state" / spec["project"] / f"{spec['name']}.json"
    identity = {"base_model": spec["base_model"], "tau_revision": REVISION,
                "tasks_sha256": splits["tasks_sha256"], "split_sha256": sha256(project_path(config["splits"]))}
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing ART experiment has different data/model; choose a new art.name")
    else:
        state = identity.copy()
    if args.stage == "grpo" and not state.get("sft_complete"):
        raise ValueError("Run SFT first using the same art.name; SFT checkpoint marker is missing")
    rows = ([json.loads(line) for line in (folder / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
            if (folder / "traces.jsonl").exists() else [])
    client = None
    pool = None
    try:
        await model.register(backend, _openai_client_config={"server_args": {"host": "127.0.0.1"}})
        step = await model.get_step()
        if args.stage in {"sft", "sft-grpo"}:
            if step or state.get("sft_complete"):
                raise ValueError("Refusing to repeat SFT on an existing checkpoint; choose a new art.name")
            original_sft = backend._train_sft

            async def recorded_sft(*positional, **keywords):
                async for metrics in original_sft(*positional, **keywords):
                    with (folder / "sft_metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics) + "\n")
                    if (not all(math.isfinite(metrics[key]) for key in ("loss", "grad_norm"))
                            or metrics["num_trainable_tokens"] <= 0):
                        raise RuntimeError("SFT produced non-finite gradients/loss or no target tokens; inspect saved metrics")
                    yield metrics

            backend._train_sft = recorded_sft
            random.seed(config["seed"])
            await train_sft_from_file(model, str(args.dataset.resolve()), epochs=args.epochs,
                                      batch_size=1, peak_lr=spec.get("sft_learning_rate", 2e-5), verbose=True)
            state.update(sft_complete=True, sft_step=await model.get_step(), dataset_sha256=sha256(args.dataset))
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        if args.stage != "sft":
            pool = ProcessPoolExecutor(max_workers=getattr(args, "workers", 2),
                                       mp_context=multiprocessing.get_context("spawn"))
        loop = asyncio.get_running_loop()

        async def episode(task_id, split, seed, temperature, group_id=None, rollout_index=None, evaluation_round=None):
            policy_step = await model.get_step()
            try:
                from .worker import episode_job
                row, history = await loop.run_in_executor(pool, episode_job, {
                    "config": config, "task_id": task_id, "split": split,
                    "source": "art", "seed": seed, "tasks_sha256": splits["tasks_sha256"],
                    "model": model.get_inference_name(step=policy_step),
                    "connection": {"base_url": model.inference_base_url, "api_key": model.inference_api_key,
                                   "timeout": 180.0, "max_retries": 2},
                    "generation": {"temperature": temperature, "max_tokens": config["agent"]["max_tokens"],
                                   "logprobs": True}})
            except Exception as exc:
                failure = {"task_id": task_id, "split": split, "policy_step": policy_step,
                           "success": False, "error": type(exc).__name__,
                           "error_frames": [{"file": f.filename, "line": f.lineno, "function": f.name}
                                            for f in traceback.extract_tb(exc.__traceback__)]}
                append_trace(folder, failure)
                rows.append(failure)
                write_summary(folder, [r for r in rows if r["split"] == split])
                raise RuntimeError(f"Episode failed ({type(exc).__name__}); see {folder}") from None
            row["policy_step"] = policy_step
            if group_id is not None:
                row.update(group_id=group_id, rollout_index=rollout_index)
            if evaluation_round is not None:
                row['evaluation_round'] = evaluation_round
            if not row.get('error'):
                breakdown = reward_breakdown(row, spec.get('reward') if split == 'train' else None)
                row['process_signals'] = breakdown['signals']
                if split == 'train':
                    row['training_reward'] = breakdown
            append_trace(folder, row)
            rows.append(row)
            write_summary(folder, [r for r in rows if r["split"] == split])
            print(f"task={task_id} split={split} step={policy_step} reward={row.get('official_reward')} error={row.get('error')}", flush=True)
            if row.get("error"):
                return None
            return art.Trajectory(messages_and_choices=history, tools=row["tools"], reward=breakdown['total'],
                                  metrics={"success": row["success"], "tool_errors": row["tool_errors"],
                                           "tool_calls": row["tool_calls"]},
                                  metadata={"task_id": task_id, "split": split, "policy_step": policy_step,
                                            **({'business_progress': row['business_progress']['score']}
                                               if 'business_progress' in row else {})})

        async def development_eval(round_id):
            """Evaluate a frozen current checkpoint without feeding dev data to training."""
            from .reporting import summarize
            done = {r['task_id'] for r in rows if r.get('evaluation_round') == round_id
                    and r['split'] == 'dev' and not r.get('error')}
            semaphore = asyncio.Semaphore(getattr(args, 'workers', 2))
            async def job(task_id):
                async with semaphore:
                    return await episode(task_id, 'dev', config['seed'], 0.0, evaluation_round=round_id)
            missing = [task_id for task_id in splits['dev'] if task_id not in done]
            for attempt in range(3):
                results = await asyncio.gather(*(job(task_id) for task_id in missing))
                missing = [task_id for task_id, result in zip(missing, results) if result is None]
                if not missing:
                    break
            if missing:
                raise RuntimeError('Incomplete development evaluation; resume before more training')
            evaluated = [r for r in rows if r.get('evaluation_round') == round_id
                         and r['split'] == 'dev' and not r.get('error')]
            result = {'round': round_id, 'policy_step': await model.get_step(),
                      'summary': summarize(evaluated),
                      'successes': sum(r['success'] for r in evaluated),
                      'budget_exhausted': sum(r['process_signals']['budget_exhausted'] for r in evaluated),
                      'repeated_calls': sum(r['process_signals']['repeated_calls'] for r in evaluated)}
            (folder / f'dev-round-{round_id:04d}.json').write_text(json.dumps(result, indent=2)+'\n')
            print('development=' + json.dumps(result), flush=True)

        if args.stage in {"evaluate", "sft-grpo"}:
            # In combined mode this is the SFT dev baseline, captured before any RL update.
            split = "dev" if args.stage == "sft-grpo" else args.split
            done = {(r["split"], r["task_id"]) for r in rows if not r.get("error")}
            targets = ["dev", "test"] if split == "all" else [split]
            jobs = [(part, task_id) for part in targets for task_id in splits[part][:args.limit]
                    if (part, task_id) not in done]
            # Keep slots busy when another conversation is long; the policy is
            # fixed throughout evaluation, so there is no batch barrier needed.
            width = getattr(args, "workers", 2)
            semaphore = asyncio.Semaphore(width)

            async def bounded_episode(part, task_id):
                async with semaphore:
                    return await episode(task_id, part, config["seed"], 0.0)

            results = await asyncio.gather(*(bounded_episode(part, task_id) for part, task_id in jobs))
            for retry in range(2):
                missing_jobs = [job for job, result in zip(jobs, results) if result is None]
                if not missing_jobs:
                    break
                print(f'evaluation_infrastructure_retry={retry + 1} episodes={len(missing_jobs)}', flush=True)
                jobs = missing_jobs
                results = await asyncio.gather(*(bounded_episode(part, task_id) for part, task_id in jobs))
            failures = sum(result is None for result in results)
            from .reporting import summarize
            summary = {part: summarize([r for r in rows if r["split"] == part and not r.get("error")]) for part in targets}
            (folder / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            if failures:
                raise RuntimeError(f"{failures} evaluation episodes need retry; resume {folder}")
        if args.stage in {"grpo", "sft-grpo"}:
            ids = splits["train"][:args.limit]
            if getattr(args, "task_ids", None):
                ids = json.loads(args.task_ids.read_text(encoding="utf-8"))
                if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(splits["train"]):
                    raise ValueError("RL curriculum must contain unique training task IDs only")
            previous_groups = ([json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
                               if (folder / "updates.jsonl").exists() else [])
            if [r["iteration"] for r in previous_groups] != list(range(len(previous_groups))):
                raise ValueError("Non-contiguous RL history; inspect before resuming")
            updates = sum(r["updated"] for r in previous_groups)
            if await model.get_step() != state["sft_step"] + updates:
                raise ValueError("Checkpoint and completed RL groups disagree; inspect before resuming")
            # Reconstruct the original curriculum order, even after checkpoints
            # have advanced. An incomplete group is resampled in its entirety.
            rng = random.Random(config["seed"] + state["sft_step"])
            eval_every = spec.get('eval_every', 0)
            if eval_every:
                # A completed group whose dev evaluation was interrupted must
                # finish its evaluation before the next policy update.
                boundary = len(previous_groups)
                if boundary > 0 and boundary % eval_every == 0:
                    await development_eval(boundary)
            if getattr(args, "resume", None):
                with (folder / "resumptions.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"completed_groups": len(previous_groups), "policy_updates": updates,
                                             "raw_rows_before_resume": len(rows),
                                             "note": "Incomplete group's old rows remain raw evidence; latest complete group is selected"}) + "\n")
            for iteration in range(args.steps):
                if iteration % len(ids) == 0:
                    rng.shuffle(ids)
                if iteration < len(previous_groups):
                    continue
                if getattr(args, "target_updates", None) and updates >= args.target_updates:
                    break
                task_id = ids[iteration % len(ids)]
                seed = (config['seed'] + 1000 + iteration if spec.get('fixed_group_seeds') else
                        config["seed"] + (await model.get_step()) * 100 + iteration)
                # Same task and user seed within the group; policy sampling is stochastic.
                trajectories = await asyncio.gather(*(episode(task_id, "train", seed, spec["temperature"], iteration, i)
                                                      for i in range(spec["rollouts_per_task"])))
                # Retry infrastructure failures only. Valid unsuccessful tasks
                # keep reward zero and are never replaced by another rollout.
                for retry in range(2):
                    missing = [i for i, trajectory in enumerate(trajectories) if trajectory is None]
                    if not missing:
                        break
                    print(f"group={iteration + 1} infrastructure_retry={retry + 1} episodes={len(missing)}", flush=True)
                    replacements = await asyncio.gather(*(episode(task_id, "train", seed, spec["temperature"], iteration, i)
                                                          for i in missing))
                    for index, trajectory in zip(missing, replacements):
                        trajectories[index] = trajectory
                if any(t is None for t in trajectories):
                    raise RuntimeError("Incomplete RL group: simulator/infrastructure errors are not rewards")
                group = art.TrajectoryGroup(trajectories, metadata={"task_id": task_id})
                await model.log([group], split="train")
                informative = group_is_informative([t.reward for t in trajectories],
                                                  [bool(t.metrics['success']) for t in trajectories],
                                                  spec.get('reward', {}),
                                                  progresses=[t.metadata.get('business_progress', 0.0) for t in trajectories])
                if informative:
                    original_train_model = backend._train_model

                    async def recorded_rl(*positional, **keywords):
                        async for batch_metrics in original_train_model(*positional, **keywords):
                            with (folder / "grpo_batch_metrics.jsonl").open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps({"group_id": iteration, "metrics": batch_metrics}) + "\n")
                            yield batch_metrics

                    backend._train_model = recorded_rl
                    try:
                        result = await backend.train(model, [group], learning_rate=spec["learning_rate"],
                                                     beta=spec.get('kl_beta', 0.0),
                                                     kl_ref_adapter_path=spec.get('kl_reference'),
                                                     ppo=True, epsilon=spec.get('epsilon', 0.2),
                                                     logprob_calculation_chunk_size=256)
                    finally:
                        backend._train_model = original_train_model
                    with (folder / "grpo_metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(result), default=str) + "\n")
                    if any(isinstance(value, (int, float)) and not math.isfinite(value)
                           for value in result.metrics.values()):
                        raise RuntimeError("GRPO produced non-finite metrics; inspect checkpoint before continuing")
                    updates += 1
                with (folder / "updates.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"iteration": iteration, "task_id": task_id,
                                             "seed": seed,
                                             "rewards": [t.reward for t in trajectories],
                                             "updated": informative, "policy_step": await model.get_step()}) + "\n")
                print(f"group={iteration + 1} updated={informative} step={await model.get_step()}", flush=True)
                if eval_every and ((iteration + 1) % eval_every == 0 or iteration + 1 == args.steps):
                    await development_eval(iteration + 1)
                if getattr(args, "target_updates", None) and updates >= args.target_updates:
                    break
        print(f"output={folder}\ncheckpoint_step={await model.get_step()}")
    finally:
        if pool is not None:
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
        # ART 0.5.16 can leave an engine orphaned after its model service exits.
        # Capture only this runner's descendants before shutting down the service.
        import psutil
        children = psutil.Process().children(recursive=True)
        try:
            if client is not None:
                await client.close()
            await backend.close()
        finally:
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = await asyncio.to_thread(psutil.wait_procs, children, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
