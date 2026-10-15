"""Verify a finished experiment from saved artifacts, without calling any API."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/retail-3b"


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def main():
    runs = json.loads((OUT / "runs.json").read_text())
    splits = json.loads((ROOT / "data/tau2/splits.json").read_text())
    expected = {(part, task_id, 42) for part in ("dev", "test") for task_id in splits[part]}
    checks, configs = {}, []
    for stage in ("base", "sft_eval", "grpo_eval"):
        assert runs.get(stage), f"{stage} has not started"
        folder = ROOT / runs[stage]
        valid = [r for r in rows(folder / "traces.jsonl") if not r.get("error")]
        assert {(r["split"], r["task_id"], r["seed"]) for r in valid} == expected, stage
        assert len(valid) == len(expected), f"{stage}: duplicate completed evaluations"
        assert len({r["policy_step"] for r in valid}) == 1, f"{stage}: mixed checkpoints"
        checks[stage] = {"completed_tasks": len(valid), "policy_step": valid[0]["policy_step"]}
        configs.append(json.loads((folder / "run.json").read_text())["config"])
    assert configs[0] == configs[1] == configs[2], "Evaluation configurations differ"
    sft_folder = ROOT / runs["sft"]
    metrics = rows(sft_folder / "sft_metrics.jsonl")
    assert len(metrics) == 116
    assert all(math.isfinite(r[k]) for r in metrics for k in ("loss", "grad_norm"))
    assert all(r["num_trainable_tokens"] > 0 for r in metrics)
    data_path = ROOT / "data/sft/experiment_train_reviewed.jsonl"
    samples = rows(data_path)
    manifest = json.loads(data_path.with_suffix(".manifest.json").read_text())
    assert len(samples) == manifest["accepted"] == 58
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest["data_sha256"]
    assert all(r["metadata"]["task_id"] in splits["train"] for r in samples)
    checks["sft"] = {"batches": len(metrics), "samples": len(samples), "data_sha256": manifest["data_sha256"]}
    grpo_folder = ROOT / runs["grpo"]
    updates = rows(grpo_folder / "updates.jsonl")
    valid = list({(r["group_id"], r["rollout_index"]): r
                  for r in rows(grpo_folder / "traces.jsonl") if not r.get("error")}.values())
    step = 1
    curriculum = json.loads((OUT / "rl_tasks.json").read_text())
    rng = random.Random(43)
    for index, group in enumerate(updates):
        if index % len(curriculum) == 0:
            rng.shuffle(curriculum)
        assert group["iteration"] == index
        assert group["task_id"] == curriculum[index % len(curriculum)]
        assert group["seed"] == 42 + step * 100 + index
        group_rows = sorted([r for r in valid if r["group_id"] == index], key=lambda r: r["rollout_index"])
        assert [r["rollout_index"] for r in group_rows] == [0, 1, 2, 3]
        assert all(r["task_id"] == group["task_id"] and r["task_id"] in splits["train"] for r in group_rows)
        assert all(r["seed"] == group["seed"] and r["policy_step"] == step for r in group_rows)
        assert [r["official_reward"] for r in group_rows] == group["rewards"]
        assert group["updated"] == (len(set(group["rewards"])) > 1)
        step += int(group["updated"])
        assert group["policy_step"] == step
    assert len(updates) == 30 or step - 1 == 10, "RL stopped before the fixed budget/target"
    assert checks["base"]["policy_step"] == 0 and checks["sft_eval"]["policy_step"] == 1
    assert checks["grpo_eval"]["policy_step"] == step
    batch_metrics = rows(grpo_folder / "grpo_batch_metrics.jsonl") if step > 1 else []
    assert len(rows(grpo_folder / "grpo_metrics.jsonl")) == step - 1 if step > 1 else True
    assert all(math.isfinite(value) for r in batch_metrics for value in r["metrics"].values() if isinstance(value, (int, float)))
    checks["grpo"] = {"groups": len(updates), "policy_updates": step - 1,
                       "optimizer_batches": len(batch_metrics), "rollouts": len(valid)}
    # Process time and sampled device memory include startup and cleanup.
    samples_gpu = []
    with (ROOT / "traces/retail/experiment/logs/gpu.csv").open() as handle:
        for r in csv.reader(handle):
            timestamp = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            samples_gpu.append((timestamp, int(r[2]), int(r[3])))
    active, resource_rows = {}, []
    for event in rows(OUT / "events.jsonl"):
        stage = event["stage"]
        if event["state"] == "started":
            active[stage] = event
        if event["state"] in {"finished", "failed"} and stage in active:
            start = datetime.fromisoformat(active[stage]["time"])
            end = datetime.fromisoformat(event["time"])
            measurements = [r for r in samples_gpu if start <= r[0] <= end]
            resource_rows.append({"stage": stage, "state": event["state"], "folder": event.get("folder"),
                                  "elapsed_seconds": (end-start).total_seconds(),
                                  "sampled_peak_gpu_mib": max((r[1] for r in measurements), default=None),
                                  "gpu_index": 0, "sample_period_seconds": 5})
    (OUT / "resources.json").write_text(json.dumps(resource_rows, indent=2) + "\n")
    (OUT / "audit.json").write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
                                                 "checks": checks, "passed": True}, indent=2) + "\n")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
