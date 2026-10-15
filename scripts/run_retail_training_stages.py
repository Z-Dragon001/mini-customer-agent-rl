"""Run the fixed SFT -> evaluation -> GRPO -> evaluation protocol, recording each phase."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/retail-3b"
LOGS = ROOT / "traces/retail/experiment/logs"


def event(stage, state, **extra):
    row = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, "state": state, **extra}
    with (OUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def base_complete():
    runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    if not runs.get("base"):
        return False
    folder = ROOT / runs["base"]
    if not (folder / "evaluation.json").exists():
        return False
    rows = [json.loads(line) for line in (folder / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = {(row["split"], row["task_id"]) for row in rows if not row.get("error")}
    splits = json.loads((ROOT / "data/tau2/splits.json").read_text(encoding="utf-8"))
    expected = {(part, task_id) for part in ("dev", "test") for task_id in splits[part]}
    if completed != expected:
        return False
    # The evaluation file is written just before backend cleanup finishes.
    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(used.strip()) < 1024


def phase(name, stage, arguments):
    runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    existing = runs.get(name)
    if existing and (ROOT / existing / "phase-complete.json").exists():
        event(name, "already_complete", folder=existing)
        return
    if existing and stage not in ("evaluate", "grpo"):
        raise RuntimeError(f"{name} already has an attempt; inspect its checkpoint before retrying training")
    env = {**os.environ, "RETAIL_CONFIG": "configs/retail.experiment.json", "CUDA_VISIBLE_DEVICES": "0",
           "PYTHONUNBUFFERED": "1"}
    for attempt in range(3 if stage == "evaluate" else 1):
        command = ["bash", "scripts/retail.sh", stage, *arguments]
        if existing:
            command += ["--resume", existing]
        before = set((ROOT / "traces/retail/experiment").glob(f"*_art_{stage}_*"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = LOGS / f"{name}-{stamp}-{attempt}.log"
        event(name, "started", command=command, log=str(log_path.relative_to(ROOT)))
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                new = set((ROOT / "traces/retail/experiment").glob(f"*_art_{stage}_*")) - before
                if not existing and new:
                    if len(new) != 1:
                        raise RuntimeError("Ambiguous experiment output directory")
                    existing = str(new.pop().relative_to(ROOT))
                    runs[name] = existing
                    (OUT / "runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
                time.sleep(5)
        event(name, "finished" if process.returncode == 0 else "failed", returncode=process.returncode, folder=existing)
        if process.returncode == 0:
            if not existing:
                raise RuntimeError("Completed command has no recorded output")
            (ROOT / existing / "phase-complete.json").write_text(json.dumps({"stage": name, "log": str(log_path.relative_to(ROOT))}) + "\n", encoding="utf-8")
            subprocess.run(["python", "scripts/retail_report.py"], cwd=ROOT, check=True)
            return
        if stage != "evaluate" or not existing:
            break
        time.sleep(10)
    raise RuntimeError(f"Phase {name} failed; remaining phases were not started")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=["sft", "sft_eval", "grpo", "grpo_eval"], default="sft")
    args = parser.parse_args()
    if args.start == "sft":
        event("base", "waiting_for_complete_evaluation")
        while not base_complete():
            time.sleep(10)
    stages = [
        ("sft", "sft", ["--dataset", "data/sft/experiment_train_reviewed.jsonl", "--epochs", "2"]),
        ("sft_eval", "evaluate", ["--split", "all", "--limit", "40", "--workers", "4"]),
        ("grpo", "grpo", ["--steps", "30", "--target-updates", "10", "--workers", "2", "--task-ids", "experiments/retail-3b/rl_tasks.json"]),
        ("grpo_eval", "evaluate", ["--split", "all", "--limit", "40", "--workers", "4"]),
    ]
    start = [row[0] for row in stages].index(args.start)
    for name, stage, arguments in stages[start:]:
        phase(name, stage, arguments)
    event("experiment", "training_and_evaluation_complete")


if __name__ == "__main__":
    main()
