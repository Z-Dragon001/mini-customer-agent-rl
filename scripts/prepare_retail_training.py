"""Build and length-filter experiment demonstrations without truncating conversations."""
import json
from collections import Counter
from pathlib import Path

from mini_customer_agent.retail.config import load_config, project_path
from mini_customer_agent.retail.data import ROOT, sha256
from mini_customer_agent.retail.dataset import build_sft, load_splits


def main():
    from transformers import AutoTokenizer
    config = load_config(ROOT / "configs/retail.experiment.json")
    runs = json.loads((ROOT / "experiments/retail-3b/runs.json").read_text(encoding="utf-8"))
    source = ROOT / runs["teacher"] / "traces.jsonl"
    attempts = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    completed = {(r["task_id"], r["seed"]): r for r in attempts if not r.get("error")}
    splits = load_splits(project_path(config["splits"]))
    candidate = ROOT / "data/sft/experiment_candidates.jsonl"
    output = ROOT / "data/sft/experiment_train.jsonl"
    if output.exists():
        raise ValueError("Training data already exists; refusing to overwrite experiment inputs")
    if not candidate.exists():
        build_sft(source, candidate, splits)
    manifest = json.loads(candidate.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != sha256(source):
        raise ValueError("Collection changed since candidate generation")
    tokenizer = AutoTokenizer.from_pretrained(config["art"]["base_model"])
    kept, excluded = [], []
    for index, line in enumerate(candidate.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        encoded = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=True,
                                                add_generation_prompt=False)
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        length = len(encoded)
        if length > config["art"]["max_seq_length"]:
            excluded.append({"candidate_row": index + 1, "task_id": row["metadata"]["task_id"], "tokens": length})
            continue
        row["metadata"]["token_count"] = length
        kept.append(row)
    if not kept:
        raise ValueError("No training samples fit the configured context")
    with output.open("x", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {**manifest, "accepted_before_length_filter": manifest["accepted"], "accepted": len(kept),
              "source_attempts": len(attempts), "completed_task_seed_pairs": len(completed),
              "successful_task_seed_pairs": sum(r["success"] for r in completed.values()),
              "infrastructure_attempts": dict(Counter(r["error"] for r in attempts if r.get("error"))),
              "data_sha256": sha256(output), "candidate_sha256": sha256(candidate),
              "max_seq_length": config["art"]["max_seq_length"], "excluded_for_length": excluded}
    output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    task_ids = {row["metadata"]["task_id"] for row in kept}
    curriculum = [task_id for task_id in splits["train"] if task_id in task_ids]
    (ROOT / "experiments/retail-3b/rl_tasks.json").write_text(json.dumps(curriculum, indent=2) + "\n", encoding="utf-8")
    (ROOT / "experiments/retail-3b/data_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": len(kept), "unique_training_tasks": len(task_ids),
                      "excluded_for_length": len(excluded), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
