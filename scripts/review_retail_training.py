"""Apply the documented Codex sample audit; never label it human review."""
import hashlib
import json
from pathlib import Path

from mini_customer_agent.retail.data import ROOT, sha256


def signature(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def main():
    source = ROOT / "data/sft/experiment_train.jsonl"
    target = ROOT / "data/sft/experiment_train_reviewed.jsonl"
    if target.exists():
        raise ValueError("Reviewed dataset already exists; refusing to overwrite")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    reviewed_tasks = {"2", "1", "58", "48", "113", "75", "11"}
    audited, seen, kept = [], set(), []
    for index, row in enumerate(rows, 1):
        task_id = row["metadata"]["task_id"]
        if task_id == "50":
            assert any("I can help you place a new order" in (m.get("content") or "")
                       for m in row["messages"] if m["role"] == "assistant")
            audited.append({"source_row": index, "task_id": task_id, "messages_sha256": signature(row["messages"]),
                            "reviewer": "Codex", "decision": "exclude",
                            "reason": "Promises to help place a new order although the environment has no order-creation tool"})
            continue
        if task_id in reviewed_tasks and task_id not in seen:
            seen.add(task_id)
            row["metadata"]["review_status"] = "codex_sample_review_not_human_review"
            audited.append({"source_row": index, "task_id": task_id, "messages_sha256": signature(row["messages"]),
                            "reviewer": "Codex", "decision": "retain",
                            "reason": "Inspected write-tool arguments and preceding explicit user confirmation; no issue found in this check"})
        kept.append(row)
    assert len(rows) - len(kept) == 2 and seen == reviewed_tasks
    with target.open("x", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = json.loads(source.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    manifest.update(accepted=len(kept), data_sha256=sha256(target), pre_audit_data_sha256=sha256(source),
                    codex_audit=audited, audit_excluded=2, audit_scope="Sample inspection by Codex, not human or exhaustive compliance review")
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    out = ROOT / "experiments/retail-3b"
    (out / "data_summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    split = json.loads((ROOT / "data/tau2/splits.json").read_text(encoding="utf-8"))
    ids = {row["metadata"]["task_id"] for row in kept}
    (out / "rl_tasks.json").write_text(json.dumps([i for i in split["train"] if i in ids], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training_rows": len(kept), "training_tasks": len(ids), "audit_excluded": 2,
                      "audited_retained": len(seen), "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
