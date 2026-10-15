from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from .data import DATA, UPSTREAM, REVISION, sha256


def create_splits(dev_fraction: float = 0.2, seed: int = 42) -> dict:
    directory = UPSTREAM / "data/tau2/domains/retail"
    official = json.loads((directory / "split_tasks.json").read_text(encoding="utf-8"))
    tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    ids = {str(task["id"]) for task in tasks}
    if not {"train", "test"} <= official.keys():
        raise ValueError("Pinned upstream has no train/test split; do not improvise benchmark splits")
    train = [str(x) for x in official["train"]]
    test = [str(x) for x in official["test"]]
    if "dev" in official:
        dev = [str(x) for x in official["dev"]]
        strategy = "upstream train/dev/test"
    else:
        if not 0 < dev_fraction < 1:
            raise ValueError("dev_fraction must be between 0 and 1")
        # Group identical user scenarios to avoid placing exact duplicates in dev/train.
        by_id = {str(task["id"]): task for task in tasks}
        groups = {}
        for task_id in train:
            key = json.dumps(by_id[task_id]["user_scenario"], sort_keys=True)
            groups.setdefault(key, []).append(task_id)
        grouped = list(groups.values())
        random.Random(seed).shuffle(grouped)
        count = max(1, round(len(grouped) * dev_fraction))
        dev = [x for group in grouped[:count] for x in group]
        train = [x for group in grouped[count:] for x in group]
        strategy = "official test; local dev held out from official train, grouped by exact user scenario (not template-family split)"
    result = {"revision": REVISION, "tasks_sha256": sha256(directory / "tasks.json"),
              "strategy": strategy, "seed": seed, "train": train, "dev": dev, "test": test}
    validate_splits(result, ids)
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / "splits.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != result:
        raise ValueError("Existing split differs; preserve it and explicitly choose a new experiment version")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def validate_splits(splits: dict, all_ids: set[str]) -> None:
    sets = []
    for name in ("train", "dev", "test"):
        values = splits[name]
        current = set(values)
        if not current or len(current) != len(values) or not current <= all_ids:
            raise ValueError(f"Invalid, empty or duplicate {name} split")
        if any(current & previous for previous in sets):
            raise ValueError("Task leakage across splits")
        sets.append(current)


def load_splits(path: Path) -> dict:
    splits = json.loads(path.read_text(encoding="utf-8"))
    tasks_path = UPSTREAM / "data/tau2/domains/retail/tasks.json"
    if splits["revision"] != REVISION or splits["tasks_sha256"] != sha256(tasks_path):
        raise ValueError("Split provenance does not match downloaded tasks")
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    validate_splits(splits, {str(t["id"]) for t in tasks})
    return splits


def validate_messages(messages: list[dict]) -> None:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT conversation must end in an assistant turn")
    pending = set()
    for message in messages:
        role = message["role"]
        if role == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending:
                raise ValueError("Orphan or duplicate tool response")
            pending.remove(call_id)
        else:
            if pending:
                raise ValueError("Missing tool responses before next turn")
            for call in message.get("tool_calls", []) or []:
                if call["id"] in pending:
                    raise ValueError("Duplicate tool call ID")
                arguments = call["function"]["arguments"]
                if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                    raise ValueError("ART requires JSON object arguments serialized as strings")
                pending.add(call["id"])
    if pending:
        raise ValueError("Conversation ends with unresolved tool calls")


def build_sft(source: Path, destination: Path, splits: dict) -> dict:
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Choose a new SFT output path to preserve existing data")
    accepted, seen = [], set()
    stats = {"accepted": 0, "failed": 0, "invalid": 0, "duplicate": 0}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "train" or row["task_id"] not in splits["train"]:
            raise ValueError("SFT input contains non-training tasks")
        if row.get("tau_revision") != REVISION or row.get("tasks_sha256") != splits["tasks_sha256"]:
            raise ValueError("Trace provenance does not match tasks")
        if row.get("source") != "teacher":
            raise ValueError("Only teacher collection traces are accepted; oracle/test traces cannot be used")
        if not row.get("success") or row.get("error") or row.get("tool_errors", 0) or row.get("truncated_generations", 0):
            stats["failed"] += 1
            continue
        messages = row["messages"]
        # The last user simulator STOP message is not part of a demonstration target.
        while messages and messages[-1]["role"] != "assistant":
            messages = messages[:-1]
        try:
            validate_messages(messages)
        except (KeyError, ValueError, TypeError):
            stats["invalid"] += 1
            continue
        signature = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        if signature in seen:
            stats["duplicate"] += 1
            continue
        seen.add(signature)
        accepted.append({"messages": messages, "tools": row["tools"], "metadata": {
            "task_id": row["task_id"], "tau_revision": REVISION, "tasks_sha256": splits["tasks_sha256"],
            "source_model": row["model"], "review_status": "auto_filtered_not_human_reviewed"}})
    if not accepted:
        raise ValueError("No eligible samples; no SFT file written")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["accepted"] = len(accepted)
    destination.with_suffix(".manifest.json").write_text(json.dumps({
        **stats, "source_sha256": sha256(source), "data_sha256": sha256(destination),
        "tasks_sha256": splits["tasks_sha256"], "tau_revision": REVISION,
        "note": "Automatic task/format/error filtering does not certify every policy requirement. Audit a sample."
    }, indent=2) + "\n", encoding="utf-8")
    return stats
