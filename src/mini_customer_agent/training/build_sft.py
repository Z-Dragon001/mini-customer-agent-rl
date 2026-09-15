from __future__ import annotations

import copy
import json
from pathlib import Path

from ..tool_schemas import TOOL_SCHEMAS


def build_sft_dataset(
    trace_path: str | Path, output_path: str | Path, *, only_success: bool = True
) -> int:
    source = Path(trace_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            trace = json.loads(line)
            if only_success and not trace.get("success"):
                continue
            messages = copy.deepcopy(trace["messages"])
            for message in messages:
                for call in message.get("tool_calls", []):
                    arguments = call.get("function", {}).get("arguments")
                    if isinstance(arguments, str):
                        try:
                            call["function"]["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass
            sample = {
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "metadata": {
                    "task_id": trace["task_id"],
                    "intent": trace["intent"],
                    "source_model": trace["model"],
                    "success": trace["success"],
                },
            }
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count
