from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .schemas import FailureType, Task


ALLOWED_TOOLS = {
    "lookup": {"query_order"},
    "change_address": {"query_order", "change_address"},
    "refund": {"query_order", "request_refund"},
    "transfer": {"transfer_to_human"},
}


def classify_failure(
    task: Task, steps: list[dict[str, Any]], success: bool
) -> tuple[FailureType, list[str]]:
    if success:
        return FailureType.NONE, []

    labels: list[str] = []
    tool_steps = [step for step in steps if step.get("kind") == "tool"]
    codes = [step.get("result", {}).get("code") for step in tool_steps]

    if any(step.get("tool_name") not in ALLOWED_TOOLS[task.intent] for step in tool_steps):
        labels.append(FailureType.WRONG_TOOL.value)
    if any(code in {"PARAMETER_ERROR", "NOT_FOUND", "POLICY_DENIED"} for code in codes):
        labels.append(FailureType.BAD_PARAMETERS.value)
    if "MISSING_INFORMATION" in codes:
        labels.append(FailureType.MISSING_INFORMATION.value)
    if any(code in {"TOOL_EXCEPTION", "MODEL_EXCEPTION"} for code in codes):
        labels.append(FailureType.UNHANDLED_EXCEPTION.value)

    failed_calls = [
        json.dumps(
            {"tool": step.get("tool_name"), "arguments": step.get("arguments")},
            ensure_ascii=False,
            sort_keys=True,
        )
        for step in tool_steps
        if not step.get("result", {}).get("ok", False)
    ]
    if any(count > 1 for count in Counter(failed_calls).values()):
        labels.append(FailureType.UNHANDLED_EXCEPTION.value)
    if not labels:
        labels.append(FailureType.PREMATURE_TERMINATION.value)

    # A stable priority makes aggregate reports reproducible while preserving all labels.
    priority = [
        FailureType.UNHANDLED_EXCEPTION,
        FailureType.WRONG_TOOL,
        FailureType.BAD_PARAMETERS,
        FailureType.MISSING_INFORMATION,
        FailureType.PREMATURE_TERMINATION,
    ]
    primary = next(item for item in priority if item.value in labels)
    return primary, list(dict.fromkeys(labels))

