from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from .environment import CustomerServiceEnvironment
from .failures import classify_failure
from .model_clients import CompletionContext, ModelClient
from .schemas import ModelUsage, Task, ToolResult
from .tool_schemas import SYSTEM_PROMPT, TOOL_SCHEMAS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentHarness:
    def __init__(self, model: ModelClient):
        self.model = model

    def run(self, task: Task) -> dict[str, Any]:
        env = CustomerServiceEnvironment(task)
        episode_id = f"{task.task_id}_{uuid.uuid4().hex[:8]}"
        started_at = _utc_now()
        started_perf = time.perf_counter()
        initial_state = env.state_dict()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.initial_message},
        ]
        steps: list[dict[str, Any]] = []
        usage = ModelUsage()
        followups = iter(task.followup_messages)
        final_response = ""
        termination = "max_steps"

        for index in range(task.max_steps):
            step_started = time.perf_counter()
            try:
                action = self.model.complete(
                    CompletionContext(messages, TOOL_SCHEMAS, task, env)
                )
                usage.add(action.usage)
            except Exception as exc:
                result = ToolResult(False, "MODEL_EXCEPTION", f"{type(exc).__name__}: {exc}")
                steps.append(
                    {
                        "index": index,
                        "kind": "tool",
                        "tool_name": "__model__",
                        "arguments": {},
                        "result": result.to_dict(),
                        "latency_ms": round((time.perf_counter() - step_started) * 1000, 3),
                    }
                )
                termination = "model_exception"
                break

            if action.kind == "tool":
                call_id = action.call_id or f"call_{uuid.uuid4().hex[:8]}"
                assistant_message = {
                    "role": "assistant",
                    "content": action.content or None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": action.tool_name,
                                "arguments": json.dumps(action.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
                messages.append(assistant_message)
                result = env.call_tool(action.tool_name or "", action.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": action.tool_name,
                        "content": json.dumps(result.to_dict(), ensure_ascii=False),
                    }
                )
                steps.append(
                    {
                        "index": index,
                        "kind": "tool",
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                        "result": result.to_dict(),
                        "latency_ms": round((time.perf_counter() - step_started) * 1000, 3),
                    }
                )
                continue

            final_response = action.content
            messages.append({"role": "assistant", "content": action.content})
            steps.append(
                {
                    "index": index,
                    "kind": "respond",
                    "content": action.content,
                    "latency_ms": round((time.perf_counter() - step_started) * 1000, 3),
                }
            )
            if env.goal_satisfied() and _response_sufficient(task, env, action.content):
                termination = "completed"
                break
            try:
                reply = next(followups)
            except StopIteration:
                termination = "agent_ended"
                break
            messages.append({"role": "user", "content": reply})
            env.observe_user_message(reply)
        else:
            termination = "max_steps"

        success = env.goal_satisfied() and termination == "completed"
        primary_failure, failure_labels = classify_failure(task, steps, success)
        tool_steps = [step for step in steps if step["kind"] == "tool"]
        return {
            "schema_version": "1.0",
            "episode_id": episode_id,
            "task_id": task.task_id,
            "split": task.split,
            "intent": task.intent,
            "model": self.model.name,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_ms": round((time.perf_counter() - started_perf) * 1000, 3),
            "termination": termination,
            "success": success,
            "primary_failure": primary_failure.value,
            "failure_labels": failure_labels,
            "illegal_operation": any(
                step["result"].get("illegal_operation", False) for step in tool_steps
            ),
            "tool_calls": len(tool_steps),
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": round(usage.cost_usd, 8),
            },
            "task": task.to_dict(),
            "initial_state": initial_state,
            "final_state": env.state_dict(),
            "initial_state_hash": _hash_state(initial_state),
            "final_state_hash": env.state_hash(),
            "messages": messages,
            "steps": steps,
            "final_response": final_response,
        }


def _hash_state(state: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _response_sufficient(
    task: Task, env: CustomerServiceEnvironment, response: str
) -> bool:
    if not response.strip():
        return False
    if task.intent != "lookup":
        return True
    order_id = task.expected.get("order_id", "")
    order = env.orders.get(order_id)
    return bool(order and (order_id in response or order.status in response.lower()))
