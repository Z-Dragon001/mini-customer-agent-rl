from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .environment import CustomerServiceEnvironment
from .schemas import AgentAction, ModelUsage, Task


@dataclass
class CompletionContext:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    task: Task
    environment: CustomerServiceEnvironment


class ModelClient(Protocol):
    name: str

    def complete(self, context: CompletionContext) -> AgentAction: ...


class ScriptedPolicyClient:
    """Oracle-like smoke-test policy; it is not an experimental baseline."""

    name = "scripted_oracle"

    def complete(self, context: CompletionContext) -> AgentAction:
        env = context.environment
        task = context.task
        last = context.messages[-1]
        if last["role"] == "tool":
            result = json.loads(last["content"])
            if result["ok"]:
                if "status" in result.get("data", {}):
                    data = result["data"]
                    return AgentAction(
                        kind="respond",
                        content=f"订单 {data['order_id']} 当前状态是 {data['status']}。",
                    )
                return AgentAction(kind="respond", content=result["message"])
            return AgentAction(kind="respond", content=f"操作未完成：{result['message']}")

        expected = task.expected
        order_id = expected.get("order_id")
        if task.intent == "lookup":
            return AgentAction(
                kind="tool",
                tool_name="query_order",
                arguments={"order_id": order_id},
                call_id=f"call_{uuid.uuid4().hex[:8]}",
            )
        if task.intent == "change_address":
            if not env.confirmed_change:
                return AgentAction(kind="respond", content="请确认是否按该新地址修改订单？")
            address = expected["address"]
            return AgentAction(
                kind="tool",
                tool_name="change_address",
                arguments={"order_id": order_id, **address, "confirmed": True},
                call_id=f"call_{uuid.uuid4().hex[:8]}",
            )
        if task.intent == "refund":
            if not env.confirmed_refund:
                return AgentAction(kind="respond", content="请确认是否提交退款申请？")
            return AgentAction(
                kind="tool",
                tool_name="request_refund",
                arguments={
                    "order_id": order_id,
                    "reason": expected["reason"],
                    "confirmed": True,
                },
                call_id=f"call_{uuid.uuid4().hex[:8]}",
            )
        return AgentAction(
            kind="tool",
            tool_name="transfer_to_human",
            arguments={"reason": "user_requested", "summary": task.initial_message},
            call_id=f"call_{uuid.uuid4().hex[:8]}",
        )


class ReplayModelClient:
    """Deterministic action sequence used by tests and failure-case reproduction."""

    def __init__(self, actions: list[AgentAction], name: str = "replay"):
        self.actions = list(actions)
        self.name = name
        self.index = 0

    def complete(self, context: CompletionContext) -> AgentAction:
        if self.index >= len(self.actions):
            return AgentAction(kind="respond", content="结束")
        action = self.actions[self.index]
        self.index += 1
        return action


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
        temperature: float = 0.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("请先安装 API 依赖：pip install -e '.[api]'") from exc
        self.name = name
        self.model = model
        self.temperature = temperature
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            timeout=60.0,
            max_retries=2,
        )

    def complete(self, context: CompletionContext) -> AgentAction:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=context.messages,
            tools=context.tools,
            tool_choice="auto",
            temperature=self.temperature,
        )
        choice = response.choices[0].message
        prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response.usage, "completion_tokens", 0) or 0)
        usage = ModelUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=(prompt_tokens * self.input_price + completion_tokens * self.output_price)
            / 1_000_000,
        )
        if choice.tool_calls:
            call = choice.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_malformed_json": call.function.arguments}
            return AgentAction(
                kind="tool",
                tool_name=call.function.name,
                arguments=arguments,
                call_id=call.id,
                usage=usage,
                raw=choice.model_dump(mode="json"),
            )
        return AgentAction(
            kind="respond",
            content=choice.content or "",
            usage=usage,
            raw=choice.model_dump(mode="json"),
        )
