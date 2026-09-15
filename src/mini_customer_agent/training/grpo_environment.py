from __future__ import annotations

import json
import random
from pathlib import Path

from ..environment import CustomerServiceEnvironment
from ..task_data import load_tasks


class CustomerServiceGRPOEnvironment:
    """TRL stateful tool environment. One fresh instance is created per rollout."""

    def __init__(self, task_path: str | Path, seed: int = 42):
        self.tasks = load_tasks(task_path)
        # Equal seeds intentionally give members of one GRPO group the same task sequence.
        self.random = random.Random(seed)
        self.env: CustomerServiceEnvironment | None = None

    def reset(self, **kwargs) -> str:
        task = self.random.choice(self.tasks)
        self.env = CustomerServiceEnvironment(task)
        return task.initial_message

    def get_reward(self) -> float:
        if self.env is None:
            return -1.0
        history = self.env.tool_history
        if self.env.goal_satisfied():
            return max(0.5, 1.0 - 0.05 * max(0, len(history) - 1))
        if any(item["result"].get("illegal_operation") for item in history):
            return -1.0
        if any(not item["result"].get("ok") for item in history):
            return -0.5
        return 0.0

    def query_order(self, order_id: str) -> str:
        """查询当前客户自己的订单。

        Args:
            order_id: 订单编号。

        Returns:
            JSON 格式的工具执行结果。
        """
        return self._call("query_order", {"order_id": order_id})

    def change_address(
        self, order_id: str, province: str, city: str, detail: str, confirmed: bool
    ) -> str:
        """修改待处理订单地址，用户必须已明确确认。

        Args:
            order_id: 订单编号。
            province: 新地址省份。
            city: 新地址城市。
            detail: 新的详细地址。
            confirmed: 用户是否已经明确确认。

        Returns:
            JSON 格式的工具执行结果。
        """
        return self._call(
            "change_address",
            {
                "order_id": order_id,
                "province": province,
                "city": city,
                "detail": detail,
                "confirmed": confirmed,
            },
        )

    def request_refund(self, order_id: str, reason: str, confirmed: bool) -> str:
        """为满足条件的订单提交退款申请。

        Args:
            order_id: 订单编号。
            reason: damaged、not_as_described 或 no_longer_needed。
            confirmed: 用户是否已经明确确认。

        Returns:
            JSON 格式的工具执行结果。
        """
        return self._call(
            "request_refund",
            {"order_id": order_id, "reason": reason, "confirmed": confirmed},
        )

    def transfer_to_human(self, reason: str, summary: str) -> str:
        """在用户明确要求人工时转接人工客服。

        Args:
            reason: 转人工原因。
            summary: 给人工客服的会话摘要。

        Returns:
            JSON 格式的工具执行结果。
        """
        return self._call("transfer_to_human", {"reason": reason, "summary": summary})

    def _call(self, name: str, arguments: dict) -> str:
        if self.env is None:
            raise RuntimeError("Environment must be reset before tool calls")
        return json.dumps(self.env.call_tool(name, arguments).to_dict(), ensure_ascii=False)
