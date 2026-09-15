from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .schemas import Task, ToolResult


@dataclass
class Order:
    order_id: str
    customer_id: str
    status: str
    total: float
    refundable: bool
    province: str
    city: str
    detail: str
    refund_requested: bool = False
    refund_reason: str | None = None


class CustomerServiceEnvironment:
    """A deterministic, resettable environment with policy-enforcing tools."""

    REFUND_REASONS = {"damaged", "not_as_described", "no_longer_needed"}

    def __init__(self, task: Task):
        self.task = task
        self.orders = {
            row["order_id"]: Order(**copy.deepcopy(row)) for row in task.seed_orders
        }
        self.confirmed_change = False
        self.confirmed_refund = False
        self.transferred = False
        self.queried_orders: set[str] = set()
        self.tool_history: list[dict[str, Any]] = []
        self.observe_user_message(task.initial_message, initial=True)

    def observe_user_message(self, message: str, *, initial: bool = False) -> None:
        normalized = message.replace(" ", "").lower()
        is_confirmation = any(
            word in normalized for word in ("确认", "同意", "可以修改", "可以退款", "yes")
        )
        if is_confirmation:
            if self.task.intent == "change_address":
                self.confirmed_change = True
            if self.task.intent == "refund":
                self.confirmed_refund = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "orders": {key: asdict(value) for key, value in sorted(self.orders.items())},
            "transferred": self.transferred,
            "queried_orders": sorted(self.queried_orders),
        }

    def state_hash(self) -> str:
        payload = json.dumps(self.state_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _own_order(self, order_id: Any) -> tuple[Order | None, ToolResult | None]:
        if not isinstance(order_id, str) or not order_id.strip():
            return None, ToolResult(False, "PARAMETER_ERROR", "order_id 必须是非空字符串。")
        order = self.orders.get(order_id)
        if order is None:
            return None, ToolResult(False, "NOT_FOUND", "订单不存在。")
        if order.customer_id != self.task.customer_id:
            return None, ToolResult(
                False,
                "POLICY_DENIED",
                "不能访问其他客户的订单。",
                illegal_operation=True,
            )
        return order, None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        before = self.state_hash()
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                result = ToolResult(False, "UNKNOWN_TOOL", f"未知工具：{name}")
            elif not isinstance(arguments, dict):
                result = ToolResult(False, "PARAMETER_ERROR", "工具参数必须是 JSON 对象。")
            else:
                result = handler(arguments)
        except Exception as exc:  # Harness should preserve environment exceptions in trace.
            result = ToolResult(False, "TOOL_EXCEPTION", f"{type(exc).__name__}: {exc}")
        self.tool_history.append(
            {
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "result": result.to_dict(),
                "state_before": before,
                "state_after": self.state_hash(),
            }
        )
        return result

    def _tool_query_order(self, args: dict[str, Any]) -> ToolResult:
        if set(args) - {"order_id"} or "order_id" not in args:
            return ToolResult(False, "PARAMETER_ERROR", "query_order 只接受必填参数 order_id。")
        order, error = self._own_order(args["order_id"])
        if error:
            return error
        assert order is not None
        self.queried_orders.add(order.order_id)
        return ToolResult(
            True,
            "OK",
            "订单查询成功。",
            {
                "order_id": order.order_id,
                "status": order.status,
                "total": order.total,
                "refundable": order.refundable,
                "address": {
                    "province": order.province,
                    "city": order.city,
                    "detail": order.detail,
                },
                "refund_requested": order.refund_requested,
            },
        )

    def _tool_change_address(self, args: dict[str, Any]) -> ToolResult:
        required = {"order_id", "province", "city", "detail", "confirmed"}
        if set(args) != required:
            return ToolResult(False, "PARAMETER_ERROR", f"参数必须且只能包含：{sorted(required)}。")
        order, error = self._own_order(args["order_id"])
        if error:
            return error
        if args["confirmed"] is not True or not self.confirmed_change:
            return ToolResult(False, "MISSING_INFORMATION", "缺少用户在本轮会话中的明确修改确认。")
        if any(not isinstance(args[key], str) or not args[key].strip() for key in ("province", "city", "detail")):
            return ToolResult(False, "PARAMETER_ERROR", "省、市、详细地址必须是非空字符串。")
        assert order is not None
        if order.status != "pending":
            return ToolResult(
                False,
                "POLICY_DENIED",
                "只有 pending 订单可以修改地址。",
                illegal_operation=True,
            )
        order.province = args["province"].strip()
        order.city = args["city"].strip()
        order.detail = args["detail"].strip()
        return ToolResult(True, "OK", "地址修改成功。", {"order_id": order.order_id})

    def _tool_request_refund(self, args: dict[str, Any]) -> ToolResult:
        required = {"order_id", "reason", "confirmed"}
        if set(args) != required:
            return ToolResult(False, "PARAMETER_ERROR", f"参数必须且只能包含：{sorted(required)}。")
        order, error = self._own_order(args["order_id"])
        if error:
            return error
        if args["confirmed"] is not True or not self.confirmed_refund:
            return ToolResult(False, "MISSING_INFORMATION", "缺少用户在本轮会话中的明确退款确认。")
        if args["reason"] not in self.REFUND_REASONS:
            return ToolResult(False, "PARAMETER_ERROR", "退款原因不在允许枚举中。")
        assert order is not None
        if order.status != "delivered" or not order.refundable or order.refund_requested:
            return ToolResult(
                False,
                "POLICY_DENIED",
                "该订单当前不满足退款申请条件。",
                illegal_operation=True,
            )
        order.refund_requested = True
        order.refund_reason = args["reason"]
        return ToolResult(True, "OK", "退款申请已提交。", {"order_id": order.order_id})

    def _tool_transfer_to_human(self, args: dict[str, Any]) -> ToolResult:
        required = {"reason", "summary"}
        if set(args) != required or any(
            not isinstance(args.get(key), str) or not args[key].strip() for key in required
        ):
            return ToolResult(False, "PARAMETER_ERROR", "reason 和 summary 均为必填非空字符串。")
        if self.task.intent != "transfer":
            return ToolResult(
                False,
                "WRONG_TOOL",
                "当前任务可以由已有业务工具处理，不应直接转人工。",
                illegal_operation=True,
            )
        self.transferred = True
        return ToolResult(True, "OK", "已转接人工客服。")

    def goal_satisfied(self) -> bool:
        expected = self.task.expected
        kind = expected.get("kind")
        order = self.orders.get(expected.get("order_id", ""))
        if kind == "queried":
            return expected.get("order_id") in self.queried_orders
        if kind == "address_changed" and order:
            address = expected["address"]
            return (order.province, order.city, order.detail) == (
                address["province"],
                address["city"],
                address["detail"],
            )
        if kind == "refund_requested" and order:
            return order.refund_requested and order.refund_reason == expected["reason"]
        if kind == "transferred":
            return self.transferred
        return False
