from __future__ import annotations

from typing import Any


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询当前已登录客户自己的订单状态与概要。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_address",
            "description": "修改待处理订单的收货地址。必须先获得用户明确确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "province": {"type": "string"},
                    "city": {"type": "string"},
                    "detail": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                },
                "required": ["order_id", "province", "city", "detail", "confirmed"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_refund",
            "description": "为已送达且可退款的订单申请退款。必须先获得用户明确确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {
                        "type": "string",
                        "enum": ["damaged", "not_as_described", "no_longer_needed"],
                    },
                    "confirmed": {"type": "boolean"},
                },
                "required": ["order_id", "reason", "confirmed"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "用户明确要求人工或现有工具无法安全解决问题时，转接人工客服。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["reason", "summary"],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_PROMPT = """你是一个电商客服工具 Agent。当前会话已经完成身份认证，你只能操作当前客户自己的订单。

可用工具只有：查订单、改地址、申请退款、转人工。
规则：
1. 写操作（改地址、退款）前必须获得用户明确确认，不得把用户最初的请求自动当作确认。
2. 改地址只允许 pending 订单；退款只允许 delivered 且 refundable 的订单。
3. 工具参数必须来自用户消息或工具结果，不得猜测。
4. 工具失败后应根据错误补充信息、改正参数或解释限制，不要重复同一失败调用。
5. 只有用户明确要求人工，或现有工具确实无法解决时才转人工。
6. 完成目标后再给简短明确的最终回复；不要声称没有实际完成的操作已经成功。
"""

