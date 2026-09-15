from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class ToolName(str, Enum):
    QUERY_ORDER = "query_order"
    CHANGE_ADDRESS = "change_address"
    REQUEST_REFUND = "request_refund"
    TRANSFER_TO_HUMAN = "transfer_to_human"


class FailureType(str, Enum):
    NONE = "none"
    WRONG_TOOL = "tool_selection_error"
    BAD_PARAMETERS = "parameter_error"
    UNHANDLED_EXCEPTION = "unhandled_exception"
    PREMATURE_TERMINATION = "premature_termination"
    MISSING_INFORMATION = "insufficient_information"


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "ModelUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd


@dataclass
class AgentAction:
    kind: Literal["tool", "respond"]
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    call_id: str | None = None
    usage: ModelUsage = field(default_factory=ModelUsage)
    raw: Any = None


@dataclass
class ToolResult:
    ok: bool
    code: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    illegal_operation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Task:
    task_id: str
    split: str
    intent: str
    customer_id: str
    initial_message: str
    followup_messages: list[str]
    expected: dict[str, Any]
    seed_orders: list[dict[str, Any]]
    max_steps: int = 8

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

