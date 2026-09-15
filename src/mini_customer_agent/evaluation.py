from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .harness import AgentHarness
from .model_clients import ModelClient
from .schemas import Task


def evaluate(model: ModelClient, tasks: list[Task]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces = [AgentHarness(model).run(task) for task in tasks]
    return traces, aggregate_metrics(traces)


def aggregate_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {"episodes": 0}
    latencies = [float(trace["elapsed_ms"]) for trace in traces]
    failure_counts = Counter(
        trace["primary_failure"] for trace in traces if not trace["success"]
    )
    return {
        "episodes": len(traces),
        "success_rate": round(sum(trace["success"] for trace in traces) / len(traces), 4),
        "illegal_operation_rate": round(
            sum(trace["illegal_operation"] for trace in traces) / len(traces), 4
        ),
        "avg_tool_calls": round(statistics.mean(trace["tool_calls"] for trace in traces), 3),
        "avg_cost_usd": round(
            statistics.mean(trace["usage"]["cost_usd"] for trace in traces), 8
        ),
        "avg_elapsed_ms": round(statistics.mean(latencies), 3),
        "p50_elapsed_ms": round(_percentile(latencies, 0.50), 3),
        "p95_elapsed_ms": round(_percentile(latencies, 0.95), 3),
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def write_run(
    output_dir: str | Path,
    model_name: str,
    traces: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in model_name)
    run_dir = Path(output_dir) / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return run_dir


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * ratio)))
    return ordered[index]

