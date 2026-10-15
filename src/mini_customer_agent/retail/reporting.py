from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import uuid

from .config import provenance, project_path


def start_run(config: dict, stage: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = project_path(config["output_dir"]) / f"{stamp}_{stage}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "run.json").write_text(json.dumps(provenance(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return folder


def append_trace(folder: Path, row: dict) -> None:
    with (folder / "traces.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"episodes": 0}
    completed = [r for r in rows if not r.get("error")]
    success = sum(r.get("success", False) for r in rows)
    return {"episodes": len(rows), "completed_episodes": len(completed),
            "infrastructure_errors": len(rows) - len(completed),
            "success_rate_all_attempts": success / len(rows),
            "success_rate_completed": success / len(completed) if completed else None,
            "mean_official_reward": statistics.mean(r["official_reward"] for r in completed) if completed else None,
            "avg_tool_calls": statistics.mean(r["tool_calls"] for r in completed) if completed else None,
            "tool_error_episode_rate": sum(r["tool_errors"] > 0 for r in completed) / len(completed) if completed else None,
            "avg_elapsed_ms": statistics.mean(r["elapsed_ms"] for r in completed) if completed else None,
            "termination_counts": dict(Counter(r.get("termination", "infrastructure_error") for r in rows)),
            "illegal_operation_rate": None,
            "note": "Tool errors are not equivalent to policy violations; no exhaustive compliance verifier is implemented."}


def write_summary(folder: Path, rows: list[dict]) -> dict:
    summary = summarize(rows)
    (folder / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
