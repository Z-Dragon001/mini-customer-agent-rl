from __future__ import annotations

import json
from pathlib import Path

from .schemas import Task


def load_tasks(path: str | Path) -> list[Task]:
    source = Path(path)
    tasks: list[Task] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    tasks.append(Task.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"任务文件 {source}:{line_number} 无法解析：{exc}") from exc
    return tasks

