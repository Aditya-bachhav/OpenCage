from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SessionLogger:
    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_step(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def write_summary(self, payload: dict[str, Any]) -> None:
        self.write_step({"type": "summary", **payload})

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows