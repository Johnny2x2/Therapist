from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AppConfig


@dataclass
class MoodEntry:
    ts: str
    scope: str  # "start" | "end" | "midpoint"
    value: int
    note: str
    session_id: str


class MoodLog:
    def __init__(self, config: AppConfig):
        self.config = config

    def record(self, scope: str, value: int, session_id: str, note: str = "") -> MoodEntry:
        if not 1 <= int(value) <= 10:
            raise ValueError("mood value must be 1..10")
        entry = MoodEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            scope=scope,
            value=int(value),
            note=note.strip(),
            session_id=session_id,
        )
        path: Path = self.config.mood_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__) + "\n")
        return entry

    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        path: Path = self.config.mood_log_path
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        if limit is not None:
            rows = rows[-limit:]
        return rows
