"""JSONL-based vulnerability log — append-only, async-safe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redteamagentloop.agent.state import AttackRecord


class JsonlStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def append(self, record: "AttackRecord") -> None:
        """Append one record as a JSON line. Async-safe via lock."""
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def read_all(self) -> list["AttackRecord"]:
        """Read and parse every line. Returns empty list if file does not exist."""
        if not self._path.exists():
            return []
        records = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
