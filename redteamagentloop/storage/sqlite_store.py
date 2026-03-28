"""SQLite metadata store — iteration stats and strategy performance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite  # noqa: F401  (used as async context manager)

if TYPE_CHECKING:
    from redteamagentloop.agent.state import AttackRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS attacks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    iteration     INTEGER NOT NULL,
    strategy      TEXT NOT NULL,
    score         REAL NOT NULL,
    was_successful INTEGER NOT NULL,
    timestamp     TEXT NOT NULL
)
"""


@dataclass
class SessionStats:
    session_id: str
    total_attempts: int
    successful_attacks: int
    strategies_used: list[str]


class SqliteStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def insert(self, record: "AttackRecord") -> None:
        async with aiosqlite.connect(self._path) as conn:
            await conn.execute(_CREATE_TABLE)
            await conn.execute(
                "INSERT INTO attacks (session_id, iteration, strategy, score, was_successful, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["session_id"],
                    record["iteration"],
                    record["strategy"],
                    record["score"],
                    int(record["was_successful"]),
                    record["timestamp"],
                ),
            )
            await conn.commit()

    async def get_session_stats(self, session_id: str) -> SessionStats:
        async with aiosqlite.connect(self._path) as conn:
            await conn.execute(_CREATE_TABLE)
            await conn.commit()
            async with conn.execute(
                "SELECT COUNT(*), SUM(was_successful) FROM attacks WHERE session_id = ?",
                (session_id,),
            ) as cur:
                row = await cur.fetchone()
                total = row[0] or 0
                successful = int(row[1] or 0)

            async with conn.execute(
                "SELECT DISTINCT strategy FROM attacks WHERE session_id = ?",
                (session_id,),
            ) as cur:
                strategies = [r[0] async for r in cur]

        return SessionStats(
            session_id=session_id,
            total_attempts=total,
            successful_attacks=successful,
            strategies_used=strategies,
        )

    async def get_strategy_performance(self) -> dict[str, float]:
        """Return average score per strategy across all sessions."""
        async with aiosqlite.connect(self._path) as conn:
            await conn.execute(_CREATE_TABLE)
            await conn.commit()
            async with conn.execute(
                "SELECT strategy, AVG(score) FROM attacks GROUP BY strategy"
            ) as cur:
                return {row[0]: row[1] async for row in cur}
