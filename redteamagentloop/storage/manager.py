"""StorageManager — single entry point composing all three stores."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from redteamagentloop.storage.chroma_store import ChromaStore
from redteamagentloop.storage.jsonl_store import JsonlStore
from redteamagentloop.storage.sqlite_store import SqliteStore

if TYPE_CHECKING:
    from redteamagentloop.agent.state import AttackRecord


class StorageManager:
    def __init__(
        self,
        jsonl_path: str,
        sqlite_path: str,
        chromadb_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        dedup_threshold: float = 0.92,
    ) -> None:
        self._jsonl = JsonlStore(jsonl_path)
        self._sqlite = SqliteStore(sqlite_path)
        self._chroma = ChromaStore(
            persist_path=chromadb_path,
            embedding_model=embedding_model,
            dedup_threshold=dedup_threshold,
        )

    # ------------------------------------------------------------------
    # Primary write path
    # ------------------------------------------------------------------

    async def log_attack(self, record: "AttackRecord") -> bool:
        """Persist a successful attack record.

        Returns True if written, False if suppressed as a near-duplicate.
        Dedup check runs first — if the prompt is too similar to an existing
        entry, the record is skipped entirely (no JSONL or SQLite write).
        """
        if self._chroma.is_duplicate(record["prompt"]):
            return False

        await self._jsonl.append(record)
        await self._sqlite.insert(record)
        self._chroma.add(
            prompt=record["prompt"],
            record_id=str(uuid.uuid4()),
            metadata={
                "session_id": record["session_id"],
                "strategy": record["strategy"],
                "score": record["score"],
                "iteration": record["iteration"],
            },
        )
        return True

    # ------------------------------------------------------------------
    # Read helpers (delegated)
    # ------------------------------------------------------------------

    def read_all_vulns(self) -> list["AttackRecord"]:
        return self._jsonl.read_all()

    async def get_session_stats(self, session_id: str):
        return await self._sqlite.get_session_stats(session_id)

    async def get_strategy_performance(self) -> dict[str, float]:
        return await self._sqlite.get_strategy_performance()

    def get_similar_prompts(self, prompt: str, top_k: int = 5):
        return self._chroma.get_similar(prompt, top_k=top_k)
