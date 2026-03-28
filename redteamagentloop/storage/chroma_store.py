"""ChromaDB-backed semantic deduplication store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

_COLLECTION_NAME = "redteam_prompts"


@dataclass
class SimilarResult:
    prompt_id: str
    similarity: float
    metadata: dict


class ChromaStore:
    def __init__(
        self,
        persist_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        dedup_threshold: float = 0.92,
    ) -> None:
        self._dedup_threshold = dedup_threshold
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, prompt: str) -> bool:
        """Return True if a stored prompt has cosine similarity >= dedup_threshold."""
        if self._collection.count() == 0:
            return False
        results = self._collection.query(
            query_texts=[prompt],
            n_results=1,
            include=["distances"],
        )
        distances = results["distances"][0]
        if not distances:
            return False
        # ChromaDB cosine distance: 0 = identical, 1 = orthogonal.
        # similarity = 1 - distance
        similarity = 1.0 - distances[0]
        return similarity >= self._dedup_threshold

    def add(self, prompt: str, record_id: str, metadata: dict) -> None:
        """Add a prompt embedding. Caller is responsible for dedup check first."""
        # ChromaDB rejects empty metadata dicts — use None instead.
        meta = metadata if metadata else None
        self._collection.add(
            documents=[prompt],
            ids=[record_id],
            metadatas=[meta],
        )

    def get_similar(self, prompt: str, top_k: int = 5) -> list[SimilarResult]:
        """Return the top-k most similar stored prompts."""
        count = self._collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = self._collection.query(
            query_texts=[prompt],
            n_results=n,
            include=["distances", "metadatas"],
        )
        similar = []
        for pid, dist, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            similar.append(
                SimilarResult(
                    prompt_id=pid,
                    similarity=round(1.0 - dist, 4),
                    metadata=meta,
                )
            )
        return similar
