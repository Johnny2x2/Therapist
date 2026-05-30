from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from .config import AppConfig
from .llm_therapist import OllamaClient


@dataclass
class RetrievedMemory:
    text: str
    distance: float
    age_days: float
    score: float


class MemoryStore:
    """Session-summary memory backed by a Chroma collection."""

    COLLECTION = "therapist_sessions"

    def __init__(self, config: AppConfig, client: OllamaClient):
        self.config = config
        self.client = client
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError:
            return None
        chroma_client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        self._collection = chroma_client.get_or_create_collection(name=self.COLLECTION)
        return self._collection

    def remember_session(self, session_id: str, summary: str) -> None:
        collection = self._ensure_collection()
        if collection is None:
            return
        embedding = self.client.embed(summary)
        now = datetime.now(timezone.utc)
        collection.upsert(
            ids=[session_id],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{"created_ts": now.timestamp(), "created_iso": now.isoformat()}],
        )
        record_path = Path(self.config.session_dir) / (session_id + ".json")
        existing: dict = {}
        if record_path.exists():
            try:
                existing = json.loads(record_path.read_text(encoding="utf-8"))
            except ValueError:
                existing = {}
        existing["summary"] = summary
        existing["summarized_at"] = now.isoformat()
        record_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def retrieve(
        self,
        text: str,
        limit: Optional[int] = None,
        apply_recency: bool = True,
    ) -> List[RetrievedMemory]:
        collection = self._ensure_collection()
        if collection is None:
            return []
        embedding = self.client.embed(text)
        n = limit if limit is not None else self.config.memory.retrieval_limit
        # Fetch a wider pool when recency reranking is enabled.
        pool = max(n * 3, n)
        result = collection.query(query_embeddings=[embedding], n_results=pool)
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        now_ts = datetime.now(timezone.utc).timestamp()
        half_life = max(self.config.memory.recency_half_life_days, 0.01)
        scored: List[RetrievedMemory] = []
        for index, doc in enumerate(docs):
            distance = float(distances[index]) if index < len(distances) else 0.0
            meta = metadatas[index] if index < len(metadatas) else {}
            created_ts = float((meta or {}).get("created_ts") or now_ts)
            age_days = max(0.0, (now_ts - created_ts) / 86400.0)
            similarity = 1.0 / (1.0 + distance)
            score = similarity * math.exp(-age_days / half_life) if apply_recency else similarity
            scored.append(RetrievedMemory(
                text=doc, distance=distance, age_days=age_days, score=score,
            ))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:n]

    def summarize(self, transcript: Iterable[Any]) -> Optional[str]:
        joined = _join_transcript(transcript)
        if not joined:
            return None
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the session in a short paragraph focusing on persistent "
                    "themes, coping concerns, and follow-up topics. Be concrete and "
                    "factual; do not add advice."
                ),
            },
            {"role": "user", "content": joined},
        ]
        summary = "".join(
            self.client.stream_chat(messages, model=self.config.models.therapist_model)
        ).strip()
        return summary or None


def _join_transcript(transcript: Iterable[Any]) -> str:
    lines: List[str] = []
    for entry in transcript:
        if isinstance(entry, str):
            lines.append(entry)
        elif isinstance(entry, dict):
            role = entry.get("role", "?")
            content = entry.get("content", "")
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()
