"""Stable high-level interface for persistent and working memory."""

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Callable

import numpy as np

import config
from memory import working_memory
from memory.extraction import extract_candidates
from memory.models import Memory, MemorySource, MemoryStatus, MemoryType
from memory.repository import MemoryRepository, user_timezone

logger = logging.getLogger(__name__)


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return 0.0 if denom == 0 else float(np.dot(va, vb) / denom)


class MemoryManager:
    def __init__(self, repository: MemoryRepository | None = None, embedder: Callable[[str], list[float]] | None = None) -> None:
        self.repository = repository or MemoryRepository()
        self._embedder = embedder

    def _embed(self, text: str) -> list[float] | None:
        try:
            if self._embedder is None:
                from llm import LocalLLM
                self._embedder = LocalLLM().embed
            return self._embedder(text)
        except Exception as exc:
            logger.warning("[MEMORY] embedding failed: %s", exc)
            return None

    @staticmethod
    def _clamp(value: float) -> float: return max(0.0, min(1.0, value))

    @staticmethod
    def _is_explicit_correction(content: str) -> bool:
        """Recognise only direct cancellation/change statements, not generic negation."""
        lowered = content.lower()
        patterns = (
            r"\bбольше\s+не\s+(?:работаю|занимаюсь|хочу|предпочитаю|люблю)\b",
            r"\bперестал(?:а)?\s+(?:работать|заниматься|хотеть|предпочитать|любить)\b",
            r"\bзабудь\s*,?\s*что\s+я\b",
            r"\bno\s+longer\s+(?:work|want|prefer|like)\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    @classmethod
    def _conflicts(cls, old: Memory, new: Memory) -> bool:
        if new.confidence < config.MEMORY_CONFLICT_MIN_CONFIDENCE or not cls._is_explicit_correction(new.content):
            return False
        old_words = set(re.findall(r"\w+", old.content.lower()))
        new_words = set(re.findall(r"\w+", new.content.lower()))
        subject_words = old_words & new_words - {"я", "над", "проектом", "работаю", "работать", "не", "больше", "занимаюсь"}
        return len(subject_words) >= 1 and _cosine(old.embedding, new.embedding) >= config.MEMORY_CONFLICT_SIMILARITY_THRESHOLD

    def remember(self, content: str, memory_type: MemoryType = MemoryType.FACT, importance: float = .5, confidence: float = .7, source: MemorySource = MemorySource.CONVERSATION, **kwargs) -> Memory:
        memory = Memory(None, content.strip(), memory_type, self._clamp(importance), self._clamp(confidence), source=source, **kwargs)
        if source is MemorySource.USER_EXPLICIT: memory.importance, memory.confidence = max(memory.importance, .95), max(memory.confidence, .98)
        memory.embedding = self._embed(memory.content)
        for candidate in self.repository.list_active():
            # Exact repetitions are always safe to merge first.
            if candidate.content.casefold() == memory.content.casefold():
                candidate.importance, candidate.confidence = max(candidate.importance, memory.importance), max(candidate.confidence, memory.confidence)
                candidate.metadata.update(memory.metadata)
                logger.info("[MEMORY] duplicate detected id=%s", candidate.id)
                return self.repository.update(candidate)
        for old in self.repository.list_active():
            if self._conflicts(old, memory):
                old.status = MemoryStatus.SUPERSEDED
                self.repository.update(old)
                memory.supersedes_id = old.id
                logger.info("[MEMORY] memory superseded id=%s", old.id)
                break
        for candidate in self.repository.list_active():
            shared_words = set(re.findall(r"\w+", candidate.content.lower())) & set(re.findall(r"\w+", memory.content.lower()))
            meaningful_shared_words = shared_words - {"я", "над", "не", "the", "a", "an", "event", "episode", "memory", "fact"}
            if _cosine(memory.embedding, candidate.embedding) >= config.MEMORY_DUPLICATE_THRESHOLD and meaningful_shared_words:
                candidate.content = memory.content if len(memory.content) > len(candidate.content) else candidate.content
                candidate.importance, candidate.confidence = max(candidate.importance, memory.importance), max(candidate.confidence, memory.confidence)
                candidate.embedding = memory.embedding or candidate.embedding
                candidate.metadata.update(memory.metadata)
                logger.info("[MEMORY] duplicate detected id=%s", candidate.id)
                return self.repository.update(candidate)
        stored = self.repository.create(memory)
        logger.info("[MEMORY] stored memory id=%s preview=%r", stored.id, stored.content[:60])
        return stored

    def recall(self, query: str, top_k: int | None = None) -> list[Memory]:
        try:
            query_embedding, now = self._embed(query), datetime.now(user_timezone())
            query_words = set(re.findall(r"\w+", query.lower())); ranked: list[Memory] = []
            for memory in self.repository.list_active():
                similarity = _cosine(query_embedding, memory.embedding)
                if query_embedding is None:
                    words = set(re.findall(r"\w+", memory.content.lower())); similarity = len(query_words & words) / max(1, len(query_words | words))
                try:
                    created = datetime.fromisoformat((memory.created_at or "").replace("Z", "+00:00")); created = created if created.tzinfo else created.replace(tzinfo=user_timezone())
                    age_days = max(0.0, (now - created).total_seconds() / 86400)
                except ValueError: age_days = config.MEMORY_RECENCY_HALF_LIFE_DAYS
                recency = math.exp(-age_days / config.MEMORY_RECENCY_HALF_LIFE_DAYS)
                explicit = 1.0 if memory.source is MemorySource.USER_EXPLICIT else 0.0
                memory.similarity = similarity
                memory.score = similarity * config.MEMORY_RANK_SIMILARITY_WEIGHT + memory.importance * config.MEMORY_RANK_IMPORTANCE_WEIGHT + recency * config.MEMORY_RANK_RECENCY_WEIGHT + memory.confidence * config.MEMORY_RANK_CONFIDENCE_WEIGHT + explicit * config.MEMORY_RANK_EXPLICIT_WEIGHT
                ranked.append(memory)
            ranked.sort(key=lambda m: m.score or 0.0, reverse=True)
            result = ranked[:top_k or config.MEMORY_RECALL_LIMIT]
            self.repository.mark_recalled(m.id for m in result if m.id is not None)
            logger.info("[MEMORY] recalled %d memories", len(result)); return result
        except Exception as exc:
            logger.warning("[MEMORY] recall failed: %s", exc); return []

    def update(self, memory: Memory) -> Memory: return self.repository.update(memory)

    def supersede(self, old_id: int, replacement: Memory) -> Memory:
        old = self.repository.get(old_id)
        if old is None:
            raise KeyError(old_id)
        old.status = MemoryStatus.SUPERSEDED
        self.repository.update(old)
        replacement.supersedes_id = old_id
        replacement.status = MemoryStatus.ACTIVE
        replacement.embedding = replacement.embedding or self._embed(replacement.content)
        return self.repository.create(replacement)

    def forget(self, memory_id: int) -> None:
        memory = self.repository.get(memory_id)
        if memory: memory.status = MemoryStatus.ARCHIVED; self.repository.update(memory)

    def extract(self, user_message: str, assistant_response: str = "", current_context: str = "") -> list[Memory]:
        stored = [self.remember(c.content, c.memory_type, c.importance, c.confidence, c.source, metadata=c.metadata) for c in extract_candidates(user_message, assistant_response, current_context)]
        logger.info("[MEMORY] extracted %d candidates", len(stored)); return stored

    def consolidate(self, days: int = config.CONSOLIDATION_DAYS) -> list[Memory]:
        logger.info("[MEMORY] consolidation started")
        cutoff = datetime.now(user_timezone()) - timedelta(days=days)
        episodes = []
        for memory in self.repository.list_active():
            if memory.memory_type is not MemoryType.EVENT or memory.consolidated_at:
                continue
            created = datetime.fromisoformat((memory.created_at or "").replace("Z", "+00:00"))
            created = created if created.tzinfo else created.replace(tzinfo=user_timezone())
            if created >= cutoff:
                episodes.append(memory)
        self.repository.mark_consolidated(m.id for m in episodes if m.id is not None)
        # Keep episodes available for audit/retrieval and add one compact semantic summary.
        if len(episodes) > 1:
            summary = "; ".join(m.content for m in episodes[:3])
            semantic = Memory(
                None, f"Консолидированное знание: {summary}", MemoryType.FACT,
                max(m.importance for m in episodes), min(m.confidence for m in episodes),
                source=MemorySource.CONSOLIDATION, embedding=self._embed(summary),
                metadata={"episode_ids": [m.id for m in episodes]},
            )
            self.repository.create(semantic)
        logger.info("[MEMORY] consolidation completed; processed=%d", len(episodes)); return episodes

    def get_working_context(self) -> str:
        try: return working_memory.read()[:config.MEMORY_WORKING_CONTEXT_CHAR_BUDGET]
        except Exception as exc: logger.warning("[MEMORY] working memory read failed: %s", exc); return ""

    def prepare_context(self, user_message: str) -> dict[str, str]:
        chunks: list[str] = []; used = 0
        for memory in self.recall(user_message):
            line = f"- [{memory.memory_type}] {memory.content}"
            if used + len(line) > config.MEMORY_CONTEXT_CHAR_BUDGET: break
            chunks.append(line); used += len(line)
        return {
            "working_memory": self.get_working_context(),
            "relevant_memories": "\n".join(chunks),
            "current_datetime": datetime.now(user_timezone()).isoformat(),
        }

    # Compatibility name retained for existing callers.
    build_context = prepare_context
