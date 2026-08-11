import sqlite3
import tempfile
import unittest
from pathlib import Path

from memory.manager import MemoryManager
from memory.models import Memory, MemorySource, MemoryStatus, MemoryType
from memory.repository import MemoryRepository


def embed(text: str) -> list[float]:
    text = text.lower()
    return [float(sum(ord(char) for char in text) % 97), float(len(text)), float(text.count("agata") + text.count("агата"))]


class MemoryV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "memory.db"
        self.repo = MemoryRepository(self.db_path)
        self.memory = MemoryManager(self.repo, embed)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_store_load_and_update(self) -> None:
        stored = self.memory.remember("User prefers concise answers", MemoryType.PREFERENCE, .7, .95)
        loaded = self.repo.get(stored.id)
        self.assertEqual(loaded.content, stored.content)
        loaded.importance = .9
        self.memory.update(loaded)
        self.assertEqual(self.repo.get(stored.id).importance, .9)

    def test_recall_ranks_explicit_and_updates_timestamp(self) -> None:
        self.memory.remember("User likes tea", MemoryType.PREFERENCE, .5, .95)
        explicit = self.memory.remember("User wants to build Agata", MemoryType.GOAL, .95, .98, MemorySource.USER_EXPLICIT)
        result = self.memory.recall("What does the user want to build?", 2)
        self.assertEqual(result[0].id, explicit.id)
        self.assertIsNotNone(self.repo.get(explicit.id).last_recalled_at)

    def test_duplicate_merges(self) -> None:
        first = self.memory.remember("User works on Agata", MemoryType.PROJECT)
        duplicate = self.memory.remember("User works on Agata", MemoryType.PROJECT, .9)
        self.assertEqual(first.id, duplicate.id)
        self.assertEqual(len(self.repo.list_active()), 1)

    def test_conflicting_memory_supersedes_old(self) -> None:
        old = self.memory.remember("Я работаю над проектом X", MemoryType.PROJECT)
        new = self.memory.remember("Я больше не занимаюсь проектом X", MemoryType.PROJECT)
        self.assertEqual(self.repo.get(old.id).status, MemoryStatus.SUPERSEDED)
        self.assertEqual(new.supersedes_id, old.id)

    def test_explicit_extraction_is_high_priority(self) -> None:
        stored = self.memory.extract("Запомни, что я хочу сделать Агату автономным ассистентом")
        self.assertEqual(stored[0].source, MemorySource.USER_EXPLICIT)
        self.assertGreaterEqual(stored[0].importance, .95)

    def test_consolidation_keeps_episodes_and_creates_semantic_memory(self) -> None:
        self.memory.remember("Episode one", MemoryType.EVENT)
        self.memory.remember("Episode two", MemoryType.EVENT)
        self.memory.consolidate()
        active = self.repo.list_active()
        self.assertEqual(sum(m.memory_type is MemoryType.EVENT for m in active), 2)
        self.assertTrue(any(m.source is MemorySource.CONSOLIDATION for m in active))

    def test_embedding_failure_is_graceful(self) -> None:
        manager = MemoryManager(self.repo, lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
        stored = manager.remember("A durable fact")
        self.assertIsNotNone(stored.id)
        self.assertEqual(len(manager.recall("durable")), 1)

    def test_legacy_diary_migrates_without_removal(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TABLE memories")
            conn.execute("CREATE TABLE diary (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB, created_at TEXT, tags TEXT, consolidated INTEGER)")
            conn.execute("INSERT INTO diary VALUES (7, 'legacy fact', NULL, '2026-01-01T00:00:00+00:00', 'legacy', 1)")
            conn.commit()
        finally:
            conn.close()
        migrated = MemoryRepository(self.db_path).get(-7)
        self.assertEqual(migrated.content, "legacy fact")
        conn = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT text FROM diary WHERE id=7").fetchone()[0], "legacy fact")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
