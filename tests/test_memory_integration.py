import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import config
from agent import Agent
from memory.manager import MemoryManager
from memory.models import MemorySource, MemoryStatus, MemoryType
from memory.repository import MemoryRepository
from memory.repository import user_timezone


def fake_embed(text: str) -> list[float]:
    lowered = text.lower()
    # Deterministic semantic-ish vector, intentionally no network/Ollama dependency.
    if any(word in lowered for word in ("агат", "agata")):
        return [1.0, 1.0, float(any(word in lowered for word in ("работ", "занима", "develop"))), 0.0, 0.0]
    if "проектом x" in lowered:
        return [0.0, 1.0, 1.0, 0.0, 0.0]
    if any(word in lowered for word in ("одеж", "style", "streetwear")):
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    index = sum(ord(char) for char in lowered) % 5
    return [
        float(index == 0), float(index == 1), float(index == 2), float(index == 3), float(index == 4),
    ]


class FakeLLM:
    def __init__(self, text="Тестовый ответ"):
        self.text = text
        self.calls = []

    def chat(self, messages, tools=None, think=None):
        self.calls.append(messages)
        return {"type": "text", "content": self.text}


class TrackingMemory:
    def __init__(self, fail_prepare=False, fail_extract=False):
        self.prepare_calls = 0
        self.extract_calls = 0
        self.fail_prepare = fail_prepare
        self.fail_extract = fail_extract

    def prepare_context(self, user_text):
        self.prepare_calls += 1
        if self.fail_prepare:
            raise RuntimeError("memory offline")
        return {"working_memory": "working", "relevant_memories": "- [GOAL] User wants Agata", "current_datetime": "2026-08-11T10:00:00+05:00"}

    def extract(self, *args):
        self.extract_calls += 1
        if self.fail_extract:
            raise RuntimeError("extract offline")
        return []


class MemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "memory.db"
        self.repository = MemoryRepository(self.path)
        self.memory = MemoryManager(self.repository, fake_embed)

    def tearDown(self):
        self.temp.cleanup()

    def test_agent_injects_automatic_recall_into_dynamic_context(self):
        self.memory.remember("Пользователь хочет сделать Агату автономным персональным AI-ассистентом.", MemoryType.GOAL, .95, .98, MemorySource.USER_EXPLICIT)
        agent = Agent(); agent.memory = self.memory; llm = FakeLLM(); agent._llm = llm
        agent.run("Что я хочу сделать с Агатой?")
        system_prompt = llm.calls[0][0]["content"]
        self.assertIn("автономным персональным AI-ассистентом", system_prompt)

    def test_memory_survives_manager_restart(self):
        saved = self.memory.remember("Пользователь предпочитает минималистичный streetwear.", MemoryType.PREFERENCE, .8, .95)
        restarted = MemoryManager(MemoryRepository(self.path), fake_embed)
        self.assertIn(saved.id, [m.id for m in restarted.recall("какой стиль одежды я предпочитаю")])

    def test_extraction_prevents_obvious_duplicate_projects(self):
        for message in ("Я сейчас работаю над Агатой.", "Я сейчас занимаюсь разработкой Агаты.", "Я продолжаю работать над проектом Агата."):
            self.memory.extract(message)
        projects = [m for m in self.repository.list_active() if m.memory_type is MemoryType.PROJECT]
        self.assertLessEqual(len(projects), 2)

    def test_explicit_remember_persists_priority(self):
        stored = self.memory.extract("Запомни, что я предпочитаю минималистичный streetwear.")[0]
        self.assertEqual(stored.source, MemorySource.USER_EXPLICIT)
        self.assertGreaterEqual(stored.importance, .95)
        self.assertGreaterEqual(stored.confidence, .98)
        restarted = MemoryManager(MemoryRepository(self.path), fake_embed)
        self.assertIn(stored.id, [m.id for m in restarted.recall("какой стиль одежды я предпочитаю")])

    def test_explicit_correction_supersedes_and_is_not_recalled_as_active(self):
        old = self.memory.extract("Я работаю над проектом X.")[0]
        new = self.memory.extract("Я больше не работаю над проектом X.")[0]
        self.assertEqual(self.repository.get(old.id).status, MemoryStatus.SUPERSEDED)
        self.assertEqual(self.repository.get(new.id).status, MemoryStatus.ACTIVE)
        self.assertNotIn(old.id, [m.id for m in self.memory.recall("над каким проектом я работаю")])

    def test_uncertain_negation_is_not_a_false_conflict(self):
        old = self.memory.extract("Я работаю над Агатой.")[0]
        self.memory.remember("Я не знаю, буду ли работать над Агатой завтра.", MemoryType.PROJECT, .55, .55)
        self.assertEqual(self.repository.get(old.id).status, MemoryStatus.ACTIVE)

    def test_consolidation_respects_window_and_preserves_episodes(self):
        now = datetime.now(user_timezone())
        recent_a = self.memory.remember("Recent alpha event", MemoryType.EVENT, created_at=(now - timedelta(days=2)).isoformat())
        recent_b = self.memory.remember("Recent beta event", MemoryType.EVENT, created_at=(now - timedelta(days=3)).isoformat())
        old = self.memory.remember("Old gamma event", MemoryType.EVENT, created_at=(now - timedelta(days=30)).isoformat())
        self.memory.consolidate(days=7)
        self.assertIsNotNone(self.repository.get(recent_a.id).consolidated_at)
        self.assertIsNotNone(self.repository.get(recent_b.id).consolidated_at)
        self.assertIsNone(self.repository.get(old.id).consolidated_at)
        self.assertTrue(any(m.source is MemorySource.CONSOLIDATION for m in self.repository.list_active()))
        self.assertIn(recent_a.id, [m.id for m in self.memory.recall("recent alpha event")])

    def test_action_route_uses_same_lifecycle(self):
        agent = Agent(); memory = TrackingMemory(); agent.memory = memory
        with patch("agent.execute_tool", return_value='{"success": true, "application": "Chrome"}'):
            agent.run("Открой Chrome")
        # Fast deterministic routes bypass slow memory retrieval (prepare_context) to eliminate latency,
        # but still record the turn via extract().
        self.assertEqual(memory.prepare_calls, 0)
        self.assertEqual(memory.extract_calls, 1)

    def test_memory_failures_do_not_prevent_response(self):
        agent = Agent(); agent.memory = TrackingMemory(fail_prepare=True, fail_extract=True); llm = FakeLLM("Ответ всё равно есть"); agent._llm = llm
        self.assertEqual(agent.run("Расскажи кратко"), "Ответ всё равно есть")
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
