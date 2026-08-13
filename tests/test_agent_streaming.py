"""
Тесты для реального LLM Streaming в agent.py и llm.py.
"""

from unittest.mock import MagicMock, patch

import pytest
from agent import Agent
from action_router import RoutedAction
from llm import LocalLLM, OllamaUnavailableError


class TestAgentStreaming:
    """Проверка streaming поведения агента."""

    @pytest.fixture
    def agent(self) -> Agent:
        return Agent()

    def test_chat_stream_yields_tokens_and_logs_metrics(self) -> None:
        """chat_stream должен сразу выдавать токены по мере поступления и логировать метрики."""
        llm = LocalLLM()
        fake_lines = [
            '{"message": {"content": "Привет! "}, "done": false}',
            '{"message": {"content": "Как дела?"}, "done": false}',
            '{"done": true, "prompt_eval_duration": 500000000, "eval_duration": 200000000, "total_duration": 700000000}',
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = fake_lines
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            tokens = list(llm.chat_stream([{"role": "user", "content": "Привет"}]))

        assert tokens == ["Привет! ", "Как дела?"]
        assert mock_post.called
        kwargs = mock_post.call_args[1]
        assert kwargs["json"]["stream"] is True
        assert "tools" not in kwargs["json"]

    def test_run_streaming_chat_uses_chat_stream_not_chat(self, agent: Agent) -> None:
        """В режиме CHAT run_streaming() обязан использовать chat_stream(), а не chat(), и не передавать tools."""
        mock_llm = MagicMock()
        mock_llm.chat_stream.return_value = iter(["Это ", "потоковый ", "ответ от ", "Агаты."])

        agent._llm = mock_llm
        emitted_sentences: list[str] = []

        def on_sentence(sentence: str) -> None:
            emitted_sentences.append(sentence)

        answer = agent.run_streaming("Расскажи шутку", on_sentence=on_sentence)

        assert answer == "Это потоковый ответ от Агаты."
        # chat() не должен вызываться вообще
        mock_llm.chat.assert_not_called()
        mock_llm.chat_stream.assert_called_once()

        # Проверяем, что в chat_stream не передавались tools
        call_kwargs = mock_llm.chat_stream.call_args
        assert call_kwargs is not None

        # Проверяем, что предложение было передано в on_sentence
        assert len(emitted_sentences) > 0
        assert "Это потоковый ответ от Агаты." in "".join(emitted_sentences)

    def test_run_streaming_flushes_remainder(self, agent: Agent) -> None:
        """Остаток неоконченного предложения в SentenceChunker должен отправляться после завершения потока."""
        mock_llm = MagicMock()
        # Текст без завершающей точки в конце
        mock_llm.chat_stream.return_value = iter(["Короткий ", "текст без точки"])

        agent._llm = mock_llm
        emitted: list[str] = []

        answer = agent.run_streaming("Привет", on_sentence=emitted.append)

        assert answer == "Короткий текст без точки"
        assert len(emitted) == 1
        assert emitted[0] == "Короткий текст без точки"

    def test_run_streaming_deterministic_action_bypasses_llm(self, agent: Agent) -> None:
        """Детерминированные действия (открой Chrome, время, 17+15) не должны обращаться к LLM."""
        mock_llm = MagicMock()
        agent._llm = mock_llm
        emitted: list[str] = []

        with patch("agent.execute_tool", return_value='{"success": true, "application": "Chrome"}'):
            answer = agent.run_streaming("открой Chrome", on_sentence=emitted.append)

        assert "открыт" in answer.lower() or "chrome" in answer.lower()
        assert emitted == [answer]
        mock_llm.chat.assert_not_called()
        mock_llm.chat_stream.assert_not_called()

    def test_run_streaming_calculation_bypasses_llm(self, agent: Agent) -> None:
        """Вычисления через калькулятор не должны вызывать LLM."""
        mock_llm = MagicMock()
        agent._llm = mock_llm
        emitted: list[str] = []

        answer = agent.run_streaming("17 + 15", on_sentence=emitted.append)

        assert "32" in answer
        assert emitted == [answer]
        mock_llm.chat.assert_not_called()
        mock_llm.chat_stream.assert_not_called()

    def test_synchronous_run_remains_functional(self, agent: Agent) -> None:
        """Синхронный run() должен продолжать работать через chat()."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {"type": "text", "content": "Привет из синхронного режима!"}
        agent._llm = mock_llm

        answer = agent.run("Привет")
        assert answer == "Привет из синхронного режима!"
        mock_llm.chat.assert_called_once()
