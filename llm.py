"""
Клиент для Ollama LLM.

Класс LocalLLM инкапсулирует всё взаимодействие с Ollama:
- chat() — генерация ответа / tool calling через /api/chat
- embed() — получение эмбеддингов через /api/embed

Обработка ошибок: таймауты, недоступность сервера, битые tool calls —
выбрасывают понятные исключения, но не роняют программу.
"""

import json
import logging
from typing import Any, Optional

import requests

import config

logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Ollama-сервер недоступен."""


class ToolCallParseError(Exception):
    """Не удалось распарсить tool call из ответа модели."""


class LocalLLM:
    """Клиент для локального Ollama API."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_URL,
        model: str = config.LLM_MODEL,
        embed_model: str = config.EMBED_MODEL,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout

    # ── Chat ───────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
        think: Optional[bool] = None,  # None = взять из config.ENABLE_THINKING
    ) -> dict[str, Any]:
        """
        Отправить сообщения в Ollama и получить ответ.

        Args:
            messages: История сообщений в формате OpenAI-like.
            tools: Список JSON Schema инструментов (или None).
            think: Переопределение режима рассуждений (think). None = взять из config.ENABLE_THINKING.

        Returns:
            {"type": "text", "content": str}
            или
            {"type": "tool_call", "name": str, "arguments": dict}

        Raises:
            OllamaUnavailableError: Ollama не отвечает.
            ToolCallParseError: Модель вернула невалидный tool call.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": think if think is not None else config.ENABLE_THINKING,
        }
        logger.info("Отправка запроса с think=%s", payload.get("think"))
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(
                f"Не удалось подключиться к Ollama ({self.base_url}). "
                "Убедись, что Ollama запущена: ollama serve"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaUnavailableError(
                f"Ollama не ответила за {self.timeout}с — "
                "возможно, модель слишком большая или GPU перегружен."
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Ollama вернула ошибку: {resp.status_code} {resp.text[:200]}"
            ) from exc

        data = resp.json()
        message = data.get("message", {})

        # ── Проверяем tool_calls ──────────────────────────────────────
        tool_calls = message.get("tool_calls")
        if tool_calls:
            try:
                tc = tool_calls[0]  # Берём первый tool call
                func = tc.get("function", tc)
                name = func["name"]
                arguments = func.get("arguments", {})
                # arguments может быть строкой (битый ответ модели)
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                logger.info("Tool call: %s(%s)", name, json.dumps(arguments, ensure_ascii=False)[:200])
                return {
                    "type": "tool_call",
                    "name": name,
                    "arguments": arguments,
                }
            except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
                logger.warning("Битый tool call от модели: %s", exc)
                raise ToolCallParseError(
                    f"Модель вернула невалидный tool call: {tool_calls}"
                ) from exc

        # ── Обычный текстовый ответ ───────────────────────────────────
        content = message.get("content", "").strip()
        # Qwen3 может обернуть текст в <think>...</think> — убираем
        content = self._strip_think_tags(content)
        logger.info("Текстовый ответ (%d символов)", len(content))
        return {"type": "text", "content": content}

    # ── Embeddings ─────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Получить эмбеддинг текста через Ollama.

        Args:
            text: Текст для эмбеддинга.

        Returns:
            Вектор эмбеддинга (list[float]).

        Raises:
            OllamaUnavailableError: Ollama не отвечает.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": text},
                timeout=60,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(
                "Не удалось подключиться к Ollama для эмбеддингов."
            ) from exc
        except requests.Timeout as exc:
            raise OllamaUnavailableError(
                "Таймаут при запросе эмбеддингов."
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Ошибка эмбеддингов: {resp.status_code} {resp.text[:200]}"
            ) from exc

        data = resp.json()
        # Ollama /api/embed возвращает {"embeddings": [[...]]}
        embeddings = data.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        # Fallback для старых версий Ollama (/api/embeddings)
        embedding = data.get("embedding", [])
        if embedding:
            return embedding
        raise OllamaUnavailableError(
            f"Ollama вернула пустой эмбеддинг. Ответ: {json.dumps(data)[:300]}"
        )

    # ── Утилиты ────────────────────────────────────────────────────────

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Убрать <think>...</think> из ответа Qwen3."""
        import re
        # Убираем блоки <think>...</think> (включая многострочные)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned if cleaned else text
