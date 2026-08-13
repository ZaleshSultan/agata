"""
Клиент для Ollama LLM.

Класс LocalLLM инкапсулирует всё взаимодействие с Ollama:
- chat() — генерация ответа / tool calling через /api/chat (stream=False)
- chat_stream() — потоковая генерация текста через /api/chat (stream=True)
- embed() — получение эмбеддингов через /api/embed

Обработка ошибок: таймауты, недоступность сервера, битые tool calls —
выбрасывают понятные исключения, но не роняют программу.
"""

import json
import logging
from time import perf_counter
from typing import Any, Generator, Optional

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
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }
        logger.info("Отправка запроса с think=%s", payload.get("think"))
        if tools:
            payload["tools"] = tools

        try:
            request_started_at = perf_counter()
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

        request_elapsed = perf_counter() - request_started_at
        data = resp.json()
        server_timings = {
            "ollama_total": data.get("total_duration"),
            "load": data.get("load_duration"),
            "prompt_eval": data.get("prompt_eval_duration"),
            "eval": data.get("eval_duration"),
        }
        timing_parts = [f"http={request_elapsed:.3f}s"]
        for name, nanoseconds in server_timings.items():
            if isinstance(nanoseconds, int):
                timing_parts.append(f"{name}={nanoseconds / 1_000_000_000:.3f}s")
        logger.info("[PERF] ollama %s messages=%d tools=%s", " ".join(timing_parts), len(messages), bool(tools))
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

    # ── Chat Stream ────────────────────────────────────────────────────

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        think: bool | None = None,
    ) -> Generator[str, None, None]:
        """
        Потоковая генерация текстового ответа от Ollama.

        В отличие от chat(), отдаёт текстовые токены по мере поступления.
        НЕ поддерживает tool calling — для этого используйте chat().

        Фильтрует содержимое <think>...</think> на лету.

        Args:
            messages: История сообщений.
            think: Переопределение режима рассуждений.

        Yields:
            Текстовые фрагменты ответа.

        Raises:
            OllamaUnavailableError: Ollama не отвечает.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think if think is not None else config.ENABLE_THINKING,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }
        
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        started_at = perf_counter()
        first_token_at: float | None = None
        last_token_at: float | None = None
        token_count = 0
        char_count = 0
        done_chunk_data: dict[str, Any] = {}

        logger.info(
            "[DIAG LLM STREAM] Sending request at %.3f | model=%s | prompt_chars=%d | think=%s",
            started_at, self.model, prompt_chars, payload.get("think")
        )

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True,
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

        # Состояние для фильтрации <think> тегов
        inside_think = False

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("[LLM STREAM] Невалидный JSON: %s", raw_line[:100])
                    continue

                # Ollama отдаёт {"done": true} в последнем чанке
                if chunk.get("done", False):
                    elapsed = perf_counter() - started_at
                    prompt_eval_ns = chunk.get("prompt_eval_duration")
                    eval_ns = chunk.get("eval_duration")
                    total_ns = chunk.get("total_duration")
                    
                    p_eval_str = f"{prompt_eval_ns / 1_000_000_000:.3f}s" if isinstance(prompt_eval_ns, int) else "N/A"
                    eval_str = f"{eval_ns / 1_000_000_000:.3f}s" if isinstance(eval_ns, int) else "N/A"
                    total_str = f"{total_ns / 1_000_000_000:.3f}s" if isinstance(total_ns, int) else f"{elapsed:.3f}s"
                    ttft_str = f"{first_token_at - started_at:.3f}s" if first_token_at is not None else "N/A"

                    logger.info(
                        "[PERF] ollama_stream http=%.3fs TTFT=%s prompt_eval=%s eval=%s total=%s tokens=%d chars=%d",
                        elapsed, ttft_str, p_eval_str, eval_str, total_str, token_count, char_count,
                    )
                    break

                message = chunk.get("message", {})
                content = message.get("content", "")
                if not content:
                    continue

                if first_token_at is None:
                    first_token_at = perf_counter()
                    ttft = first_token_at - started_at
                    logger.info("[LLM STREAM] first token received after %.3fs | TTFT=%.3fs", ttft, ttft)

                token_count += 1

                # ── Фильтрация <think>...<think> ──────────────────────
                filtered = self._filter_stream_think(content, inside_think)
                text_out, inside_think = filtered

                if text_out:
                    char_count += len(text_out)
                    yield text_out

        finally:
            resp.close()

    @staticmethod
    def _filter_stream_think(token: str, inside_think: bool) -> tuple[str, bool]:
        """Отфильтровать <think>...</think> из потокового токена.

        Returns:
            (filtered_text, new_inside_think_state)
        """
        result_parts: list[str] = []
        i = 0

        while i < len(token):
            if inside_think:
                close_pos = token.find("</think>", i)
                if close_pos == -1:
                    break  # весь остаток внутри think
                inside_think = False
                i = close_pos + len("</think>")
            else:
                open_pos = token.find("<think>", i)
                if open_pos == -1:
                    result_parts.append(token[i:])
                    break
                result_parts.append(token[i:open_pos])
                inside_think = True
                i = open_pos + len("<think>")

        return "".join(result_parts), inside_think

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
                json={
                    "model": self.embed_model,
                    "input": text,
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                },
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
