"""
Agent loop Агаты.

Центральная логика: получает текст пользователя, подаёт в LLM
с рабочей памятью, короткой историей диалога и инструментами,
обрабатывает tool calls в цикле, возвращает финальный текстовый ответ.

Принцип: LLM сама решает, какой инструмент вызвать — никакого
жёсткого if/else роутинга.
"""

import json
import logging
from time import perf_counter
from typing import Any

import config
from action_router import (
    RoutedAction,
    route_calculation,
    route_content_play,
    route_datetime,
    route_explicit_action,
    route_media,
    route_volume,
    route_web_intent,
)
from llm import LocalLLM, OllamaUnavailableError, ToolCallParseError
from memory import MemoryManager
from prompts import build_agent_system_prompt, build_web_system_prompt
from tools import get_tool_schemas, execute_tool, requires_confirmation

logger = logging.getLogger(__name__)

# Сколько последних пар (user, assistant) хранить в сессионной истории.
# Больше — лучше помнит контекст разговора, но длиннее промпт → медленнее
# инференс на слабом железе. 5 пар — разумный баланс для голосового
# ассистент реального времени при num_ctx=8192.
MAX_HISTORY_TURNS: int = config.MEMORY_RECENT_CONVERSATION_TURNS


# Паттерны, однозначно указывающие на сложный запрос — thinking нужен
_COMPLEX_PATTERNS = (
    "сравни", "объясни", "спланируй", "проанализируй", "почему",
    "как думаешь", "посоветуй", "в чём разница", "что лучше",
    "разработай", "составь план", "напиши код",
)

_APP_DISPLAY_NAMES = {
    "chrome": "Chrome",
    "firefox": "Firefox",
    "vscode": "VS Code",
    "discord": "Discord",
    "telegram": "Telegram",
    "notepad": "Блокнот",
    "explorer": "Проводник",
}

# Инструменты, которые action_router.py уже перехватывает детерминированно
# ДО обращения к LLM — в разговорной ветке (CHAT/UNKNOWN) их не нужно
# посылать в tools, это чистый лишний вес в промпте.
_ROUTER_HANDLED_TOOLS: set[str] = {
    "calculate",
    "open_app",
    "close_app",
    "close_target",
    "open_browser",
    "web_search",
    "get_current_datetime",
    "control_volume",
    "control_media",
    "play_content",
}

class Agent:
    """
    Агент с сессионной памятью диалога.

    Один экземпляр Agent живёт всё время работы main.py — history
    накапливается между вызовами run() и обнуляется при перезапуске
    программы (это ожидаемо: это "оперативная" память разговора,
    не долгосрочная).
    """

    def __init__(self) -> None:
        self._llm: LocalLLM | None = None
        self.memory = MemoryManager()
        # Короткая история диалога: только чистые пары user/assistant,
        # БЕЗ внутренних tool_calls шагов — они нужны только внутри
        # одного agent loop, не в долгой истории.
        self.history: list[dict[str, Any]] = []

    def _get_llm(self) -> LocalLLM:
        if self._llm is None:
            self._llm = LocalLLM()
        return self._llm

    def _ask_confirmation(self, tool_name: str, arguments: dict) -> bool:
        args_str = json.dumps(arguments, ensure_ascii=False, indent=2)
        print(f"\n⚠️  {config.ASSISTANT_NAME} хочет выполнить: {tool_name}")
        print(f"   Аргументы: {args_str}")
        try:
            answer = input("   Выполнить? [y/n]: ").strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes", "д", "да")

    def reset_history(self) -> None:
        """Явно очистить сессионную историю (например, по голосовой команде
        'забудь этот разговор' — можно завести под это отдельный intent позже)."""
        self.history.clear()
        logger.info("Сессионная история диалога очищена.")

    def _remember_turn(self, user_text: str, answer: str) -> str:
        """Store only the user-visible exchange, including routed actions."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
        # Memory is best-effort: a storage/extraction outage must never suppress an answer.
        try:
            self.memory.extract(user_text, answer)
        except Exception as exc:
            logger.warning("[MEMORY] turn extraction failed: %s", exc)
        return answer

    @staticmethod
    def _tool_payload(result: str) -> dict[str, Any] | None:
        """Read the structured result returned by tools with side effects."""
        try:
            payload = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _execution_failure(cls, tool_name: str, result: str) -> str | None:
        """Return a truthful error for structured tool failures, if any."""
        payload = cls._tool_payload(result)
        if payload is not None and payload.get("success") is False:
            return payload.get("error") or f"Инструмент {tool_name} не выполнил действие."
        return None

    @staticmethod
    def _log_perf(started_at: float, **stages: float | str) -> None:
        """Log durations and branch decisions without logging user content."""
        parts = []
        for name, value in stages.items():
            rendered = f"{value:.3f}s" if isinstance(value, float) else str(value)
            parts.append(f"{name}={rendered}")
        parts.append(f"total={perf_counter() - started_at:.3f}s")
        logger.info("[PERF] %s", " ".join(parts))

    def _run_routed_action(self, user_text: str, action: RoutedAction) -> tuple[str, float]:
        """Execute a deterministic local action and render its verified result."""
        logger.info("Action router: %s", action.kind)
        if action.arguments:
            logger.info("application/url: %s", next(iter(action.arguments.values())))

        if requires_confirmation(action.tool_name) and not self._ask_confirmation(action.tool_name, action.arguments):
            return self._remember_turn(user_text, "Действие отменено."), 0.0

        logger.info("Tool call: %s(%s)", action.tool_name, json.dumps(action.arguments, ensure_ascii=False))
        tool_started_at = perf_counter()
        result = execute_tool(action.tool_name, action.arguments)
        tool_elapsed = perf_counter() - tool_started_at
        failure = self._execution_failure(action.tool_name, result)
        if failure:
            if failure == "target_not_individually_closeable":
                target_name = action.arguments.get("target", "эту вкладку")
                return self._remember_turn(
                    user_text,
                    f"Не могу закрыть отдельную вкладку {target_name} в браузере. Вы можете закрыть браузер целиком командой 'закрой Chrome'."
                ), tool_elapsed
            logger.info("Tool result: failure; %s", failure)
            return self._remember_turn(user_text, f"Не удалось выполнить действие: {failure}"), tool_elapsed

        payload = self._tool_payload(result)
        if action.tool_name in {"open_app", "close_app", "open_browser", "close_target"}:
            if payload is None or payload.get("success") is not True:
                error = payload.get("error") if payload else f"Инструмент {action.tool_name} не подтвердил успешное выполнение."
                if error == "target_not_individually_closeable":
                    target_name = action.arguments.get("target", "эту вкладку")
                    return self._remember_turn(
                        user_text,
                        f"Не могу закрыть отдельную вкладку {target_name} в браузере. Вы можете закрыть браузер целиком командой 'закрой Chrome'."
                    ), tool_elapsed
                logger.error("Tool result: unverified success for %s", action.tool_name)
                return self._remember_turn(user_text, f"Не удалось выполнить действие: {error}"), tool_elapsed

        logger.info("Tool result: success")
        if action.tool_name == "open_app":
            app = payload.get("application") if payload else None
            if not app:
                app = _APP_DISPLAY_NAMES.get(action.arguments["app_name"], action.arguments["app_name"])
            return self._remember_turn(user_text, f"{app} открыт."), tool_elapsed
        if action.tool_name in {"close_app", "close_target"}:
            app = payload.get("application") if payload else None
            if not app:
                raw_arg = action.arguments.get("app_name") or action.arguments.get("target") or "Приложение"
                app = _APP_DISPLAY_NAMES.get(raw_arg.lower(), raw_arg)
            return self._remember_turn(user_text, f"{app} закрыт."), tool_elapsed
        if action.tool_name == "open_browser":
            return self._remember_turn(user_text, f"Открыла {action.arguments['url']} в браузере."), tool_elapsed
        if action.tool_name == "get_current_datetime":
            return self._remember_turn(user_text, result), tool_elapsed
        if action.tool_name == "control_volume":
            volume = payload.get("volume") if payload else None
            if action.arguments.get("action") == "volume_set":
                return self._remember_turn(user_text, f"Громкость установлена на {volume}%."), tool_elapsed
            if action.arguments.get("action") == "volume_up":
                return self._remember_turn(user_text, f"Громкость увеличена до {volume}%."), tool_elapsed
            return self._remember_turn(user_text, f"Громкость уменьшена до {volume}%."), tool_elapsed
        if action.tool_name == "control_media":
            labels = {
                "play_pause": "Воспроизведение переключено.",
                "pause": "Поставила на паузу.",
                "stop": "Остановила воспроизведение.",
                "resume": "Продолжаю воспроизведение.",
                "play": "Продолжаю воспроизведение.",
                "next": "Переключила на следующий трек.",
                "previous": "Переключила на предыдущий трек.",
            }
            act_key = action.arguments.get("action", "play_pause")
            return self._remember_turn(user_text, labels.get(act_key, "Готово.")), tool_elapsed
        if action.tool_name == "play_content":
            if payload and payload.get("success"):
                title = payload.get("title") or action.arguments.get("query", "контент")
                return self._remember_turn(user_text, f"Включила {title}."), tool_elapsed
            reason = (payload.get("reason") or "Не удалось воспроизвести контент.") if payload else "Не удалось воспроизвести контент."
            return self._remember_turn(user_text, f"Не удалось выполнить действие: {reason}"), tool_elapsed
        if result.startswith("Ошибка"):
            return self._remember_turn(user_text, f"Не удалось выполнить вычисление: {result}"), tool_elapsed
        return self._remember_turn(user_text, result), tool_elapsed

    def _run_routed_web(self, user_text: str, action: RoutedAction, memory_context: dict[str, str]) -> tuple[str, float, float]:
        """Run one web search followed by exactly one LLM synthesis call."""
        logger.info("Web router: %s", action.kind)
        logger.info("Tool call: %s(%s)", action.tool_name, json.dumps(action.arguments, ensure_ascii=False))
        search_started_at = perf_counter()
        result = execute_tool(action.tool_name, action.arguments)
        search_elapsed = perf_counter() - search_started_at

        failure = self._execution_failure(action.tool_name, result)
        payload = self._tool_payload(result)
        results = payload.get("results") if payload else None
        if failure:
            logger.info("Tool result: failure; %s", failure)
            return self._remember_turn(user_text, f"Не удалось выполнить поиск: {failure}"), search_elapsed, 0.0
        if not isinstance(results, list) or not results:
            logger.info("Tool result: search returned no verified results")
            return (
                self._remember_turn(user_text, "Не нашла надёжных результатов для этого запроса."),
                search_elapsed,
                0.0,
            )

        search_context = json.dumps(results, ensure_ascii=False)
        web_system_prompt = build_web_system_prompt(
            search_context,
            memory_context.get("relevant_memories", ""),
            memory_context.get("current_datetime", ""),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": web_system_prompt},
            {"role": "user", "content": user_text},
        ]
        logger.info("WEB synthesis context: results=%d chars=%d", len(results), len(search_context))

        synthesis_started_at = perf_counter()
        try:
            response = self._get_llm().chat(messages, tools=None, think=False)
        except OllamaUnavailableError as exc:
            logger.error("Ollama недоступна для WEB synthesis: %s", exc)
            return self._remember_turn(user_text, f"Поиск выполнен, но не могу сформировать ответ: {exc}"), search_elapsed, perf_counter() - synthesis_started_at

        synthesis_elapsed = perf_counter() - synthesis_started_at
        if response.get("type") != "text":
            logger.error("WEB synthesis unexpectedly returned a tool call")
            return (
                self._remember_turn(user_text, "Поиск выполнен, но не удалось безопасно сформировать ответ по его результатам."),
                search_elapsed,
                synthesis_elapsed,
            )
        answer = response.get("content") or "Не удалось сформировать ответ по результатам поиска."
        return self._remember_turn(user_text, answer), search_elapsed, synthesis_elapsed

    def _decide_thinking(self, user_text: str) -> bool:
        """Choose thinking deterministically; this method never calls an LLM."""
        if not config.ENABLE_ADAPTIVE_THINKING:
            return config.ENABLE_THINKING

        text_lower = user_text.lower()
        if any(p in text_lower for p in _COMPLEX_PATTERNS):
            logger.info("Router: эвристика -> СЛОЖНЫЙ (без LLM)")
            return True
        logger.info("Router: эвристика -> ОБЫЧНЫЙ (classifier skipped)")
        return False

    def run(self, user_text: str) -> str:
        """
        Выполнить один цикл агента: от текста пользователя до финального ответа.

        Теперь messages строятся как:
        system + working_memory  →  self.history (последние N пар)  →  текущая реплика

        Внутренние tool-call шаги текущего запроса живут только в локальной
        messages этого вызова и НЕ попадают в self.history — туда попадает
        только финальный чистый обмен репликами.
        """
        started_at = perf_counter()
        router_started_at = perf_counter()

        # Every route starts with the same best-effort memory lifecycle.  Routed
        # actions may not use it for reasoning, but recall and failure isolation
        # are intentional rather than accidental CHAT-only behaviour.
        try:
            memory_context = self.memory.prepare_context(user_text)
        except Exception as exc:
            logger.warning("[MEMORY] context preparation failed: %s", exc)
            memory_context = {"working_memory": "", "relevant_memories": "", "current_datetime": ""}

        # 1. MEDIA TRANSPORT (Highest priority — stop, pause, resume, next, prev)
        media_action = route_media(user_text)
        if media_action is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", media_action.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, media_action)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 2. CLOSE / OPEN / SYSTEM ACTION
        action = route_explicit_action(user_text)
        if action is not None:
            router_elapsed = perf_counter() - router_started_at
            answer, tool_elapsed = self._run_routed_action(user_text, action)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 3. VOLUME
        volume_action = route_volume(user_text)
        if volume_action is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", volume_action.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, volume_action)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 4. CONTENT PLAY (YouTube / SoundCloud)
        content_action = route_content_play(user_text)
        if content_action is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", content_action.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, content_action)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 5. WEB SEARCH (incl. weather)
        web_action = route_web_intent(user_text)
        if web_action is not None:
            router_elapsed = perf_counter() - router_started_at
            answer, search_elapsed, synthesis_elapsed = self._run_routed_web(user_text, web_action, memory_context)
            self._log_perf(
                started_at,
                router=router_elapsed,
                classifier="SKIPPED",
                web_search=search_elapsed,
                final_llm=synthesis_elapsed,
                response=0.0,
            )
            return answer

        # 6. CALCULATION
        calculation = route_calculation(user_text)
        if calculation is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", calculation.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, calculation)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 7. DATETIME
        datetime_action = route_datetime(user_text)
        if datetime_action is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", datetime_action.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, datetime_action)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        router_elapsed = perf_counter() - router_started_at
        logger.info("Router: CHAT/UNKNOWN (classifier skipped)")
        llm = self._get_llm()
        use_thinking = self._decide_thinking(user_text)

        working_mem = memory_context["working_memory"]
        system_content = build_agent_system_prompt(
            working_mem,
            memory_context["relevant_memories"],
            memory_context["current_datetime"],
        )

        # ── Собираем messages: system + история диалога + текущая реплика ──
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        messages.extend(self.history)  # прошлые чистые пары user/assistant
        messages.append({"role": "user", "content": user_text})
        logger.info(
            "LLM context: messages=%d history_messages=%d working_memory_chars=%d relevant_memory_chars=%d",
            len(messages),
            len(self.history),
            len(working_mem),
            len(memory_context["relevant_memories"]),
        )

        tool_schemas = get_tool_schemas(exclude=_ROUTER_HANDLED_TOOLS)
        final_content: str | None = None
        llm_elapsed = 0.0
        tool_elapsed = 0.0

        for step in range(config.MAX_AGENT_STEPS):
            logger.info("Agent step %d/%d", step + 1, config.MAX_AGENT_STEPS)

            try:
                llm_started_at = perf_counter()
                response = llm.chat(messages, tools=tool_schemas, think=use_thinking)
                llm_elapsed += perf_counter() - llm_started_at
            except OllamaUnavailableError as exc:
                logger.error("Ollama недоступна: %s", exc)
                final_content = f"Извини, не могу связаться с Ollama: {exc}"
                break
            except ToolCallParseError as exc:
                logger.warning("Битый tool call, пробую без инструментов: %s", exc)
                try:
                    llm_started_at = perf_counter()
                    response = llm.chat(messages, tools=None, think=use_thinking)
                    llm_elapsed += perf_counter() - llm_started_at
                except OllamaUnavailableError as inner_exc:
                    final_content = f"Извини, не могу связаться с Ollama: {inner_exc}"
                    break

            if final_content is not None:
                break

            if response["type"] == "text":
                final_content = response["content"] or "Хм, у меня нет ответа на это."
                break

            tool_name: str = response["name"]
            tool_args: dict = response["arguments"]
            logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_args, ensure_ascii=False)[:200])

            if requires_confirmation(tool_name):
                if not self._ask_confirmation(tool_name, tool_args):
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": tool_name, "arguments": tool_args}}],
                    })
                    messages.append({"role": "tool", "content": f"Пользователь отказался выполнять {tool_name}."})
                    continue

            tool_started_at = perf_counter()
            tool_result = execute_tool(tool_name, tool_args)
            tool_elapsed += perf_counter() - tool_started_at
            logger.info("Результат %s: %s", tool_name, tool_result[:200])

            failure = self._execution_failure(tool_name, tool_result)
            if failure:
                # Do not let a later text turn claim a failed system action succeeded.
                final_content = f"Не удалось выполнить действие: {failure}"
                break

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": tool_name, "arguments": tool_args}}],
            })
            messages.append({"role": "tool", "content": tool_result})

        if final_content is None:
            logger.warning("Превышен лимит шагов агента (%d)", config.MAX_AGENT_STEPS)
            final_content = (
                f"Извини, я превысила лимит в {config.MAX_AGENT_STEPS} шагов "
                "и не смогла завершить задачу. Попробуй сформулировать проще."
            )

        # ── Сохраняем ЧИСТУЮ пару в сессионную историю (без tool-шагов) ──
        answer = self._remember_turn(user_text, final_content)
        self._log_perf(
            started_at,
            router=router_elapsed,
            classifier="SKIPPED",
            llm=llm_elapsed,
            tool=tool_elapsed,
            response=0.0,
        )
        return answer
