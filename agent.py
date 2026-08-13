"""
Agent loop Агаты.

Центральная логика: получает текст пользователя, подаёт в LLM
с рабочей памятью, короткой историей диалога и инструментами,
обрабатывает tool calls в цикле, возвращает финальный текстовый ответ.

run() — синхронный режим (как было).
run_streaming() — потоковый режим: текст LLM стримится через SentenceChunker
и отправляется в TTS по предложениям.

Принцип: LLM сама решает, какой инструмент вызвать — никакого
жёсткого if/else роутинга.
"""

import json
import logging
from time import perf_counter
from typing import Any, Callable, Optional

import config
from action_router import (
    RoutedAction,
    route_calculation,
    route_datetime,
    route_explicit_action,
    route_web_intent,
)
from llm import LocalLLM, OllamaUnavailableError, ToolCallParseError
from memory import MemoryManager
from prompts import build_agent_system_prompt, build_web_system_prompt
from sentence_chunker import SentenceChunker
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

    # ── Паттерны голосового подтверждения ────────────────────────────────
    _CONFIRM_YES = {"да", "подтверждаю", "выполняй", "давай", "ок", "окей",
                    "yes", "конечно", "ага", "угу", "точно", "валяй"}
    _CONFIRM_NO = {"нет", "отмена", "не надо", "отменяй", "стоп", "отмени",
                   "no", "не нужно", "нельзя", "не стоит", "отказ"}

    # Человекочитаемые описания действий для голосового вопроса
    _TOOL_VOICE_LABELS: dict[str, str] = {
        "close_app": "Закрыть {app_name}?",
        "close_target": "Закрыть {target}?",
        "open_app": "Открыть {app_name}?",
        "open_browser": "Открыть {url}?",
        "copy_project_snapshot": "Скопировать проект на флешку?",
        "create_reminder": "Создать напоминание?",
    }

    def _build_voice_question(self, tool_name: str, arguments: dict) -> str:
        """Сформировать голосовой вопрос для подтверждения."""
        template = self._TOOL_VOICE_LABELS.get(tool_name)
        if template:
            try:
                question = template.format(**arguments)
            except KeyError:
                question = f"Выполнить {tool_name}?"
        else:
            question = f"Выполнить {tool_name}?"
        return question + " Скажите да или нет."

    def _ask_confirmation(self, tool_name: str, arguments: dict) -> bool:
        """Запросить подтверждение действия — голосом или с клавиатуры.

        При VOICE_CONFIRMATION=True:
          1. TTS озвучивает вопрос.
          2. STT записывает ответ с микрофона.
          3. Ответ матчится по паттернам да/нет.
          4. Если STT не распознал — fallback на input().

        При VOICE_CONFIRMATION=False:
          Как раньше — текстовый input("y/n").
        """
        args_str = json.dumps(arguments, ensure_ascii=False, indent=2)
        print(f"\n⚠️  {config.ASSISTANT_NAME} хочет выполнить: {tool_name}")
        print(f"   Аргументы: {args_str}")

        if config.VOICE_CONFIRMATION and config.TTS_ENABLED:
            return self._ask_confirmation_voice(tool_name, arguments)

        return self._ask_confirmation_keyboard()

    def _ask_confirmation_voice(self, tool_name: str, arguments: dict) -> bool:
        """Голосовое подтверждение: TTS вопрос → STT ответ → паттерн-матчинг."""
        import stt
        import tts as tts_module

        question = self._build_voice_question(tool_name, arguments)
        logger.info("[CONFIRM] Voice question: '%s'", question)

        # Озвучиваем вопрос (синхронно — speak() блокирует до конца воспроизведения)
        tts_module.speak(question)

        # Записываем ответ с микрофона
        logger.info("[CONFIRM] Listening for voice answer...")
        print("   🎤 Слушаю ваш ответ...")
        spoken_text = stt.record_and_transcribe()

        if not spoken_text:
            logger.info("[CONFIRM] STT returned empty — fallback to keyboard")
            print("   (Не удалось распознать речь — введите ответ вручную)")
            return self._ask_confirmation_keyboard()

        spoken_lower = spoken_text.lower().strip().rstrip(".")
        logger.info("[CONFIRM] Voice answer: '%s'", spoken_text)

        # Точное совпадение со словарями
        if spoken_lower in self._CONFIRM_YES:
            logger.info("[CONFIRM] Accepted (exact match: '%s')", spoken_lower)
            print(f"   ✔ Подтверждено голосом: \"{spoken_text}\"")
            return True

        if spoken_lower in self._CONFIRM_NO:
            logger.info("[CONFIRM] Rejected (exact match: '%s')", spoken_lower)
            print(f"   ✖ Отменено голосом: \"{spoken_text}\"")
            return False

        # Подстрочный поиск (на случай "ну да давай" или "нет не надо")
        words = set(spoken_lower.split())
        if words & self._CONFIRM_YES:
            logger.info("[CONFIRM] Accepted (substring match in: '%s')", spoken_text)
            print(f"   ✔ Подтверждено голосом: \"{spoken_text}\"")
            return True

        if words & self._CONFIRM_NO:
            logger.info("[CONFIRM] Rejected (substring match in: '%s')", spoken_text)
            print(f"   ✖ Отменено голосом: \"{spoken_text}\"")
            return False

        # Не распознали как да/нет — fallback на клавиатуру
        logger.info("[CONFIRM] Unrecognized voice answer '%s' — fallback to keyboard", spoken_text)
        print(f"   (Не поняла ответ: \"{spoken_text}\" — введите вручную)")
        return self._ask_confirmation_keyboard()

    @staticmethod
    def _ask_confirmation_keyboard() -> bool:
        """Клавиатурное подтверждение (fallback)."""
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

        # 1. CLOSE / OPEN / SYSTEM ACTION
        action = route_explicit_action(user_text)
        if action is not None:
            router_elapsed = perf_counter() - router_started_at
            answer, tool_elapsed = self._run_routed_action(user_text, action)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 2. CALCULATION
        calculation = route_calculation(user_text)
        if calculation is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", calculation.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, calculation)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 3. DATETIME
        datetime_action = route_datetime(user_text)
        if datetime_action is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", datetime_action.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, datetime_action)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # 4. WEB SEARCH
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

    # ── Streaming ─────────────────────────────────────────────────────────

    def run_streaming(
        self,
        user_text: str,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Выполнить цикл агента с потоковой отдачей предложений.

        Детерминированные маршруты (открытие/закрытие, калькулятор,
        дата/время, веб-поиск) работают как прежде — ответ
        отправляется в on_sentence одной строкой.

        Путь CHAT использует LLM streaming: tool calls обрабатываются
        через обычный chat() (не стриминг), финальный текстовый
        ответ стримится через SentenceChunker в on_sentence().

        Args:
            user_text: Текст пользователя.
            on_sentence: Коллбэк, вызываемый для каждого готового
                предложения (tts.enqueue).

        Returns:
            Полный текст ответа (для истории и памяти).
        """
        started_at = perf_counter()
        router_started_at = perf_counter()

        def _emit(sentence: str) -> None:
            """Safe callback wrapper."""
            if on_sentence is not None:
                try:
                    on_sentence(sentence)
                except Exception as exc:
                    logger.error("[STREAM] on_sentence callback error: %s", exc)

        # Контекст памяти (best-effort)
        try:
            memory_context = self.memory.prepare_context(user_text)
        except Exception as exc:
            logger.warning("[MEMORY] context preparation failed: %s", exc)
            memory_context = {"working_memory": "", "relevant_memories": "", "current_datetime": ""}

        # ── 1. Детерминированные маршруты (без стриминга LLM) ────────

        # SYSTEM ACTIONS (open/close)
        action = route_explicit_action(user_text)
        if action is not None:
            router_elapsed = perf_counter() - router_started_at
            answer, tool_elapsed = self._run_routed_action(user_text, action)
            _emit(answer)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # CALCULATION
        calculation = route_calculation(user_text)
        if calculation is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", calculation.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, calculation)
            _emit(answer)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # DATETIME
        datetime_action = route_datetime(user_text)
        if datetime_action is not None:
            router_elapsed = perf_counter() - router_started_at
            logger.info("Fast router: %s", datetime_action.kind)
            answer, tool_elapsed = self._run_routed_action(user_text, datetime_action)
            _emit(answer)
            self._log_perf(started_at, router=router_elapsed, classifier="SKIPPED", tool=tool_elapsed, response=0.0)
            return answer

        # WEB SEARCH
        web_action = route_web_intent(user_text)
        if web_action is not None:
            router_elapsed = perf_counter() - router_started_at
            answer, search_elapsed, synthesis_elapsed = self._run_routed_web(
                user_text, web_action, memory_context,
            )
            _emit(answer)
            self._log_perf(
                started_at,
                router=router_elapsed,
                classifier="SKIPPED",
                web_search=search_elapsed,
                final_llm=synthesis_elapsed,
                response=0.0,
            )
            return answer

        # ── 2. CHAT/UNKNOWN: direct streaming response (no blocking tools) ──

        router_elapsed = perf_counter() - router_started_at
        logger.info("Router: CHAT/UNKNOWN (streaming mode)")
        llm = self._get_llm()
        use_thinking = self._decide_thinking(user_text)

        working_mem = memory_context["working_memory"]
        system_content = build_agent_system_prompt(
            working_mem,
            memory_context["relevant_memories"],
            memory_context["current_datetime"],
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        logger.info(
            "LLM context: messages=%d history_messages=%d",
            len(messages), len(self.history),
        )

        final_content: str | None = None
        llm_started_at = perf_counter()

        try:
            final_content = self._stream_final_response(
                llm, messages, use_thinking, _emit,
            )
            if not final_content:
                final_content = "Хм, у меня нет ответа на это."
                _emit(final_content)
        except OllamaUnavailableError as exc:
            logger.error("Ollama недоступна для streaming: %s", exc)
            final_content = f"Извини, не могу связаться с Ollama: {exc}"
            _emit(final_content)

        llm_elapsed = perf_counter() - llm_started_at

        answer = self._remember_turn(user_text, final_content)
        self._log_perf(
            started_at,
            router=router_elapsed,
            classifier="SKIPPED",
            llm=llm_elapsed,
            tool=0.0,
            mode="streaming",
        )
        return answer

    def _stream_final_response(
        self,
        llm: LocalLLM,
        messages: list[dict[str, Any]],
        use_thinking: bool,
        emit: Callable[[str], None],
    ) -> str:
        """Потоково получить финальный текстовый ответ от LLM.

        Токены проходят через SentenceChunker, который собирает
        их в предложения и отправляет через emit() в TTS.

        Returns:
            Полный накопленный текст ответа.
        """
        chunker = SentenceChunker()
        full_text_parts: list[str] = []
        llm_stream_started_at = perf_counter()
        first_chunker_at: float | None = None
        first_tts_emit_at: float | None = None

        def _chunker_emit(sentence: str) -> None:
            nonlocal first_tts_emit_at
            if first_tts_emit_at is None:
                first_tts_emit_at = perf_counter()
                logger.info(
                    "[DIAG AGENT] First sentence emitted to TTS at +%.3fs from LLM stream start",
                    first_tts_emit_at - llm_stream_started_at,
                )
            emit(sentence)

        logger.info("[STREAM] Начало потокового ответа")

        for token in llm.chat_stream(messages, think=use_thinking):
            if first_chunker_at is None:
                first_chunker_at = perf_counter()
                logger.info(
                    "[DIAG AGENT] First token fed to SentenceChunker at +%.3fs from LLM stream start",
                    first_chunker_at - llm_stream_started_at,
                )
            full_text_parts.append(token)

            # Передаём токен в chunker
            sentences = chunker.feed(token)
            for sentence in sentences:
                _chunker_emit(sentence)

        # Отправляем остаток буфера
        remainder = chunker.flush()
        if remainder:
            _chunker_emit(remainder)

        full_text = "".join(full_text_parts).strip()
        # Убираем <think> теги из полного текста для истории
        full_text = LocalLLM._strip_think_tags(full_text)
        logger.info("[STREAM] Полный ответ: %d символов", len(full_text))
        return full_text
