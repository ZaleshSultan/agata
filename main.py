"""
Главный модуль запуска голосового ассистента Агата.

Интерактивный цикл:
1. Нажмите Enter для начала записи речи (или введите текст вручную при необходимости/тестировании).
2. STT: запись с микрофона и распознавание в текст через faster-whisper.
3. Agent loop: обработка запроса с памятью и вызовом инструментов.
4. Вывод ответа в консоль и озвучивание через TTS.
"""

import logging
import sys
from time import perf_counter

from agent import Agent
import config
import stt
import tts

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Главный интерактивный цикл работы с Агатой."""
    print("=" * 60)
    print(f"  Голосовой ассистент {config.ASSISTANT_NAME} запущен.")
    print("  Убедитесь, что Ollama запущена на localhost:11434")
    print("  Нажмите Ctrl+C для выхода.")
    print("=" * 60)
    print()

    agent = Agent()

    while True:
        try:
            print("\nНажмите ENTER для записи с микрофона (или введите текст вручную для консольного ввода):")
            user_input = input("> ").strip()

            if user_input:
                # Введён текст напрямую (для удобства отладки без микрофона)
                text = user_input
                print(f"[Вы (текст)]: {text}")
            else:
                # Нажат Enter -> Запись с микрофона
                text = stt.record_and_transcribe()
                if not text:
                    print("[Речь не распознана или тишина, попробуйте ещё раз]")
                    continue
                print(f"[Вы (голос)]: {text}")

            print(f"Размышляю...")
            request_started_at = perf_counter()
            answer = agent.run(text)
            agent_elapsed = perf_counter() - request_started_at
            print(f"\n{config.ASSISTANT_NAME}: {answer}\n")

            # Озвучивание через TTS
            tts_started_at = perf_counter()
            tts.speak(answer)
            tts_elapsed = perf_counter() - tts_started_at
            logger.info(
                "[PERF] agent=%0.3fs tts=%0.3fs total=%0.3fs",
                agent_elapsed,
                tts_elapsed,
                perf_counter() - request_started_at,
            )

        except (KeyboardInterrupt, EOFError):
            print(f"\nДо встречи! {config.ASSISTANT_NAME} отключается.")
            break
        except Exception as exc:
            logger.error("Непредвиденная ошибка в главном цикле: %s", exc, exc_info=True)
            print(f"\nПроизошла ошибка: {exc}\n")


if __name__ == "__main__":
    main()
