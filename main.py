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
    tts.start()

    try:
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
                tts_enqueue_count = 0
                first_tts_enqueue_at: float | None = None

                def on_sentence(sentence: str) -> None:
                    nonlocal tts_enqueue_count, first_tts_enqueue_at
                    now = perf_counter()
                    if first_tts_enqueue_at is None:
                        first_tts_enqueue_at = now
                        logger.info(
                            "[DIAG MAIN] First TTS sentence enqueued at +%.3fs from request start",
                            first_tts_enqueue_at - request_started_at,
                        )
                    tts_enqueue_count += 1
                    logger.info("[MAIN STREAM] Sentence ready for TTS (%d): '%s'", tts_enqueue_count, sentence[:60])
                    tts.enqueue(sentence)

                answer = agent.run_streaming(text, on_sentence=on_sentence)
                agent_elapsed = perf_counter() - request_started_at
                print(f"\n{config.ASSISTANT_NAME}: {answer}\n")

                # Дожидаемся завершения озвучивания предложений в очереди
                tts_started_at = perf_counter()
                tts.wait_until_done()
                tts_elapsed = perf_counter() - tts_started_at

                logger.info(
                    "[DIAG MAIN STATS] agent_stream=%0.3fs | first_tts_latency=%s | total_tts_chunks=%d | tts_drain=%0.3fs | total=%0.3fs",
                    agent_elapsed,
                    f"{first_tts_enqueue_at - request_started_at:.3f}s" if first_tts_enqueue_at is not None else "N/A",
                    tts_enqueue_count,
                    tts_elapsed,
                    perf_counter() - request_started_at,
                )

            except (KeyboardInterrupt, EOFError):
                print(f"\nДо встречи! {config.ASSISTANT_NAME} отключается.")
                break
            except Exception as exc:
                logger.error("Непредвиденная ошибка в главном цикле: %s", exc, exc_info=True)
                print(f"\nПроизошла ошибка: {exc}\n")
    finally:
        tts.stop()


if __name__ == "__main__":
    main()
