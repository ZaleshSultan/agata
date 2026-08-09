"""
Скрипт консолидации памяти ("сон" Агаты).

Периодический процесс:
1. Читает неконсолидированные записи из дневника за N дней.
2. Группирует похожие записи по косинусной схожести эмбеддингов (threshold-кластеризация).
3. Сжимает и объединяет похожие записи через LLM в емкие выводы.
4. Помечает обработанные записи как consolidated = 1.
5. Просит LLM обновить working_memory.md — убрать устаревшее, дополнить актуальным.

Запуск: python scripts/consolidate.py
"""

import logging
import sys
from pathlib import Path

# Добавляем родительскую директорию в sys.path для импорта модулей agata
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from llm import LocalLLM, OllamaUnavailableError
from memory import diary, working_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("consolidate")


def _cluster_entries(entries: list[dict], threshold: float = config.SIMILARITY_THRESHOLD) -> list[list[dict]]:
    """
    Простая threshold-кластеризация записей на основе косинусной схожести эмбеддингов.

    Args:
        entries: Список записей дневника (с полями embedding, text, id).
        threshold: Порог косинусной схожести для объединения в кластер.

    Returns:
        Список кластеров (каждый кластер — список записей dict).
    """
    clusters: list[list[dict]] = []

    for entry in entries:
        emb = entry.get("embedding")
        if emb is None:
            # Запись без эмбеддинга — отдельный кластер из 1 элемента
            clusters.append([entry])
            continue

        vec = np.array(emb, dtype=np.float32)
        norm_v = np.linalg.norm(vec)
        if norm_v == 0:
            clusters.append([entry])
            continue

        assigned = False
        for cluster in clusters:
            # Сравниваем с центроидом/первым элементом кластера
            c_emb = cluster[0].get("embedding")
            if c_emb is None:
                continue

            c_vec = np.array(c_emb, dtype=np.float32)
            norm_c = np.linalg.norm(c_vec)
            if norm_c == 0:
                continue

            sim = float(np.dot(vec, c_vec) / (norm_v * norm_c))
            if sim >= threshold:
                cluster.append(entry)
                assigned = True
                break

        if not assigned:
            clusters.append([entry])

    return clusters


def consolidate() -> None:
    """Запустить процесс консолидации памяти."""
    logger.info("Starting memory consolidation ('sleep' phase)...")
    llm = LocalLLM()

    # 1. Получаем неконсолидированные записи за последние N дней
    entries = diary.all_entries_since(days=config.CONSOLIDATION_DAYS)
    if not entries:
        logger.info("Нет новых записей в дневнике за последние %d дней для консолидации.", config.CONSOLIDATION_DAYS)
        return

    logger.info("Найдено %d записей в дневнике для обработки.", len(entries))

    # 2. Группируем по схожести
    clusters = _cluster_entries(entries, threshold=config.SIMILARITY_THRESHOLD)
    logger.info("Записи сгруппированы в %d кластеров.", len(clusters))

    ids_to_mark: list[int] = []
    summarized_facts: list[str] = []

    # 3. Обрабатываем каждый кластер через LLM
    for i, cluster in enumerate(clusters, 1):
        ids = [e["id"] for e in cluster]
        ids_to_mark.extend(ids)

        if len(cluster) == 1:
            # Одиночная запись — переносим как факт без лишних LLM вызовов
            summarized_facts.append(cluster[0]["text"])
            continue

        texts = [f"• {e['created_at']}: {e['text']}" for e in cluster]
        cluster_prompt = (
            "Объедини и сожми следующие похожие записи из дневника в один короткий, емкий и точный факт о пользователе или событиях:\n"
            + "\n".join(texts)
        )

        try:
            resp = llm.chat(
                messages=[
                    {"role": "system", "content": "Ты — модуль консолидации памяти. Пиши кратко, только суть."},
                    {"role": "user", "content": cluster_prompt},
                ],
                tools=None,
            )
            summary_text = resp.get("content", "").strip()
            if summary_text:
                summarized_facts.append(summary_text)
                logger.info("Кластер %d (%d записей) сжат в: %s", i, len(cluster), summary_text[:100])
        except OllamaUnavailableError as exc:
            logger.error("Ошибка обращения к LLM во время консолидации: %s", exc)
            return

    # 4. Обновляем рабочую память (working_memory.md) через LLM
    current_wm = working_memory.read()
    new_facts_str = "\n".join(f"- {fact}" for fact in summarized_facts)

    wm_update_prompt = f"""Ниже приведена текущая рабочая память и новые консолидированные факты из дневника.
Обнови и перепиши структуру рабочей памяти (Markdown).
Убери устаревшую информацию, залей актуальные новые факты, сохрани структуру (О пользователе, Текущие задачи, Важные факты).

<current_working_memory>
{current_wm}
</current_working_memory>

<new_consolidated_facts>
{new_facts_str}
</new_consolidated_facts>
"""

    try:
        wm_resp = llm.chat(
            messages=[
                {"role": "system", "content": "Ты — редактор рабочей памяти. Верни только обновлённый Markdown без комментариев."},
                {"role": "user", "content": wm_update_prompt},
            ],
            tools=None,
        )
        updated_wm_text = wm_resp.get("content", "").strip()
        if updated_wm_text:
            working_memory.write(updated_wm_text)
            logger.info("Рабочая память успешно обновлена.")
    except OllamaUnavailableError as exc:
        logger.error("Ошибка при обновлении рабочей памяти: %s", exc)
        return

    # 5. Помечаем записи как consolidated = 1
    diary.mark_consolidated(ids_to_mark)
    logger.info("Консолидация завершена. Обработано записей: %d.", len(ids_to_mark))


if __name__ == "__main__":
    consolidate()
