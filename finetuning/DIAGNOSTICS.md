# Диагностика запоминания и поведения

Этап `diagnostics_v1` проверяет работоспособность обучения отдельно от качества обучения книге.
Локальный итог: [REPORT.md](evaluation/reports/diagnostics_v1/REPORT.md).
Один новый идентификатор воспроизведён после 20 шагов; восемь разных фактов — на 16/16
тренировочных вопросах после 50 шагов. Перенос на новые формулировки — 5/8, ответы о
неизвестном — 0/2. Это успешная диагностика запоминания, а не готовый адаптер AGATA.
Стилевой прогон не дал совокупного улучшения на маленьком независимом наборе.

## Границы и сохранность

Все модели загружаются из закреплённой базы `unsloth/Qwen3-1.7B-bnb-4bit`, revision
`d1d7781199f38692496fab70012c39598f0c858f`. Каждый прогон начинает новый LoRA r16/alpha16.
Рабочие AGATA, Ollama, память и Modelfile не используются для записи. Push не выполняется.
Запуски идут последовательно в `agata-ft`; запускать несколько GPU-процессов одновременно не нужно.
Данные, полный отчёт, ответы и checkpoints остаются локальными в игнорируемых Git каталогах.
Синтетические исходники и диагностические инструменты находятся в Git.

`prepare_diagnostics.py` создаёт 3/1/9 примеров train/validation/benchmark для одного факта,
16/8/27 для восьми фактов и 18/6/22 для характера. Числа вопросов не являются числом независимых
фактов. В factual benchmark exact-вопросы намеренно совпадают с тренировочными; paraphrase и
validation проверяют новые формулировки тех же фактов. Unknown и format оцениваются отдельно.
Идентификатор генерируется один раз и сохраняется до baseline. Существующие данные не перезаписываются.

## Инструменты аудита

`audit_tokenization.py` проверяет реальные токены всех A/B/C train/validation, assistant-only labels,
EOS, padding, одинаковость префиксов обучения и генерации и saved tokenizers. CUDA не нужна.
`audit_adapter_runtime.py` на GPU проверяет каждый сохранённый LoRA tensor, отключение адаптера,
обнуление LoRA-B только в оперативной памяти, восстановление и ручной causal loss против fused loss.
Архив оригинального `train.py` лежит в локальном `audit_original/`.

В `train.py` исправлено чтение `training.max_steps`: CLI имеет приоритет, smoke без явного
переопределения использует 5, обычный запуск читает YAML и использует -1 при отсутствии значения.
Исторические A/B/C запускались с явным `--max-steps 32`, поэтому эта ошибка не меняла их бюджет.

```powershell
conda activate agata-ft
python -m pytest finetuning/tests -q
python -m pip check
python finetuning/audit_tokenization.py --output evaluation/reports/diagnostics_repeat/tokenization.json
# GPU-аудит одного прежнего адаптера; A/C задаются аналогично.
python finetuning/audit_adapter_runtime.py --config datasets/abc_v1/configs/B.yaml --adapter outputs/abc_v1/B/adapter --output evaluation/reports/diagnostics_repeat/adapter_B.json
```

## Повтор диагностического прогона

Ниже отдельный новый каталог; запуск всех команд занимает GPU и создаёт новые артефакты.
Для точного повтора используйте замороженные `datasets/diagnostics_v1`, а не заново случайно
сгенерированный идентификатор. Другой идентификатор — новый эксперимент с собственным baseline.

```powershell
# Из корня репозитория. Параметры путей скриптов относительны к finetuning/.
python finetuning/evaluate_diagnostics.py --benchmark datasets/diagnostics_v1/one_fact_benchmark.jsonl --output evaluation/reports/diagnostics_repeat/one_fact_baseline
python finetuning/train_diagnostics.py --variant one_fact --output-dir outputs/diagnostics_repeat/one_fact
python finetuning/evaluate_diagnostics.py --benchmark datasets/diagnostics_v1/one_fact_benchmark.jsonl --adapter outputs/diagnostics_repeat/one_fact/checkpoint-20/adapter --output evaluation/reports/diagnostics_repeat/one_fact_step20
```

Следующий этап допускается только после положительного `one_fact_gate` в полном отчёте:
точный вопрос должен вызвать весь идентификатор без добавлений и завершиться EOS.
Одной низкой loss или наличия подстроки недостаточно. Mini trainer проверяет gate и его происхождение.

```powershell
python finetuning/evaluate_diagnostics.py --benchmark datasets/diagnostics_v1/mini_benchmark.jsonl --output evaluation/reports/diagnostics_repeat/mini_baseline
python finetuning/train_diagnostics.py --variant mini --gate-report evaluation/reports/diagnostics_repeat/one_fact_step20.json --output-dir outputs/diagnostics_repeat/mini
python finetuning/evaluate_diagnostics.py --benchmark datasets/diagnostics_v1/style_benchmark.jsonl --output evaluation/reports/diagnostics_repeat/style_baseline
python finetuning/train_diagnostics.py --variant style --output-dir outputs/diagnostics_repeat/style

foreach ($variant in @('one_fact', 'mini', 'style')) {
    $steps = if ($variant -eq 'style') { @(20, 50) } else { @(20, 50, 100) }
    foreach ($step in $steps) {
        if (($variant -eq 'one_fact') -and ($step -eq 20)) { continue }
        python finetuning/evaluate_diagnostics.py --benchmark "datasets/diagnostics_v1/${variant}_benchmark.jsonl" --adapter "outputs/diagnostics_repeat/$variant/checkpoint-$step/adapter" --output "evaluation/reports/diagnostics_repeat/${variant}_step$step"
    }
}
```

Обучение: 100 шагов facts, 50 style; seed3407, BF16, batch1/accumulation4, длина 1024,
AdamW8bit, weight decay0.01, warmup2, постоянный LR2e-4 facts /1e-4 style. LoRA применяется
к q/k/v/o/gate/up/down projection. Датасет не размножается; повторные эпохи намеренны для overfit.
Реальное число предъявлений учитывается после каждого выполненного training microstep,
поскольку неполные последние накопления дают меньше примеров, чем steps×4.
Checkpoints включают состояние Trainer и самостоятельный адаптер в `checkpoint-N/adapter/`.

Inference во всех точках одинаков: greedy, thinking выключен, max_new_tokens128, общая system-инструкция,
без эталона в запросе. Сохраняются исходные IDs, raw decoded, признаки EOS/лимита, хэши входных
токенов, lexical exact/inclusion и NLL с teacher forcing. Низкая NLL не считается правильным ответом.

## Оценка и отчёт

```powershell
python finetuning/report_diagnostics.py --variant mini --reports-dir evaluation/reports/diagnostics_repeat --runs-dir outputs/diagnostics_repeat --output evaluation/reports/diagnostics_repeat/mini_comparison
```

Команда создаёт таблицы, примеры, loss/VRAM/NLL PNG и шаблон смысловых оценок. Смысловую оценку
нужно заполнить отдельно по замороженной рубрике, предпочтительно скрыв имя checkpoint от проверяющего.
Не превращайте exact-match или NLL в автоматический балл характера, честности или общей правильности.
Для отчёта с оценками укажите `--ratings PATH` и новый `--output`; файл содержит reviewer,
review_method, SHA256 каждого исходного отчёта по step и ratings с полями step/id/score/reason/unsupported_claim.
Существующие отчёты не перезаписываются. Reviewer проверяет все вопросы и отдельно отмечает
галлюцинации, естественность русского и копирование запроса.

В итоговом `diagnostics_v1` оценены все 210 ответов; для 30 ответов о навыках/честности выполнена
вторая слепая AI-проверка. Это не независимое исследование с людьми и не статистическое доказательство
превосходства модели. Следующий эксперимент с книгой должен оставаться небольшим, проверять перенос
и неизвестные факты и не начинаться с простого наращивания числа эпох.
