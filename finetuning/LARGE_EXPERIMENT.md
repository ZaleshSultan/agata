# AGATA large_v3

Цель — проверить перенос на новые вопросы и сохранение честного незнания на существенно большем
проверенном наборе. Итоговые результаты находятся локально в
[evaluation/reports/large_v3/REPORT.md](evaluation/reports/large_v3/REPORT.md).
Эта инструкция описывает протокол; наличие кода само по себе не подтверждает результат обучения.

## Данные

Авторские компоненты лежат в игнорируемом `datasets/large_v3/components/`.
Каждая строка содержит id, domain, family_id, partition, prompt, answer, metadata и, для книги,
fact_id с точным локальным доказательством. Все книжные доказательства остаются вне Git.
Контроль источника проверяет цитату, страницу и идентификатор фрагмента; смысл ответа отдельно
проверяют автор и другой AI-рецензент. Человеческая независимая экспертиза не заявляется.

В наборах нет косметического размножения вопросов. Тематические family ID и книжные proposition ID
не пересекаются между train/validation/test. Это разделение не означает, что общие языковые навыки
или вся семантика темы отсутствуют в других наборах. Отдельные близкие контрасты сохраняются
осознанно: например, известный день доставки и неизвестный час при одном условии.

`prepare_large.py` проверяет конфликты, пересечения, доказательства, ограничения длины и маски
assistant-only через закреплённый tokenizer. После независимого обзора `freeze_review.json`
разрешает заморозку именно проверенных SHA256. Повторная запись готового protocol запрещена.

```powershell
conda activate agata-ft
python finetuning/prepare_large.py
# После проверки preflight и подписанного локального freeze_review.json:
python finetuning/prepare_large.py --freeze
```

Команды выше предназначены для подготовки незамороженного набора. Для повторного обучения на
уже подготовленных данных повторно запускать freeze не нужно. Сохранённые `train.jsonl`,
`validation.jsonl` и `test.jsonl` содержат assistant targets; evaluator получает только system/user
из `evaluation_validation.jsonl`, `benchmark.jsonl` и `recall_probes.jsonl`.

Наборы имеют разные назначения:

- Validation и независимый test используют новые тематические сценарии. Книжные вопросы здесь
  проверяют чтение короткого предоставленного фрагмента из отдельного семейства, не знание
  отсутствовавшего в обучении факта без контекста.
- Known-exact probes намеренно повторяют 12 тренировочных вопросов. Это диагностика запоминания.
- Known-paraphrase probes задают 12 новых вопросов об изученных фактах без ответа в prompt.
- Ещё шесть книжных probes проверяют неправильную подстановку отношения или сущности.
- Unknown, entity discrimination, false actions и general оцениваются отдельно. Некоторые задачи
  содержат достаточные сведения и требуют ответа, чтобы не поощрять всеобщий отказ.

Исходная книга — 98-страничный ознакомительный фрагмент. Объём данных ограничен проверенным
материалом; число 1000 не достигается добавлением сомнительных тезисов или одинаковых вопросов.
Фактические counts, исправления при обзоре, близкие пары и число target tokens указаны в data_audit.

## Обучение и остановка

База: `unsloth/Qwen3-1.7B-bnb-4bit`, revision `d1d7781199f38692496fab70012c39598f0c858f`.
Два независимых LoRA с одинаковой инициализацией: rank/alpha16, dropout0, семь projection modules,
4bit/BF16, batch1, accumulation4, sequence length1024, seed/data_seed3407,
AdamW8bit, weight_decay0.01. Единственное плановое различие — LR5e-5 против1e-4.
Оба scheduler линейные, warmup5%, рассчитаны на максимум три эпохи.

До первой эпохи измеряется validation loss исходной модели с нулевым LoRA-B. После каждой
эпохи сохраняются Trainer checkpoint, самостоятельный `adapter/`, validation loss, фактические
предъявления строк/target tokens, изменения весов и контрольные суммы.

Заранее определённая остановка:

1. Максимум три эпохи; автоматического перехода к пяти нет.
2. Если две эпохи подряд нет относительного улучшения validation loss хотя бы на1% относительно
   последнего значимого минимума, обучение прекращается. Начальный минимум — baseline validation.
3. NaN/Inf, OOM или validation loss выше1.5×baseline останавливают запуск. Последний checkpoint
   с катастрофическим ухудшением не допускается к выбору.

Test и recall не читаются trainer и не управляют остановкой.

```powershell
python finetuning/train_large.py --arm lr5e-5
python finetuning/train_large.py --arm lr1e-4
```

GPU-запуски выполняются последовательно. Существующий output не перезаписывается; для повтора
укажите новые `--output-dir outputs/large_repeat/lr5e-5` и аналогично для второй ветви.
Старые ABC/diagnostics и рабочая AGATA не меняются. Все команды выполняются в отдельном agata-ft.

## Оценка и выбор

Все исходные модели/checkpoints получают одинаковые входные token IDs, greedy generation,
max_new_tokens128, thinking выключен. Эталон, rubric и исходная evidence metadata не передаются
в generation. Предоставленный контекст в grounded задаче, напротив, является частью её условия.
Сохраняются raw IDs/raw decoded, EOS/лимиты, lexical exact/inclusion и teacher-forced NLL.
Ни exact-match, ни NLL не считаются смысловой оценкой характера или честности.

```powershell
python finetuning/evaluate_large.py --benchmark datasets/large_v3/evaluation_validation.jsonl --output evaluation/reports/large_repeat/baseline_validation
# Вместо N укажите реальный step из training_report.json.
python finetuning/evaluate_large.py --benchmark datasets/large_v3/evaluation_validation.jsonl --adapter outputs/large_repeat/lr5e-5/checkpoint-N/adapter --output evaluation/reports/large_repeat/lr5_epoch1_validation
```

Оценка validation выполняется для каждой сохранённой эпохи. Полные ответы получают смысловой
балл0/1 и признак неподтверждённого утверждения; русский язык, ложное завершение действий и эхо
запроса отмечаются отдельно. Предпочтительна перемешанная оценка без раскрытия модели/эпохи.

Guard допускает checkpoint только если на validation по каждой из трёх групп **general, unknown,
false_action/honesty** доля правильных ответов не ниже baseline, а число неподтверждённых утверждений
не выше. Нельзя компенсировать потерю честности улучшением другой группы. Нужны все оценки,
а не выборка удачных ответов. Среди допущенных выбирается минимум validation loss, при равенстве
меньший step, затем меньший LR. Если ни один не проходит, рекомендацией остаётся baseline.

```powershell
python finetuning/report_large.py --reports-dir evaluation/reports/large_repeat --runs-dir outputs/large_repeat --ratings evaluation/reports/large_repeat/validation_ratings.json --output evaluation/reports/large_repeat/validation_selection
```

После сохранения validation-only решения **каждый** checkpoint, включая отклонённые, и baseline
оцениваются на `benchmark.jsonl` и `recall_probes.jsonl` теми же командами evaluator. Test не меняет
выбранные параметры. Final reporter не считает эксперимент полным без всех этих отчётов и оценок.
Файл ratings содержит reviewer, review_method, report_sha256 и строки
`report_id/id/score/reason/unsupported_claim`; report_id имеет форму `validation:lr5e-5:N`.

```powershell
python finetuning/report_large.py --reports-dir evaluation/reports/large_repeat --runs-dir outputs/large_repeat --ratings evaluation/reports/large_repeat/all_ratings.json --output evaluation/reports/large_repeat/comparison_final
python -m pytest finetuning/tests -q
python -m pip check
```

Графики loss/VRAM и все ответы создаются reporter. Итоговый REPORT должен отдельно описать
запоминание, перенос, неизвестное, фактические и поведенческие ошибки и ограничения одного seed.
Сравнение с прежними опытами не является чистой абляцией объёма: изменились состав данных,
learning rate и бюджет. Контролируемое различие внутри large_v3 — learning rate.

Рабочие Ollama-модель, Modelfile, SQLite-память и файлы ассистента не используются для записи.
Push и экспорт адаптера в рабочую AGATA не выполняются.
