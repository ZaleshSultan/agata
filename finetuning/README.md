# AGATA Fine-Tuning Lab

Изолированная лаборатория QLoRA для RTX 4050 Laptop 6 GB, Windows 11 и Python 3.12.
Рабочие `agent.py`, `prompts.py`, `config.py`, `Modelfile`, инструменты и SQLite-память AGATA не используются для побочных действий и не изменяются.
Предыдущий эксперимент `agata_finetune_knowledge_v2/` сохранён. Новая ветка: `codex/agata-finetuning-lab-20261110`.

Фактические результаты установки, CUDA, генерации и обучения приведены в `STATUS.md` и локальных `logs/`, `outputs/`, `evaluation/reports/`.
Наличие скрипта само по себе не означает успешного обучения или экспорта.

## Окружение

Окружение называется `agata-ft`; оно расположено в `miniconda3/envs/agata-ft`, вне репозитория и OneDrive.
Не устанавливайте эти зависимости в Python голосового ассистента: его `requirements.txt` содержит другой диапазон Transformers.

```powershell
# Из корня репозитория; установка только в отдельное окружение.
.\finetuning\setup_windows.ps1
conda activate agata-ft
python finetuning/diagnose.py --cuda
python finetuning/download_model.py
```

Вместо активации можно использовать `conda run --no-capture-output -n agata-ft python ...`
либо `.\finetuning\run.ps1 diagnose.py --cuda`.
`setup_windows.ps1` не изменяет системные драйверы, политику PowerShell или настройки WSL.

Выбор версий основан на [официальной инструкции Unsloth для Windows](https://unsloth.ai/docs/get-started/install/windows-installation),
метаданных wheel [Unsloth 2026.11.3](https://pypi.org/project/unsloth/2026.11.3/)
и [таблице PyTorch/Triton](https://github.com/triton-lang/triton-windows).
PyTorch 2.14 требует Triton 3.8; ограничения Unsloth исключают PyTorch 2.15 и ограничивают Transformers/TRL.
Конкретные прямые зависимости закреплены в `requirements.txt`, полный проверенный набор — в `requirements.lock.txt`.
`environment-win64.lock.txt` фиксирует Python 3.12.13 и системные библиотеки внутри Conda; установщик использует оба lock-файла.
Установка PyTorch использует официальный индекс `https://download.pytorch.org/whl/cu130`.
Надпись CUDA 13.3 в `nvidia-smi` означает возможности драйвера; CUDA runtime установленного PyTorch — отдельная версия 13.0.
Полный CUDA Toolkit и административная установка не нужны для используемых готовых wheel и Triton.

На этой машине уже обнаружены Conda, WSL2 Ubuntu, Visual Studio Build Tools и Ollama.
Предпочтительный путь — нативная Windows Conda. WSL2 остаётся альтернативой только при воспроизводимой несовместимости;
для Linux нужен отдельный lock с `triton`, а Windows lock туда переносить нельзя.

## Модель и память

Используется только [unsloth/Qwen3-1.7B-bnb-4bit](https://huggingface.co/unsloth/Qwen3-1.7B-bnb-4bit),
revision `d1d7781199f38692496fab70012c39598f0c858f`. Веса остаются в пользовательском кэше Hugging Face.
Русский токенизатор проверяется обратным преобразованием строки, а работоспособность — реальной генерацией на CUDA.

Параметры `config.yaml`: 4-bit QLoRA, rank/alpha 16/16, dropout 0, batch 1, accumulation 4,
learning rate 2e-4, 2 эпохи, длина 1024, seed 3407, gradient checkpointing, `adamw_8bit`.
BF16/FP16 выбирается по GPU. Упаковка нескольких диалогов отключена.
Только параметры LoRA обучаемы. Полные диалоги длиннее лимита отклоняются с записью причин, а не обрезаются посреди tool call.

Не запускайте параллельно несколько GPU-моделей на 6 GB. При CUDA OOM повторите smoke в новой папке с `--max-seq-length 512`.
Сокращение длины может исключить длинные диалоги; смотрите отчёт. 256 может быть слишком мало для системных правил.
Недостаток RAM диагностируется отдельно от VRAM; скрипты не закрывают пользовательские программы и не меняют файл подкачки.

## Данные и книга

`seed_data.py` содержит специально написанные разнообразные диалоги; реальные личные разговоры и память пользователя не импортируются.
Системные правила кратко адаптированы из `prompts.py`, его SHA256 записывается в manifest.
Схемы инструментов извлекаются из исходников через AST без импорта и выполнения AGATA.
Подробности состава, разбиения и ограничений: [DATASET.md](DATASET.md).

```powershell
# Первый запуск без книги. Уже существующие данные не перезаписываются.
python finetuning/prepare_dataset.py
```

В текущей локальной подготовке также обработан ранее сохранённый ознакомительный PDF Поливалиной 2013 года.
Он не является полной книгой или изданием 1980–1990-х. Отобранные ответы передают позицию автора с указанием источника.
224 автоматически извлечённых черновика нельзя считать проверенным датасетом.
Книга, производные данные, медицинские/правовые спорные карточки и отчёты находятся в игнорируемых Git каталогах.
Новая книга помещается в `finetuning/books/`; полный конвейер и точные команды находятся в [BOOKS.md](BOOKS.md).

Локальный объединённый набор: 40 personality, 24 general_chat, 23 tool_calling, 6 вымышленных обучающих фактов,
2 полностью отложенных вымышленных факта, 10 проверенных книжных примеров. Train/validation/test: 85/8/12.
Smoke: 20 примеров исключительно из train. Все примеры одной книжной главы остаются в одной части.
Файлы наборов `personality.jsonl`, `general_chat.jsonl`, `tool_calling.jsonl`, `book_knowledge.jsonl`, `train.jsonl`,
`validation.jsonl`, `test.jsonl` генерируются локально и не коммитятся.
Не используйте `--overwrite` для старого эксперимента; для новой версии можно передать другую папку и benchmark.

## Проверка до и после обучения

```powershell
# Все этапы: диагностика, базовая генерация/benchmark, 5 шагов обучения,
# загрузка сохранённого адаптера, тот же benchmark и сравнение.
.\finetuning\smoke_test.ps1

# Или запуск по отдельности в agata-ft:
python finetuning/inference.py --output outputs/base-generation.json
python finetuning/evaluate.py --output evaluation/reports/before --max-new-tokens 96
python finetuning/train.py --smoke --max-steps 5 --output-dir outputs/smoke
python finetuning/inference.py --adapter outputs/smoke/adapter --output outputs/reload.json
python finetuning/evaluate.py --adapter outputs/smoke/adapter --output evaluation/reports/after --max-new-tokens 96
python finetuning/compare_models.py evaluation/reports/before.json evaluation/reports/after.json
```

После неудачного запуска используйте новое имя папки; старые журналы полезны для воспроизведения ошибки.
`train.py` сохраняет конечные loss, число изменившихся LoRA-тензоров, точную конфигурацию, хеш данных,
checkpoint, адаптер и CSV измерений `nvidia-smi`. `inference.py` повторно загружает адаптер в отдельном процессе
и фиксирует успешную проверку в его training report. Одна успешная команда save не считается проверкой загрузки.

Обучение использует собственную маску целевых токенов на настоящем Qwen3 chat template:
assistant payload, tool call и EOS участвуют в loss; system/user/tool результаты и padding — нет.
Такой подход не зависит от наличия Jinja generation blocks в конкретной версии шаблона.

Benchmark A–F отделяет личность, воспроизведение обученных фактов, перефразировки,
полностью исключённые главы, неизвестные сведения и русский язык. В текущем объединённом наборе 34 вопроса.
Только B намеренно повторяет обучающие вопросы; это измерение запоминания.
Синтетические факты явно вымышлены; книжные ответы оцениваются отдельно.
Лексические совпадения и фразы неопределённости — индикаторы для ручной проверки, а не автоматическое доказательство истинности.
Для неизвестных сведений отдельно отмечаются кандидаты неподтверждённых ответов.
Пять шагов и один воспроизведённый факт не доказывают надёжного приобретения знаний.

## Полное обучение и checkpoint

Полное обучение разрешается скриптом только после успешного smoke той же конфигурации модели/LoRA и повторной загрузки адаптера.
Сначала расширьте и проверьте маленький начальный корпус: он не достаточен для готовой замены AGATA.

```powershell
python finetuning/train.py --smoke-report outputs/smoke/training_report.json --output-dir outputs/full-run-01

# Возобновление из реально созданного checkpoint, с тем же output-dir и конфигурацией.
python finetuning/train.py --smoke-report outputs/smoke/training_report.json --output-dir outputs/full-run-01 --resume outputs/full-run-01/checkpoint-25

python finetuning/evaluate.py --adapter outputs/full-run-01/adapter --output evaluation/reports/full-after --max-new-tokens 96
python finetuning/compare_models.py evaluation/reports/before.json evaluation/reports/full-after.json --output evaluation/reports/full-comparison
```

Номер checkpoint зависит от фактического запуска. Сохранение optimizer/trainer state позволяет продолжить обучение,
а `adapter/` предназначен для инференса. Benchmark, параметры генерации и базовая ревизия должны совпадать для сравнения.
На этом ноутбуке также проверено продолжение smoke с шага 4 до 5 в отдельной папке `outputs/smoke-resume`.
Loss совпал, но веса не побитово идентичны; воспроизводимость окружения не означает побитовой детерминированности CUDA.

Для независимой проверки изменения сохранённых LoRA-весов без загрузки GPU-модели:

```powershell
python finetuning/verify_adapter_update.py outputs/smoke/checkpoint-4 outputs/smoke/checkpoint-5 --output evaluation/reports/new-update-proof.json
```

## Экспорт и сохранность проекта

Экспорт FP16 → GGUF → Q4_K_M — отдельная операция с дополнительными требованиями к RAM/диску.
См. [EXPORT.md](EXPORT.md) и `python finetuning/export_ollama.py --help`.
Создавать можно только новое имя `agata-finetuned-test` или его суффикс; существующая модель не перезаписывается.
Рабочий `Modelfile` не используется для записи. После регистрации нужны реальные проверки русского ответа,
tool calling и реакции на ошибку инструмента; наличие GGUF само по себе не подтверждает совместимость.

Все исходники книг, веса, checkpoints, кэши и результаты исключены из Git.
Ничего автоматически не отправляется в Hugging Face, GitHub, внешние модели или трекеры экспериментов.
Загрузка публичной базовой модели — отдельная входящая операция. `.gitignore` не снимает уже отслеживаемые файлы;
старые изменения браузерного профиля оставлены в рабочей копии и не входят в изменения лаборатории.

```powershell
python -m pytest finetuning -q
python -m pip check
```

## Контролируемое сравнение A/B/C

Второй этап сравнивает обучение характеру, проверенному содержанию книги и их объединению.
Общий benchmark, неизменяемый протокол, бюджет по 32 шага, слепая смысловая оценка и команды
запуска описаны в [ABC_EXPERIMENT.md](ABC_EXPERIMENT.md).
Локальные результаты находятся в `evaluation/reports/abc_v1/`, отдельные адаптеры и графики —
в `outputs/abc_v1/A`, `outputs/abc_v1/B`, `outputs/abc_v1/C`. Эти приватные артефакты исключены из Git.

## Диагностика неудачного обучения

Аудит токенизации/LoRA, one-fact и mini-dataset overfit, отдельная проверка характера и команды
повтора описаны в [DIAGNOSTICS.md](DIAGNOSTICS.md). Реальные результаты сохранены локально в
[diagnostics_v1/REPORT.md](evaluation/reports/diagnostics_v1/REPORT.md).

## Расширенный набор large_v3

Протокол большего проверенного датасета, трёх эпох, сравнения двух learning rate и выбора
checkpoint только по validation описан в [LARGE_EXPERIMENT.md](LARGE_EXPERIMENT.md).
Результаты сохраняются локально в [large_v3/REPORT.md](evaluation/reports/large_v3/REPORT.md).

## Сравнение ослабленного адаптера

Свежая проверка baseline, исходного checkpoint и `manual_selected_half`, доказательство загрузки
и правила переноса оценок описаны в [ADAPTER_COMPARISON.md](ADAPTER_COMPARISON.md).
Реальные ответы и попарные результаты: [half_compare_v1/REPORT.md](evaluation/reports/half_compare_v1/REPORT.md).
