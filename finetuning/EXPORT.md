# Экспорт в Ollama

Экспорт выполняется отдельно после проверки адаптера. Нужны свободные RAM и около
10 ГиБ диска для FP16 и GGUF; рабочая модель AGATA не переключается автоматически.
Команды ниже запускаются из корня репозитория в PowerShell после `conda activate agata-ft`.

1. Подготовить переносимый конвертер:

```powershell
python finetuning/setup_export.py --install-converter
```

Скрипт скачивает официальный CPU-архив llama.cpp `b12110`, который соответствует
стабильному `v0.9.0` от 4 ноября 2026, и проверяет SHA-256
`98901e327c252875accaf2d1bf364e10e868843b0b63468baad1aa945cdc5da2`.
Исходники фиксируются на commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`.
CPU выбран для квантования без дополнительного CUDA runtime. Все файлы находятся
в игнорируемой папке `finetuning/outputs/tooling/`; драйверы и PATH не меняются.

Конвертер получает отдельный `venv`: его официальные зависимости включают
`torch==2.11.0` и `transformers==4.57.6`, поэтому они не устанавливаются в Conda
окружение обучения. Сам venv находится в коротком ASCII-пути
`%USERPROFILE%/.cache/agata-ft/converter-env`, вне OneDrive. Проверки `pip check`, `convert_hf_to_gguf.py --help` и
`llama-quantize.exe --help` сохраняют сведения в `export_tooling.json`.
Проверенные версии закреплены в `requirements-export.lock.txt`; setup автоматически
использует этот lock в конверсионном venv. Фактический `pip freeze` дополнительно
сохраняется в `outputs/tooling/converter-requirements.lock.txt`. Без `--install-converter` скачиваются только portable binaries
и закреплённые исходники; это ещё не готовое Python-окружение конверсии.

2. Объединить базовые веса с проверенным адаптером:

```powershell
python finetuning/export_ollama.py merge --adapter outputs/smoke/adapter --output-dir outputs/export-merged
```

Замените путь smoke на адаптер полноценного эксперимента. Папка назначения должна
быть новой. Используется официальный Unsloth `save_pretrained_merged(...,
save_method="merged_16bit", maximum_memory_usage=0.5)`. Успешный merge сам по себе
не означает успешный экспорт или сохранение качества.

3. Создать F16 GGUF и Q4_K_M:

```powershell
$exportTools = Get-Content finetuning/outputs/tooling/export_tooling.json -Raw | ConvertFrom-Json
python finetuning/export_ollama.py convert --merged-dir outputs/export-merged --llama-cpp-dir $exportTools.source --quantizer $exportTools.quantizer --converter-python $exportTools.converter_python --output-dir outputs/export-gguf
```

Конверсия использует локальные исходники llama.cpp и параметр `--use-temp-file`;
большой GPU model load на этом этапе не нужен. FP16 промежуточный файл остаётся
для диагностики. Шаблон Qwen3 переносится из tokenizer через GGUF metadata.

4. Создать только тестовую модель и проверить реальные ответы:

```powershell
python finetuning/export_ollama.py register --gguf outputs/export-gguf/agata-q4_k_m.gguf --name agata-finetuned-test
python finetuning/export_ollama.py verify --name agata-finetuned-test
```

Регистрация откажется перезаписывать существующую модель. Для нового эксперимента
используйте новое имя `agata-finetuned-test-2` и новую `--output-dir`.
Исходный `Modelfile` и модель `agata` не затрагиваются. Проверка проверяет русский
ответ, нативный `tool_calls` и честное сообщение об ошибке инструмента;
предложенные инструменты не выполняются. Ответы и шаблон сохраняются в
`finetuning/evaluation/reports/ollama-test.json` и требуют просмотра человеком.
Если инструмент вернулся только текстовым XML либо шаблон не соответствует Qwen3,
экспорт нельзя считать совместимым с AGATA до исправления и повторной проверки.

Источники, проверенные 10 ноября 2026:

- [Официальный release v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0)
- [Артефакты b12110](https://github.com/ggml-org/llama.cpp/releases/tag/b12110)
- [Зависимости конвертера закреплённого исходника](https://github.com/ggml-org/llama.cpp/blob/cd959bae99ff4bed22bef2604f45241c09bbb8c8/requirements/requirements-convert_hf_to_gguf.txt)
- [Unsloth: merged weights и GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [Ollama: импорт GGUF](https://docs.ollama.com/import)
