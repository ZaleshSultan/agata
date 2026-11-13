# Сравнение исходной модели и силы адаптера

`evaluate_adapter_comparison.py` выполняет свежую генерацию для baseline, исходного
`large_v3/lr1e-4/checkpoint-188/adapter` и предоставленного `manual_selected_half/adapter`.
Результаты текущего сравнения остаются локальными:
[half_compare_v1/REPORT.md](evaluation/reports/half_compare_v1/REPORT.md).

Это оценка фиксированного коэффициента 0.5, без обучения и без поиска коэффициента по test.
Уменьшение LoRA B вдвое при неизменных A и alpha/r уменьшает линейную добавку каждого слоя
вдвое. Оно не означает половину знаний или половину изменения logits всей сети.

Протокол хранит pinned base revision, SHA кода/двух benchmark/адаптеров, seed3407,
greedy decoding, thinking=false, max_new_tokens128 и прежние входные token IDs.
Тест содержит76 сценариев и30 отдельных книжных probes. Он уже раскрыт предыдущим
экспериментом, поэтому сравнение не является новой независимой проверкой обобщения.

Перед генерацией проверяются:

- хеши обязательных исходников, конфигураций и весов;
- все входные token IDs и отсутствие обрезания;
- фактически загруженные LoRA tensors против safetensors;
- активные адаптеры и все LoRA layers;
- изменение logits при отключении и точное восстановление после включения;
- отпечаток фактической базы, включая quantization state, без LoRA;
- полная эффективная generation config для последующего сравнения трёх запусков.

Модифицированные веса не объявляются новым trained checkpoint. Их происхождение проверяется
отдельным сравнением с оригиналом и прежним training manifest; старый `large_checkpoint.json`
не подделывается и не переписывается.

```powershell
conda activate agata-ft
# Используются уже подготовленные локальные protocol/data/weights.
python finetuning/evaluate_adapter_comparison.py --protocol evaluation/reports/half_compare_v1/protocol.json --variant baseline
python finetuning/evaluate_adapter_comparison.py --protocol evaluation/reports/half_compare_v1/protocol.json --variant original
python finetuning/evaluate_adapter_comparison.py --protocol evaluation/reports/half_compare_v1/protocol.json --variant half
python -m pytest finetuning/tests/test_adapter_comparison.py -q
```

Команды отказываются перезаписывать существующие результаты. Для повторения нужны новый
каталог отчётов и новый protocol с проверенными текущими SHA. GPU-запуски последовательные.
`--proof-only` сохраняет только отдельную проверку загрузки без scored generation.

Смысловые оценки переносятся из запечатанного large_v3 только при точном совпадении задания,
полной metadata, input/rendered prompt, raw/decoded/generated tokens и finish flags. Нормализация
для такого переноса запрещена. Новые ответы проверяются по тем же рубрикам без меток моделей,
в смеси с контрольными ответами. Старые оценки не изменяются, reuse и новый review считаются
отдельно. Прежние ответы могут быть узнаваемыми, поэтому идеальная слепота не заявляется.

Итог содержит все318 ответов, отдельные категории, три попарных сравнения с ID приобретений
и потерь, галлюцинации/ложные действия и совместный успех двух формулировок12 книжных фактов.
Lexical match и teacher-forced NLL не заменяют семантическую правильность. Числа описывают
этот небольшой набор; статистическая значимость или пригодность к внедрению автоматически
из них не следуют.

Операции ограничены лабораторией. Рабочие AGATA, Ollama, Modelfile и SQLite не изменяются.
Данные, цитаты, ответы, журналы и веса остаются в ignored-каталогах; push не выполняется.
