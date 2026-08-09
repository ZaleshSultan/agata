"""Regenerate Modelfile SYSTEM block from prompts.py."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from prompts import SYSTEM_PROMPT

MODELFILE = BASE_DIR / "Modelfile"

HEADER = f"""# Modelfile для модели "agata"
# Создание: ollama create agata -f Modelfile
# Использование: LLM_MODEL = "agata" в config.py
#
# SYSTEM-блок генерируется из prompts.py:
#   python scripts/sync_modelfile.py

FROM huihui_ai/qwen3-abliterated:30b-a3b

SYSTEM \"\"\"
"""

FOOTER = """
\"\"\"

PARAMETER temperature 0.75
PARAMETER num_ctx 8192
"""


def main() -> None:
    if config.ASSISTANT_NAME not in SYSTEM_PROMPT:
        raise SystemExit("SYSTEM_PROMPT does not contain ASSISTANT_NAME — check prompts.py")
    MODELFILE.write_text(HEADER + SYSTEM_PROMPT + FOOTER, encoding="utf-8")
    print(f"Updated {MODELFILE}")


if __name__ == "__main__":
    main()
