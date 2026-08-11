"""CLI entry point for Memory v2 consolidation.

Legacy diary data is migrated automatically by MemoryRepository.  Consolidated
episodes remain ACTIVE; `consolidated_at` means processed, never hidden.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import MemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def consolidate() -> None:
    processed = MemoryManager().consolidate()
    logging.getLogger(__name__).info("Consolidation processed %d episodic memories.", len(processed))


if __name__ == "__main__":
    consolidate()
