"""One folder per investigation; raw data stays outside conversational memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
from osint.context import InvestigationContext
from osint.graph import export_graph
from osint.report import render_report
from osint.timeline import build_timeline


class InvestigationStorage:
    def __init__(self, root: Path = config.OSINT_STORAGE_DIR) -> None: self.root = root

    def save(self, context: InvestigationContext) -> Path:
        folder = self.root / context.investigation_id; folder.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"investigation_id": context.investigation_id, "target": context.target, "target_type": context.target_type, "depth": context.depth, "metrics": {"runtime_seconds": context.budget.elapsed_seconds, "requests": context.budget.requests, "cache_hits": context.cache.hits}, "errors": context.errors}
        self._write(folder / "investigation.json", data)
        self._write(folder / "entities.json", [item.to_dict() for item in context.entities.values()])
        self._write(folder / "relations.json", [item.to_dict() for item in context.relations.values()])
        self._write(folder / "observations.json", [item.to_dict() for item in context.observations.values()])
        self._write(folder / "evidence.json", [item.to_dict() for item in context.evidence.values()])
        self._write(folder / "timeline.json", build_timeline(context)); self._write(folder / "graph.json", export_graph(context))
        (folder / "report.md").write_text(render_report(context), encoding="utf-8")
        return folder

    @staticmethod
    def _write(path: Path, value: Any) -> None: path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
