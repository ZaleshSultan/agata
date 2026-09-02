import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from osint.collectors.base import CollectorSpec
from osint.engine import OSINTEngine
from osint.models import CollectorOutcome, Entity, EntityType, Evidence
from osint.storage import InvestigationStorage


class FakeCollector:
    spec = CollectorSpec("username_discovery", (EntityType.USERNAME,))
    async def collect(self, context, entity_id):
        child = Entity(EntityType.SOCIAL, "https://example.test/neo", label="Example", attributes={"username": "neo"})
        proof = Evidence("Public test profile", "https://example.test/neo", source="fake")
        child.evidence_ids.append(proof.id)
        return CollectorOutcome("username_discovery", True, [child], [proof], requests=1)


def test_engine_lifecycle_saves_investigation(tmp_path: Path) -> None:
    engine = OSINTEngine({"username_discovery": FakeCollector()}, InvestigationStorage(tmp_path))
    result = asyncio.run(engine.investigate("Neo", target_type="username", depth="quick"))
    assert result["success"] and result["metrics"]["entities_discovered"] == 2
    assert Path(result["report_path"]).exists()


def test_engine_rejects_bad_depth(tmp_path: Path) -> None:
    result = asyncio.run(OSINTEngine(storage=InvestigationStorage(tmp_path)).investigate("neo", depth="forever"))
    assert result["success"] is False


def test_provider_registry_is_safe_metadata(tmp_path: Path) -> None:
    engine = OSINTEngine({"username_discovery": FakeCollector()}, InvestigationStorage(tmp_path))
    assert engine.provider_registry()[0]["name"] == "username_discovery"


def test_agata_bridge_exposes_one_structured_result() -> None:
    from tools.osint import investigate_person
    with patch("tools.osint.investigate", new=AsyncMock(return_value={"success": True, "investigation_id": "test"})):
        assert json.loads(investigate_person("neo"))["investigation_id"] == "test"
