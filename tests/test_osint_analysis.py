from io import BytesIO
from pathlib import Path

from PIL import Image

from osint.context import InvestigationContext
from osint.correlator import Correlator
from osint.image_intel.metadata import extract_metadata
from osint.limits import Budget, BudgetExceeded, InvestigationLimits
from osint.models import Entity, EntityType
from osint.scoring import confidence


def test_budget_terminates_requests() -> None:
    budget = Budget(InvestigationLimits(max_requests=1))
    budget.consume_request()
    try: budget.consume_request()
    except BudgetExceeded: pass
    else: raise AssertionError("request budget must stop collection")


def test_correlation_never_uses_username_alone() -> None:
    context = InvestigationContext("neo", "username", "standard", InvestigationLimits())
    first = Entity(EntityType.SOCIAL, "https://a.test/neo", attributes={"username": "neo"})
    second = Entity(EntityType.SOCIAL, "https://b.test/neo", attributes={"username": "neo"})
    context.add_entity(first); context.add_entity(second)
    assert Correlator().correlate(context) == []
    assert confidence({"username"}) < 0.35


def test_image_metadata_extracts_without_network() -> None:
    raw = BytesIO(); Image.new("RGB", (7, 5), "green").save(raw, "PNG")
    metadata, evidence, observations = extract_metadata(raw.getvalue(), "image")
    assert metadata["width"] == 7 and evidence and observations
