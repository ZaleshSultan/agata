from osint.deduplicator import Deduplicator
from osint.models import Entity, EntityType, Evidence, Relation
from osint.normalizer import infer_target_type, normalize


def test_normalization_and_target_type() -> None:
    assert normalize("@Neo_Sultan", EntityType.USERNAME) == "neo_sultan"
    assert normalize("HTTPS://Example.COM/A/?q=1", EntityType.WEBSITE) == "https://example.com/A"
    assert infer_target_type("user@example.com") is EntityType.EMAIL


def test_deduplication_merges_evidence_not_identity() -> None:
    first = Entity(EntityType.USERNAME, "Neo")
    second = Entity(EntityType.USERNAME, "@neo", evidence_ids=["ev_1"])
    entities = {first.id: first}
    merged, created = Deduplicator().merge(entities, second)
    assert not created and merged.id == first.id and merged.evidence_ids == ["ev_1"]


def test_relation_and_evidence_are_serializable() -> None:
    evidence = Evidence("Observed public profile", source_url="https://example.test")
    relation = Relation("a", "b", "associated_with", 0.5, [evidence.id])
    assert relation.to_dict()["evidence_ids"] == [evidence.id]
