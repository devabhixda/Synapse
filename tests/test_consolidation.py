import uuid
from datetime import UTC, datetime

from synapse.consolidation.extractor import derive_semantic_plan
from synapse.db.models import Episode, EpisodeKind, SourceType


def episode(source_id: str, content: str, participants: list[str]) -> Episode:
    return Episode(
        id=uuid.uuid4(),
        kind=EpisodeKind.SOURCE,
        source_type=SourceType.SLACK,
        source_id=source_id,
        conversation_id="test",
        content=content,
        participants=participants,
        details={},
        occurred_at=datetime.now(UTC),
    )


def test_semantic_plan_resolves_titan_alias_and_evidence() -> None:
    plan = derive_semantic_plan(
        [
            episode(
                "decision",
                "Maya: We decided the auth database will be called Titan. "
                "Atlas API should use it for all new sessions.",
                ["Maya", "Jon"],
            ),
            episode(
                "blocker",
                "Maya: Atlas API staging is blocked because the session backfill is failing.",
                ["Maya", "Priya"],
            ),
        ]
    )

    titan = next(entity for entity in plan.entities if entity.canonical_name == "Titan")
    uses = next(
        relationship for relationship in plan.relationships if relationship.predicate == "uses"
    )

    assert titan.aliases == ["auth database"]
    assert uses.evidence_source_ids == ["decision"]
    assert any(entity.kind == "Blocker" for entity in plan.entities)