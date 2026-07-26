import uuid
from datetime import UTC, datetime

from synapse.consolidation.extractor import derive_semantic_plan
from synapse.consolidation.llm_extractor import apply_alias_map, normalize_raw_plan
from synapse.consolidation.models import RawConsolidationPlan
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


def test_llm_plan_rejects_unknown_evidence_and_normalizes_predicates() -> None:
    episodes = [episode("decision", "Atlas API uses Titan.", ["Maya"])]
    raw_plan = RawConsolidationPlan.model_validate(
        {
            "entities": [
                {"kind": "System", "canonical_name": "Atlas API"},
                {"kind": "System", "canonical_name": "Titan"},
            ],
            "relationships": [
                {
                    "subject_kind": "System",
                    "subject_name": "Atlas API",
                    "predicate": "USES DATABASE",
                    "object_kind": "System",
                    "object_name": "Titan",
                    "evidence_source_ids": ["decision", "invented"],
                }
            ],
        }
    )

    plan = normalize_raw_plan(raw_plan, episodes)

    assert plan.relationships[0].predicate == "uses_database"
    assert plan.relationships[0].evidence_source_ids == ["decision"]


def test_adjudication_alias_map_only_targets_canonical_entities() -> None:
    plan = derive_semantic_plan(
        [episode("decision", "The auth database will be called Titan.", ["Maya"])]
    )

    apply_alias_map(plan, {"Titan": "authentication store", "invented": "unknown"})

    titan = next(entity for entity in plan.entities if entity.canonical_name == "Titan")
    assert "authentication store" in titan.aliases
    assert all(entity.canonical_name != "unknown" for entity in plan.entities)