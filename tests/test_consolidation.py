import uuid
from datetime import UTC, datetime

from synapse.consolidation.extractor import derive_semantic_plan
from synapse.consolidation.llm_extractor import apply_alias_map, normalize_raw_plan
from synapse.consolidation.models import RawConsolidationPlan
from synapse.db.models import Episode, EpisodeKind, SourceType


def episode(
    source_id: str,
    content: str,
    participants: list[str],
    details: dict[str, object] | None = None,
) -> Episode:
    return Episode(
        id=uuid.uuid4(),
        kind=EpisodeKind.SOURCE,
        source_type=SourceType.SLACK,
        source_id=source_id,
        conversation_id="test",
        content=content,
        participants=participants,
        details=details or {},
        occurred_at=datetime.now(UTC),
    )


def test_semantic_plan_merges_generic_assertions_and_evidence() -> None:
    plan = derive_semantic_plan(
        [
            episode(
                "decision",
                "Maya: Orbit Forge offers Prism for the Nova project.",
                ["Maya", "Jon"],
                {
                    "semantic_assertions": {
                        "entities": [
                            {
                                "kind": "Company",
                                "name": "Orbit Forge",
                                "aliases": ["Orbit"],
                            },
                            {"kind": "Product", "name": "Prism"},
                            {"kind": "Project", "name": "Nova"},
                        ],
                        "relationships": [
                            {
                                "subject_kind": "Company",
                                "subject_name": "Orbit Forge",
                                "predicate": "offers",
                                "object_kind": "Product",
                                "object_name": "Prism",
                            },
                            {
                                "subject_kind": "Product",
                                "subject_name": "Prism",
                                "predicate": "evaluated_for",
                                "object_kind": "Project",
                                "object_name": "Nova",
                            },
                        ],
                    }
                },
            ),
            episode(
                "follow-up",
                "Priya: Prism remains under evaluation for Nova.",
                ["Priya"],
                {
                    "semantic_assertions": {
                        "relationships": [
                            {
                                "subject_kind": "Product",
                                "subject_name": "Prism",
                                "predicate": "evaluated_for",
                                "object_kind": "Project",
                                "object_name": "Nova",
                                "confidence": 0.9,
                            }
                        ]
                    }
                },
            ),
        ]
    )

    company = next(
        entity for entity in plan.entities if entity.canonical_name == "Orbit Forge"
    )
    evaluated = next(
        relationship
        for relationship in plan.relationships
        if relationship.predicate == "evaluated_for"
    )

    assert company.aliases == ["Orbit"]
    assert evaluated.evidence_source_ids == ["decision", "follow-up"]
    assert {entity.canonical_name for entity in plan.entities} >= {
        "Orbit Forge",
        "Prism",
        "Nova",
        "Maya",
        "Jon",
        "Priya",
    }


def test_llm_plan_rejects_unknown_evidence_and_normalizes_predicates() -> None:
    episodes = [episode("decision", "Prism supports Nova.", ["Maya"])]
    raw_plan = RawConsolidationPlan.model_validate(
        {
            "entities": [
                {"kind": "Product", "canonical_name": "Prism"},
                {"kind": "Project", "canonical_name": "Nova"},
            ],
            "relationships": [
                {
                    "subject_kind": "Product",
                    "subject_name": "Prism",
                    "predicate": "SUPPORTS PROJECT",
                    "object_kind": "Project",
                    "object_name": "Nova",
                    "evidence_source_ids": ["decision", "invented"],
                }
            ],
        }
    )

    plan = normalize_raw_plan(raw_plan, episodes)

    assert plan.relationships[0].predicate == "supports_project"
    assert plan.relationships[0].evidence_source_ids == ["decision"]


def test_adjudication_alias_map_only_targets_canonical_entities() -> None:
    plan = derive_semantic_plan(
        [
            episode(
                "decision",
                "Orbit Forge offers Prism.",
                ["Maya"],
                {
                    "semantic_assertions": {
                        "entities": [{"kind": "Product", "name": "Prism"}]
                    }
                },
            )
        ]
    )

    apply_alias_map(plan, {"Prism": "Prism AI", "invented": "unknown"})

    product = next(entity for entity in plan.entities if entity.canonical_name == "Prism")
    assert "Prism AI" in product.aliases
    assert all(entity.canonical_name != "unknown" for entity in plan.entities)