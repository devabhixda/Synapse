from synapse.consolidation.extractor import derive_semantic_plan
from synapse.db.models import Episode, EpisodeKind
from synapse.sources.mock import mock_events


def test_rich_mock_events_are_unique_chronological_and_multi_source() -> None:
    events = mock_events()

    assert len(events) == 14
    assert len({event.source_id for event in events}) == len(events)
    assert [event.occurred_at for event in events] == sorted(
        event.occurred_at for event in events
    )
    assert {event.source_type.value for event in events} == {"slack", "meeting"}
    assert len({event.conversation_id for event in events}) >= 8


def test_rich_mock_events_produce_security_repository_and_rollout_context() -> None:
    episodes = [
        Episode(
            kind=EpisodeKind.SOURCE,
            source_type=event.source_type,
            source_id=event.source_id,
            conversation_id=event.conversation_id,
            content=event.content,
            participants=event.participants,
            details=event.metadata,
            occurred_at=event.occurred_at,
        )
        for event in mock_events()
    ]

    plan = derive_semantic_plan(episodes)
    names = {entity.canonical_name for entity in plan.entities}
    predicates = {relationship.predicate for relationship in plan.relationships}

    assert {
        "Titan",
        "Atlas API",
        "KMS auth-prod",
        "services/atlas-api",
        "Titan session dual-write",
        "Session backfill repair",
        "Titan production rollout",
        "Titan production monitoring",
        "Security",
        "Platform",
    } <= names
    assert {"encrypted_with", "implemented_in", "resolves", "part_of", "owns"} <= predicates

    crustdata_relationship = next(
        relationship
        for relationship in plan.relationships
        if relationship.predicate == "proposed_for"
    )
    assert crustdata_relationship.evidence_source_ids == [
        "meet-partnerships-001",
        "slack-partnership-002",
    ]