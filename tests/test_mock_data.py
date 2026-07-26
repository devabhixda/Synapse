from synapse.consolidation.extractor import derive_semantic_plan
from synapse.db.models import Episode, EpisodeKind
from synapse.sources.mock import mock_events


def test_rich_mock_events_are_unique_chronological_and_multi_source() -> None:
    events = mock_events()

    assert len(events) == 18
    assert len({event.source_id for event in events}) == len(events)
    assert [event.occurred_at for event in events] == sorted(
        event.occurred_at for event in events
    )
    assert {event.source_type.value for event in events} == {"slack", "meeting"}
    assert len({event.conversation_id for event in events}) >= 8
    assert not any("crustdata" in event.content.casefold() for event in events)


def test_rich_mock_events_produce_sponsor_products_and_fit_decisions() -> None:
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
        "Channel3",
        "Channel3 API",
        "Allus AI",
        "Allus AI vision platform",
        "Random Labs",
        "Slate",
        "Hexclave",
        "Hexclave platform",
        "Synapse",
        "Channel3 developer API key unavailable",
        "Keep Synapse MCP read-only",
    } <= names
    assert {
        "offers",
        "evaluated_for",
        "optional_for",
        "deferred_for",
        "complements",
        "identity_candidate_for",
        "blocks",
        "governs",
    } <= predicates

    channel3_relationship = next(
        relationship
        for relationship in plan.relationships
        if relationship.predicate == "optional_for"
    )
    assert "slack-channel3-fit-001" in channel3_relationship.evidence_source_ids