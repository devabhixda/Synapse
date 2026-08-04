"""Isolation between accounts, checked against a real database.

These run the real services against a real database, because the property under
test is the absence of something — a missing ``where`` clause — and a mock will
happily return whatever it was told to. Two accounts are always populated, so a
query that forgot its filter returns the other account's rows and the
assertion fails rather than passing vacuously against an empty table.

Only the model calls and the vector store are faked. Their behaviour is not
what is being tested, and the fake vector store is deliberately made to lie in
one test, to prove the database is what enforces ownership.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from synapse.consolidation.models import ConsolidationPlan, EntitySpec, ExtractionResult
from synapse.consolidation.service import ConsolidationService
from synapse.db.models import (
    Episode,
    RecallEvent,
    SemanticEntity,
    SkillSpec,
    SourceType,
    WorkflowPattern,
)
from synapse.domain.models import FastObservation, SourceEvent
from synapse.episodic.service import SystemOneService
from synapse.procedural.service import (
    ProceduralMemoryService,
    get_skill_by_name,
    list_patterns,
    list_skills,
)
from synapse.recall.models import EpisodeSelection
from synapse.recall.service import RecallService
from synapse.semantic.repository import load_entities, load_relationships

ACME = "acme"
GLOBEX = "globex"
NOW = datetime.now(UTC)


class StubInterpreter:
    """Returns a fixed observation so ingest never calls a model."""

    async def observe(self, event: SourceEvent) -> FastObservation:
        return FastObservation(
            summary=f"summary of {event.content}",
            intent="share_update",
            salience=0.5,
            provider="stub",
        )


class StubMemoryStore:
    """A vector store that records what it was asked, and can be told to lie.

    ``hits`` is what ``search`` returns regardless of the account, which is how
    the recall test simulates an index that has gone stale or was written
    before tenancy existed.
    """

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.added: list[tuple[str, str]] = []
        self.searched: list[tuple[str, str]] = []

    async def add_episode(
        self, *, episode_id: str, group_id: str, summary: str, event: SourceEvent
    ) -> str:
        self.added.append((group_id, episode_id))
        return f"mem0-{episode_id}"

    async def add_retrieval_episode(
        self, *, episode_id: str, group_id: str, answer: str, query: str, occurred_at: str
    ) -> str:
        self.added.append((group_id, episode_id))
        return f"mem0-{episode_id}"

    async def search(self, query: str, group_id: str) -> list[dict[str, Any]]:
        self.searched.append((group_id, query))
        return self.hits


class StubJudge:
    """Answers "yes, the first candidate" so recall reaches the episodic path."""

    async def select(
        self, query: str, candidates: list[dict[str, object]]
    ) -> EpisodeSelection:
        if not candidates:
            return EpisodeSelection(answerable=False, confidence=0.0)
        return EpisodeSelection(
            answerable=True,
            selected_episode_id=str(candidates[0]["episode_id"]),
            confidence=0.9,
        )


def event(source_id: str, content: str, *, occurred_at: datetime | None = None) -> SourceEvent:
    return SourceEvent(
        source_id=source_id,
        source_type=SourceType.SLACK,
        conversation_id="thread-1",
        content=content,
        participants=["Maya"],
        occurred_at=occurred_at or NOW,
    )


async def ingest(group_id: str, session: Any, *events: SourceEvent) -> list[Episode]:
    service = SystemOneService(StubInterpreter(), StubMemoryStore())
    episodes = []
    for item in events:
        episode, _ = await service.process(item, group_id, session)
        episodes.append(episode)
    return episodes


async def test_the_same_slack_thread_in_two_accounts_is_two_episodes(session) -> None:
    """Dedupe keys on source_id, which is only unique within one export."""
    await ingest(ACME, session, event("slack-1", "Acme is on 2xH100"))
    await ingest(GLOBEX, session, event("slack-1", "Globex is on 4xH200"))

    total = await session.scalar(select(func.count()).select_from(Episode))
    assert total == 2, "one account's ingest overwrote the other's"

    acme_content = await session.scalar(
        select(Episode.content).where(Episode.group_id == ACME)
    )
    assert acme_content == "Acme is on 2xH100"


async def test_re_ingesting_within_one_account_still_dedupes(session) -> None:
    await ingest(ACME, session, event("slack-1", "first"))
    await ingest(ACME, session, event("slack-1", "first"))

    total = await session.scalar(select(func.count()).select_from(Episode))
    assert total == 1


async def test_two_accounts_can_hold_the_same_entity_name(session) -> None:
    """uq_semantic_entity_identity is per account, so "the gateway" is not shared."""
    for group in (ACME, GLOBEX):
        session.add(
            SemanticEntity(
                group_id=group,
                kind="System",
                canonical_name="the gateway",
                normalized_name="the gateway",
            )
        )
    await session.commit()

    acme_entities = await load_entities(ACME, session)
    globex_entities = await load_entities(GLOBEX, session)

    assert len(acme_entities) == 1
    assert len(globex_entities) == 1
    assert acme_entities[0].id != globex_entities[0].id


async def test_graph_reads_return_only_the_requested_account(session) -> None:
    session.add(
        SemanticEntity(
            group_id=GLOBEX, kind="Company", canonical_name="Globex", normalized_name="globex"
        )
    )
    await session.commit()

    assert await load_entities(ACME, session) == []
    assert await load_relationships(ACME, session) == []
    assert len(await load_entities(GLOBEX, session)) == 1


async def test_recall_ignores_a_vector_hit_from_another_account(session) -> None:
    """The database, not Mem0, decides who owns an episode.

    The stub returns Globex's episode no matter who is asking, standing in for
    a stale vector or metadata written before tenancy. Recall must not serve it
    to Acme.
    """
    (globex_episode,) = await ingest(GLOBEX, session, event("slack-1", "Globex secret"))

    leaking_store = StubMemoryStore(
        hits=[{"metadata": {"episode_id": str(globex_episode.id)}, "score": 0.99}]
    )
    result = await RecallService(memory_store=leaking_store, judge=StubJudge()).recall(
        "what is the secret?", ACME, session
    )

    assert result.answer_origin == "miss"
    assert result.answer is None
    assert "Globex secret" not in (result.answer or "")


async def test_recall_still_answers_from_its_own_account(session) -> None:
    """The guard above must not be so strict that legitimate recall breaks."""
    (episode,) = await ingest(ACME, session, event("slack-1", "Acme is on 2xH100"))

    store = StubMemoryStore(
        hits=[{"metadata": {"episode_id": str(episode.id)}, "score": 0.99}]
    )
    result = await RecallService(memory_store=store, judge=StubJudge()).recall(
        "what hardware?", ACME, session
    )

    assert result.answer_origin == "episodic"
    assert result.answer == "summary of Acme is on 2xH100"
    assert store.searched == [(ACME, "what hardware?")]


async def test_recall_events_are_recorded_against_the_asking_account(session) -> None:
    await RecallService(memory_store=StubMemoryStore(), judge=StubJudge()).recall(
        "anything?", ACME, session
    )

    groups = list(await session.scalars(select(RecallEvent.group_id)))
    assert groups == [ACME]


async def test_consolidation_only_reads_its_own_episodes(session) -> None:
    await ingest(ACME, session, event("slack-1", "Acme ships Prism"))
    await ingest(GLOBEX, session, event("slack-2", "Globex ships Nova"))

    class OneEntityExtractor:
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def extract(self, episodes: list[Episode]) -> ExtractionResult:
            self.seen = [episode.content for episode in episodes]
            return ExtractionResult(
                plan=ConsolidationPlan(
                    entities=[
                        EntitySpec(key="Company:acme", kind="Company", canonical_name="Acme")
                    ],
                    relationships=[],
                ),
                provider="stub",
            )

    extractor = OneEntityExtractor()
    run = await ConsolidationService(extractor=extractor, enricher=None).run(ACME, session)

    assert extractor.seen == ["Acme ships Prism"], "consolidation crossed accounts"
    assert run.group_id == ACME
    assert run.episode_count == 1
    entity_groups = list(await session.scalars(select(SemanticEntity.group_id)))
    assert entity_groups == [ACME]


async def test_procedural_detection_only_scans_its_own_recall_events(session) -> None:
    for index in range(3):
        session.add(
            RecallEvent(
                group_id=GLOBEX,
                query="what is the project status?",
                normalized_intent=None,
                answer_origin="episodic",
                created_at=NOW + timedelta(seconds=index),
            )
        )
    await session.commit()

    result = await ProceduralMemoryService().detect(ACME, session)

    assert result.recall_events_scanned == 0
    assert result.patterns_created == 0
    assert await list_patterns(ACME, session) == []


async def test_two_accounts_can_hold_skills_with_the_same_name(session) -> None:
    for group in (ACME, GLOBEX):
        pattern = WorkflowPattern(
            group_id=group,
            key="project_status_briefing",
            name="Status",
            description="d",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        session.add(pattern)
        await session.flush()
        session.add(
            SkillSpec(
                group_id=group,
                pattern_id=pattern.id,
                name="project_status_briefing",
                status="approved",
                purpose="p",
            )
        )
    await session.commit()

    acme_skills = await list_skills(ACME, session)
    assert len(acme_skills) == 1
    assert (await get_skill_by_name(ACME, session, "project_status_briefing")).id == acme_skills[
        0
    ].id


async def test_a_skill_cannot_be_approved_from_another_account(session) -> None:
    pattern = WorkflowPattern(
        group_id=GLOBEX,
        key="project_status_briefing",
        name="Status",
        description="d",
        first_seen_at=NOW,
        last_seen_at=NOW,
    )
    session.add(pattern)
    await session.flush()
    skill = SkillSpec(
        group_id=GLOBEX,
        pattern_id=pattern.id,
        name="project_status_briefing",
        status="candidate",
        purpose="p",
    )
    session.add(skill)
    await session.commit()

    # Knowing the id is not authority over it; it must read as absent.
    with pytest.raises(ValueError, match="was not found"):
        await ProceduralMemoryService().set_status(str(skill.id), "approved", ACME, session)

    await session.refresh(skill)
    assert skill.status == "candidate"
