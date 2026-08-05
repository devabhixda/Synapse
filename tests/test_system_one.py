import sqlite3
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from synapse.config import Settings
from synapse.db.base import Base
from synapse.db.models import Episode, SourceType
from synapse.db.pragmas import configure_sqlite
from synapse.domain.models import FastObservation, SourceEvent
from synapse.episodic.service import SystemOneService
from synapse.inference.chat import (
    FastClassification,
    LLMInterpreter,
    assemble_observation,
    deterministic_observation,
    parse_json_object,
)


def test_deterministic_observation_marks_decision() -> None:
    event = SourceEvent(
        source_id="test-1",
        source_type=SourceType.SLACK,
        conversation_id="thread-1",
        content="We decided Orbit Forge will provide Prism.",
        participants=["Maya"],
        occurred_at=datetime.now(UTC),
        metadata={
            "semantic_assertions": {
                "entities": [
                    {"kind": "Company", "name": "Orbit Forge"},
                    {"kind": "Product", "name": "Prism"},
                ]
            }
        },
    )

    observation: FastObservation = deterministic_observation(event)

    assert observation.intent == "record_decision"
    assert observation.entity_mentions == ["Orbit Forge", "Prism"]
    assert observation.salience == 0.9


def test_a_compact_classification_assembles_a_safe_observation() -> None:
    event = SourceEvent(
        source_id="test-2",
        source_type=SourceType.SLACK,
        conversation_id="thread-2",
        content="Maya: We decided Orbit Forge will provide Prism.",
        participants=["Maya"],
        occurred_at=datetime.now(UTC),
        metadata={
            "semantic_assertions": {
                "entities": [
                    {"kind": "Company", "name": "Orbit Forge"},
                    {"kind": "Product", "name": "Prism"},
                ]
            }
        },
    )
    classification = FastClassification.model_validate(
        parse_json_object('{"intent":"record_decision","salience":0.9}')
    )

    observation = assemble_observation(
        event,
        classification,
        provider="google/gemma-4-E2B-it",
        confidence=0.88,
        latency_ms=120,
    )

    assert observation.provider == "google/gemma-4-E2B-it"
    assert observation.tags == ["slack", "record_decision"]
    assert observation.entity_mentions == ["Orbit Forge", "Prism"]


async def test_fake_mode_is_the_only_explicit_deterministic_path() -> None:
    event = SourceEvent(
        source_id="test-3",
        source_type=SourceType.MEETING,
        conversation_id="meeting-1",
        content="Status update for Nova.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
    )

    observation = await LLMInterpreter(Settings(inference_mode="fake")).observe(event)

    assert observation.provider == "deterministic-fallback"


class FakeMemoryStore:
    async def add_episode(
        self, *, episode_id: str, group_id: str, summary: str, event: SourceEvent
    ) -> str:
        return f"mem0-{episode_id}"

    async def search(self, query: str, group_id: str) -> list[dict[str, Any]]:
        return []

class FlakyMemoryStore(FakeMemoryStore):
    """Fails every index write until ``fixed`` is set, like a locked vector file."""

    def __init__(self) -> None:
        self.fixed = False
        self.indexed: list[tuple[str, str]] = []

    async def add_episode(
        self, *, episode_id: str, group_id: str, summary: str, event: SourceEvent
    ) -> str:
        if not self.fixed:
            raise sqlite3.OperationalError("database is locked")
        self.indexed.append((episode_id, summary))
        return f"mem0-{episode_id}"


async def _episode_store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}")
    configure_sqlite(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def test_an_episode_survives_a_failed_index_but_is_left_unsearchable(tmp_path) -> None:
    """The behavior that hid the bug: storing succeeds, indexing quietly doesn't."""
    engine, sessions = await _episode_store(tmp_path)
    store = FlakyMemoryStore()
    service = SystemOneService(LLMInterpreter(Settings(inference_mode="fake")), store)
    event = SourceEvent(
        source_id="s1",
        source_type=SourceType.SLACK,
        conversation_id="c1",
        content="Nova ships Thursday.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
    )

    async with sessions() as session:
        episode, created = await service.process(event, "acme", session)

        assert created is True
        assert episode.mem0_id is None
        assert episode.details["memory_index_error"] == "OperationalError"

    await engine.dispose()


async def test_reindex_missing_repairs_episodes_a_watermark_would_never_resend(
    tmp_path,
) -> None:
    engine, sessions = await _episode_store(tmp_path)
    store = FlakyMemoryStore()
    service = SystemOneService(LLMInterpreter(Settings(inference_mode="fake")), store)

    async with sessions() as session:
        for i in range(3):
            await service.process(
                SourceEvent(
                    source_id=f"s{i}",
                    source_type=SourceType.SLACK,
                    conversation_id="c1",
                    content=f"message {i}",
                    participants=["Priya"],
                    occurred_at=datetime.now(UTC),
                ),
                "acme",
                session,
            )
        # An episode belonging to someone else must not be swept up.
        await service.process(
            SourceEvent(
                source_id="other",
                source_type=SourceType.SLACK,
                conversation_id="c9",
                content="not yours",
                participants=[],
                occurred_at=datetime.now(UTC),
            ),
            "widgets",
            session,
        )

    store.fixed = True
    async with sessions() as session:
        counts = await service.reindex_missing("acme", session)

        assert counts == {"indexed": 3, "failed": 0}
        rows = (
            await session.scalars(select(Episode).where(Episode.group_id == "acme"))
        ).all()
        assert all(r.mem0_id is not None for r in rows)
        assert all("memory_index_error" not in (r.details or {}) for r in rows)

        # Idempotent: nothing is left to repair on a second pass.
        assert await service.reindex_missing("acme", session) == {"indexed": 0, "failed": 0}

        orphan = await session.scalar(
            select(Episode).where(Episode.group_id == "widgets")
        )
        assert orphan.mem0_id is None

    await engine.dispose()
