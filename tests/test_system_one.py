from datetime import UTC, datetime
from typing import Any

from synapse.config import Settings
from synapse.db.models import SourceType
from synapse.domain.models import FastObservation, SourceEvent
from synapse.inference.cactus import (
    CactusInterpreter,
    FastClassification,
    assemble_observation,
    deterministic_observation,
    parse_json_object,
)


def test_deterministic_observation_marks_decision() -> None:
    event = SourceEvent(
        source_id="test-1",
        source_type=SourceType.SLACK,
        conversation_id="thread-1",
        content="We decided the auth database will be called Titan.",
        participants=["Maya"],
        occurred_at=datetime.now(UTC),
    )

    observation: FastObservation = deterministic_observation(event)

    assert observation.intent == "record_decision"
    assert "Titan" in observation.entity_mentions
    assert observation.salience == 0.9


def test_compact_gemma_classification_assembles_safe_observation() -> None:
    event = SourceEvent(
        source_id="test-2",
        source_type=SourceType.SLACK,
        conversation_id="thread-2",
        content="Maya: We decided the auth database will be called Titan.",
        participants=["Maya"],
        occurred_at=datetime.now(UTC),
    )
    classification = FastClassification.model_validate(
        parse_json_object('{"intent":"record_decision","salience":0.9}')
    )

    observation = assemble_observation(
        event,
        classification,
        provider="cactus-gemma",
        confidence=0.88,
        latency_ms=120,
    )

    assert observation.provider == "cactus-gemma"
    assert observation.tags == ["slack", "record_decision"]
    assert observation.entity_mentions == ["Titan", "auth database"]


async def test_fake_mode_is_the_only_explicit_deterministic_path() -> None:
    event = SourceEvent(
        source_id="test-3",
        source_type=SourceType.MEETING,
        conversation_id="meeting-1",
        content="Status update for Titan.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
    )

    observation = await CactusInterpreter(Settings(inference_mode="fake")).observe(event)

    assert observation.provider == "deterministic-fallback"


class FakeMemoryStore:
    async def add_episode(
        self, *, episode_id: str, summary: str, event: SourceEvent
    ) -> str:
        return f"mem0-{episode_id}"

    async def search(self, query: str) -> list[dict[str, Any]]:
        return []