from datetime import UTC, datetime
from typing import Any

from synapse.db.models import SourceType
from synapse.domain.models import FastObservation, SourceEvent
from synapse.inference.cactus import deterministic_observation


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


class FakeMemoryStore:
    async def add_episode(
        self, *, episode_id: str, summary: str, event: SourceEvent
    ) -> str:
        return f"mem0-{episode_id}"

    async def search(self, query: str) -> list[dict[str, Any]]:
        return []