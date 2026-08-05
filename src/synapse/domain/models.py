from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from synapse.db.models import SourceType


class SourceEvent(BaseModel):
    source_id: str
    source_type: SourceType
    conversation_id: str
    content: str
    participants: list[str]
    occurred_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticEntityAssertion(BaseModel):
    kind: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)


class SemanticRelationshipAssertion(BaseModel):
    subject_kind: str
    subject_name: str
    predicate: str
    object_kind: str
    object_name: str
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)


class SemanticAssertions(BaseModel):
    entities: list[SemanticEntityAssertion] = Field(default_factory=list)
    relationships: list[SemanticRelationshipAssertion] = Field(default_factory=list)


def semantic_assertions(metadata: dict[str, Any]) -> SemanticAssertions:
    payload = metadata.get("semantic_assertions")
    if not isinstance(payload, dict):
        return SemanticAssertions()
    return SemanticAssertions.model_validate(payload)


class FastObservation(BaseModel):
    summary: str
    intent: str
    salience: float = Field(ge=0, le=1)
    entity_mentions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    provider: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float | None = Field(default=None, ge=0)


class FastInterpreter(Protocol):
    async def observe(self, event: SourceEvent) -> FastObservation: ...


class EpisodeMemoryStore(Protocol):
    async def add_episode(
        self, *, episode_id: str, group_id: str, summary: str, event: SourceEvent
    ) -> list[str]: ...

    async def search(self, query: str, group_id: str) -> list[dict[str, Any]]: ...

    async def forget_group(self, group_id: str) -> None: ...

    async def forget_memories(self, memory_ids: list[str]) -> None: ...
