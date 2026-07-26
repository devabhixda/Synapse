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
        self, *, episode_id: str, summary: str, event: SourceEvent
    ) -> str: ...

    async def search(self, query: str) -> list[dict[str, Any]]: ...
