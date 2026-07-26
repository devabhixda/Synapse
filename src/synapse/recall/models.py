from typing import Any, Literal

from pydantic import BaseModel, Field


class RecallCitation(BaseModel):
    kind: Literal["episode", "external"]
    id: str
    label: str
    excerpt: str | None = None
    url: str | None = None


class RecallResult(BaseModel):
    query: str
    answer: str | None
    answer_origin: Literal["episodic", "semantic_fallback", "miss"]
    citations: list[RecallCitation] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    score_details: dict[str, Any] = Field(default_factory=dict)
    created_episode_id: str | None = None
    latency_ms: float


class EpisodeSelection(BaseModel):
    answerable: bool
    selected_episode_id: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)


class EpisodeDetail(BaseModel):
    id: str
    kind: str
    source_type: str
    content: str
    summary: str | None
    occurred_at: str
    participants: list[str]
    details: dict[str, Any]