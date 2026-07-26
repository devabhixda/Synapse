from typing import Any, Literal

from pydantic import BaseModel, Field


class SkillTemplate(BaseModel):
    name: str
    purpose: str
    triggers: list[str]
    required_queries: list[str]
    steps: list[str]
    guardrails: list[str]
    expected_output: dict[str, Any]


class DetectionResult(BaseModel):
    recall_events_scanned: int
    candidate_families: int
    patterns_created: int
    patterns_updated: int
    revisions_created: int


class PatternView(BaseModel):
    id: str
    key: str
    name: str
    description: str
    status: str
    occurrence_count: int
    confidence: float
    evidence_recall_ids: list[str]
    first_seen_at: str
    last_seen_at: str


class SkillSpecView(BaseModel):
    id: str
    pattern_id: str
    name: str
    version: int
    status: str
    purpose: str
    triggers: list[str]
    required_queries: list[str]
    steps: list[str]
    guardrails: list[str]
    expected_output: dict[str, Any]
    confidence: float


class SkillStatusUpdate(BaseModel):
    status: Literal["candidate", "approved", "rejected"]


class SkillSpecSnapshot(BaseModel):
    name: str
    purpose: str
    triggers: list[str]
    required_queries: list[str]
    steps: list[str]
    guardrails: list[str]
    expected_output: dict[str, Any]
    confidence: float = Field(ge=0, le=1)