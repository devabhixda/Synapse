from datetime import datetime

from pydantic import BaseModel, Field


class EntitySpec(BaseModel):
    key: str
    kind: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)


class RelationshipSpec(BaseModel):
    subject_key: str
    predicate: str
    object_key: str
    evidence_source_ids: list[str]
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)
    # The words in each evidence episode that support the edge, by source id.
    # A quote must appear in its episode, and is what the evidence row keeps;
    # without one the row keeps the whole episode, as it always has.
    evidence_quotes: dict[str, str] = Field(default_factory=dict)
    # When the edge became true. Defaults to when its newest evidence episode
    # began, which is wrong for a long thread: a reply today on a thread from
    # August dates the edge to August. A caller that knows when the deciding
    # message was said passes it here.
    asserted_at: datetime | None = None


class ConsolidationPlan(BaseModel):
    entities: list[EntitySpec]
    relationships: list[RelationshipSpec]


class ExtractionResult(BaseModel):
    plan: ConsolidationPlan
    provider: str
    fallback_reason: str | None = None


class GraphAdjudication(BaseModel):
    approved: bool
    alias_map: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class RawEntitySpec(BaseModel):
    kind: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class RawRelationshipSpec(BaseModel):
    subject_kind: str
    subject_name: str
    predicate: str
    object_kind: str
    object_name: str
    evidence_source_ids: list[str]
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class RawConsolidationPlan(BaseModel):
    entities: list[RawEntitySpec]
    relationships: list[RawRelationshipSpec]