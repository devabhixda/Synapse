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


class ConsolidationPlan(BaseModel):
    entities: list[EntitySpec]
    relationships: list[RelationshipSpec]