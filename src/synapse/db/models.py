"""Schema for episodic memory, the semantic graph, and procedural skills.

Every table carries ``group_id``, the account the row belongs to. It is on all
of them rather than only the roots so that no query has to reach through a join
to be safe — "filter by group_id" is then a rule with no exceptions, and a
missing filter is visible in the query itself instead of in a join two tables
away.

Which unique constraints grew a ``group_id`` follows one rule: a constraint gets
it when its columns are not already scoped by a foreign key to a tenant-owned
row. ``uq_semantic_entity_identity`` needs it, because two accounts can each
have an entity called "the gateway". ``uq_relationship_evidence`` does not,
because it keys on a relationship and an episode that are themselves already
scoped. Adding it there would suggest the pair could otherwise collide across
accounts, which is not true, and would leave the impression that the constraint
is what prevents mismatched parents when nothing does.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from synapse.db.base import Base


class EpisodeKind(StrEnum):
    SOURCE = "source"
    RETRIEVAL = "retrieval"


class SourceType(StrEnum):
    SLACK = "slack"
    MEETING = "meeting"
    RECALL = "recall"


episode_kind_enum = Enum(
    EpisodeKind,
    name="episode_kind",
    values_callable=lambda members: [member.value for member in members],
)
source_type_enum = Enum(
    SourceType,
    name="source_type",
    values_callable=lambda members: [member.value for member in members],
)

GROUP_ID_LENGTH = 120


class Tenanted:
    """Mixin marking a table as owned by one account.

    Inherit it rather than declaring the column, so that adding a table without
    tenancy is a visible omission instead of an easy oversight.
    """

    group_id: Mapped[str] = mapped_column(String(GROUP_ID_LENGTH), index=True)


class Episode(Tenanted, Base):
    __tablename__ = "episodes"
    __table_args__ = (
        UniqueConstraint("group_id", "source_type", "source_id", name="uq_episode_source"),
        Index("ix_episodes_group_occurred", "group_id", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[EpisodeKind] = mapped_column(episode_kind_enum)
    source_type: Mapped[SourceType] = mapped_column(source_type_enum)
    source_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    participants: Mapped[list[str]] = mapped_column(JSONB, default=list)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    mem0_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    stale_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    observations: Mapped[list["IntentObservation"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )


class IntentObservation(Tenanted, Base):
    __tablename__ = "intent_observations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), index=True
    )
    intent: Mapped[str] = mapped_column(String(160))
    salience: Mapped[float] = mapped_column(Float)
    entity_mentions: Mapped[list[str]] = mapped_column(JSONB, default=list)
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list)
    provider: Mapped[str] = mapped_column(String(80))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    episode: Mapped[Episode] = relationship(back_populates="observations")


class RecallEvent(Tenanted, Base):
    __tablename__ = "recall_events"
    __table_args__ = (Index("ix_recall_events_group_created", "group_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text)
    normalized_intent: Mapped[str | None] = mapped_column(String(160), nullable=True)
    answer_origin: Mapped[str] = mapped_column(String(40))
    selected_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    score_details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_episode_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class WorkflowPattern(Tenanted, Base):
    __tablename__ = "workflow_patterns"
    __table_args__ = (
        # The key is a workflow family like "status_query", so every account
        # grows the same handful — unique per account, never globally.
        UniqueConstraint("group_id", "key", name="uq_workflow_pattern_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(160), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_recall_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    skill_spec: Mapped["SkillSpec | None"] = relationship(
        back_populates="pattern", cascade="all, delete-orphan", uselist=False
    )


class SkillSpec(Tenanted, Base):
    __tablename__ = "skill_specs"
    __table_args__ = (UniqueConstraint("group_id", "name", name="uq_skill_spec_name"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_patterns.id", ondelete="CASCADE"), unique=True, index=True
    )
    name: Mapped[str] = mapped_column(String(160), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="candidate", index=True)
    purpose: Mapped[str] = mapped_column(Text)
    triggers: Mapped[list[str]] = mapped_column(JSONB, default=list)
    required_queries: Mapped[list[str]] = mapped_column(JSONB, default=list)
    steps: Mapped[list[str]] = mapped_column(JSONB, default=list)
    guardrails: Mapped[list[str]] = mapped_column(JSONB, default=list)
    expected_output: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    pattern: Mapped[WorkflowPattern] = relationship(back_populates="skill_spec")
    revisions: Mapped[list["SkillSpecRevision"]] = relationship(
        back_populates="skill_spec", cascade="all, delete-orphan"
    )


class SkillSpecRevision(Tenanted, Base):
    __tablename__ = "skill_spec_revisions"
    __table_args__ = (
        UniqueConstraint("skill_spec_id", "version", name="uq_skill_spec_revision"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    skill_spec_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("skill_specs.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    skill_spec: Mapped[SkillSpec] = relationship(back_populates="revisions")


class ConsolidationRun(Tenanted, Base):
    __tablename__ = "consolidation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    provider: Mapped[str] = mapped_column(String(80), default="deterministic")
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    relationship_count: Mapped[int] = mapped_column(Integer, default=0)
    graph_diff: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    snapshots: Mapped[list["ConsolidationEpisode"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ConsolidationEpisode(Tenanted, Base):
    __tablename__ = "consolidation_episodes"
    __table_args__ = (
        UniqueConstraint("run_id", "episode_id", name="uq_consolidation_episode"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("consolidation_runs.id", ondelete="CASCADE"), index=True
    )
    episode_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), index=True
    )

    run: Mapped[ConsolidationRun] = relationship(back_populates="snapshots")
    episode: Mapped[Episode] = relationship()


class SemanticEntity(Tenanted, Base):
    __tablename__ = "semantic_entities"
    __table_args__ = (
        UniqueConstraint(
            "group_id", "kind", "normalized_name", name="uq_semantic_entity_identity"
        ),
        Index("ix_semantic_entities_group_name", "group_id", "normalized_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(64))
    canonical_name: Mapped[str] = mapped_column(String(255))
    normalized_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    aliases: Mapped[list["EntityAlias"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )
    external_attributes: Mapped[list["ExternalAttribute"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )


class EntityAlias(Tenanted, Base):
    __tablename__ = "entity_aliases"
    __table_args__ = (
        UniqueConstraint("group_id", "normalized_alias", name="uq_entity_alias_normalized"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("semantic_entities.id", ondelete="CASCADE"), index=True
    )
    alias: Mapped[str] = mapped_column(String(255))
    normalized_alias: Mapped[str] = mapped_column(String(255), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    entity: Mapped[SemanticEntity] = relationship(back_populates="aliases")


class ExternalAttribute(Tenanted, Base):
    __tablename__ = "external_attributes"
    __table_args__ = (
        UniqueConstraint(
            "entity_id", "provider", "field_group", name="uq_external_attribute"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("semantic_entities.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(80), index=True)
    field_group: Mapped[str] = mapped_column(String(120))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    entity: Mapped[SemanticEntity] = relationship(back_populates="external_attributes")


class SemanticRelationship(Tenanted, Base):
    """A subject-predicate-object edge, true over a period rather than forever.

    ``valid_at`` is when the edge became true — the time of the episode that
    asserted it, not the time it was written down. ``invalid_at`` is when it
    stopped; ``None`` means it still holds. A contradicting assertion closes
    the earlier edge instead of deleting it, so "who owns this?" and "who owned
    it in March?" are both answerable.

    Only predicates that genuinely replace their predecessor are closed this
    way; see ``SUBJECT_FUNCTIONAL`` and ``OBJECT_FUNCTIONAL`` in the
    consolidation service. A subject can block many things at once, and
    retiring those would erase history that is still true.
    """

    __tablename__ = "semantic_relationships"
    __table_args__ = (
        UniqueConstraint(
            "subject_id", "predicate", "object_id", name="uq_semantic_relationship"
        ),
        Index("ix_semantic_relationships_group_predicate", "group_id", "predicate"),
        # Reads default to current edges, so the common query is
        # group_id + invalid_at IS NULL.
        Index("ix_semantic_relationships_group_current", "group_id", "invalid_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("semantic_entities.id", ondelete="CASCADE"), index=True
    )
    predicate: Mapped[str] = mapped_column(String(120))
    object_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("semantic_entities.id", ondelete="CASCADE"), index=True
    )
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    valid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    invalid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    subject: Mapped[SemanticEntity] = relationship(foreign_keys=[subject_id])
    object: Mapped[SemanticEntity] = relationship(foreign_keys=[object_id])
    evidence: Mapped[list["RelationshipEvidence"]] = relationship(
        back_populates="semantic_relationship", cascade="all, delete-orphan"
    )


class RelationshipEvidence(Tenanted, Base):
    __tablename__ = "relationship_evidence"
    __table_args__ = (
        UniqueConstraint("relationship_id", "episode_id", name="uq_relationship_evidence"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relationship_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("semantic_relationships.id", ondelete="CASCADE"), index=True
    )
    episode_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), index=True
    )
    excerpt: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    semantic_relationship: Mapped[SemanticRelationship] = relationship(back_populates="evidence")
    episode: Mapped[Episode] = relationship()
