import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String, Text, UniqueConstraint
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


class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_episode_source"),
        Index("ix_episodes_occurred_at", "occurred_at"),
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


class IntentObservation(Base):
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


class RecallEvent(Base):
    __tablename__ = "recall_events"

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
