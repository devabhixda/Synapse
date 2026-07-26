"""Create episodic memory tables.

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    episode_kind = postgresql.ENUM("source", "retrieval", name="episode_kind", create_type=False)
    source_type = postgresql.ENUM(
        "slack", "meeting", "recall", name="source_type", create_type=False
    )
    episode_kind.create(op.get_bind(), checkfirst=True)
    source_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "episodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", episode_kind, nullable=False),
        sa.Column("source_type", source_type, nullable=False),
        sa.Column("source_id", sa.String(255)),
        sa.Column("conversation_id", sa.String(255)),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text()),
        sa.Column("participants", postgresql.JSONB(), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("mem0_id", sa.String(255), unique=True),
        sa.Column("stale_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("source_type", "source_id", name="uq_episode_source"),
    )
    op.create_index("ix_episodes_occurred_at", "episodes", ["occurred_at"])

    op.create_table(
        "intent_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "episode_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("intent", sa.String(160), nullable=False),
        sa.Column("salience", sa.Float(), nullable=False),
        sa.Column("entity_mentions", postgresql.JSONB(), nullable=False),
        sa.Column("tags", postgresql.JSONB(), nullable=False),
        sa.Column("provider", sa.String(80), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_intent_observations_episode_id", "intent_observations", ["episode_id"])

    op.create_table(
        "recall_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("normalized_intent", sa.String(160)),
        sa.Column("answer_origin", sa.String(40), nullable=False),
        sa.Column("selected_ids", postgresql.JSONB(), nullable=False),
        sa.Column("score_details", postgresql.JSONB(), nullable=False),
        sa.Column("latency_ms", sa.Float()),
        sa.Column(
            "created_episode_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("episodes.id", ondelete="SET NULL"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_recall_events_created_at", "recall_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("recall_events")
    op.drop_table("intent_observations")
    op.drop_table("episodes")
    op.execute("DROP TYPE IF EXISTS source_type")
    op.execute("DROP TYPE IF EXISTS episode_kind")
