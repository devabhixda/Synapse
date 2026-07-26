"""Create System 2 consolidation and semantic graph tables.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "consolidation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(80), nullable=False),
        sa.Column("episode_count", sa.Integer(), nullable=False),
        sa.Column("entity_count", sa.Integer(), nullable=False),
        sa.Column("relationship_count", sa.Integer(), nullable=False),
        sa.Column("graph_diff", postgresql.JSONB(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_consolidation_runs_status", "consolidation_runs", ["status"])
    op.create_index("ix_consolidation_runs_created_at", "consolidation_runs", ["created_at"])

    op.create_table(
        "consolidation_episodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("consolidation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "episode_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.UniqueConstraint("run_id", "episode_id", name="uq_consolidation_episode"),
    )
    op.create_index("ix_consolidation_episodes_run_id", "consolidation_episodes", ["run_id"])
    op.create_index(
        "ix_consolidation_episodes_episode_id", "consolidation_episodes", ["episode_id"]
    )

    op.create_table(
        "semantic_entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("canonical_name", sa.String(255), nullable=False),
        sa.Column("normalized_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("attributes", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("kind", "normalized_name", name="uq_semantic_entity_identity"),
    )
    op.create_index("ix_semantic_entities_name", "semantic_entities", ["normalized_name"])

    op.create_table(
        "entity_aliases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("semantic_entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alias", sa.String(255), nullable=False),
        sa.Column("normalized_alias", sa.String(255), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.UniqueConstraint("normalized_alias", name="uq_entity_alias_normalized"),
    )
    op.create_index("ix_entity_aliases_entity_id", "entity_aliases", ["entity_id"])
    op.create_index("ix_entity_aliases_normalized_alias", "entity_aliases", ["normalized_alias"])

    op.create_table(
        "semantic_relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "subject_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("semantic_entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("predicate", sa.String(120), nullable=False),
        sa.Column(
            "object_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("semantic_entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attributes", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "subject_id", "predicate", "object_id", name="uq_semantic_relationship"
        ),
    )
    op.create_index(
        "ix_semantic_relationships_subject_id", "semantic_relationships", ["subject_id"]
    )
    op.create_index(
        "ix_semantic_relationships_object_id", "semantic_relationships", ["object_id"]
    )
    op.create_index(
        "ix_semantic_relationships_predicate", "semantic_relationships", ["predicate"]
    )

    op.create_table(
        "relationship_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "relationship_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("semantic_relationships.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "episode_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.UniqueConstraint(
            "relationship_id", "episode_id", name="uq_relationship_evidence"
        ),
    )
    op.create_index(
        "ix_relationship_evidence_relationship_id",
        "relationship_evidence",
        ["relationship_id"],
    )
    op.create_index(
        "ix_relationship_evidence_episode_id", "relationship_evidence", ["episode_id"]
    )


def downgrade() -> None:
    op.drop_table("relationship_evidence")
    op.drop_table("semantic_relationships")
    op.drop_table("entity_aliases")
    op.drop_table("semantic_entities")
    op.drop_table("consolidation_episodes")
    op.drop_table("consolidation_runs")