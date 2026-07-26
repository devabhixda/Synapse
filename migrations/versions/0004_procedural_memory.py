"""Create procedural workflow and skill specification tables.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_patterns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("key", sa.String(160), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence_recall_ids", postgresql.JSONB(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_workflow_patterns_key", "workflow_patterns", ["key"])
    op.create_index("ix_workflow_patterns_status", "workflow_patterns", ["status"])

    op.create_table(
        "skill_specs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "pattern_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_patterns.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("name", sa.String(160), nullable=False, unique=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("triggers", postgresql.JSONB(), nullable=False),
        sa.Column("required_queries", postgresql.JSONB(), nullable=False),
        sa.Column("steps", postgresql.JSONB(), nullable=False),
        sa.Column("guardrails", postgresql.JSONB(), nullable=False),
        sa.Column("expected_output", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_skill_specs_pattern_id", "skill_specs", ["pattern_id"])
    op.create_index("ix_skill_specs_name", "skill_specs", ["name"])
    op.create_index("ix_skill_specs_status", "skill_specs", ["status"])

    op.create_table(
        "skill_spec_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "skill_spec_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skill_specs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("skill_spec_id", "version", name="uq_skill_spec_revision"),
    )
    op.create_index(
        "ix_skill_spec_revisions_skill_spec_id",
        "skill_spec_revisions",
        ["skill_spec_id"],
    )


def downgrade() -> None:
    op.drop_table("skill_spec_revisions")
    op.drop_table("skill_specs")
    op.drop_table("workflow_patterns")