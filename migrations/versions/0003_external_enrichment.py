"""Create external semantic enrichment table.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "external_attributes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("semantic_entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(80), nullable=False),
        sa.Column("field_group", sa.String(120), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("source_url", sa.Text()),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "entity_id", "provider", "field_group", name="uq_external_attribute"
        ),
    )
    op.create_index("ix_external_attributes_entity_id", "external_attributes", ["entity_id"])
    op.create_index("ix_external_attributes_provider", "external_attributes", ["provider"])


def downgrade() -> None:
    op.drop_table("external_attributes")