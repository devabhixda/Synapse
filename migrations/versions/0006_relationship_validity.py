"""Give semantic relationships a period of validity.

Existing edges are treated as true since they were first written and still
true now: ``valid_at`` is backfilled from ``created_at`` and ``invalid_at``
stays null. That is the only honest reading — nothing recorded before this
migration says when a fact stopped holding, and assuming any of them are
already retired would silently drop them from every answer.

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "semantic_relationships",
        sa.Column("valid_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "semantic_relationships",
        sa.Column("invalid_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("update semantic_relationships set valid_at = created_at where valid_at is null")
    op.alter_column("semantic_relationships", "valid_at", nullable=False)
    op.create_index(
        "ix_semantic_relationships_group_current",
        "semantic_relationships",
        ["group_id", "invalid_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_semantic_relationships_group_current", table_name="semantic_relationships"
    )
    op.drop_column("semantic_relationships", "invalid_at")
    op.drop_column("semantic_relationships", "valid_at")
