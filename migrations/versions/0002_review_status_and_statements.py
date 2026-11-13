"""review status on edges, and statements

Hosts that embed Synapse get the same change from
``synapse.db.schema.create_schema``; this is for databases managed by Alembic.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12 17:40:00
"""

from collections.abc import Sequence

from alembic import op

from synapse.db.schema import create_schema

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    create_schema(op.get_bind())


def downgrade() -> None:
    op.drop_table("statement_evidence")
    op.drop_table("semantic_statements")
    with op.batch_alter_table("semantic_relationships") as batch:
        batch.drop_column("status")
