"""per-episode consolidation state, for incremental runs

Hosts that embed Synapse get the same change from
``synapse.db.schema.create_schema``; this is for databases managed by Alembic.
A database that already has the consolidation tables but no runs needs nothing
else: every episode starts out due.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-22 20:00:00
"""

from collections.abc import Sequence

from alembic import op

from synapse.db.schema import create_schema

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    create_schema(op.get_bind())


def downgrade() -> None:
    op.drop_table("consolidation_episode_states")
