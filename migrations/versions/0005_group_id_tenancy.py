"""Scope every table to an account with group_id.

Existing rows predate tenancy and cannot be attributed after the fact, so they
are all assigned to one account. Set ``SYNAPSE_BACKFILL_GROUP_ID`` before
running this to choose the name; it defaults to ``default``. Getting it wrong
is recoverable with an UPDATE, since the value is an opaque label rather than a
key into anything.

The column is added nullable, backfilled, and only then made NOT NULL, so the
migration works on a populated database rather than only an empty one.

Constraints that keyed on globally-unique-by-accident values are rebuilt to
include the account: two accounts can both have a workflow called
"status_query" or an entity called "the gateway", and under the old
constraints the second one to arrive would have been rejected or merged.

Revision ID: 0005
Revises: 0004
"""

import os
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES: tuple[str, ...] = (
    "episodes",
    "intent_observations",
    "recall_events",
    "workflow_patterns",
    "skill_specs",
    "skill_spec_revisions",
    "consolidation_runs",
    "consolidation_episodes",
    "semantic_entities",
    "entity_aliases",
    "external_attributes",
    "semantic_relationships",
    "relationship_evidence",
)

GROUP_ID_LENGTH = 120


def _backfill_group() -> str:
    return os.environ.get("SYNAPSE_BACKFILL_GROUP_ID", "default")


def upgrade() -> None:
    group = _backfill_group()

    for table in TABLES:
        op.add_column(table, sa.Column("group_id", sa.String(GROUP_ID_LENGTH), nullable=True))
        op.execute(sa.text(f"update {table} set group_id = :group").bindparams(group=group))
        op.alter_column(table, "group_id", nullable=False)
        op.create_index(f"ix_{table}_group_id", table, ["group_id"])

    # Episodes: dedupe key was (source_type, source_id) across all accounts.
    op.drop_constraint("uq_episode_source", "episodes", type_="unique")
    op.create_unique_constraint(
        "uq_episode_source", "episodes", ["group_id", "source_type", "source_id"]
    )
    op.drop_index("ix_episodes_occurred_at", table_name="episodes")
    op.create_index("ix_episodes_group_occurred", "episodes", ["group_id", "occurred_at"])

    op.create_index("ix_recall_events_group_created", "recall_events", ["group_id", "created_at"])

    # Entity identity: "Acme" in two accounts is two entities, not one.
    op.drop_constraint("uq_semantic_entity_identity", "semantic_entities", type_="unique")
    op.create_unique_constraint(
        "uq_semantic_entity_identity",
        "semantic_entities",
        ["group_id", "kind", "normalized_name"],
    )
    op.drop_index("ix_semantic_entities_name", table_name="semantic_entities")
    op.create_index(
        "ix_semantic_entities_group_name", "semantic_entities", ["group_id", "normalized_name"]
    )

    op.drop_constraint("uq_entity_alias_normalized", "entity_aliases", type_="unique")
    op.create_unique_constraint(
        "uq_entity_alias_normalized", "entity_aliases", ["group_id", "normalized_alias"]
    )

    op.drop_index("ix_semantic_relationships_predicate", table_name="semantic_relationships")
    op.create_index(
        "ix_semantic_relationships_group_predicate",
        "semantic_relationships",
        ["group_id", "predicate"],
    )

    # These two were column-level unique=True, so Postgres named them itself.
    op.execute("alter table workflow_patterns drop constraint if exists workflow_patterns_key_key")
    op.create_unique_constraint("uq_workflow_pattern_key", "workflow_patterns", ["group_id", "key"])

    op.execute("alter table skill_specs drop constraint if exists skill_specs_name_key")
    op.create_unique_constraint("uq_skill_spec_name", "skill_specs", ["group_id", "name"])


def downgrade() -> None:
    op.drop_constraint("uq_skill_spec_name", "skill_specs", type_="unique")
    op.create_unique_constraint("skill_specs_name_key", "skill_specs", ["name"])

    op.drop_constraint("uq_workflow_pattern_key", "workflow_patterns", type_="unique")
    op.create_unique_constraint("workflow_patterns_key_key", "workflow_patterns", ["key"])

    op.drop_index(
        "ix_semantic_relationships_group_predicate", table_name="semantic_relationships"
    )
    op.create_index(
        "ix_semantic_relationships_predicate", "semantic_relationships", ["predicate"]
    )

    op.drop_constraint("uq_entity_alias_normalized", "entity_aliases", type_="unique")
    op.create_unique_constraint(
        "uq_entity_alias_normalized", "entity_aliases", ["normalized_alias"]
    )

    op.drop_index("ix_semantic_entities_group_name", table_name="semantic_entities")
    op.create_index("ix_semantic_entities_name", "semantic_entities", ["normalized_name"])
    op.drop_constraint("uq_semantic_entity_identity", "semantic_entities", type_="unique")
    op.create_unique_constraint(
        "uq_semantic_entity_identity", "semantic_entities", ["kind", "normalized_name"]
    )

    op.drop_index("ix_recall_events_group_created", table_name="recall_events")

    op.drop_index("ix_episodes_group_occurred", table_name="episodes")
    op.create_index("ix_episodes_occurred_at", "episodes", ["occurred_at"])
    op.drop_constraint("uq_episode_source", "episodes", type_="unique")
    op.create_unique_constraint("uq_episode_source", "episodes", ["source_type", "source_id"])

    for table in reversed(TABLES):
        op.drop_index(f"ix_{table}_group_id", table_name=table)
        op.drop_column(table, "group_id")
