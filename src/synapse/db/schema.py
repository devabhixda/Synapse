"""Bring a database up to the current schema, for hosts that do not run Alembic.

A host that embeds Synapse (fde-assist, basemem) creates the tables on first
start with ``Base.metadata.create_all``. That creates tables that are missing
and never touches one that exists, so the first release to add a column to an
existing table would break every database already out there: each query names
a column the file does not have.

``create_schema`` is ``create_all`` plus the columns existing tables lack. It
is additive only, which is all SQLite's ``ALTER TABLE ADD COLUMN`` can do
safely: a new column must be nullable or carry a constant ``server_default``.
Anything else (a dropped column, a changed type or constraint) needs a real
migration and fails loudly here instead of being skipped.
"""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CreateColumn

from synapse.db.base import Base


def create_schema(connection: Connection) -> list[str]:
    """Create missing tables and add missing columns. Returns the columns added.

    Synchronous, for ``await conn.run_sync(create_schema)``.
    """
    from synapse.db import models  # noqa: F401 — registers every table on Base.metadata

    existing = set(inspect(connection).get_table_names())
    Base.metadata.create_all(connection)

    added: list[str] = []
    inspector = inspect(connection)
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue
        have = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in have:
                continue
            if not column.nullable and column.server_default is None:
                raise RuntimeError(
                    f"{table.name}.{column.name} is NOT NULL with no server_default, "
                    "so it cannot be added to a table that already has rows"
                )
            ddl = CreateColumn(column).compile(dialect=connection.dialect)
            connection.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN {ddl}'))
            added.append(f"{table.name}.{column.name}")
    return added
