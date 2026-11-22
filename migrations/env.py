import asyncio
from logging.config import fileConfig

import sqlalchemy as sa
from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

from synapse.config import get_settings
from synapse.db import models  # noqa: F401
from synapse.db.base import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def compare_type(
    context: object,
    inspected_column: object,
    metadata_column: object,
    inspected_type: sa.types.TypeEngine[object],
    metadata_type: sa.types.TypeEngine[object],
) -> bool | None:
    """Ignore a change in an enum column's VARCHAR length.

    Synapse stores memory in SQLite, where an enum is a VARCHAR sized to its
    longest member with no CHECK constraint, so adding a longer member (such as
    ``whatsapp``) reads as a type change. SQLite does not enforce VARCHAR
    lengths, so there is nothing to migrate, and the only way to "apply" it is
    rebuilding the table, which with ON DELETE CASCADE children is not something
    to do for nothing. Every other comparison is Alembic's default.
    """
    if isinstance(metadata_type, sa.Enum) and isinstance(inspected_type, sa.String):
        return False
    return None


def do_run_migrations(connection: object) -> None:
    # SQLite cannot ALTER most of a column definition, so any future revision
    # that changes one has to rebuild the table. render_as_batch makes Alembic
    # emit that copy-and-swap instead of an ALTER that SQLite would reject.
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=compare_type,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(config.get_section(config.config_ini_section) or {})
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


asyncio.run(run_async_migrations())
