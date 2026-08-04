"""SQLite connection settings, kept apart from the engine that uses them.

``session.py`` builds an engine at import time from this process's settings.
Hosts that embed Synapse build their own engine against their own URL — see
fde-assist — and need these pragmas without also constructing Synapse's engine
and reading Synapse's environment as a side effect of the import.
"""

from typing import Any

from sqlalchemy import event


def configure_sqlite(engine: Any) -> None:
    """Set the pragmas that make one SQLite file safe for two writers.

    Mem0 opens the same file on its own ``sqlite3`` connection, so the engine
    is not the only writer. WAL lets its reads proceed while a write is in
    flight, and ``busy_timeout`` turns the remaining collisions into a wait
    instead of an immediate "database is locked". Foreign keys are off by
    default in SQLite, and the schema leans on ``ON DELETE CASCADE``.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
