"""Database fixtures for tests that need the real schema.

Every test here used to be a pure unit test or an OpenAPI schema check, so
nothing ever opened a session. That was survivable while the schema was global,
but tenancy is enforced by ``group_id`` filters and composite unique
constraints, and neither can be proven against mocks — a missing ``where``
clause looks exactly like a correct one until two accounts share a table. These
fixtures exist so isolation can be tested against a real database rather than
asserted.

Each test gets its own SQLite file in a temporary directory, which is why there
is no ``_test`` database-name guard and no skip: there is no server to be
absent and nothing of anyone's to drop. The tenancy tests that used to skip
without Postgres now always run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from synapse.db import models  # noqa: F401 — imported so Base.metadata sees every table
from synapse.db.base import Base
from synapse.db.pragmas import configure_sqlite


@pytest_asyncio.fixture
async def sessions(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory over an empty schema, rebuilt for each test."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'synapse.db'}")
    configure_sqlite(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def session(
    sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """One session over an empty schema, for tests that only need the one."""
    async with sessions() as open_session:
        yield open_session


@pytest_asyncio.fixture
async def memory_db_path(tmp_path: Path) -> Any:
    """A path for a Mem0 vector store file, isolated per test."""
    return str(tmp_path / "memory.db")
