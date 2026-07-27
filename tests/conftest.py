"""Postgres fixtures for tests that need the real database.

Every test here used to be a pure unit test or an OpenAPI schema check, so
nothing ever opened a session. That was survivable while the schema was global,
but tenancy is enforced by ``group_id`` filters and composite unique
constraints, and neither can be proven against mocks — a missing ``where``
clause looks exactly like a correct one until two accounts share a table. These
fixtures exist so isolation can be tested against Postgres rather than asserted.

The database name always ends in ``_test``, appended if the configured URL
doesn't already say so, because the schema is dropped between tests and
pointing ``DATABASE_URL`` at a development database should not be destructive.

Tests requesting a fixture from this module skip when no server is reachable,
so ``pytest`` still runs for someone without Postgres. ``make test`` supplies
one via Compose.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from synapse.db import models  # noqa: F401 — imported so Base.metadata sees every table
from synapse.db.base import Base

DEFAULT_URL = "postgresql+asyncpg://synapse:synapse@localhost:5432/synapse"


def _with_database(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


def _database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/") or "synapse"


def test_database_url() -> str:
    """The configured URL, forced onto a ``_test`` database."""
    url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    name = _database_name(url)
    if not name.endswith("_test"):
        name = f"{name}_test"
    return _with_database(url, name)


@lru_cache(maxsize=1)
def _bootstrap() -> str:
    """Create the test database if absent. Returns "" on success, else why not.

    Compose points the test service at ``synapse_test`` but only ever creates
    ``synapse``, so the database this connects to genuinely may not exist yet.
    """
    import psycopg

    url = test_database_url()
    admin = _with_database(url, "postgres").replace("postgresql+asyncpg://", "postgresql://", 1)
    name = _database_name(url)
    try:
        with psycopg.connect(admin, connect_timeout=5, autocommit=True) as conn:
            exists = conn.execute(
                "select 1 from pg_database where datname = %s", (name,)
            ).fetchone()
            if not exists:
                conn.execute(f'create database "{name}"')
        target = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        with psycopg.connect(target, connect_timeout=5, autocommit=True) as conn:
            conn.execute("create extension if not exists vector")
    except Exception as exc:  # noqa: BLE001 — any failure means "no database", not a test error
        return f"{type(exc).__name__}: {exc}"
    return ""


@pytest_asyncio.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory over an empty schema, rebuilt for each test."""
    problem = _bootstrap()
    if problem:
        pytest.skip(f"no Postgres at {test_database_url()} ({problem})")

    engine = create_async_engine(test_database_url())
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
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
