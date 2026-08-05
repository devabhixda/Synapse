"""How memory is laid out on disk, and why it is two files rather than one.

The relational schema and the vector store were originally pointed at the same
SQLite file. That looked safe — both writers set WAL and a busy timeout, and a
test hammered them concurrently without a failure. It was not safe, because
that test had each writer commit its own work: neither ever held the write lock
while the other tried to take it.

The real caller does. ``SystemOneService.process`` flushes the episode and only
commits after the vector write returns, so a vector store on the same file
waits on a lock that cannot be released until the call it is blocking finishes.
Every index write failed with "database is locked", and because indexing is
best-effort the episodes were saved anyway with a NULL ``mem0_id`` — a silently
unsearchable memory. Hence the sibling file, and hence
``test_the_vector_store_can_write_while_a_transaction_is_open`` below.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from synapse.config import Settings, sqlite_path
from synapse.db import models
from synapse.db.base import Base
from synapse.db.pragmas import configure_sqlite
from synapse.memory.sqlite_vec_store import SqliteVecStore

WRITES = 100


async def test_the_engine_sets_the_pragmas_two_writers_depend_on(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'synapse.db'}")
    configure_sqlite(engine)
    try:
        async with engine.connect() as conn:
            journal = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
            busy = (await conn.execute(text("PRAGMA busy_timeout"))).scalar()
            keys = (await conn.execute(text("PRAGMA foreign_keys"))).scalar()
    finally:
        await engine.dispose()

    assert journal == "wal"
    assert busy == 5000
    # The schema declares ON DELETE CASCADE, which SQLite ignores by default.
    assert keys == 1


async def test_both_writers_can_hammer_their_files(tmp_path: Path) -> None:
    path = str(tmp_path / "synapse.db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    configure_sqlite(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqliteVecStore(
        path=str(tmp_path / "synapse.vectors.db"),
        collection_name="vec",
        embedding_model_dims=8,
    )

    failures: list[str] = []

    def write_vectors() -> None:
        try:
            for i in range(WRITES):
                store.insert(
                    vectors=[[float(i), 0, 0, 0, 0, 0, 0, 1]],
                    payloads=[{"data": f"m{i}", "text_lemmatized": f"m{i}", "user_id": "acme"}],
                    ids=[f"v{i}"],
                )
        except Exception as exc:  # noqa: BLE001 — the point is to report, not to raise here
            failures.append(f"vector store: {exc!r}")

    async def write_episodes() -> None:
        for i in range(WRITES):
            async with sessions() as session:
                session.add(
                    models.Episode(
                        group_id="acme",
                        kind=models.EpisodeKind.SOURCE,
                        source_type=models.SourceType.SLACK,
                        source_id=f"s{i}",
                        content=f"content {i}",
                        participants=[],
                        details={},
                        occurred_at=datetime.now(UTC),
                    )
                )
                await session.commit()

    writer = threading.Thread(target=write_vectors)
    writer.start()
    try:
        await write_episodes()
    finally:
        writer.join()
        await engine.dispose()

    assert not failures
    assert store.col_info()["count"] == WRITES

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == WRITES
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


async def test_cascade_delete_is_enforced(tmp_path: Path) -> None:
    """Foreign keys are off by default in SQLite, so this would silently orphan."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'synapse.db'}")
    configure_sqlite(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    episode_id = uuid.uuid4()
    async with sessions() as session:
        session.add(
            models.Episode(
                id=episode_id,
                group_id="acme",
                kind=models.EpisodeKind.SOURCE,
                source_type=models.SourceType.SLACK,
                source_id="s1",
                content="content",
                participants=[],
                details={},
                occurred_at=datetime.now(UTC),
            )
        )
        await session.commit()
        session.add(
            models.IntentObservation(
                group_id="acme",
                episode_id=episode_id,
                intent="ask",
                salience=1.0,
                entity_mentions=[],
                tags=[],
                provider="test",
            )
        )
        await session.commit()

        await session.execute(text("DELETE FROM episodes WHERE id = :id"), {"id": episode_id.hex})
        await session.commit()
        remaining = (
            await session.execute(text("SELECT COUNT(*) FROM intent_observations"))
        ).scalar()

    await engine.dispose()
    assert remaining == 0


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("sqlite+aiosqlite:///./synapse.db", "./synapse.db"),
        ("sqlite+aiosqlite:////var/lib/synapse.db", "/var/lib/synapse.db"),
        ("sqlite:///relative.db", "relative.db"),
        ("sqlite+aiosqlite://", ":memory:"),
    ],
)
def test_mem0_gets_a_path_where_sqlalchemy_gets_a_url(url: str, expected: str) -> None:
    """Mem0 opens the file with ``sqlite3``, which does not understand a driver."""
    assert sqlite_path(url) == expected


def test_a_non_sqlite_url_is_rejected_rather_than_silently_mishandled() -> None:
    with pytest.raises(ValueError, match="sqlite"):
        sqlite_path("postgresql+asyncpg://synapse:synapse@localhost:5432/synapse")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("sqlite+aiosqlite:////tmp/synapse.db", "/tmp/synapse.vectors.db"),
        ("sqlite+aiosqlite:///./memory.db", "memory.vectors.db"),
        ("sqlite+aiosqlite:////var/lib/fde/memory.db", "/var/lib/fde/memory.vectors.db"),
        # No extension to preserve, and a directory with a dot in it must not
        # be mistaken for one.
        ("sqlite+aiosqlite:////var/lib/synapse", "/var/lib/synapse.vectors"),
        ("sqlite+aiosqlite:////var/my.data/memory.db", "/var/my.data/memory.vectors.db"),
    ],
)
def test_the_vector_store_gets_its_own_file_beside_the_relational_one(
    url: str, expected: str
) -> None:
    assert Settings(database_url=url).mem0_database_url == expected


def test_an_in_memory_url_stays_in_memory() -> None:
    """Nothing to sit beside, and two anonymous in-memory DBs are already separate."""
    assert Settings(database_url="sqlite+aiosqlite://").mem0_database_url == ":memory:"


async def test_the_vector_store_can_write_while_a_transaction_is_open(
    tmp_path: Path,
) -> None:
    """The regression: index during an uncommitted flush, as `process` does.

    On one shared file this raised "database is locked" after burning the full
    busy timeout. It must not wait, and it must not fail.
    """
    settings = Settings(database_url=f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}")
    engine = create_async_engine(settings.database_url)
    configure_sqlite(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqliteVecStore(
        path=settings.mem0_database_url,
        collection_name="vec",
        embedding_model_dims=8,
    )

    async with sessions() as session:
        session.add(
            models.Episode(
                group_id="acme",
                kind=models.EpisodeKind.SOURCE,
                source_type=models.SourceType.SLACK,
                source_id="s1",
                content="content",
                participants=[],
                details={},
                occurred_at=datetime.now(UTC),
            )
        )
        # Takes the write lock on memory.db and holds it past the vector write.
        await session.flush()

        started = time.monotonic()
        store.insert(
            vectors=[[1.0, 0, 0, 0, 0, 0, 0, 1]],
            payloads=[{"data": "m", "text_lemmatized": "m", "user_id": "acme"}],
            ids=["v1"],
        )
        waited = time.monotonic() - started

        await session.commit()

    await engine.dispose()

    assert store.col_info()["count"] == 1
    # A shared file would have blocked here for the full 5s busy timeout.
    assert waited < 1.0
