"""The relational schema and the vector store share one SQLite file.

Mem0 opens that file on its own ``sqlite3`` connection while SQLAlchemy holds
another, so there are genuinely two writers. SQLite serializes them, but only
if WAL and ``busy_timeout`` are set — without them the second writer gets an
immediate "database is locked" rather than waiting its turn, and the failure
only shows up under concurrency, which is to say in front of a user.
"""

from __future__ import annotations

import sqlite3
import threading
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


async def test_both_writers_can_hammer_the_same_file(tmp_path: Path) -> None:
    path = str(tmp_path / "synapse.db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    configure_sqlite(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqliteVecStore(path=path, collection_name="vec", embedding_model_dims=8)

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


def test_the_default_settings_point_both_layers_at_one_file() -> None:
    settings = Settings(database_url="sqlite+aiosqlite:////tmp/synapse.db")

    assert settings.mem0_database_url == "/tmp/synapse.db"
