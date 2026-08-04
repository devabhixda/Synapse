"""A Mem0 vector store backed by SQLite, so memory needs no external database.

Mem0 has no SQLite provider, and the two it ships that need no server —
Chroma and FAISS — each cost something this project is not willing to pay.
Chroma pulls 389 MB of site-packages for a venv the desktop app builds on
first launch, and neither implements ``keyword_search``, so Mem0 silently
drops BM25 from hybrid scoring and recall regresses with only a log warning.
This provider is one dependency with a 67 KB dylib, and it backs
``keyword_search`` with FTS5 to keep parity with the pgvector store it
replaces.

Three tables per collection. The items table is the source of truth and holds
the payload and the raw embedding; the ``vec0`` and FTS5 tables are indexes
derived from it, and either can be rebuilt by replaying the items table. That
costs one extra copy of each vector, and buys a file that opens correctly
whether or not the interpreter can load the extension — see ``_vec_available``.

Scores are returned as ``max(0, 1 - cosine_distance)``, the conversion the base
class mandates and the one pgvector uses, so ``memory_hit_threshold`` keeps
meaning roughly what it meant before. ``keyword_search`` returns ``-bm25()``:
FTS5 reports the negated Okapi score, and negating it back lands on the 0-20
range Mem0's sigmoid normalization is tuned for.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import threading
from typing import Any

from mem0.vector_stores.base import VectorStoreBase  # type: ignore[import-untyped]
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Mem0's filter grammar, mirrored from the pgvector provider so that a filter
# that worked there keeps working here. The bool is whether the comparison is
# numeric, which decides how the extracted JSON value is cast.
_OPERATORS: dict[str, tuple[str, bool]] = {
    "eq": ("{field} = ?", False),
    "ne": ("{field} != ?", False),
    "gt": ("{field} > ?", True),
    "gte": ("{field} >= ?", True),
    "lt": ("{field} < ?", True),
    "lte": ("{field} <= ?", True),
    "in": ("{field} IN ({placeholders})", False),
    "nin": ("{field} NOT IN ({placeholders})", False),
    "contains": ("{field} LIKE ? ESCAPE '\\'", False),
    "icontains": ("LOWER({field}) LIKE LOWER(?) ESCAPE '\\'", False),
}


class SqliteVecConfig(BaseModel):
    """Config for the sqlite_vec provider.

    ``path`` is the SQLite file. Mem0's config validator fills in a ``/tmp``
    default for any provider whose config declares ``path``, which is why this
    is annotated rather than required — Synapse always passes one.
    """

    path: str = "mem0_sqlite_vec.db"
    collection_name: str = "mem0"
    embedding_model_dims: int = 1536


class OutputData(BaseModel):
    id: str | None
    score: float | None
    payload: dict[str, Any] | None


# Aliased at module scope because the store defines a method named ``list``,
# which shadows the builtin for annotations written inside the class body.
SearchResults = list[OutputData]
ListResults = list[list[OutputData]]
StrList = list[str]


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _json_field(key: str) -> str:
    """A JSON path expression for ``key``, quoted so dots and spaces are safe."""
    escaped = key.replace('"', '""')
    return f"json_extract(payload, '$.\"{escaped}\"')"


def _like_pattern(value: Any) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _comparand(value: Any) -> Any:
    """Render a filter value the way ``json_extract`` will return it.

    SQLite gives back native types, so a JSON ``true`` reads as ``1`` and a
    number as a number. Everything else is compared as text, which is what
    pgvector's ``payload->>`` does.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _build_filters(filters: dict[str, Any] | None) -> tuple[list[str], list[Any]]:
    """Translate Mem0's filter dict into WHERE fragments and bound parameters."""
    conditions: list[str] = []
    params: list[Any] = []

    if not filters:
        return conditions, params

    for key, value in filters.items():
        if key == "$or":
            groups = []
            for sub in value:
                sub_conds, sub_params = _build_filters(sub)
                if sub_conds:
                    groups.append("(" + " AND ".join(sub_conds) + ")")
                    params.extend(sub_params)
            if groups:
                conditions.append("(" + " OR ".join(groups) + ")")
            continue

        if key == "$not":
            groups = []
            for sub in value:
                sub_conds, sub_params = _build_filters(sub)
                if sub_conds:
                    groups.append("(" + " AND ".join(sub_conds) + ")")
                    params.extend(sub_params)
            if groups:
                conditions.append("NOT (" + " OR ".join(groups) + ")")
            continue

        field = _json_field(key)

        if value == "*":
            conditions.append(f"json_type(payload, '$.\"{key}\"') IS NOT NULL")
            continue

        if isinstance(value, dict):
            for op, operand in value.items():
                if op not in _OPERATORS:
                    raise ValueError(f"Unsupported filter operator: {op}")
                template, numeric = _OPERATORS[op]
                cast = f"CAST({field} AS REAL)" if numeric else field
                if op in ("in", "nin"):
                    placeholders = ", ".join("?" for _ in operand)
                    conditions.append(
                        template.format(field=cast, placeholders=placeholders or "NULL")
                    )
                    params.extend(_comparand(item) for item in operand)
                elif op in ("contains", "icontains"):
                    conditions.append(template.format(field=cast))
                    params.append(_like_pattern(operand))
                else:
                    conditions.append(template.format(field=cast))
                    params.append(float(operand) if numeric else _comparand(operand))
            continue

        if isinstance(value, list):
            placeholders = ", ".join("?" for _ in value)
            conditions.append(f"{field} IN ({placeholders or 'NULL'})")
            params.extend(_comparand(item) for item in value)
            continue

        conditions.append(f"{field} = ?")
        params.append(_comparand(value))

    return conditions, params


def _cosine_similarity(query: list[float], matrix: Any) -> Any:
    import numpy as np

    q = np.asarray(query, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    m_norms = np.linalg.norm(matrix, axis=1)
    denom = m_norms * q_norm
    # A zero vector has no direction, so it is no more similar to the query
    # than anything else; call it orthogonal rather than dividing by zero.
    safe = np.where(denom == 0, 1.0, denom)
    sims = (matrix @ q) / safe
    return np.where(denom == 0, 0.0, sims)


class SqliteVecStore(VectorStoreBase):  # type: ignore[misc]
    def __init__(
        self,
        path: str = "mem0_sqlite_vec.db",
        collection_name: str = "mem0",
        embedding_model_dims: int = 1536,
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        # One connection shared across Mem0's worker threads. Writes to a
        # SQLite file serialize regardless, so a lock costs no throughput and
        # saves reasoning about a pool.
        self._lock = threading.RLock()

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # The SQLAlchemy engine writes to this same file, so a writer here can
        # find it locked. Wait rather than failing the call.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._vec_available = self._load_extension()
        self.create_col()

    # -- setup -------------------------------------------------------------

    def _load_extension(self) -> bool:
        """Load sqlite-vec, reporting whether KNN can run in SQL.

        Some interpreters are built without ``enable_load_extension``. That is
        a property of the machine and not something the caller can fix, so it
        degrades to the numpy path instead of raising.
        """
        if not hasattr(self._conn, "enable_load_extension"):
            logger.warning(
                "This interpreter cannot load SQLite extensions; "
                "sqlite-vec is unavailable and vector search will use numpy."
            )
            return False
        try:
            import sqlite_vec  # type: ignore[import-untyped]

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            return True
        except Exception as exc:
            logger.warning(
                "Could not load sqlite-vec (%s); vector search will use numpy.", exc
            )
            return False
        finally:
            try:
                self._conn.enable_load_extension(False)
            except (AttributeError, sqlite3.Error):
                pass

    @property
    def _items(self) -> str:
        return f'"{self.collection_name}"'

    @property
    def _vec(self) -> str:
        return f'"{self.collection_name}_vec"'

    @property
    def _fts(self) -> str:
        return f'"{self.collection_name}_fts"'

    def create_col(self, name: str | None = None, vector_size: int | None = None,
                   distance: str | None = None) -> None:
        """Create the collection's tables.

        The signature carries the base class's arguments, but the collection
        this store manages is fixed at construction, so they are accepted and
        ignored the way the pgvector provider does.
        """
        with self._lock, self._conn:
            self._conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._items} (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )"""
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS "
                f'"{self.collection_name}_memory_id_idx" ON {self._items}(memory_id)'
            )
            if self._vec_available:
                self._conn.execute(
                    f"""CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec} USING vec0(
                        embedding float[{self.embedding_model_dims}] distance_metric=cosine
                    )"""
                )
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._fts} USING fts5(content)"
            )

    # -- helpers -----------------------------------------------------------

    def _rowid(self, memory_id: str) -> int | None:
        row = self._conn.execute(
            f"SELECT rowid FROM {self._items} WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return int(row[0]) if row else None

    @staticmethod
    def _lemmatized(payload: dict[str, Any] | None) -> str:
        """The text Mem0 wants BM25 to match against."""
        if not payload:
            return ""
        return str(payload.get("text_lemmatized") or payload.get("data") or "")

    def _candidate_rowids(self, filters: dict[str, Any] | None) -> list[int] | None:
        """Rowids matching ``filters``, or None when there is nothing to filter."""
        conditions, params = _build_filters(filters)
        if not conditions:
            return None
        sql = f"SELECT rowid FROM {self._items} WHERE " + " AND ".join(conditions)
        return [int(r[0]) for r in self._conn.execute(sql, params).fetchall()]

    def _row_to_output(self, memory_id: str, payload: str, score: float | None) -> OutputData:
        return OutputData(id=memory_id, score=score, payload=json.loads(payload))

    def _reindex(
        self, rowid: int, blob: bytes | None, payload: dict[str, Any] | None
    ) -> None:
        """Point the derived indexes at the current state of one item.

        Both index tables are rowid-keyed and neither supports upsert, so each
        one is cleared and rewritten rather than updated in place.
        """
        if blob is not None and self._vec_available:
            self._conn.execute(f"DELETE FROM {self._vec} WHERE rowid = ?", (rowid,))
            self._conn.execute(
                f"INSERT INTO {self._vec}(rowid, embedding) VALUES (?, ?)", (rowid, blob)
            )
        if payload is not None:
            self._conn.execute(f"DELETE FROM {self._fts} WHERE rowid = ?", (rowid,))
            self._conn.execute(
                f"INSERT INTO {self._fts}(rowid, content) VALUES (?, ?)",
                (rowid, self._lemmatized(payload)),
            )

    # -- writes ------------------------------------------------------------

    def insert(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        payloads = payloads or [{} for _ in vectors]
        if ids is None:
            raise ValueError("sqlite_vec requires explicit ids")

        with self._lock, self._conn:
            for memory_id, vector, payload in zip(ids, vectors, payloads, strict=True):
                blob = _serialize(vector)
                # Reuse the existing rowid on re-insert. INSERT OR REPLACE
                # would allocate a fresh one and strand this memory's rows in
                # the two index tables, which are keyed by rowid.
                rowid = self._rowid(str(memory_id))
                if rowid is None:
                    cursor = self._conn.execute(
                        f"INSERT INTO {self._items} "
                        f"(memory_id, payload, embedding) VALUES (?, ?, ?)",
                        (str(memory_id), json.dumps(payload), blob),
                    )
                    rowid = int(cursor.lastrowid or 0)
                else:
                    self._conn.execute(
                        f"UPDATE {self._items} SET payload = ?, embedding = ? WHERE rowid = ?",
                        (json.dumps(payload), blob, rowid),
                    )
                self._reindex(rowid, blob, payload)

    def update(
        self,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._conn:
            rowid = self._rowid(str(vector_id))
            if rowid is None:
                return
            blob = _serialize(vector) if vector is not None else None
            if blob is not None:
                self._conn.execute(
                    f"UPDATE {self._items} SET embedding = ? WHERE rowid = ?", (blob, rowid)
                )
            if payload is not None:
                self._conn.execute(
                    f"UPDATE {self._items} SET payload = ? WHERE rowid = ?",
                    (json.dumps(payload), rowid),
                )
            self._reindex(rowid, blob, payload)

    def delete(self, vector_id: str) -> None:
        with self._lock, self._conn:
            rowid = self._rowid(str(vector_id))
            if rowid is None:
                return
            self._conn.execute(f"DELETE FROM {self._items} WHERE rowid = ?", (rowid,))
            if self._vec_available:
                self._conn.execute(f"DELETE FROM {self._vec} WHERE rowid = ?", (rowid,))
            self._conn.execute(f"DELETE FROM {self._fts} WHERE rowid = ?", (rowid,))

    # -- reads -------------------------------------------------------------

    def search(
        self,
        query: str,
        vectors: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> SearchResults:
        with self._lock:
            candidates = self._candidate_rowids(filters)
            if candidates is not None and not candidates:
                return []
            if self._vec_available:
                return self._search_vec(vectors, top_k, candidates)
            return self._search_numpy(vectors, top_k, candidates)

    def _search_vec(
        self, vectors: list[float], top_k: int, candidates: list[int] | None
    ) -> SearchResults:
        params: list[Any] = [_serialize(vectors), top_k]
        restrict = ""
        if candidates is not None:
            restrict = " AND v.rowid IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(candidates))
        rows = self._conn.execute(
            f"""SELECT i.memory_id, i.payload, v.distance
                FROM {self._vec} v JOIN {self._items} i ON i.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?{restrict}
                ORDER BY v.distance""",
            params,
        ).fetchall()
        return [
            self._row_to_output(r[0], r[1], max(0.0, 1.0 - float(r[2]))) for r in rows
        ]

    def _search_numpy(
        self, vectors: list[float], top_k: int, candidates: list[int] | None
    ) -> SearchResults:
        import numpy as np

        sql = f"SELECT memory_id, payload, embedding FROM {self._items}"
        params: list[Any] = []
        if candidates is not None:
            sql += " WHERE rowid IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(candidates))
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        matrix = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32).reshape(
            len(rows), -1
        )
        sims = _cosine_similarity(vectors, matrix)
        order = np.argsort(-sims)[:top_k]
        return [
            self._row_to_output(rows[i][0], rows[i][1], max(0.0, float(sims[i])))
            for i in order
        ]

    def keyword_search(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> SearchResults | None:
        try:
            with self._lock:
                match = _fts_query(query)
                if not match:
                    return []
                candidates = self._candidate_rowids(filters)
                if candidates is not None and not candidates:
                    return []
                params: list[Any] = [match]
                restrict = ""
                if candidates is not None:
                    restrict = " WHERE m.rid IN (SELECT value FROM json_each(?))"
                    params.append(json.dumps(candidates))
                params.append(top_k)
                # FTS5 resolves MATCH and bm25() against the table name, not
                # against a join alias, so the match runs in a subquery and the
                # payload is joined onto its result.
                rows = self._conn.execute(
                    f"""SELECT i.memory_id, i.payload, m.rank FROM (
                            SELECT rowid AS rid, bm25({self._fts}) AS rank
                            FROM {self._fts} WHERE {self._fts} MATCH ?
                        ) m JOIN {self._items} i ON i.rowid = m.rid
                        {restrict}
                        ORDER BY m.rank LIMIT ?""",
                    params,
                ).fetchall()
            # FTS5 reports the negated Okapi score, so more negative is a
            # better match. Mem0 discards anything <= 0, so flip the sign.
            return [self._row_to_output(r[0], r[1], -float(r[2])) for r in rows]
        except Exception as exc:
            logger.debug("Keyword search failed: %s", exc)
            return None

    def get(self, vector_id: str) -> OutputData | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT memory_id, payload FROM {self._items} WHERE memory_id = ?",
                (str(vector_id),),
            ).fetchone()
        if not row:
            return None
        return self._row_to_output(row[0], row[1], None)

    def list(
        self, filters: dict[str, Any] | None = None, top_k: int | None = 100
    ) -> ListResults:
        """List memories.

        Wrapped in an outer list because that is the shape Mem0 unpacks, the
        same as the pgvector provider returns.
        """
        with self._lock:
            conditions, params = _build_filters(filters)
            sql = f"SELECT memory_id, payload FROM {self._items}"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY rowid"
            if top_k is not None:
                sql += " LIMIT ?"
                params = [*params, top_k]
            rows = self._conn.execute(sql, params).fetchall()
        return [[self._row_to_output(r[0], r[1], None) for r in rows]]

    # -- collection management --------------------------------------------

    def list_cols(self) -> StrList:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return [r[0] for r in rows]

    def col_info(self) -> dict[str, Any]:
        with self._lock:
            count = self._conn.execute(f"SELECT COUNT(*) FROM {self._items}").fetchone()[0]
        return {
            "name": self.collection_name,
            "count": int(count),
            "dimensions": self.embedding_model_dims,
            "backend": "sqlite-vec" if self._vec_available else "numpy",
        }

    def delete_col(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(f"DROP TABLE IF EXISTS {self._fts}")
            if self._vec_available:
                self._conn.execute(f"DROP TABLE IF EXISTS {self._vec}")
            self._conn.execute(f"DROP TABLE IF EXISTS {self._items}")

    def reset(self) -> None:
        self.delete_col()
        self.create_col()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _fts_query(query: str) -> str:
    """Turn free text into an FTS5 MATCH expression.

    Every token is quoted so that punctuation Mem0 leaves in a lemmatized
    query is read as text rather than as FTS5 operator syntax, which would
    otherwise raise and drop the whole BM25 leg of the search.

    Tokens are joined with OR, where pgvector's ``plainto_tsquery`` joins with
    AND. AND means one absent term zeroes the keyword signal for the whole
    query, which on a lemmatized ten-word question is the normal case rather
    than the exception. OR lets every term contribute and leaves the
    discrimination to BM25, which is what Mem0 uses the score for.
    """
    tokens = [token for token in "".join(
        ch if ch.isalnum() or ch.isspace() else " " for ch in query
    ).split() if token]
    return " OR ".join(f'"{token}"' for token in tokens)


def _provider_configs(config_class: type) -> dict[str, str]:
    """Mem0's provider-to-config-class map, reached through Pydantic.

    ``_provider_configs`` is a Pydantic private attribute, so reading it off
    the class yields the ``ModelPrivateAttr`` descriptor rather than the dict
    every instance is seeded from. The dict is shared by reference, so
    mutating the default registers the provider for instances not yet built.
    """
    attr = config_class._provider_configs  # type: ignore[attr-defined]
    mapping = getattr(attr, "default", attr)
    if not isinstance(mapping, dict):
        raise TypeError(
            f"Expected Mem0's _provider_configs to be a dict, got {type(mapping).__name__}"
        )
    return mapping


def register() -> None:
    """Make ``sqlite_vec`` selectable as a Mem0 vector store provider.

    Mem0 has two registries that both reject unknown providers, and its config
    validator imports the config class from ``mem0.configs.vector_stores.
    <provider>``. Rather than fork Mem0 to add a module there, this publishes
    the current module under that name. ``tests/test_sqlite_vec_registry.py``
    fails if any of the three assumptions stops holding.
    """
    import sys

    from mem0.utils.factory import VectorStoreFactory  # type: ignore[import-untyped]
    from mem0.vector_stores.configs import (  # type: ignore[import-untyped]
        VectorStoreConfig,
    )

    VectorStoreFactory.provider_to_class["sqlite_vec"] = (
        "synapse.memory.sqlite_vec_store.SqliteVecStore"
    )
    _provider_configs(VectorStoreConfig)["sqlite_vec"] = "SqliteVecConfig"

    module_name = "mem0.configs.vector_stores.sqlite_vec"
    if module_name not in sys.modules:
        sys.modules[module_name] = sys.modules[__name__]
        import mem0.configs.vector_stores as configs  # type: ignore[import-untyped]

        configs.sqlite_vec = sys.modules[__name__]


__all__ = ["SqliteVecConfig", "SqliteVecStore", "register"]
