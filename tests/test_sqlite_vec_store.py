"""Tests for the SQLite vector store that replaced pgvector.

The store is the one piece of the SQLite move that is new code rather than a
type substitution, and Mem0 reaches it through an interface rather than
through Synapse, so nothing else in this suite would notice if it broke.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapse.memory.sqlite_vec_store import SqliteVecStore


def payload(text: str, user: str, status: str = "active") -> dict[str, object]:
    return {"data": text, "text_lemmatized": text, "user_id": user, "status": status}


@pytest.fixture
def store(tmp_path: Path) -> SqliteVecStore:
    store = SqliteVecStore(
        path=str(tmp_path / "memory.db"), collection_name="c", embedding_model_dims=4
    )
    store.insert(
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0], [1, 0, 0, 0]],
        payloads=[
            payload("gateway timeout error", "acme"),
            payload("deploy pipeline green", "acme"),
            payload("gateway latency spike", "acme"),
            payload("gateway timeout error", "globex"),
        ],
        ids=["a", "b", "c", "d"],
    )
    return store


def test_search_is_ordered_by_similarity(store: SqliteVecStore) -> None:
    results = store.search("gateway", [1, 0, 0, 0], top_k=3, filters={"user_id": "acme"})

    assert [r.id for r in results] == ["a", "c", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score > results[1].score > results[2].score


def test_search_never_crosses_a_user_boundary(store: SqliteVecStore) -> None:
    """The identical vector under another user must not come back."""
    results = store.search("gateway", [1, 0, 0, 0], top_k=10, filters={"user_id": "acme"})

    assert "d" not in [r.id for r in results]


def test_search_with_no_matching_filter_is_empty(store: SqliteVecStore) -> None:
    assert store.search("gateway", [1, 0, 0, 0], filters={"user_id": "nobody"}) == []


def test_scores_are_similarities_not_distances(store: SqliteVecStore) -> None:
    """The base class requires higher to mean more similar, in [0, 1]."""
    results = store.search("x", [1, 0, 0, 0], top_k=10, filters={"user_id": "acme"})

    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_keyword_search_ranks_by_bm25(store: SqliteVecStore) -> None:
    results = store.keyword_search("gateway timeout", top_k=5, filters={"user_id": "acme"})

    assert results is not None
    assert [r.id for r in results][0] == "a"
    # Mem0 discards non-positive scores before normalizing, so the sign of
    # FTS5's negated BM25 has to be flipped back on the way out.
    assert all(r.score > 0 for r in results)


def test_keyword_search_is_scoped_by_filters(store: SqliteVecStore) -> None:
    results = store.keyword_search("gateway timeout", top_k=10, filters={"user_id": "acme"})

    assert results is not None
    assert "d" not in [r.id for r in results]


def test_keyword_search_survives_punctuation(store: SqliteVecStore) -> None:
    """FTS5 operator characters in a query must not blow up the BM25 leg."""
    results = store.keyword_search('gateway "OR" (timeout)*', top_k=5)

    assert results is not None


def test_keyword_search_with_no_hits_is_empty(store: SqliteVecStore) -> None:
    assert store.keyword_search("nonexistentterm", filters={"user_id": "acme"}) == []


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"user_id": "acme"}, {"a", "b", "c"}),
        ({"user_id": {"ne": "acme"}}, {"d"}),
        ({"user_id": {"in": ["acme", "globex"]}}, {"a", "b", "c", "d"}),
        ({"user_id": {"nin": ["acme"]}}, {"d"}),
        ({"data": {"icontains": "GATEWAY"}}, {"a", "c", "d"}),
        ({"data": {"contains": "pipeline"}}, {"b"}),
        ({"status": "*"}, {"a", "b", "c", "d"}),
        ({"$or": [{"user_id": "globex"}, {"data": {"icontains": "pipeline"}}]}, {"b", "d"}),
        ({"$not": [{"user_id": "acme"}]}, {"d"}),
    ],
)
def test_filter_operators(
    store: SqliteVecStore, filters: dict[str, object], expected: set[str]
) -> None:
    assert {r.id for r in store.list(filters=filters)[0]} == expected


def test_list_is_wrapped_the_way_mem0_unpacks_it(store: SqliteVecStore) -> None:
    results = store.list()

    assert isinstance(results, list)
    assert isinstance(results[0], list)


def test_get_returns_none_for_an_unknown_id(store: SqliteVecStore) -> None:
    assert store.get("a") is not None
    assert store.get("nope") is None


def test_update_moves_the_vector_and_the_keyword_index(store: SqliteVecStore) -> None:
    store.update("a", vector=[0, 0, 1, 0], payload=payload("entirely unrelated", "acme"))

    moved = store.search("x", [0, 0, 1, 0], top_k=1, filters={"user_id": "acme"})
    assert [r.id for r in moved] == ["a"]
    # The old text must stop matching, or the FTS index is serving a stale row.
    assert store.keyword_search("timeout", filters={"user_id": "acme"}) == []


def test_reinserting_an_id_does_not_strand_index_rows(store: SqliteVecStore) -> None:
    """Both indexes are rowid-keyed, so a re-insert must reuse the rowid."""
    store.insert(vectors=[[0, 0, 0, 1]], payloads=[payload("replaced text", "acme")], ids=["a"])

    assert store.col_info()["count"] == 4
    assert _index_counts(store) == (4, 4)
    hits = store.keyword_search("replaced", filters={"user_id": "acme"})
    assert hits is not None and [r.id for r in hits] == ["a"]


def test_delete_clears_every_index(store: SqliteVecStore) -> None:
    store.delete("b")

    assert store.col_info()["count"] == 3
    assert _index_counts(store) == (3, 3)
    assert store.get("b") is None


def test_reset_empties_the_collection(store: SqliteVecStore) -> None:
    store.reset()

    assert store.col_info()["count"] == 0
    assert store.list()[0] == []


def test_data_survives_reopening_the_file(tmp_path: Path) -> None:
    path = str(tmp_path / "memory.db")
    first = SqliteVecStore(path=path, collection_name="c", embedding_model_dims=4)
    first.insert(vectors=[[1, 0, 0, 0]], payloads=[payload("persisted", "acme")], ids=["a"])
    first.close()

    reopened = SqliteVecStore(path=path, collection_name="c", embedding_model_dims=4)
    assert reopened.get("a") is not None


def test_numpy_fallback_matches_the_extension(tmp_path: Path) -> None:
    """The fallback exists for interpreters that cannot load extensions, so it
    has to rank identically rather than merely return something."""
    path = str(tmp_path / "memory.db")
    with_vec = SqliteVecStore(path=path, collection_name="c", embedding_model_dims=4)
    if not with_vec._vec_available:
        pytest.skip("sqlite-vec did not load, so there is nothing to compare against")
    with_vec.insert(
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]],
        payloads=[payload(t, "acme") for t in ("one", "two", "three")],
        ids=["a", "b", "c"],
    )

    fallback = SqliteVecStore(path=path, collection_name="c", embedding_model_dims=4)
    fallback._vec_available = False

    query = [1, 0, 0, 0]
    expected = with_vec.search("q", query, top_k=3, filters={"user_id": "acme"})
    actual = fallback.search("q", query, top_k=3, filters={"user_id": "acme"})

    assert [r.id for r in actual] == [r.id for r in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.score == pytest.approx(want.score, abs=1e-6)


def _index_counts(store: SqliteVecStore) -> tuple[int, int]:
    """Row counts of the two derived indexes, which must track the items table."""
    items = store._conn.execute('SELECT COUNT(*) FROM "c"').fetchone()[0]
    fts = store._conn.execute('SELECT COUNT(*) FROM "c_fts"').fetchone()[0]
    return items, fts
