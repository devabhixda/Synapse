"""Guards on the private Mem0 internals the sqlite_vec provider registers through.

Mem0 ships no SQLite vector store and no public way to add one, so
``register()`` reaches into three internals: the factory's provider map, the
config model's provider map, and the module path its validator derives from
the provider name. ``mem0ai`` is pinned exactly because of that, but a pin only
stops the version moving by accident — it does not stop someone raising it.
These tests make that upgrade fail here, loudly and with the reason attached,
rather than at runtime as memory silently declining to record anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapse.memory.sqlite_vec_store import (
    SqliteVecConfig,
    SqliteVecStore,
    _provider_configs,
    register,
)


def test_factory_registry_still_exists_and_accepts_the_provider() -> None:
    from mem0.utils.factory import VectorStoreFactory

    register()

    assert isinstance(VectorStoreFactory.provider_to_class, dict)
    assert (
        VectorStoreFactory.provider_to_class["sqlite_vec"]
        == "synapse.memory.sqlite_vec_store.SqliteVecStore"
    )


def test_config_registry_still_exists_and_accepts_the_provider() -> None:
    from mem0.vector_stores.configs import VectorStoreConfig

    register()

    assert _provider_configs(VectorStoreConfig)["sqlite_vec"] == "SqliteVecConfig"


def test_config_validator_resolves_the_provider_to_our_config_class() -> None:
    """Mem0 imports the config class from ``mem0.configs.vector_stores.<provider>``."""
    from mem0.vector_stores.configs import VectorStoreConfig

    register()

    config = VectorStoreConfig(
        provider="sqlite_vec",
        config={"path": "unused.db", "collection_name": "c", "embedding_model_dims": 8},
    )

    assert isinstance(config.config, SqliteVecConfig)


def test_the_factory_builds_our_store(tmp_path: Path) -> None:
    from mem0.utils.factory import VectorStoreFactory
    from mem0.vector_stores.configs import VectorStoreConfig

    register()
    config = VectorStoreConfig(
        provider="sqlite_vec",
        config={
            "path": str(tmp_path / "memory.db"),
            "collection_name": "c",
            "embedding_model_dims": 8,
        },
    )

    store = VectorStoreFactory.create("sqlite_vec", config.config)

    assert isinstance(store, SqliteVecStore)


def test_mem0_can_tell_that_keyword_search_is_implemented() -> None:
    """Mem0 disables hybrid BM25 scoring with only a log warning when a store
    leaves ``keyword_search`` inherited, which is the silent recall regression
    that ruled out Chroma. It must see ours as overridden."""
    from mem0.vector_stores.base import VectorStoreBase

    assert SqliteVecStore.keyword_search is not VectorStoreBase.keyword_search


def test_the_abstract_interface_has_not_grown_a_method_we_do_not_implement() -> None:
    """A Mem0 upgrade adding an abstract method would otherwise surface as a
    TypeError the first time memory is used, not at import."""
    from mem0.vector_stores.base import VectorStoreBase

    missing = {
        name
        for name in getattr(VectorStoreBase, "__abstractmethods__", set())
        if getattr(SqliteVecStore, name, None) is getattr(VectorStoreBase, name, None)
    }

    assert not missing, f"SqliteVecStore does not implement: {sorted(missing)}"
    assert not getattr(SqliteVecStore, "__abstractmethods__", set())


def test_unknown_providers_are_still_rejected() -> None:
    """Registration must not have made the validator permissive."""
    from mem0.vector_stores.configs import VectorStoreConfig

    register()

    with pytest.raises(ValueError):
        VectorStoreConfig(provider="not_a_real_provider", config={})
