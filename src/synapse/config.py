"""Configuration, including which model server answers each kind of call.

Inference is addressed as two OpenAI-compatible endpoints — one for chat, one
for embeddings — rather than as a named provider. The defaults point at a local
Cactus serving Gemma, which is how the project is meant to be run, but nothing
below the settings knows that: pointing the URLs at a hosted gateway and
supplying its key is the whole of what switching providers involves.

Chat and embeddings are configured separately because they are not always the
same service. A gateway that serves chat may serve no embedding model at all,
leaving embeddings to a second deployment with its own URL, key, and dimension.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


def sqlite_path(database_url: str) -> str:
    """The filesystem path inside a SQLAlchemy SQLite URL.

    A URL with no database part is SQLAlchemy's spelling of an anonymous
    in-memory database, so it maps to the ``sqlite3`` spelling of the same.
    """
    url = make_url(database_url)
    if not url.get_backend_name().startswith("sqlite"):
        raise ValueError(
            "Synapse stores memory in SQLite; database_url must be a sqlite URL, "
            f"got {database_url!r}"
        )
    return url.database or ":memory:"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Synapse"
    environment: str = "development"
    database_url: str = "sqlite+aiosqlite:///./synapse.db"
    chat_base_url: str = "http://localhost:8080/v1"
    embedding_base_url: str = "http://localhost:8080/v1"
    chat_model: str = "google/gemma-4-E2B-it"
    embedding_model: str = "google/gemma-4-E2B-it"
    # A local Cactus ignores the bearer token, so the default is a placeholder
    # rather than a secret. A hosted endpoint checks it.
    chat_api_key: str = "cactus-local"
    embedding_api_key: str = "cactus-local"
    embedding_dimensions: int = 1536
    # Extra completion budget for models that think before answering. Every
    # call site sizes its own budget for the answer it expects, which is all a
    # model like Gemma spends tokens on. A reasoning model bills its private
    # reasoning against the same budget and can exhaust it before writing a
    # single character, so those budgets need to be raised — by a lot, and
    # without changing what each call site asks for. Set this to a few thousand
    # when pointing at one; it costs nothing on a model that doesn't reason,
    # which stops when it is done rather than when it hits the cap.
    chat_reasoning_tokens: int = Field(default=0, ge=0)
    inference_mode: str = "auto"
    consolidation_mode: str = "auto"
    crustdata_api_key: str | None = None
    crustdata_base_url: str = "https://api.crustdata.com"
    crustdata_api_version: str = "2025-11-01"
    enrichment_cache_hours: int = Field(default=24, ge=1)
    procedural_min_occurrences: int = Field(default=2, ge=2)
    memory_hit_threshold: float = Field(default=0.62, ge=0, le=1)
    mem0_collection: str = "synapse_episodes_gemma_1536"
    # An episode is indexed as several vectors, one per chunk of its body, so a
    # long conversation is searchable throughout rather than by its opening
    # alone. This is the size of those chunks; see ``synapse.memory.chunking``.
    memory_chunk_chars: int = Field(default=700, ge=100)
    # Vector hits fetched per search, before they are folded back to one
    # candidate per episode. Several chunks of one long episode can occupy these
    # slots, so this is deliberately larger than the number of distinct episodes
    # a query is expected to need.
    memory_search_top_k: int = Field(default=20, ge=1)

    @property
    def mem0_database_url(self) -> str:
        """The SQLite file the vector store writes to: a sibling of the main one.

        Mem0 opens the file with ``sqlite3`` rather than through SQLAlchemy, so
        it needs a filesystem path where ``database_url`` carries a driver.

        Deliberately not the same file. ``SystemOneService.process`` flushes the
        new episode — taking SQLite's one write lock — and commits only after
        the vector write has returned, so a vector store on that file waits for
        a lock held by the call it is blocking. That is a deadlock, not
        contention: every index write fails with "database is locked" no matter
        how long the busy timeout is, and the episode lands with a NULL
        ``mem0_id`` because indexing is best-effort.
        """
        path = sqlite_path(self.database_url)
        if path == ":memory:":
            return path
        p = Path(path)
        return str(p.with_name(f"{p.stem}.vectors{p.suffix}"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
