"""Configuration, including which model server answers each kind of call.

Inference is addressed as two OpenAI-compatible endpoints — one for chat, one
for embeddings — rather than as a named provider. Nothing below the settings
knows which server answers: a URL, a model name and a key are the whole of
what choosing one involves. There is no default model; set both.

Chat and embeddings are configured separately because they are not always the
same service. A gateway that serves chat may serve no embedding model at all,
leaving embeddings to a second deployment with its own URL, key, and dimension.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

# The kind an extracted entity gets when the model names one outside the
# configured list; always allowed, whatever the list.
FALLBACK_ENTITY_KIND = "Concept"

DEFAULT_ENTITY_KINDS: tuple[str, ...] = (
    "Person",
    "Team",
    "Company",
    "Product",
    "Project",
    "System",
    "Repository",
    "Decision",
    "Task",
    "Blocker",
    "Concept",
)

type PredicateCardinality = Literal["many", "one_per_subject", "one_per_object"]
"""How many current edges a relationship predicate admits.

``many``: edges accumulate, and asserting one closes nothing (a person likes
many things). ``one_per_subject``: a subject has one current object, so a new
assertion closes the subject's previous edge (a person lives in one place at a
time). ``one_per_object``: an object has one current subject (a thing has one
owner at a time; ``owns`` points from owner to thing).
"""

# The predicate an extracted relationship gets when the model names one outside
# the configured set; always allowed, and always ``many``, because it absorbs
# unrelated relationships that must not retire one another.
FALLBACK_PREDICATE = "related_to"

DEFAULT_RELATIONSHIP_PREDICATES: dict[str, PredicateCardinality] = {
    "owns": "one_per_object",
    "assigned_to": "one_per_subject",
    "blocks": "many",
    "resolves": "many",
    "decides": "many",
    "proposes": "many",
    "requests": "many",
    "updates": "many",
    "implements": "many",
    "uses": "many",
    "part_of": "many",
    "affects": "many",
    "depends_on": "many",
    "related_to": "many",
}


def snake_case(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


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
    chat_model: str = ""
    embedding_model: str = ""
    # A local server that doesn't check keys accepts the empty default; a
    # hosted endpoint needs a real one.
    chat_api_key: str = ""
    embedding_api_key: str = ""
    # A second key for the same endpoint, used only when the primary one is out
    # of quota (HTTP 429 / RESOURCE_EXHAUSTED): a free-tier key first, and a
    # paid one once it runs dry. After a quota error the primary key is skipped
    # for ``fallback_key_cooldown`` seconds and then tried again, so a
    # per-minute limit costs a minute on the fallback rather than the rest of
    # the day. Unset, a quota error fails the call exactly as before.
    chat_fallback_api_key: str | None = None
    embedding_fallback_api_key: str | None = None
    fallback_key_cooldown: float = Field(default=60, ge=0)
    embedding_dimensions: int = 1536
    # Extra completion budget for models that think before answering. Every
    # call site sizes its own budget for the answer it expects, which is all a
    # model that answers directly spends tokens on. A reasoning model bills its private
    # reasoning against the same budget and can exhaust it before writing a
    # single character, so those budgets need to be raised — by a lot, and
    # without changing what each call site asks for. Set this to a few thousand
    # when pointing at one; it costs nothing on a model that doesn't reason,
    # which stops when it is done rather than when it hits the cap.
    chat_reasoning_tokens: int = Field(default=0, ge=0)
    # Seconds System 1 waits to label one conversation. Eight is plenty for a
    # model that answers directly; a reasoning model on a long conversation
    # (a busy channel's whole day) can think past it, and a timeout costs the
    # episode its model label (it falls back to the rule-based one).
    chat_classify_timeout: float = Field(default=8, gt=0)
    inference_mode: str = "auto"
    consolidation_mode: str = "auto"
    crustdata_api_key: str | None = None
    crustdata_base_url: str = "https://api.crustdata.com"
    crustdata_api_version: str = "2025-11-01"
    enrichment_cache_hours: int = Field(default=24, ge=1)
    procedural_min_occurrences: int = Field(default=2, ge=2)
    memory_hit_threshold: float = Field(default=0.62, ge=0, le=1)
    # One collection per embedding model: vectors from two models can't be
    # compared, so a new model needs a new collection and a re-index.
    mem0_collection: str = "synapse_episodes"
    # An episode is indexed as several vectors, one per chunk of its body, so a
    # long conversation is searchable throughout rather than by its opening
    # alone. This is the size of those chunks; see ``synapse.memory.chunking``.
    memory_chunk_chars: int = Field(default=700, ge=100)
    # Vector hits fetched per search, before they are folded back to one
    # candidate per episode. Several chunks of one long episode can occupy these
    # slots, so this is deliberately larger than the number of distinct episodes
    # a query is expected to need.
    memory_search_top_k: int = Field(default=20, ge=1)
    # Writer mode: besides indexing each episode's chunks verbatim, have Mem0
    # extract standalone facts from it with the chat model ("Asha moved to Pune
    # in March 2026") and index those too, tagged ``memory_kind: fact``. Off by
    # default; it costs one extra chat call and a few embeddings per episode.
    # See ``Mem0EpisodeStore.extract_facts``.
    mem0_writer_mode: bool = False
    # Extra guidance for that extraction, added after Synapse's own (which
    # explains that the input is a dated transcript). Mem0 gives it the highest
    # priority, e.g. "Ignore small talk; keep plans, preferences and contacts."
    mem0_fact_instructions: str | None = None
    # How many earlier episodes the extraction sees as context. Mem0 would show
    # the last ten; for Synapse each is a whole conversation, so that multiplies
    # the prompt by ten. The related facts it already holds are shown either
    # way. Held in memory, so it starts empty after a restart.
    mem0_fact_context_episodes: int = Field(default=0, ge=0, le=10)
    # The entity kinds the model may assign when it extracts the graph. The
    # default suits workplace conversations; a host with other material (a
    # personal assistant, say) names its own. A kind the model invents outside
    # this list is stored as ``Concept``, which is therefore always allowed.
    entity_kinds: list[str] = Field(default_factory=lambda: list(DEFAULT_ENTITY_KINDS))
    # The relationship predicates extraction may use, and how many current
    # edges each admits, which is what decides supersession (see
    # ``PredicateCardinality``). A list means every predicate accumulates. In
    # the environment, JSON: '{"lives_in": "one_per_subject", "likes": "many"}'.
    # A predicate the model invents outside this set is stored as
    # ``related_to``, which is therefore always present and always ``many``.
    relationship_predicates: dict[str, PredicateCardinality] = Field(
        default_factory=lambda: dict(DEFAULT_RELATIONSHIP_PREDICATES)
    )

    @field_validator("entity_kinds")
    @classmethod
    def _normalize_entity_kinds(cls, kinds: list[str]) -> list[str]:
        # Kinds match case-insensitively, so two spellings of one kind would
        # split its entities across two keys; keep the first, and spell the
        # fallback kind exactly as the rest of the code expects it.
        unique: dict[str, str] = {}
        for kind in kinds:
            cleaned = " ".join(kind.split())
            if not cleaned:
                raise ValueError("entity kinds must be non-empty")
            folded = cleaned.casefold()
            if folded == FALLBACK_ENTITY_KIND.casefold():
                cleaned = FALLBACK_ENTITY_KIND
            unique.setdefault(folded, cleaned)
        unique.setdefault(FALLBACK_ENTITY_KIND.casefold(), FALLBACK_ENTITY_KIND)
        return list(unique.values())

    @field_validator("relationship_predicates", mode="before")
    @classmethod
    def _predicates_from_list(cls, value: Any) -> Any:
        if isinstance(value, list | tuple | set | frozenset):
            return {str(predicate): "many" for predicate in value}
        return value

    @field_validator("relationship_predicates")
    @classmethod
    def _normalize_predicates(
        cls, predicates: dict[str, PredicateCardinality]
    ) -> dict[str, PredicateCardinality]:
        normalized: dict[str, PredicateCardinality] = {}
        for predicate, cardinality in predicates.items():
            name = snake_case(predicate)
            if not name:
                raise ValueError(f"relationship predicate {predicate!r} is empty once normalized")
            if normalized.get(name, cardinality) != cardinality:
                raise ValueError(f"relationship predicate {name!r} is given two cardinalities")
            normalized[name] = cardinality
        if normalized.setdefault(FALLBACK_PREDICATE, "many") != "many":
            raise ValueError(
                f"{FALLBACK_PREDICATE!r} catches every unknown predicate, so it must be 'many'"
            )
        return normalized

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
