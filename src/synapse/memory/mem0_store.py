import asyncio
import copy
import json
from datetime import UTC, datetime
from typing import Any

from mem0 import Memory  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from synapse.config import Settings, get_settings
from synapse.domain.models import SourceEvent
from synapse.inference.chat import chat_completion, message_content, parse_json_object
from synapse.inference.keys import with_fallback_key_sync
from synapse.memory.chunking import chunk_text
from synapse.memory.sqlite_vec_store import register as register_sqlite_vec

register_sqlite_vec()

# Each pass of a group purge clears one page of the vector store. The bound only
# exists so a store that reports memories it will not delete fails loudly rather
# than spinning; at 100 memories a page this covers far more than any real
# account holds.
_MAX_PURGE_PAGES = 500


# Fact extraction is told this, before any instructions the host adds. Mem0's
# prompt expects a chat between a user and an assistant, dated today; what it
# gets here is a stored conversation between named people, dated when it
# happened.
FACT_INSTRUCTIONS = (
    "The new message is a stored conversation transcript, not a message to you. "
    "Its first line gives the date the conversation took place: use that date as "
    "the Observation Date when resolving relative times such as 'yesterday' or "
    "'next Friday'. Its other lines read 'Speaker: text'; attribute each fact to "
    "the speaker by name."
)

# Facts are written under this mem0 ``run_id`` as well as the group's
# ``user_id``. Mem0 shows the extraction the memories it already holds that are
# nearest the new conversation, filtered by those ids only; without a run of
# their own that would be the conversation's own chunks, and the model would
# skip facts as already known.
FACT_RUN_ID = "synapse-facts"


class _WithFallbackKey:
    """One of Mem0's API clients, with a second key for when the first is out of quota.

    Mem0 builds its embedder and LLM from a single key each and calls them
    directly, so the fallback has to sit in front of those objects: this one
    forwards each call to the copy holding whichever key
    ``synapse.inference.keys`` prefers. The copy differs only in its key, so the
    model, URL and dimensions stay in step.
    """

    def __init__(
        self, primary: Any, *, primary_key: str, fallback_key: str, cooldown: float, purpose: str
    ) -> None:
        secondary = copy.copy(primary)
        secondary.client = primary.client.with_options(api_key=fallback_key)
        self._primary = primary
        self._primary_key = primary_key
        self._fallback_key = fallback_key
        self._cooldown = cooldown
        self._purpose = purpose
        self._by_key = {primary_key: primary, fallback_key: secondary}
        self.config = primary.config

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return with_fallback_key_sync(
            self._primary_key,
            self._fallback_key,
            lambda key: getattr(self._by_key[key], method)(*args, **kwargs),
            cooldown=self._cooldown,
            purpose=self._purpose,
        )

    def __getattr__(self, name: str) -> Any:
        # Anything Mem0 reads other than the calls wrapped below.
        return getattr(self.__dict__["_primary"], name)


class FallbackKeyEmbedder(_WithFallbackKey):
    def embed(self, text: Any, memory_action: Any = None) -> Any:
        return self._call("embed", text, memory_action)

    def embed_batch(self, texts: Any, memory_action: Any = "add") -> Any:
        return self._call("embed_batch", texts, memory_action)


class FallbackKeyLLM(_WithFallbackKey):
    def generate_response(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("generate_response", *args, **kwargs)


SUPERSESSION_PROMPT = """You maintain a memory of facts taken from one person's
conversations. Each NEW fact comes with the EXISTING facts most similar to it. Decide
which pairs conflict: they cannot both be true now, because one corrects, updates,
changes or replaces the other (a corrected name, date or detail; a changed plan;
a move; a new job; a changed preference). Facts that are merely related, add detail,
or can both be true do not conflict. Do not decide which one is newer.
Return exactly one JSON object: {"conflicts": [{"new": <id>, "existing": <id>}]},
listing only conflicting pairs, or {"conflicts": []}. Output JSON only."""


class _Conflict(BaseModel):
    new: int
    existing: int


class _Conflicts(BaseModel):
    conflicts: list[_Conflict] = Field(default_factory=list)


def _said_at(payload: dict[str, Any]) -> datetime:
    """When a fact was said: its conversation's date, not when it was stored."""
    value = datetime.fromisoformat(str(payload["occurred_at"]))
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def fact_message(event: SourceEvent) -> str:
    """The conversation as fact extraction reads it: dated, then the transcript."""
    return (
        f"Conversation date: {event.occurred_at.date().isoformat()} "
        f"({event.source_type.value})\n{event.content}"
    )


class Mem0EpisodeStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._memory: Memory | None = None

    def _get_memory(self) -> Memory:
        if self._memory is None:
            settings = self.settings
            self._memory = Memory.from_config(
                {
                    "version": "v1.1",
                    "history_db_path": ":memory:",
                    "vector_store": {
                        "provider": "sqlite_vec",
                        "config": {
                            "path": settings.mem0_database_url,
                            "collection_name": settings.mem0_collection,
                            "embedding_model_dims": settings.embedding_dimensions,
                        },
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": settings.embedding_model,
                            "api_key": settings.embedding_api_key,
                            "openai_base_url": settings.embedding_base_url,
                            "embedding_dims": settings.embedding_dimensions,
                        },
                    },
                    # Only writer mode reaches the LLM: every other write is
                    # ``infer=False`` and search never calls it.
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": settings.chat_model,
                            "api_key": settings.chat_api_key,
                            "openai_base_url": settings.chat_base_url,
                            "max_tokens": 4000 + settings.chat_reasoning_tokens,
                        },
                    },
                }
            )
            memory = self._memory
            embedding_fallback = settings.embedding_fallback_api_key
            if embedding_fallback and embedding_fallback != settings.embedding_api_key:
                memory.embedding_model = FallbackKeyEmbedder(
                    memory.embedding_model,
                    primary_key=settings.embedding_api_key,
                    fallback_key=embedding_fallback,
                    cooldown=settings.fallback_key_cooldown,
                    purpose="embedding",
                )
            chat_fallback = settings.chat_fallback_api_key
            if chat_fallback and chat_fallback != settings.chat_api_key:
                memory.llm = FallbackKeyLLM(
                    memory.llm,
                    primary_key=settings.chat_api_key,
                    fallback_key=chat_fallback,
                    cooldown=settings.fallback_key_cooldown,
                    purpose="chat",
                )
            # Mem0 asks for ten; see ``Settings.mem0_fact_context_episodes``.
            context = settings.mem0_fact_context_episodes
            last_messages = memory.db.get_last_messages
            memory.db.get_last_messages = lambda scope, limit=10: (
                last_messages(scope, limit=min(limit, context)) if context else []
            )
        return self._memory

    @property
    def writes_facts(self) -> bool:
        return self.settings.mem0_writer_mode

    async def extract_facts(
        self, *, episode_id: str, group_id: str, event: SourceEvent
    ) -> list[str]:
        """Writer mode: index the standalone facts the chat model reads off ``event``.

        One chat call reads the conversation, alongside the group's facts
        nearest to it, and returns new facts; each is embedded and stored with
        ``memory_kind: fact`` and the episode's id, so recall folds a fact hit
        back to the conversation it came from and cites that. Returns the new
        facts' memory ids (none when writer mode is off).

        Mem0 2.0's extraction only ever adds. It drops a fact it already holds
        word for word, and never edits or deletes a memory, so the source
        chunks and recall's write-backs are not at risk from it. Retiring a fact
        that stops being true is ``settle_facts``'s job.
        """
        if not self.writes_facts or not (event.content or "").strip():
            return []
        instructions = FACT_INSTRUCTIONS
        if self.settings.mem0_fact_instructions:
            instructions = f"{instructions}\n{self.settings.mem0_fact_instructions}"
        result = await asyncio.to_thread(
            self._get_memory().add,
            [{"role": "user", "content": fact_message(event)}],
            user_id=group_id,
            run_id=FACT_RUN_ID,
            metadata={
                "episode_id": episode_id,
                "group_id": group_id,
                "memory_kind": "fact",
                "source_type": event.source_type.value,
                "conversation_id": event.conversation_id,
                "occurred_at": event.occurred_at.isoformat(),
                "status": "active",
            },
            infer=True,
            prompt=instructions,
        )
        return [
            str(item["id"])
            for item in result.get("results", [])
            if item.get("event", "ADD") == "ADD"
        ]

    async def add_episode(
        self, *, episode_id: str, group_id: str, summary: str, event: SourceEvent
    ) -> list[str]:
        """Index one episode as a vector per chunk of its body.

        The body is what gets embedded, not ``summary``: a summary is short by
        construction, and for a long conversation it describes the opening and
        nothing else. ``summary`` is still the fallback for an event with no
        content of its own.

        Every chunk carries the same ``episode_id``, so retrieval can fold the
        hits back to the conversation they came from. Returns every chunk's
        memory id, in order: the caller keeps them so the episode's vectors can
        be replaced when its body changes.
        """
        text = event.content if (event.content or "").strip() else summary
        chunks = chunk_text(text, self.settings.memory_chunk_chars) or [summary]

        ids: list[str] = []
        for index, chunk in enumerate(chunks):
            result = await asyncio.to_thread(
                self._get_memory().add,
                chunk,
                user_id=group_id,
                metadata={
                    "episode_id": episode_id,
                    "group_id": group_id,
                    "memory_kind": "source",
                    "source_type": event.source_type.value,
                    "conversation_id": event.conversation_id,
                    "occurred_at": event.occurred_at.isoformat(),
                    "status": "active",
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
                infer=False,
            )
            results = result.get("results", [])
            if not results:
                raise RuntimeError("Mem0 did not return an indexed memory")
            ids.append(str(results[0]["id"]))
        return ids

    async def forget_memories(self, memory_ids: list[str]) -> None:
        """Drop specific vectors, for replacing one episode's index in place.

        Deleting is best-effort per id: a vector that is already gone is the
        state the caller wanted, and one stubborn id must not strand the rest.
        """
        memory = self._get_memory()
        for memory_id in memory_ids:
            try:
                await asyncio.to_thread(memory.delete, memory_id)
            except Exception:  # noqa: BLE001 — already absent is success enough
                continue

    async def search(self, query: str, group_id: str) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._get_memory().search,
            query,
            filters={"user_id": group_id, "status": "active"},
            top_k=self.settings.memory_search_top_k,
            threshold=self.settings.memory_hit_threshold,
            explain=True,
        )
        return list(result.get("results", []))

    async def forget_group(self, group_id: str) -> None:
        """Drop every vector held for ``group_id``.

        Re-indexing writes new vectors but cannot recognise the ones it
        replaces, so without this a rebuild leaves the old ones behind to keep
        matching and to crowd out the new. The episodes themselves are untouched
        — they are the durable copy, and the index is derived from them.

        Mem0's ``delete_all`` deletes a page at a time: it lists the group's
        memories with the store's default limit and deletes what came back, so a
        single call empties a group only if the group is small. Called once
        against a real account it silently leaves the remainder behind, which is
        worse than not clearing at all — the survivors are exactly the stale
        vectors a rebuild was meant to remove. So it is called until the group
        reads empty.
        """
        memory = self._get_memory()

        def _purge() -> None:
            for _ in range(_MAX_PURGE_PAGES):
                remaining = memory.vector_store.list(
                    filters={"user_id": group_id}, top_k=1
                )[0]
                if not remaining:
                    return
                memory.delete_all(user_id=group_id)
            raise RuntimeError(
                f"could not empty the index for {group_id}: still not clear after "
                f"{_MAX_PURGE_PAGES} passes"
            )

        await asyncio.to_thread(_purge)

    async def add_retrieval_episode(
        self,
        *,
        episode_id: str,
        group_id: str,
        answer: str,
        query: str,
        occurred_at: str,
    ) -> str:
        result = await asyncio.to_thread(
            self._get_memory().add,
            answer,
            user_id=group_id,
            metadata={
                "episode_id": episode_id,
                "group_id": group_id,
                "memory_kind": "retrieval",
                "source_type": "recall",
                "conversation_id": "recall",
                "query": query,
                "occurred_at": occurred_at,
                "status": "active",
            },
            infer=False,
        )
        results = result.get("results", [])
        if not results:
            raise RuntimeError("Mem0 did not return an indexed retrieval memory")
        return str(results[0]["id"])

    async def release_facts(self, fact_ids: list[str]) -> list[str]:
        """Reinstate the facts that ``fact_ids`` retired; returns their ids.

        Called before those facts are deleted, when their conversation is read
        again: a fact retired by one that no longer exists would otherwise stay
        retired with nothing current in its place. The reinstated facts are
        settled again with the replacements (``settle_facts``), which retires
        them anew if the conversation still contradicts them.
        """
        if not fact_ids:
            return []

        def _release() -> list[str]:
            store = self._get_memory().vector_store
            rows = store.list(
                filters={"memory_kind": "fact", "superseded_by": {"in": list(fact_ids)}},
                top_k=None,
            )[0]
            released: list[str] = []
            for row in rows:
                payload = {
                    key: value
                    for key, value in (row.payload or {}).items()
                    if key not in ("superseded_by", "invalid_at")
                }
                store.update(vector_id=row.id, payload={**payload, "status": "active"})
                released.append(str(row.id))
            return released

        return await asyncio.to_thread(_release)

    async def settle_facts(self, *, group_id: str, fact_ids: list[str]) -> int:
        """Retire the current facts that ``fact_ids`` contradict, or the other way round.

        Mem0 only adds, so without this a correction would sit beside the fact
        it corrects. Each fact is compared with its nearest current facts from
        other conversations in the group, and one chat call, for all of them at
        once, names the pairs that cannot both be true. The model judges only
        the conflict; when each was said decides which one gives way, as in the
        graph: the older is retired as of the newer, so a conversation imported
        late cannot retire a fact stated after it — its own fact is the one
        retired. Retiring keeps the fact, marked ``status: superseded`` with
        ``invalid_at`` and ``superseded_by``; search reads current facts only.

        Returns how many facts were retired. Raises if the check fails, so the
        caller can try it again; it is safe to repeat.
        """
        if not fact_ids or self.settings.mem0_supersession_candidates == 0:
            return 0
        new, existing, pairs = await asyncio.to_thread(
            self._conflict_candidates, group_id, fact_ids
        )
        if not pairs:
            return 0
        payload = {
            "new": [
                {
                    "id": index,
                    "fact": fact["data"],
                    "said_on": _said_at(fact).date().isoformat(),
                    "compare_with": sorted(e for n, e in pairs if n == index),
                }
                for index, (_, fact) in enumerate(new)
                if any(n == index for n, _ in pairs)
            ],
            "existing": [
                {"id": index, "fact": fact["data"], "said_on": _said_at(fact).date().isoformat()}
                for index, (_, fact) in enumerate(existing)
            ],
        }
        body = await chat_completion(
            self.settings,
            [
                {"role": "system", "content": SUPERSESSION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=40 + 16 * len(pairs),
            timeout=45,
        )
        decided = _Conflicts.model_validate(parse_json_object(message_content(body)))
        conflicts = {(c.new, c.existing) for c in decided.conflicts} & pairs
        return await asyncio.to_thread(self._retire_conflicts, new, existing, conflicts)

    def _conflict_candidates(
        self, group_id: str, fact_ids: list[str]
    ) -> tuple[
        list[tuple[str, dict[str, Any]]],
        list[tuple[str, dict[str, Any]]],
        set[tuple[int, int]],
    ]:
        memory = self._get_memory()
        store = memory.vector_store
        new: list[tuple[str, dict[str, Any]]] = []
        for fact_id in dict.fromkeys(fact_ids):
            row = store.get(fact_id)
            fact = row.payload if row is not None else None
            if (
                fact
                and fact.get("memory_kind") == "fact"
                and fact.get("group_id") == group_id
                and fact.get("status", "active") == "active"
                and fact.get("data")
            ):
                new.append((str(fact_id), fact))
        if not new:
            return [], [], set()

        vectors = memory.embedding_model.embed_batch([fact["data"] for _, fact in new], "search")
        limit = self.settings.mem0_supersession_candidates
        threshold = self.settings.mem0_supersession_threshold
        existing: list[tuple[str, dict[str, Any]]] = []
        position: dict[str, int] = {}
        pairs: set[tuple[int, int]] = set()
        judged: set[frozenset[str]] = set()
        for index, ((fact_id, fact), vector) in enumerate(zip(new, vectors, strict=True)):
            hits = store.search(
                query=fact["data"],
                vectors=vector,
                top_k=limit,
                filters={
                    "group_id": group_id,
                    "memory_kind": "fact",
                    "status": "active",
                    # A conversation's own facts were extracted together, and
                    # Mem0 already reconciled them against each other.
                    "episode_id": {"ne": fact.get("episode_id")},
                },
            )
            for hit in hits:
                pair = frozenset({fact_id, str(hit.id)})
                if (hit.score or 0.0) < threshold or pair in judged or not hit.payload:
                    continue
                judged.add(pair)
                if str(hit.id) not in position:
                    position[str(hit.id)] = len(existing)
                    existing.append((str(hit.id), hit.payload))
                pairs.add((index, position[str(hit.id)]))
        return new, existing, pairs

    def _retire_conflicts(
        self,
        new: list[tuple[str, dict[str, Any]]],
        existing: list[tuple[str, dict[str, Any]]],
        conflicts: set[tuple[int, int]],
    ) -> int:
        store = self._get_memory().vector_store
        retired = 0
        for n, e in sorted(conflicts):
            (new_id, new_fact), (old_id, old_fact) = new[n], existing[e]
            # Ties go to the fact being settled: it was read later.
            if _said_at(old_fact) <= _said_at(new_fact):
                loser, winner, winner_fact = old_id, new_id, new_fact
            else:
                loser, winner, winner_fact = new_id, old_id, old_fact
            row = store.get(loser)
            if row is None or not row.payload:
                continue
            current = row.payload
            ends = _said_at(winner_fact)
            if current.get("status", "active") != "active":
                # Already retired: keep the earliest successor, as the graph does.
                if "invalid_at" in current and datetime.fromisoformat(
                    str(current["invalid_at"])
                ) <= ends:
                    continue
            else:
                retired += 1
            store.update(
                vector_id=loser,
                payload={
                    **current,
                    "status": "superseded",
                    "invalid_at": ends.isoformat(),
                    "superseded_by": winner,
                },
            )
        return retired
