import asyncio
import copy
from typing import Any

from mem0 import Memory  # type: ignore[import-untyped]

from synapse.config import Settings, get_settings
from synapse.domain.models import SourceEvent
from synapse.inference.keys import with_fallback_key_sync
from synapse.memory.chunking import chunk_text
from synapse.memory.sqlite_vec_store import register as register_sqlite_vec

register_sqlite_vec()

# Each pass of a group purge clears one page of the vector store. The bound only
# exists so a store that reports memories it will not delete fails loudly rather
# than spinning; at 100 memories a page this covers far more than any real
# account holds.
_MAX_PURGE_PAGES = 500


class FallbackKeyEmbedder:
    """Mem0's OpenAI embedder, with a second key for when the first is out of quota.

    Mem0 builds its embedder from a single key and calls ``embed`` and
    ``embed_batch`` on it directly, so the fallback has to sit in front of that
    object: this one forwards each call to the embedder holding whichever key
    ``synapse.inference.keys`` prefers. The second embedder is a copy of the
    first with only the key changed, so model, URL and dimensions stay in step.
    """

    def __init__(self, primary: Any, settings: Settings, fallback_api_key: str) -> None:
        secondary = copy.copy(primary)
        secondary.client = primary.client.with_options(api_key=fallback_api_key)
        self._primary_key = settings.embedding_api_key
        self._fallback_key = fallback_api_key
        self._cooldown = settings.fallback_key_cooldown
        self._embedders = {self._primary_key: primary, fallback_api_key: secondary}
        self.config = primary.config

    def embed(self, text: Any, memory_action: Any = None) -> Any:
        return with_fallback_key_sync(
            self._primary_key,
            self._fallback_key,
            lambda key: self._embedders[key].embed(text, memory_action),
            cooldown=self._cooldown,
            purpose="embedding",
        )

    def embed_batch(self, texts: Any, memory_action: Any = "add") -> Any:
        return with_fallback_key_sync(
            self._primary_key,
            self._fallback_key,
            lambda key: self._embedders[key].embed_batch(texts, memory_action),
            cooldown=self._cooldown,
            purpose="embedding",
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
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": settings.chat_model,
                            "api_key": settings.chat_api_key,
                            "openai_base_url": settings.chat_base_url,
                        },
                    },
                }
            )
            fallback = settings.embedding_fallback_api_key
            if fallback and fallback != settings.embedding_api_key:
                self._memory.embedding_model = FallbackKeyEmbedder(
                    self._memory.embedding_model, settings, fallback
                )
        return self._memory

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
