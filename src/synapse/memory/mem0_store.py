import asyncio
from typing import Any

from mem0 import Memory  # type: ignore[import-untyped]

from synapse.config import Settings, get_settings
from synapse.domain.models import SourceEvent


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
                        "provider": "pgvector",
                        "config": {
                            "connection_string": settings.mem0_database_url,
                            "collection_name": settings.mem0_collection,
                            "embedding_model_dims": settings.embedding_dimensions,
                            "hnsw": True,
                        },
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": settings.cactus_embedding_model,
                            "api_key": "cactus-local",
                            "openai_base_url": settings.cactus_embedding_base_url,
                            "embedding_dims": settings.embedding_dimensions,
                        },
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": settings.cactus_chat_model,
                            "api_key": "cactus-local",
                            "openai_base_url": settings.cactus_chat_base_url,
                        },
                    },
                }
            )
        return self._memory

    async def add_episode(
        self, *, episode_id: str, summary: str, event: SourceEvent
    ) -> str:
        result = await asyncio.to_thread(
            self._get_memory().add,
            summary,
            user_id=self.settings.memory_user_id,
            metadata={
                "episode_id": episode_id,
                "memory_kind": "source",
                "source_type": event.source_type.value,
                "conversation_id": event.conversation_id,
                "occurred_at": event.occurred_at.isoformat(),
                "status": "active",
            },
            infer=False,
        )
        results = result.get("results", [])
        if not results:
            raise RuntimeError("Mem0 did not return an indexed memory")
        return str(results[0]["id"])

    async def search(self, query: str) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._get_memory().search,
            query,
            filters={"user_id": self.settings.memory_user_id, "status": "active"},
            top_k=5,
            threshold=self.settings.memory_hit_threshold,
            explain=True,
        )
        return list(result.get("results", []))
