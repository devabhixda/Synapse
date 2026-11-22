"""Writer mode: Mem0 extracts facts from each episode, alongside its chunks.

Three layers are pinned here. The service runs the extraction after the episode
is committed and keeps the fact ids on the row so a rewrite can replace them;
recall folds fact hits into their episode's candidate and still cites the
conversation; and the store drives Mem0's real write path with the prompt,
scope and metadata the other two depend on.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from synapse.config import Settings
from synapse.db.models import Episode, SourceType
from synapse.domain.models import FastObservation, SourceEvent
from synapse.episodic.service import SystemOneService
from synapse.memory.mem0_store import FACT_INSTRUCTIONS, Mem0EpisodeStore
from synapse.recall.models import EpisodeSelection
from synapse.recall.service import RecallService

OWNER = "owner"
SAID_ON = datetime(2026, 11, 20, 18, 30, tzinfo=UTC)
THREAD = "Asha: we're moving to Pune in March\nAbhi: that's next year, right?"
FACT = "Asha is moving to Pune in March 2027"


def whatsapp(content: str = THREAD) -> SourceEvent:
    return SourceEvent(
        source_id="whatsapp-2026-11-20",
        source_type=SourceType.WHATSAPP,
        conversation_id=OWNER,
        content=content,
        participants=["Asha", "Abhi"],
        occurred_at=SAID_ON,
    )


class StubInterpreter:
    async def observe(self, event: SourceEvent) -> FastObservation:
        return FastObservation(
            summary=event.content[:40], intent="share_update", salience=0.5, provider="stub"
        )


class FactStore:
    """Indexes nothing real; records what writer mode asked of it."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        writes_facts: bool = True,
        fail: bool = False,
        hits: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sessions = sessions
        self.writes_facts = writes_facts
        self.fail = fail
        self.hits = hits or []
        self.extracted: list[str] = []
        self.committed_first: list[bool] = []
        self.forgotten: list[str] = []

    async def add_episode(
        self, *, episode_id: str, group_id: str, summary: str, event: SourceEvent
    ) -> list[str]:
        return [f"chunk-{episode_id}"]

    async def extract_facts(
        self, *, episode_id: str, group_id: str, event: SourceEvent
    ) -> list[str]:
        # Another connection sees the episode only once it is committed.
        async with self.sessions() as other:
            self.committed_first.append(await other.get(Episode, episode_id) is not None)
        if self.fail:
            raise RuntimeError("model unavailable")
        self.extracted.append(event.content)
        return [f"fact-{len(self.extracted)}"]

    async def forget_memories(self, memory_ids: list[str]) -> None:
        self.forgotten.extend(memory_ids)

    async def forget_group(self, group_id: str) -> None:
        pass

    async def search(self, query: str, group_id: str) -> list[dict[str, Any]]:
        return self.hits


async def test_facts_are_extracted_after_the_episode_is_committed(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    store = FactStore(sessions)
    async with sessions() as session:
        episode, created = await SystemOneService(StubInterpreter(), store).process(
            whatsapp(), OWNER, session
        )

    assert created
    assert store.committed_first == [True]
    assert episode.details["mem0_fact_ids"] == ["fact-1"]
    assert episode.mem0_id == f"chunk-{episode.id}"


async def test_without_writer_mode_nothing_is_extracted(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    store = FactStore(sessions, writes_facts=False)
    async with sessions() as session:
        episode, _ = await SystemOneService(StubInterpreter(), store).process(
            whatsapp(), OWNER, session
        )

    assert store.extracted == []
    assert "mem0_fact_ids" not in episode.details


async def test_a_failed_extraction_costs_the_facts_not_the_episode(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    service = SystemOneService(StubInterpreter(), FactStore(sessions, fail=True))
    async with sessions() as session:
        episode, _ = await service.process(whatsapp(), OWNER, session)
    async with sessions() as session:
        stored = await session.scalar(select(Episode))

    assert stored is not None and stored.mem0_id == f"chunk-{episode.id}"
    assert stored.details["memory_fact_error"] == "RuntimeError"
    assert "mem0_fact_ids" not in stored.details

    # The same conversation arriving again is the retry.
    service.memory_store = FactStore(sessions)
    async with sessions() as session:
        episode, _ = await service.process(whatsapp(), OWNER, session)
    assert episode.details["mem0_fact_ids"] == ["fact-1"]
    assert "memory_fact_error" not in episode.details


async def test_a_longer_conversation_replaces_its_facts(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    store = FactStore(sessions)
    service = SystemOneService(StubInterpreter(), store)
    async with sessions() as session:
        await service.process(whatsapp(), OWNER, session)
        episode, _ = await service.process(
            whatsapp(THREAD + "\nAsha: actually April"), OWNER, session
        )

    assert "fact-1" in store.forgotten
    assert store.extracted[-1].endswith("actually April")
    assert episode.details["mem0_fact_ids"] == ["fact-2"]


async def stored_episode(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> Episode:
    episode, _ = await SystemOneService(
        StubInterpreter(), FactStore(sessions, writes_facts=False)
    ).process(whatsapp(), OWNER, session)
    return episode


class FirstCandidateJudge:
    def __init__(self) -> None:
        self.seen: list[dict[str, object]] = []

    async def select(self, query: str, candidates: list[dict[str, object]]) -> EpisodeSelection:
        self.seen = candidates
        return EpisodeSelection(
            answerable=True, selected_episode_id=str(candidates[0]["episode_id"]), confidence=0.9
        )


async def test_recall_reads_facts_with_their_date_and_cites_the_conversation(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    episode = await stored_episode(sessions, session)
    meta = {"episode_id": str(episode.id)}
    store = FactStore(
        sessions,
        hits=[
            {
                "metadata": {**meta, "chunk_index": 0},
                "memory": "Asha: moving to Pune",
                "score": 0.6,
            },
            {"metadata": {**meta, "memory_kind": "fact"}, "memory": FACT, "score": 0.9},
        ],
    )
    judge = FirstCandidateJudge()

    result = await RecallService(memory_store=store, judge=judge).recall(
        "when is Asha moving?", OWNER, session
    )

    assert len(judge.seen) == 1
    assert judge.seen[0]["text"] == (
        f"{FACT} (from a conversation on 2026-11-20)\nAsha: moving to Pune"
    )
    assert result.score_details["memory_score"] == 0.9
    assert result.score_details["fact_matches"] == 1
    # The paraphrase is never presented as a quotation.
    assert result.citations[0].excerpt == "Asha: moving to Pune"


async def test_a_fact_alone_cites_the_whole_conversation(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    episode = await stored_episode(sessions, session)
    store = FactStore(
        sessions,
        hits=[
            {
                "metadata": {"episode_id": str(episode.id), "memory_kind": "fact"},
                "memory": FACT,
                "score": 0.9,
            }
        ],
    )

    result = await RecallService(memory_store=store, judge=FirstCandidateJudge()).recall(
        "when is Asha moving?", OWNER, session
    )

    assert result.answer is not None and FACT in result.answer
    assert result.citations[0].excerpt == THREAD


class ModelServer:
    """Answers chat with one fact and embeddings with a constant vector."""

    def __init__(self, exhausted: frozenset[str] = frozenset()) -> None:
        self.exhausted = exhausted
        self.prompts: list[str] = []
        self.chat_keys: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"].removeprefix("Bearer ")
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            data = [
                {"object": "embedding", "index": i, "embedding": [0.1] * 8}
                for i in range(len(body["input"]))
            ]
            return httpx.Response(
                200, json={"object": "list", "model": "e", "data": data, "usage": {}}
            )
        self.chat_keys.append(key)
        if key in self.exhausted:
            return httpx.Response(429, json={}, headers={"x-should-retry": "false"})
        self.prompts.append(body["messages"][-1]["content"])
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"memory": [{"text": FACT}]}),
                        },
                    }
                ],
            },
        )


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> ModelServer:
    model_server = ModelServer()
    original = httpx.Client.__init__

    def patched(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        transport = httpx.MockTransport(model_server.handle)
        original(self, *args, **{**kwargs, "transport": transport})

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    return model_server


def writer(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'synapse.db'}",
        chat_base_url="https://models.example.com/v1",
        chat_model="flash",
        chat_api_key="free",
        embedding_base_url="https://models.example.com/v1",
        embedding_model="embed",
        embedding_api_key="free",
        embedding_dimensions=8,
        mem0_writer_mode=True,
        **overrides,
    )


def section(prompt: str, title: str) -> str:
    body = prompt.split(f"## {title}\n", 1)[1]
    return re.split(r"\n\n#", body, maxsplit=1)[0]


async def test_mem0_writes_dated_facts_tagged_with_their_episode(
    server: ModelServer, tmp_path: Path
) -> None:
    store = Mem0EpisodeStore(writer(tmp_path, mem0_fact_instructions="Skip small talk."))
    await store.add_episode(episode_id="e1", group_id=OWNER, summary="", event=whatsapp())

    ids = await store.extract_facts(episode_id="e1", group_id=OWNER, event=whatsapp())
    await store.extract_facts(episode_id="e2", group_id=OWNER, event=whatsapp("Asha: hi"))

    assert len(ids) == 1
    first, second = server.prompts
    assert section(first, "New Messages").count("Conversation date: 2026-11-20 (whatsapp)") == 1
    assert section(first, "Custom Instructions") == f"{FACT_INSTRUCTIONS}\nSkip small talk."
    # The episode's own chunks are not shown as facts already known...
    assert "moving to Pune in March" not in section(first, "Existing Memories")
    # ...the facts are...
    assert FACT in section(second, "Existing Memories")
    # ...and earlier conversations are not replayed whole.
    assert "Asha:" not in section(second, "Last k Messages")

    hits = await store.search("when is Asha moving?", OWNER)
    facts = [hit for hit in hits if hit["metadata"].get("memory_kind") == "fact"]
    assert [hit["memory"] for hit in facts] == [FACT]
    assert facts[0]["metadata"]["episode_id"] == "e1"
    assert facts[0]["metadata"]["source_type"] == "whatsapp"


async def test_mem0_fact_extraction_falls_back_to_the_second_key(
    server: ModelServer, tmp_path: Path
) -> None:
    server.exhausted = frozenset({"free"})
    store = Mem0EpisodeStore(writer(tmp_path, chat_fallback_api_key="paid"))

    ids = await store.extract_facts(episode_id="e1", group_id=OWNER, event=whatsapp())

    assert len(ids) == 1
    assert server.chat_keys == ["free", "paid"]


async def test_without_writer_mode_mem0_is_not_asked(server: ModelServer, tmp_path: Path) -> None:
    store = Mem0EpisodeStore(writer(tmp_path).model_copy(update={"mem0_writer_mode": False}))

    assert await store.extract_facts(episode_id="e1", group_id=OWNER, event=whatsapp()) == []
    assert server.chat_keys == []


async def test_reindex_missing_retries_facts_whose_extraction_failed(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The chunks embedded but the chat call failed: nothing would resend it."""
    service = SystemOneService(StubInterpreter(), FactStore(sessions, fail=True))
    async with sessions() as session:
        await service.process(whatsapp(), OWNER, session)

    service.memory_store = FactStore(sessions)
    async with sessions() as session:
        counts = await service.reindex_missing(OWNER, session)
        again = await service.reindex_missing(OWNER, session)
        stored = await session.scalar(select(Episode))

    assert counts == {"indexed": 0, "failed": 0, "facts": 1}
    assert again == {"indexed": 0, "failed": 0, "facts": 0}
    assert stored is not None and stored.details["mem0_fact_ids"] == ["fact-1"]
