"""Writer mode: Mem0 extracts facts from each episode, alongside its chunks.

Three layers are pinned here. The service runs the extraction after the episode
is committed and keeps the fact ids on the row so a rewrite can replace them;
recall folds fact hits into their episode's candidate and still cites the
conversation; and the store drives Mem0's real write path with the prompt,
scope and metadata the other two depend on.
"""

import json
import re
from collections.abc import Callable
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
    # The answer is the fact alone; the transcript it came from is the citation.
    assert result.answer == FACT
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

    assert result.answer == FACT
    assert result.citations[0].excerpt == THREAD


class ModelServer:
    """Answers chat with one fact and embeddings with a constant vector."""

    def __init__(self) -> None:
        # Gemini's OpenAI-compatible endpoint leaves each embedding's index null.
        self.null_indices = False
        self.prompts: list[str] = []
        self.chat_keys: list[str] = []
        # What each extraction returns, in turn; FACT once they run out.
        self.facts: list[list[str]] = []
        # Whether two facts conflict, for the supersession check.
        self.conflicting: Callable[[str, str], bool] = lambda new, old: False
        self.judgements: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"].removeprefix("Bearer ")
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            data = [
                {
                    "object": "embedding",
                    "index": None if self.null_indices else i,
                    "embedding": [0.1] * 8,
                }
                for i in range(len(body["input"]))
            ]
            return httpx.Response(
                200, json={"object": "list", "model": "e", "data": data, "usage": {}}
            )
        if body["messages"][0]["content"].startswith("You maintain a memory of facts"):
            return self.judge(json.loads(body["messages"][-1]["content"]))
        self.chat_keys.append(key)
        self.prompts.append(body["messages"][-1]["content"])
        facts = self.facts.pop(0) if self.facts else [FACT]
        return self.reply({"memory": [{"text": text} for text in facts]})

    def judge(self, payload: dict[str, Any]) -> httpx.Response:
        self.judgements.append(payload)
        existing = {item["id"]: item["fact"] for item in payload["existing"]}
        return self.reply(
            {
                "conflicts": [
                    {"new": item["id"], "existing": other}
                    for item in payload["new"]
                    for other in item["compare_with"]
                    if self.conflicting(item["fact"], existing[other])
                ]
            }
        )

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        return self.handle(request)

    def reply(self, content: dict[str, Any]) -> httpx.Response:
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
                        "message": {"role": "assistant", "content": json.dumps(content)},
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
    original_async = httpx.AsyncClient.__init__

    def patched_async(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        transport = httpx.MockTransport(model_server.handle_async)
        original_async(self, *args, **{**kwargs, "transport": transport})

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_async)
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


# Fact supersession: Mem0 only adds, so a correction has to retire what it corrects.

NOV_20 = datetime(2026, 11, 20, 9, tzinfo=UTC)
NOV_22 = datetime(2026, 11, 22, 9, tzinfo=UTC)
KATHY = "Kathy is flying via Singapore from San Francisco on December 26, 2026"
KRATI = "Krati, not Kathy, is flying via Singapore from San Francisco on December 26, 2026"
KRATI_AGAIN = "The person flying via Singapore on December 26, 2026 is Krati"
AISLE = "Abhi prefers aisle seats when booking flights"


def flying(new: str, old: str) -> bool:
    """The two flight facts contradict each other; nothing else does."""
    return "flying" in new and "flying" in old and ("Krati" in new) != ("Krati" in old)


def chat_on(day: datetime, source_id: str, content: str) -> SourceEvent:
    return SourceEvent(
        source_id=source_id,
        source_type=SourceType.WHATSAPP,
        conversation_id=OWNER,
        content=content,
        participants=["Abhi"],
        occurred_at=day,
    )


def fact_payload(store: Mem0EpisodeStore, fact_id: str) -> dict[str, Any]:
    row = store._get_memory().vector_store.get(fact_id)
    assert row is not None
    payload: dict[str, Any] = row.payload
    return payload


async def current_facts(store: Mem0EpisodeStore) -> list[str]:
    hits = await store.search("When is Krati flying?", OWNER)
    return sorted(hit["memory"] for hit in hits if hit["metadata"].get("memory_kind") == "fact")


async def learn(
    store: Mem0EpisodeStore, episode_id: str, day: datetime
) -> list[str]:
    ids = await store.extract_facts(
        episode_id=episode_id, group_id=OWNER, event=chat_on(day, episode_id, "Abhi: flights")
    )
    await store.settle_facts(group_id=OWNER, fact_ids=ids)
    return ids


async def test_a_correction_retires_the_fact_it_corrects(
    server: ModelServer, tmp_path: Path
) -> None:
    server.facts, server.conflicting = [[KATHY], [KRATI]], flying
    store = Mem0EpisodeStore(writer(tmp_path))

    [kathy] = await learn(store, "e1", NOV_20)
    [krati] = await learn(store, "e2", NOV_22)

    retired = fact_payload(store, kathy)
    assert retired["status"] == "superseded"
    assert retired["superseded_by"] == krati
    assert retired["invalid_at"] == NOV_22.isoformat()
    assert retired["data"] == KATHY  # kept, not deleted
    assert fact_payload(store, krati)["status"] == "active"
    assert await current_facts(store) == [KRATI]
    # One call for the second episode; the first had nothing to compare with.
    assert len(server.judgements) == 1


async def test_facts_that_can_both_be_true_both_stay(
    server: ModelServer, tmp_path: Path
) -> None:
    server.facts, server.conflicting = [[KATHY], [AISLE]], flying
    store = Mem0EpisodeStore(writer(tmp_path))

    await learn(store, "e1", NOV_20)
    await learn(store, "e2", NOV_22)

    assert len(server.judgements) == 1
    assert await current_facts(store) == sorted([KATHY, AISLE])


async def test_an_older_conversation_imported_later_does_not_retire_a_newer_fact(
    server: ModelServer, tmp_path: Path
) -> None:
    server.facts, server.conflicting = [[KRATI], [KATHY]], flying
    store = Mem0EpisodeStore(writer(tmp_path))

    [krati] = await learn(store, "e2", NOV_22)
    [kathy] = await learn(store, "e1", NOV_20)

    assert fact_payload(store, krati)["status"] == "active"
    late = fact_payload(store, kathy)
    assert (late["status"], late["superseded_by"]) == ("superseded", krati)
    assert late["invalid_at"] == NOV_22.isoformat()
    assert await current_facts(store) == [KRATI]


async def test_without_candidates_the_check_is_off(server: ModelServer, tmp_path: Path) -> None:
    server.facts, server.conflicting = [[KATHY], [KRATI]], flying
    store = Mem0EpisodeStore(writer(tmp_path, mem0_supersession_candidates=0))

    await learn(store, "e1", NOV_20)
    await learn(store, "e2", NOV_22)

    assert server.judgements == []
    assert await current_facts(store) == sorted([KATHY, KRATI])


class PickingJudge:
    """Selects the candidate whose text mentions ``word``."""

    def __init__(self, word: str) -> None:
        self.word = word
        self.seen: list[dict[str, object]] = []

    async def select(self, query: str, candidates: list[dict[str, object]]) -> EpisodeSelection:
        self.seen = candidates
        chosen = next(c for c in candidates if self.word in str(c["text"]))
        return EpisodeSelection(
            answerable=True, selected_episode_id=str(chosen["episode_id"]), confidence=0.9
        )


async def test_recall_answers_with_the_correction_through_a_growing_thread(
    server: ModelServer, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    server.facts, server.conflicting = [[KATHY], [KRATI], [KRATI_AGAIN, AISLE]], flying
    store = Mem0EpisodeStore(writer(tmp_path))
    service = SystemOneService(StubInterpreter(), store)
    first = chat_on(NOV_20, "wa-nov-20", "Abhi: Kathy flies via Singapore on the 26th")
    correction = "Abhi: sorry, it's Krati not Kathy"
    async with sessions() as session:
        older, _ = await service.process(first, OWNER, session)
        newer, _ = await service.process(
            chat_on(NOV_22, "wa-nov-22", correction), OWNER, session
        )
    [kathy] = older.details["mem0_fact_ids"]
    [krati] = newer.details["mem0_fact_ids"]
    assert fact_payload(store, kathy)["superseded_by"] == krati

    judge = PickingJudge("Krati")
    async with sessions() as session:
        result = await RecallService(memory_store=store, judge=judge).recall(
            "When is Krati flying?", OWNER, session
        )
    assert result.answer == KRATI
    assert not any(KATHY in str(candidate["text"]) for candidate in judge.seen)

    # The thread grows: its facts are re-read, and the old ones deleted. The
    # Kathy fact must not stay retired by a fact that is gone — it is
    # reinstated and settled against the new reading, which still corrects it.
    async with sessions() as session:
        grown, _ = await service.process(
            chat_on(NOV_22, "wa-nov-22", correction + "\nAbhi: aisle seat please"),
            OWNER,
            session,
        )
    again, aisle = grown.details["mem0_fact_ids"]
    assert store._get_memory().vector_store.get(krati) is None
    assert fact_payload(store, kathy)["superseded_by"] == again
    assert "mem0_facts_unsettled" not in grown.details
    assert await current_facts(store) == sorted([KRATI_AGAIN, AISLE])


async def test_a_fact_retired_by_a_reading_that_is_gone_is_reinstated(
    server: ModelServer, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    server.facts, server.conflicting = [[KATHY], [KRATI], [AISLE]], flying
    store = Mem0EpisodeStore(writer(tmp_path))
    service = SystemOneService(StubInterpreter(), store)
    async with sessions() as session:
        older, _ = await service.process(
            chat_on(NOV_20, "wa-nov-20", "Abhi: Kathy flies on the 26th"), OWNER, session
        )
        await service.process(chat_on(NOV_22, "wa-nov-22", "Abhi: Krati?"), OWNER, session)
        # Read again, the conversation no longer yields the correction.
        await service.process(
            chat_on(NOV_22, "wa-nov-22", "Abhi: Krati?\nAbhi: aisle seat"), OWNER, session
        )
    [kathy] = older.details["mem0_fact_ids"]

    reinstated = fact_payload(store, kathy)
    assert reinstated["status"] == "active"
    assert "superseded_by" not in reinstated and "invalid_at" not in reinstated
    assert await current_facts(store) == sorted([KATHY, AISLE])


class SettlingStore(FactStore):
    def __init__(self, sessions: async_sessionmaker[AsyncSession], *, fail: bool) -> None:
        super().__init__(sessions)
        self.fail_settling = fail
        self.settled: list[list[str]] = []

    async def settle_facts(self, *, group_id: str, fact_ids: list[str]) -> int:
        if self.fail_settling:
            raise RuntimeError("quota")
        self.settled.append(fact_ids)
        return 0

    async def release_facts(self, fact_ids: list[str]) -> list[str]:
        return []


async def test_a_failed_supersession_check_is_recorded_and_retried(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    service = SystemOneService(StubInterpreter(), SettlingStore(sessions, fail=True))
    async with sessions() as session:
        episode, _ = await service.process(whatsapp(), OWNER, session)

    assert episode.details["mem0_fact_ids"] == ["fact-1"]
    assert episode.details["mem0_facts_unsettled"] == ["fact-1"]
    assert episode.details["memory_supersession_error"] == "RuntimeError"

    working = SettlingStore(sessions, fail=False)
    service.memory_store = working
    async with sessions() as session:
        counts = await service.reindex_missing(OWNER, session)
        stored = await session.scalar(select(Episode))

    assert counts == {"indexed": 0, "failed": 0, "facts": 0, "settled": 1}
    assert working.settled == [["fact-1"]]
    assert working.extracted == []  # the facts were kept; only the check ran again
    assert stored is not None
    assert "mem0_facts_unsettled" not in stored.details
    assert "memory_supersession_error" not in stored.details


async def test_facts_settle_when_the_endpoint_leaves_embedding_indices_null(
    server: ModelServer, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Gemini's batch embeddings carry ``index: null``; Mem0 sorted on it.

    Any episode with two or more new facts then failed its check with
    "'<' not supported between instances of 'NoneType' and 'int'", left its
    facts unsettled, and ``reindex_missing`` could not recover them.
    """
    server.null_indices, server.conflicting = True, flying
    server.facts = [[KATHY], [KRATI, AISLE]]
    store = Mem0EpisodeStore(writer(tmp_path))
    service = SystemOneService(StubInterpreter(), store)
    async with sessions() as session:
        older, _ = await service.process(
            chat_on(NOV_20, "wa-nov-20", "Abhi: Kathy flies on the 26th"), OWNER, session
        )
        newer, _ = await service.process(
            chat_on(NOV_22, "wa-nov-22", "Abhi: it's Krati. Aisle seat."), OWNER, session
        )

    assert "memory_supersession_error" not in newer.details
    assert "mem0_facts_unsettled" not in newer.details
    [kathy] = older.details["mem0_fact_ids"]
    assert fact_payload(store, kathy)["status"] == "superseded"
    assert await current_facts(store) == sorted([KRATI, AISLE])


async def test_reindex_missing_settles_facts_a_failed_check_left_behind(
    server: ModelServer, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Episodes stored by the broken release keep ``mem0_facts_unsettled``."""
    server.conflicting = flying
    server.facts = [[KATHY], [KRATI, AISLE]]
    store = Mem0EpisodeStore(writer(tmp_path, mem0_supersession_candidates=0))
    service = SystemOneService(StubInterpreter(), store)
    async with sessions() as session:
        older, _ = await service.process(
            chat_on(NOV_20, "wa-nov-20", "Abhi: Kathy flies on the 26th"), OWNER, session
        )
        newer, _ = await service.process(
            chat_on(NOV_22, "wa-nov-22", "Abhi: it's Krati. Aisle seat."), OWNER, session
        )
        newer.details = {
            **newer.details,
            "mem0_facts_unsettled": newer.details["mem0_fact_ids"],
            "memory_supersession_error": "TypeError",
        }
        await session.commit()

    server.null_indices = True
    store.settings = writer(tmp_path)
    async with sessions() as session:
        counts = await service.reindex_missing(OWNER, session)
        stored = await session.get(Episode, newer.id)

    assert counts["settled"] == 1
    assert stored is not None and "mem0_facts_unsettled" not in stored.details
    assert "memory_supersession_error" not in stored.details
    [kathy] = older.details["mem0_fact_ids"]
    assert fact_payload(store, kathy)["status"] == "superseded"
