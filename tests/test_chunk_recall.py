"""Retrieval over an episode indexed as several chunks.

An episode is stored whole and indexed in pieces, so more than one vector can
point at the same conversation. These cover what that asymmetry has to
guarantee: a conversation is still one candidate, it is judged on the passage
that actually matched, and the index can be rebuilt from episodes already
stored without re-reading the source or asking a model anything.
"""

from datetime import UTC, datetime
from typing import Any

from synapse.db.models import Episode, SourceType
from synapse.domain.models import FastObservation, SourceEvent
from synapse.episodic.service import SystemOneService
from synapse.recall.models import EpisodeSelection
from synapse.recall.service import RecallService

ACME = "acme"
NOW = datetime.now(UTC)

THREAD = "\n".join(
    ["caleb: getting a GB300 replica up tonight"]
    + [f"user{i}: {'x' * 80}" for i in range(20)]
    + ["rishi: GB300:4 c=4 0.51 / 0.72"]
)


class StubInterpreter:
    async def observe(self, event: SourceEvent) -> FastObservation:
        # Mirrors the real truncation: a summary is the body's opening.
        return FastObservation(
            summary=event.content[:60],
            intent="share_update",
            salience=0.5,
            provider="stub",
        )


class RecordingStore:
    """Records indexing calls and replays fixed hits."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.indexed: list[str] = []
        self.forgotten: list[str] = []

    async def add_episode(
        self, *, episode_id: str, group_id: str, summary: str, event: SourceEvent
    ) -> list[str]:
        self.indexed.append(episode_id)
        return [f"mem0-{episode_id}-{len(self.indexed)}-{n}" for n in range(2)]

    async def forget_memories(self, memory_ids: list[str]) -> None:
        self.forgotten.extend(memory_ids)

    async def search(self, query: str, group_id: str) -> list[dict[str, Any]]:
        return self.hits

    async def forget_group(self, group_id: str) -> None:
        self.forgotten.append(group_id)


class CapturingJudge:
    """Selects the first candidate and remembers what it was shown."""

    def __init__(self) -> None:
        self.seen: list[dict[str, object]] = []

    async def select(
        self, query: str, candidates: list[dict[str, object]]
    ) -> EpisodeSelection:
        self.seen = candidates
        if not candidates:
            return EpisodeSelection(answerable=False, confidence=0.0)
        return EpisodeSelection(
            answerable=True,
            selected_episode_id=str(candidates[0]["episode_id"]),
            confidence=0.9,
        )


def event(source_id: str, content: str) -> SourceEvent:
    return SourceEvent(
        source_id=source_id,
        source_type=SourceType.SLACK,
        conversation_id="thread-1",
        content=content,
        participants=["rishi"],
        occurred_at=NOW,
    )


async def ingest(session: Any, content: str, source_id: str = "slack-1") -> Episode:
    service = SystemOneService(StubInterpreter(), RecordingStore())
    episode, _ = await service.process(event(source_id, content), ACME, session)
    return episode


async def test_several_chunks_of_one_episode_become_one_candidate(session) -> None:
    """Otherwise a single long thread crowds every other conversation out.

    The judge is shown a handful of candidates. If each chunk arrived as its
    own, one thread could fill the list and the answer might sit in a
    conversation that never got looked at.
    """
    episode = await ingest(session, THREAD)
    meta = {"episode_id": str(episode.id)}
    store = RecordingStore(
        hits=[
            {"metadata": {**meta, "chunk_index": 0}, "memory": "an early chunk", "score": 0.71},
            {"metadata": {**meta, "chunk_index": 7}, "memory": "the best chunk", "score": 0.93},
            {"metadata": {**meta, "chunk_index": 3}, "memory": "a middling chunk", "score": 0.80},
        ]
    )
    judge = CapturingJudge()

    result = await RecallService(memory_store=store, judge=judge).recall(
        "what TTFT on GB300?", ACME, session
    )

    assert len(judge.seen) == 1
    # The candidate scores as its strongest chunk...
    assert result.score_details["memory_score"] == 0.93
    assert result.score_details["chunk_matches"] == 3


async def test_a_folded_candidate_keeps_every_matched_chunk_in_order(session) -> None:
    """Similarity ranks what reads like the question, not what answers it.

    A benchmark's setup message is prose about the topic and outscores the
    table of numbers posted two messages later. Keeping only the best chunk
    would hand the judge the preamble and drop the answer, so every matched
    span is kept, in the order it was said.
    """
    episode = await ingest(session, THREAD)
    meta = {"episode_id": str(episode.id)}
    store = RecordingStore(
        hits=[
            {
                "metadata": {**meta, "chunk_index": 15},
                "memory": "rishi: ran new evals, a conversation-replay benchmark",
                "score": 0.47,
            },
            {
                "metadata": {**meta, "chunk_index": 16},
                "memory": "rishi: GB300:4 c=4 0.51 / 0.72",
                "score": 0.39,
            },
        ]
    )
    judge = CapturingJudge()

    result = await RecallService(memory_store=store, judge=judge).recall(
        "what TTFT on GB300?", ACME, session
    )

    text = str(judge.seen[0]["text"])
    assert "ran new evals" in text
    assert "GB300:4 c=4 0.51 / 0.72" in text
    assert text.index("ran new evals") < text.index("GB300:4")
    assert "0.51 / 0.72" in result.answer


async def test_the_answer_comes_from_the_chunk_not_the_thread_opening(session) -> None:
    """The failure this whole design exists to prevent.

    Indexed whole, the thread is described by its first line and the passage
    holding the numbers is unreachable. Indexed in chunks, the hit carries that
    passage and it is what the judge weighs and what the answer quotes.
    """
    episode = await ingest(session, THREAD)
    assert episode.summary.startswith("caleb:")  # the opening, as stored

    store = RecordingStore(
        hits=[
            {
                "metadata": {"episode_id": str(episode.id), "chunk_index": 21},
                "memory": "rishi: GB300:4 c=4 0.51 / 0.72",
                "score": 0.9,
            }
        ]
    )
    judge = CapturingJudge()

    result = await RecallService(memory_store=store, judge=judge).recall(
        "what TTFT did we see on GB300?", ACME, session
    )

    assert "0.51 / 0.72" in result.answer
    assert "0.51 / 0.72" in result.citations[0].excerpt
    assert "getting a GB300 replica up tonight" not in result.answer


async def test_an_episode_indexed_before_chunking_still_answers(session) -> None:
    """Hits written by an older index carry no chunk text; recall must cope."""
    episode = await ingest(session, "acme is on 2xH100")
    store = RecordingStore(
        hits=[{"metadata": {"episode_id": str(episode.id)}, "score": 0.9}]
    )

    result = await RecallService(memory_store=store, judge=CapturingJudge()).recall(
        "what hardware?", ACME, session
    )

    assert result.answer_origin == "episodic"
    assert result.answer == episode.summary


async def test_reindex_group_rebuilds_from_stored_episodes(session) -> None:
    """A change to how episodes are indexed has to reach the ones already stored.

    ``process`` returns early for an episode it has seen, so re-running an
    ingest cannot do this. Nothing is re-read from Slack and no model is called
    — the bodies are already on the rows.
    """
    first = await ingest(session, THREAD)
    second = await ingest(session, "a second conversation", source_id="slack-2")

    store = RecordingStore()
    counts = await SystemOneService(StubInterpreter(), store).reindex_group(ACME, session)

    # The stale index is dropped before the new one is built, or the old
    # vectors would linger and keep matching.
    assert store.forgotten == [ACME]
    assert counts == {"indexed": 2, "failed": 0}
    assert set(store.indexed) == {str(first.id), str(second.id)}


async def test_reindex_group_leaves_other_accounts_alone(session) -> None:
    await ingest(session, "acme thread")
    store = RecordingStore()

    counts = await SystemOneService(StubInterpreter(), store).reindex_group(
        "globex", session
    )

    assert counts == {"indexed": 0, "failed": 0}
    assert store.indexed == []
    assert store.forgotten == ["globex"]


class DecliningJudge:
    """Never finds a candidate answerable, as a judge flake does."""

    async def select(
        self, query: str, candidates: list[dict[str, object]]
    ) -> EpisodeSelection:
        return EpisodeSelection(answerable=False, confidence=0.4)


async def test_a_declining_judge_still_returns_what_matched(session) -> None:
    """A spurious "no" must not throw the evidence away.

    The judge is one LLM call deciding yes or no. When it declines, no single
    conversation is being put forward as the answer — but the conversations
    that matched are still the best evidence there is, and a caller that
    renders sources or writes its own answer needs them.
    """
    episode = await ingest(session, THREAD)
    store = RecordingStore(
        hits=[
            {
                "metadata": {"episode_id": str(episode.id), "chunk_index": 16},
                "memory": "rishi: GB300:4 c=4 0.51 / 0.72",
                "score": 0.39,
            }
        ]
    )

    result = await RecallService(memory_store=store, judge=DecliningJudge()).recall(
        "what TTFT on GB300?", ACME, session
    )

    # No graph to fall back to here, so the origin is a miss — and the matched
    # conversation still comes back with the passage that matched.
    assert result.answer_origin == "miss"
    assert [c.id for c in result.citations] == [str(episode.id)]
    assert "0.51 / 0.72" in result.citations[0].excerpt


async def test_a_thread_that_grew_replaces_its_stored_version(session) -> None:
    """The reason late replies were being lost.

    A thread keeps collecting replies for days. The source hands back the whole
    thread each time, so the same source_id arriving with a longer body means
    the conversation grew — and the stored version has to move with it, or the
    episode is frozen at whatever had been said when it was first seen.
    """
    first = await ingest(session, "caleb: getting a GB300 replica up tonight")
    episode_id = first.id

    grown = "caleb: getting a GB300 replica up tonight\nrishi: GB300:4 c=4 0.51 / 0.72"
    service = SystemOneService(StubInterpreter(), RecordingStore())
    again, created = await service.process(event("slack-1", grown), ACME, session)

    # The same row, carrying the new reply — not a second episode.
    assert again.id == episode_id
    assert created is False
    assert "0.51 / 0.72" in again.content


async def test_an_updated_episode_drops_the_vectors_of_its_old_body(session) -> None:
    """Otherwise the index keeps matching text the episode no longer says."""
    episode = await ingest(session, "caleb: replica up tonight")
    written = list((episode.details or {}).get("mem0_ids") or [])
    assert written, "the first indexing should have recorded its vector ids"

    store = RecordingStore()
    service = SystemOneService(StubInterpreter(), store)
    updated, _ = await service.process(
        event("slack-1", "caleb: replica up tonight\nrishi: 0.51 / 0.72"), ACME, session
    )

    # Exactly the vectors of the old body are dropped, and the episode is
    # indexed again under the same id.
    assert store.forgotten == written
    assert store.indexed == [str(episode.id)]
    assert (updated.details or {}).get("mem0_ids")


async def test_an_unchanged_conversation_is_not_rewritten(session) -> None:
    """Re-ingesting an overlapping window must stay cheap and idempotent.

    The interpreter is a model call. Re-running it for a conversation that has
    not changed would spend one on every episode of every overlapping window.
    """
    await ingest(session, "caleb: replica up tonight")

    class CountingInterpreter(StubInterpreter):
        calls = 0

        async def observe(self, event: SourceEvent) -> FastObservation:
            CountingInterpreter.calls += 1
            return await super().observe(event)

    store = RecordingStore()
    service = SystemOneService(CountingInterpreter(), store)
    _, created = await service.process(
        event("slack-1", "caleb: replica up tonight"), ACME, session
    )

    assert created is False
    assert CountingInterpreter.calls == 0
    assert store.indexed == [] and store.forgotten == []
