import hashlib
import re
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from synapse.config import Settings, get_settings
from synapse.db.models import Episode, EpisodeKind, IntentObservation, RecallEvent, SourceType
from synapse.memory.mem0_store import Mem0EpisodeStore
from synapse.recall.answerability import EpisodeAnswerabilityJudge
from synapse.recall.models import RecallCitation, RecallResult
from synapse.semantic.repository import search_semantic_context


def normalize_query(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _score(memory: dict[str, Any]) -> float:
    """A hit's similarity, treating a missing score as the weakest possible."""
    try:
        return float(memory.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def matched_text(episode: Episode, memory: dict[str, Any]) -> str:
    """The text a vector hit actually matched on.

    An episode is indexed chunk by chunk, so the hit names a span of the
    conversation rather than the whole of it. That span is what the judge should
    weigh and what the answer should come from — judging the episode's summary
    instead is how a thread gets rejected for not answering a question that its
    tenth message answers plainly.

    Falls back to the episode's own text for a hit carrying no memory body, and
    for episodes indexed before chunking existed.
    """
    text = str(memory.get("memory") or "").strip()
    return text or episode.summary or episode.content


def citation_excerpt(episode: Episode, memory: dict[str, Any]) -> str | None:
    """What a citation should quote for a candidate: words from the conversation.

    A writer-mode fact is the model's paraphrase, so it is evidence for the
    judge but not a quotation. A candidate that matched on facts alone cites
    the whole conversation; one that also matched chunks quotes those.
    """
    if "excerpt" in memory:
        excerpt = memory["excerpt"]
        return excerpt if isinstance(excerpt, str) and excerpt.strip() else None
    return matched_text(episode, memory)


def answer_text(episode: Episode, memory: dict[str, Any]) -> str:
    """The answer recall returns for a selected candidate.

    The judge reads a writer-mode candidate as its facts followed by the chunks
    they came from, but the facts are the answer: the transcript behind them is
    supporting evidence, and it stays in the citation. A candidate with no
    facts answers with its matched chunks, as before writer mode.
    """
    facts = memory.get("facts")
    if isinstance(facts, list) and any(isinstance(f, str) and f.strip() for f in facts):
        return "\n".join(f for f in facts if isinstance(f, str) and f.strip())
    return matched_text(episode, memory)


def needs_substantive_update(query: str) -> bool:
    normalized = normalize_query(query)
    if any(phrase in normalized for phrase in ("who asked", "who requested")):
        return False
    return any(
        term in normalized
        for term in ("status", "blocker", "next action", "owner", "readiness")
    )


class RecallService:
    def __init__(
        self,
        memory_store: Mem0EpisodeStore | None = None,
        judge: EpisodeAnswerabilityJudge | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self.memory_store = memory_store or Mem0EpisodeStore(resolved)
        self.judge = judge or EpisodeAnswerabilityJudge(resolved)

    async def recall(self, query: str, group_id: str, session: AsyncSession) -> RecallResult:
        started = time.perf_counter()
        normalized = normalize_query(query)
        if not normalized:
            raise ValueError("Recall query cannot be empty")

        memories = await self.memory_store.search(query, group_id)
        candidates = await self._durable_candidates(query, group_id, memories, session)
        selection = await self.judge.select(
            query,
            [
                {
                    "episode_id": str(episode.id),
                    "text": matched_text(episode, memory),
                }
                for episode, memory in candidates
            ],
        )
        if selection.answerable and selection.selected_episode_id:
            selected = next(
                (
                    (episode, memory)
                    for episode, memory in candidates
                    if str(episode.id) == selection.selected_episode_id
                ),
                None,
            )
            if selected is not None:
                episode, memory = selected
                result = RecallResult(
                    query=query,
                    answer=answer_text(episode, memory),
                    answer_origin="episodic",
                    citations=episode_citations(episode, citation_excerpt(episode, memory)),
                    selected_ids=[str(episode.id)],
                    score_details={
                        "memory_score": memory.get("score"),
                        "answerability_confidence": selection.confidence,
                        "memory_kind": episode.kind.value,
                        # How much of the conversation the query actually hit,
                        # and where it started: a single chunk deep into a long
                        # thread reads very differently from ten scattered ones.
                        "chunk_index": (memory.get("metadata") or {}).get("chunk_index"),
                        "chunk_matches": (memory.get("metadata") or {}).get("chunk_matches"),
                        "fact_matches": (memory.get("metadata") or {}).get("fact_matches"),
                    },
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                await self._record_event(result, group_id, normalized, session)
                return result

        # The judge declined, so no single conversation is being put forward as
        # the answer. The conversations that matched are still the best evidence
        # there is, and they are carried through as citations rather than
        # dropped: a caller that renders sources, or composes its own answer
        # from them, should not be left with only the graph's abstraction
        # because one yes/no went the other way.
        retrieved = [
            citation
            for episode, memory in candidates
            for citation in episode_citations(episode, citation_excerpt(episode, memory))
        ]

        semantic = await search_semantic_context(query, group_id, session)
        if semantic.answer is None:
            result = RecallResult(
                query=query,
                answer=None,
                answer_origin="miss",
                citations=retrieved,
                score_details={"episodic_candidates": len(candidates)},
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            await self._record_event(result, group_id, normalized, session)
            return result

        retrieval_episode = await self._write_back(
            query=query,
            group_id=group_id,
            normalized_query=normalized,
            answer=semantic.answer,
            citations=semantic.citations,
            session=session,
        )
        result = RecallResult(
            query=query,
            answer=semantic.answer,
            answer_origin="semantic_fallback",
            # Graph citations first — they are what the answer was drawn from —
            # followed by the conversations that matched but were not selected.
            citations=[*semantic.citations, *retrieved],
            selected_ids=semantic.selected_ids,
            score_details={
                "episodic_candidates": len(candidates),
                "semantic_matches": len(semantic.selected_ids),
            },
            created_episode_id=str(retrieval_episode.id),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        await self._record_event(result, group_id, normalized, session)
        return result

    async def _durable_candidates(
        self,
        query: str,
        group_id: str,
        memories: list[dict[str, Any]],
        session: AsyncSession,
    ) -> list[tuple[Episode, dict[str, Any]]]:
        # Several chunks of one episode can match a query. They are folded back
        # into a single candidate, so the judge weighs each conversation once
        # instead of spending its slots re-reading one thread.
        #
        # The fold keeps every matched chunk, not just the best-scoring one.
        # Similarity ranks the passage that reads most like the question, which
        # is not always the passage that answers it: a benchmark's setup
        # ("ran new evals, here is the workload") is prose about the topic and
        # outscores the table of numbers two messages later. Keeping both, in
        # the order they were said, is what puts the answer in front of the
        # judge — and the surrounding chunks are the context that makes a bare
        # table legible anyway.
        #
        # Writer-mode facts fold into the same candidate as their episode's
        # chunks. They compete for the same search slots on the same scores, and
        # a candidate ranks by its best hit of either kind. The judge reads the
        # facts first, each dated by its conversation, then the spans. Facts a
        # later one superseded are not searched, so they never get this far.
        matches: dict[str, list[tuple[int, float, str]]] = {}
        facts: dict[str, list[tuple[float, str]]] = {}
        episodes: dict[str, Episode] = {}
        for memory in memories:
            metadata = memory.get("metadata")
            if not isinstance(metadata, dict):
                continue
            episode_id = metadata.get("episode_id")
            if not isinstance(episode_id, str):
                continue
            episode = await session.get(Episode, episode_id)
            if episode is None or episode.stale_at is not None:
                continue
            # Mem0 filters on user_id, but it is a separate store with its own
            # index: a vector left behind by a deleted episode, or metadata
            # written before tenancy, would otherwise hand back another
            # account's text. The episode row is the authority on ownership.
            if episode.group_id != group_id:
                continue
            episodes[episode_id] = episode
            if metadata.get("memory_kind") == "fact":
                facts.setdefault(episode_id, []).append(
                    (_score(memory), str(memory.get("memory") or ""))
                )
                continue
            index = metadata.get("chunk_index")
            matches.setdefault(episode_id, []).append(
                (
                    index if isinstance(index, int) else 0,
                    _score(memory),
                    str(memory.get("memory") or ""),
                )
            )

        folded: list[tuple[Episode, dict[str, Any]]] = []
        for episode_id, episode in episodes.items():
            hits = matches.get(episode_id, [])
            fact_hits = facts.get(episode_id, [])
            best = max([hit[1] for hit in hits] + [hit[0] for hit in fact_hits])
            spans = [text for _, _, text in sorted(hits) if text.strip()]
            folded_memory: dict[str, Any] = {
                "score": best,
                "memory": "\n".join(spans),
                "metadata": {
                    "episode_id": episode_id,
                    "chunk_index": min(hit[0] for hit in hits) if hits else None,
                    "chunk_matches": len(hits),
                },
            }
            if fact_hits:
                said_on = episode.occurred_at.date().isoformat()
                lines = [
                    f"{text} (from a conversation on {said_on})"
                    for _, text in sorted(fact_hits, key=lambda hit: hit[0], reverse=True)
                    if text.strip()
                ]
                folded_memory["memory"] = "\n".join([*lines, *spans])
                folded_memory["facts"] = [
                    text
                    for _, text in sorted(fact_hits, key=lambda hit: hit[0], reverse=True)
                    if text.strip()
                ]
                folded_memory["excerpt"] = "\n".join(spans)
                folded_memory["metadata"]["fact_matches"] = len(fact_hits)
            folded.append((episode, folded_memory))

        candidates: list[tuple[Episode, dict[str, Any]]] = sorted(
            folded, key=lambda pair: _score(pair[1]), reverse=True
        )

        if needs_substantive_update(query) and candidates:
            request_episode_ids = set(
                await session.scalars(
                    select(IntentObservation.episode_id).where(
                        IntentObservation.group_id == group_id,
                        IntentObservation.episode_id.in_(
                            episode.id for episode, _ in candidates
                        ),
                        IntentObservation.intent == "request_status",
                    )
                )
            )
            candidates = [
                (episode, memory)
                for episode, memory in candidates
                if episode.id not in request_episode_ids
            ]
        return candidates

    async def _write_back(
        self,
        *,
        query: str,
        group_id: str,
        normalized_query: str,
        answer: str,
        citations: list[RecallCitation],
        session: AsyncSession,
    ) -> Episode:
        citation_ids = sorted(citation.id for citation in citations)
        digest = hashlib.sha256(
            f"{normalized_query}|{'|'.join(citation_ids)}".encode()
        ).hexdigest()[:24]
        source_id = f"recall-{digest}"
        existing = await session.scalar(
            select(Episode).where(
                Episode.group_id == group_id,
                Episode.source_type == SourceType.RECALL,
                Episode.source_id == source_id,
            )
        )
        if existing is not None:
            return existing

        now = datetime.now(UTC)
        episode = Episode(
            group_id=group_id,
            kind=EpisodeKind.RETRIEVAL,
            source_type=SourceType.RECALL,
            source_id=source_id,
            conversation_id="recall",
            content=query,
            summary=answer,
            participants=["local-user"],
            details={
                "query": query,
                "normalized_query": normalized_query,
                "citations": [citation.model_dump(mode="json") for citation in citations],
                "answer_origin": "semantic_fallback",
            },
            occurred_at=now,
        )
        session.add(episode)
        await session.flush()
        try:
            episode.mem0_id = await self.memory_store.add_retrieval_episode(
                episode_id=str(episode.id),
                group_id=group_id,
                answer=answer,
                query=query,
                occurred_at=now.isoformat(),
            )
        except Exception as error:
            episode.details = {**episode.details, "memory_index_error": type(error).__name__}
        await session.commit()
        await session.refresh(episode)
        return episode

    async def _record_event(
        self,
        result: RecallResult,
        group_id: str,
        normalized_intent: str,
        session: AsyncSession,
    ) -> None:
        session.add(
            RecallEvent(
                group_id=group_id,
                query=result.query,
                normalized_intent=normalized_intent,
                answer_origin=result.answer_origin,
                selected_ids=result.selected_ids,
                score_details=result.score_details,
                latency_ms=result.latency_ms,
                created_episode_id=result.created_episode_id,
            )
        )
        await session.commit()


def episode_citations(
    episode: Episode, excerpt: str | None = None
) -> list[RecallCitation]:
    """Citations for one episode, quoting ``excerpt`` when a span is known.

    A caller that retrieved the episode by one of its chunks should pass that
    chunk: quoting the conversation from its beginning would show the reader
    something other than the passage the answer came from.
    """
    if episode.kind == EpisodeKind.RETRIEVAL:
        stored_citations = episode.details.get("citations")
        if isinstance(stored_citations, list):
            citations = [
                RecallCitation.model_validate(item)
                for item in stored_citations
                if isinstance(item, dict)
            ]
            if citations:
                return citations
    return [
        RecallCitation(
            kind="episode",
            id=str(episode.id),
            label=f"{episode.source_type.value} episode",
            excerpt=excerpt or episode.content,
        )
    ]