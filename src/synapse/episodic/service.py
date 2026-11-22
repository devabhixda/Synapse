import logging
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from synapse.db.models import Episode, EpisodeKind, IntentObservation
from synapse.domain.models import EpisodeMemoryStore, FastInterpreter, SourceEvent

logger = logging.getLogger(__name__)

# Fact ids an episode still has to settle against the group's current facts.
_UNSETTLED = "mem0_facts_unsettled"


def _event_from(episode: Episode) -> SourceEvent:
    """Rebuild the event an episode was made from, for a second indexing attempt.

    Only the fields ``add_episode`` reads for its metadata need to survive the
    round trip; the summary it indexes is passed separately and is already on
    the row.
    """
    return SourceEvent(
        source_id=episode.source_id or "",
        source_type=episode.source_type,
        conversation_id=episode.conversation_id or "",
        content=episode.content,
        participants=list(episode.participants or []),
        occurred_at=episode.occurred_at,
        metadata=dict(episode.details or {}),
    )


class SystemOneService:
    def __init__(self, interpreter: FastInterpreter, memory_store: EpisodeMemoryStore) -> None:
        self.interpreter = interpreter
        self.memory_store = memory_store

    async def process(
        self, event: SourceEvent, group_id: str, session: AsyncSession
    ) -> tuple[Episode, bool]:
        existing = await session.scalar(
            select(Episode).where(
                Episode.group_id == group_id,
                Episode.source_type == event.source_type,
                Episode.source_id == event.source_id,
            )
        )
        if existing is not None:
            # A conversation is not finished when it is first seen. A thread
            # picks up replies for days, and the source it came from hands back
            # the whole thing each time, so the same source_id arriving with a
            # longer body means the conversation grew. Keeping the first version
            # would freeze it at whatever had been said by then.
            if event.content and existing.content != event.content:
                await self._rewrite(existing, event, group_id, session)
                return existing, False

            if existing.mem0_id is None and existing.summary:
                try:
                    await self._index(existing, event, group_id, existing.summary)
                    await session.commit()
                    await session.refresh(existing)
                except Exception as error:
                    existing.details = {
                        **existing.details,
                        "memory_index_error": type(error).__name__,
                    }
                    await session.commit()
                    await session.refresh(existing)
            await self._extract_facts(existing, event, group_id, session, missing_only=True)
            return existing, False

        observation = await self.interpreter.observe(event)
        episode = Episode(
            group_id=group_id,
            kind=EpisodeKind.SOURCE,
            source_type=event.source_type,
            source_id=event.source_id,
            conversation_id=event.conversation_id,
            content=event.content,
            summary=observation.summary,
            participants=event.participants,
            details=event.metadata,
            occurred_at=event.occurred_at,
        )
        episode.observations.append(
            IntentObservation(
                group_id=group_id,
                intent=observation.intent,
                salience=observation.salience,
                entity_mentions=observation.entity_mentions,
                tags=observation.tags,
                provider=observation.provider,
                confidence=observation.confidence,
                latency_ms=observation.latency_ms,
            )
        )
        session.add(episode)
        await session.flush()

        try:
            await self._index(episode, event, group_id, observation.summary)
        except Exception as error:
            episode.details = {**episode.details, "memory_index_error": type(error).__name__}

        await session.commit()
        await session.refresh(episode)
        await self._extract_facts(episode, event, group_id, session)
        return episode, True

    async def _index(
        self, episode: Episode, event: SourceEvent, group_id: str, summary: str
    ) -> None:
        """Write the episode's vectors and remember which ones they are.

        An episode becomes several vectors, one per chunk of its body. Their ids
        are kept on the row so the set can be replaced wholesale when the body
        changes — nothing else can identify them afterwards, and vectors left
        behind go on matching text the episode no longer contains. ``mem0_id``
        keeps holding the first of them, which is what marks the row indexed.
        """
        written = await self.memory_store.add_episode(
            episode_id=str(episode.id),
            group_id=group_id,
            summary=summary,
            event=event,
        )
        # A store written against the pre-chunking protocol returns the one id
        # it made rather than a list. Indexing that as a sequence would take its
        # first character for the id, so the shape is normalised here instead.
        ids = [written] if isinstance(written, str) else list(written or [])
        episode.mem0_id = ids[0] if ids else None
        episode.details = {
            key: value
            for key, value in (episode.details or {}).items()
            if key != "memory_index_error"
        }
        episode.details = {**episode.details, "mem0_ids": list(ids)}

    @property
    def _writes_facts(self) -> bool:
        return hasattr(self.memory_store, "extract_facts") and bool(
            getattr(self.memory_store, "writes_facts", False)
        )

    async def _extract_facts(
        self,
        episode: Episode,
        event: SourceEvent,
        group_id: str,
        session: AsyncSession,
        *,
        missing_only: bool = False,
        released: Sequence[str] = (),
    ) -> None:
        """Writer mode: add the facts the model reads off the episode, if the store can.

        Runs after the episode is committed, never inside that transaction. The
        extraction is a chat call that can take seconds, and SQLite has one
        write lock: holding it across the call would stall every other writer
        past its busy timeout. The ids land on the row (``mem0_fact_ids``) so a
        rewrite can replace them; a failure is recorded and leaves them unset,
        which is what ``missing_only`` callers look for to try again. Failing
        here costs the facts, never the episode or its chunks.

        The new facts, and any ``released`` by a rewrite, are then settled
        against the group's current facts (:meth:`_settle_facts`).
        """
        if not self._writes_facts or episode.kind != EpisodeKind.SOURCE:
            return
        extract = self.memory_store.extract_facts  # type: ignore[attr-defined]
        if missing_only and "mem0_fact_ids" in (episode.details or {}):
            await self._settle_facts(episode, group_id, session)
            return
        details = {
            key: value
            for key, value in (episode.details or {}).items()
            if key != "memory_fact_error"
        }
        pending = [str(i) for i in details.get(_UNSETTLED) or []] + list(released)
        try:
            ids = [str(i) for i in await extract(
                episode_id=str(episode.id), group_id=group_id, event=event
            )]
            details["mem0_fact_ids"] = ids
            pending += ids
        except Exception as error:  # noqa: BLE001 — facts are an extra; the episode stands
            details["memory_fact_error"] = type(error).__name__
            logger.warning("extracting facts from episode %s failed: %s", episode.id, error)
        if pending and self._settles_facts:
            details[_UNSETTLED] = list(dict.fromkeys(pending))
        episode.details = details
        await session.commit()
        await session.refresh(episode)
        await self._settle_facts(episode, group_id, session)

    @property
    def _settles_facts(self) -> bool:
        return hasattr(self.memory_store, "settle_facts")

    async def _settle_facts(
        self, episode: Episode, group_id: str, session: AsyncSession
    ) -> None:
        """Retire the current facts this episode's facts contradict (or theirs, if older).

        The ids still to check wait on the row (``mem0_facts_unsettled``) until
        the check succeeds, so a failure — recorded as
        ``memory_supersession_error`` — costs only the check, and
        :meth:`reindex_missing` runs it again. The check is safe to repeat.
        """
        pending = [str(i) for i in (episode.details or {}).get(_UNSETTLED) or []]
        if not pending or not self._settles_facts:
            return
        details = {
            key: value
            for key, value in (episode.details or {}).items()
            if key not in (_UNSETTLED, "memory_supersession_error")
        }
        try:
            await self.memory_store.settle_facts(  # type: ignore[attr-defined]
                group_id=group_id, fact_ids=pending
            )
            episode.details = details
        except Exception as error:  # noqa: BLE001 — the facts stand; only the check is lost
            episode.details = {
                **details,
                _UNSETTLED: pending,
                "memory_supersession_error": type(error).__name__,
            }
            logger.warning("settling facts from episode %s failed: %s", episode.id, error)
        await session.commit()
        await session.refresh(episode)

    async def _rewrite(
        self, episode: Episode, event: SourceEvent, group_id: str, session: AsyncSession
    ) -> None:
        """Replace a stored episode's body, and its index, with a newer version.

        The interpreter runs again because what it produces — the summary, the
        intent, how salient this is — is read off the content, and the content
        is what changed. The old vectors are dropped first: they were built from
        text that is no longer what the episode says.

        A failure to re-index leaves the new body stored with a NULL ``mem0_id``,
        which is the same state a failed first indexing leaves behind, and
        :meth:`reindex_missing` repairs it.
        """
        observation = await self.interpreter.observe(event)
        stale = [str(i) for i in (episode.details or {}).get("mem0_ids") or []]
        if not stale and episode.mem0_id:
            stale = [episode.mem0_id]
        # Facts read off the old body go too: re-reading the longer one would
        # otherwise add each of them again in slightly different words.
        stale_facts = [str(i) for i in (episode.details or {}).get("mem0_fact_ids") or []]
        stale += stale_facts
        released: list[str] = []

        episode.content = event.content
        episode.summary = observation.summary
        episode.participants = event.participants
        episode.occurred_at = event.occurred_at
        episode.details = {
            key: value
            for key, value in {**(episode.details or {}), **(event.metadata or {})}.items()
            # Pending ids stay: some may be other episodes' facts still to
            # settle. Ids of facts deleted below are skipped when settled.
            if key != "mem0_fact_ids"
        }
        # Added against the foreign key rather than through
        # ``episode.observations``: the episode was loaded from the database and
        # its observations are lazy, so touching the collection would go back to
        # the database from inside a context that cannot await. The earlier
        # observations are kept — how a conversation was read as it grew is part
        # of its history.
        session.add(
            IntentObservation(
                episode_id=episode.id,
                group_id=group_id,
                intent=observation.intent,
                salience=observation.salience,
                entity_mentions=observation.entity_mentions,
                tags=observation.tags,
                provider=observation.provider,
                confidence=observation.confidence,
                latency_ms=observation.latency_ms,
            )
        )

        try:
            # Facts the old ones retired are reinstated before those go, and
            # settled again with the new ones; otherwise they would stay retired
            # by a fact that no longer exists.
            release = getattr(self.memory_store, "release_facts", None)
            if release is not None and stale_facts:
                released = list(await release(stale_facts))
            await self.memory_store.forget_memories(stale)
            await self._index(episode, event, group_id, observation.summary)
        except Exception as error:  # noqa: BLE001 — the body is the durable half
            episode.mem0_id = None
            episode.details = {
                **(episode.details or {}),
                "memory_index_error": type(error).__name__,
            }
            logger.warning("reindexing updated episode %s failed: %s", episode.id, error)

        await session.commit()
        await session.refresh(episode)
        await self._extract_facts(episode, event, group_id, session, released=released)
        logger.info("episode %s updated from a longer source event", episode.id)

    async def reindex_missing(
        self, group_id: str, session: AsyncSession, *, limit: int | None = None
    ) -> dict[str, int]:
        """Index episodes that were stored but never made it into the vector store.

        Indexing is best-effort inside :meth:`process`: an episode whose vector
        write fails is still committed, carrying a ``memory_index_error`` and a
        NULL ``mem0_id``. :meth:`process` retries such an episode the next time
        the same source event arrives — but an ingest that tracks a watermark
        never sends it again, so a failure that hit a whole run leaves memory
        permanently unsearchable with nothing to notice it.

        This is the repair. It re-reads the summary already on the row, so it
        costs one embedding per episode and no model calls. Returns the counts
        of episodes ``indexed`` and still ``failed``.

        In writer mode it also repairs facts: every source episode without
        ``mem0_fact_ids`` — its extraction failed, typically on chat quota while
        the embeddings went through, or it predates writer mode — gets its one
        extraction call, and ``facts`` counts those that now have them. Facts
        whose supersession check failed are checked again; ``settled`` counts
        the episodes whose check now succeeded.
        """
        query = (
            select(Episode)
            .where(
                Episode.group_id == group_id,
                Episode.mem0_id.is_(None),
                Episode.summary.is_not(None),
            )
            .order_by(Episode.occurred_at)
        )
        if limit is not None:
            query = query.limit(limit)
        episodes = list((await session.scalars(query)).all())

        counts = {"indexed": 0, "failed": 0}
        for episode in episodes:
            try:
                await self._index(
                    episode, _event_from(episode), group_id, episode.summary or ""
                )
                counts["indexed"] += 1
            except Exception as error:  # noqa: BLE001 — one bad row must not stop the repair
                episode.details = {
                    **(episode.details or {}),
                    "memory_index_error": type(error).__name__,
                }
                counts["failed"] += 1
                logger.warning("reindexing episode %s failed: %s", episode.id, error)
            # Commit per episode: a repair interrupted halfway keeps its work,
            # and the next run picks up exactly what is still missing.
            await session.commit()
            await session.refresh(episode)
            await self._extract_facts(
                episode, _event_from(episode), group_id, session, missing_only=True
            )
        if self._writes_facts:
            counts["facts"] = await self._extract_missing_facts(group_id, session, limit=limit)
            if self._settles_facts:
                counts["settled"] = await self._settle_missing_facts(
                    group_id, session, limit=limit
                )
        return counts

    async def _settle_missing_facts(
        self, group_id: str, session: AsyncSession, *, limit: int | None
    ) -> int:
        query = (
            select(Episode)
            .where(
                Episode.group_id == group_id,
                Episode.kind == EpisodeKind.SOURCE,
                func.json_extract(Episode.details, f"$.{_UNSETTLED}").is_not(None),
            )
            .order_by(Episode.occurred_at)
        )
        if limit is not None:
            query = query.limit(limit)
        settled = 0
        for episode in list((await session.scalars(query)).all()):
            await self._settle_facts(episode, group_id, session)
            settled += _UNSETTLED not in (episode.details or {})
        return settled

    async def _extract_missing_facts(
        self, group_id: str, session: AsyncSession, *, limit: int | None
    ) -> int:
        query = (
            select(Episode)
            .where(
                Episode.group_id == group_id,
                Episode.kind == EpisodeKind.SOURCE,
                Episode.mem0_id.is_not(None),
                func.json_extract(Episode.details, "$.mem0_fact_ids").is_(None),
            )
            .order_by(Episode.occurred_at)
        )
        if limit is not None:
            query = query.limit(limit)
        repaired = 0
        for episode in list((await session.scalars(query)).all()):
            await self._extract_facts(
                episode, _event_from(episode), group_id, session, missing_only=True
            )
            repaired += "mem0_fact_ids" in (episode.details or {})
        return repaired

    async def reindex_group(
        self, group_id: str, session: AsyncSession, *, limit: int | None = None
    ) -> dict[str, int]:
        """Rebuild the whole vector index for ``group_id`` from stored episodes.

        Unlike :meth:`reindex_missing`, which repairs the rows that never made
        it into the index, this discards the index and builds it again from
        every episode the group has. It exists because how an episode is indexed
        can change — the chunk size, the embedding model, the decision to embed
        bodies rather than summaries — and none of those changes reach episodes
        already stored. ``process`` recognises an episode it has seen and returns
        early without rewriting it, so an ingest re-run would not do this.

        Nothing is re-read from the source and no model is asked anything: the
        bodies are already on the rows. The cost is one embedding per chunk. In
        writer mode the facts are dropped with everything else and extracted
        again, one chat call per episode, which is also how episodes stored
        before writer mode was turned on get theirs.

        The old vectors are dropped first, since they cannot be matched up with
        the ones replacing them and would otherwise linger and keep matching.
        Episodes are untouched, so a failure here costs a re-run, not data.
        """
        await self.memory_store.forget_group(group_id)

        query = (
            select(Episode)
            .where(Episode.group_id == group_id, Episode.kind == EpisodeKind.SOURCE)
            .order_by(Episode.occurred_at)
        )
        if limit is not None:
            query = query.limit(limit)
        episodes = list((await session.scalars(query)).all())

        counts = {"indexed": 0, "failed": 0}
        for episode in episodes:
            episode.details = {
                key: value
                for key, value in (episode.details or {}).items()
                if key
                not in (
                    "mem0_fact_ids",
                    "memory_fact_error",
                    _UNSETTLED,
                    "memory_supersession_error",
                )
            }
            try:
                await self._index(
                    episode, _event_from(episode), group_id, episode.summary or ""
                )
                counts["indexed"] += 1
            except Exception as error:  # noqa: BLE001 — one bad row must not stop the rebuild
                episode.mem0_id = None
                episode.details = {
                    **(episode.details or {}),
                    "memory_index_error": type(error).__name__,
                }
                counts["failed"] += 1
                logger.warning("reindexing episode %s failed: %s", episode.id, error)
            await session.commit()
            await session.refresh(episode)
            await self._extract_facts(episode, _event_from(episode), group_id, session)
        return counts
