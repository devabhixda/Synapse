import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from synapse.db.models import Episode, EpisodeKind, IntentObservation
from synapse.domain.models import EpisodeMemoryStore, FastInterpreter, SourceEvent

logger = logging.getLogger(__name__)


def _event_from(episode: Episode) -> SourceEvent:
    """Rebuild the event an episode was made from, for a second indexing attempt.

    Only the fields ``add_episode`` reads for its metadata need to survive the
    round trip; the summary it indexes is passed separately and is already on
    the row.
    """
    return SourceEvent(
        source_id=episode.source_id,
        source_type=episode.source_type,
        conversation_id=episode.conversation_id,
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

        episode.content = event.content
        episode.summary = observation.summary
        episode.participants = event.participants
        episode.occurred_at = event.occurred_at
        episode.details = {**(episode.details or {}), **(event.metadata or {})}
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
        return counts
