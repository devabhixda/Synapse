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
            if existing.mem0_id is None and existing.summary:
                try:
                    existing.mem0_id = await self.memory_store.add_episode(
                        episode_id=str(existing.id),
                        group_id=group_id,
                        summary=existing.summary,
                        event=event,
                    )
                    existing.details = {
                        key: value
                        for key, value in existing.details.items()
                        if key != "memory_index_error"
                    }
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
            episode.mem0_id = await self.memory_store.add_episode(
                episode_id=str(episode.id),
                group_id=group_id,
                summary=observation.summary,
                event=event,
            )
        except Exception as error:
            episode.details = {**episode.details, "memory_index_error": type(error).__name__}

        await session.commit()
        await session.refresh(episode)
        return episode, True

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
                episode.mem0_id = await self.memory_store.add_episode(
                    episode_id=str(episode.id),
                    group_id=group_id,
                    summary=episode.summary or "",
                    event=_event_from(episode),
                )
                episode.details = {
                    key: value
                    for key, value in (episode.details or {}).items()
                    if key != "memory_index_error"
                }
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
