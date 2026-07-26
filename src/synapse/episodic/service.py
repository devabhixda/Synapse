from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from synapse.db.models import Episode, EpisodeKind, IntentObservation
from synapse.domain.models import EpisodeMemoryStore, FastInterpreter, SourceEvent


class SystemOneService:
    def __init__(self, interpreter: FastInterpreter, memory_store: EpisodeMemoryStore) -> None:
        self.interpreter = interpreter
        self.memory_store = memory_store

    async def process(self, event: SourceEvent, session: AsyncSession) -> tuple[Episode, bool]:
        existing = await session.scalar(
            select(Episode).where(
                Episode.source_type == event.source_type,
                Episode.source_id == event.source_id,
            )
        )
        if existing is not None:
            if existing.mem0_id is None and existing.summary:
                try:
                    existing.mem0_id = await self.memory_store.add_episode(
                        episode_id=str(existing.id), summary=existing.summary, event=event
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
                episode_id=str(episode.id), summary=observation.summary, event=event
            )
        except Exception as error:
            episode.details = {**episode.details, "memory_index_error": type(error).__name__}

        await session.commit()
        await session.refresh(episode)
        return episode, True
