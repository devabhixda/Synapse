from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from synapse.consolidation.extractor import derive_semantic_plan, normalize_name
from synapse.db.models import (
    ConsolidationEpisode,
    ConsolidationRun,
    EntityAlias,
    Episode,
    EpisodeKind,
    RelationshipEvidence,
    SemanticEntity,
    SemanticRelationship,
)


class ConsolidationService:
    async def run(self, session: AsyncSession) -> ConsolidationRun:
        run = ConsolidationRun(
            status="running",
            stage="snapshotting",
            provider="deterministic-system2",
            started_at=datetime.now(UTC),
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        run_id = run.id

        try:
            episodes = list(
                await session.scalars(
                    select(Episode)
                    .where(Episode.kind == EpisodeKind.SOURCE)
                    .order_by(Episode.occurred_at, Episode.id)
                )
            )
            run.episode_count = len(episodes)
            session.add_all(
                [
                    ConsolidationEpisode(run_id=run_id, episode_id=episode.id)
                    for episode in episodes
                ]
            )
            run.stage = "extracting"
            await session.commit()

            plan = derive_semantic_plan(episodes)
            run.stage = "linking"
            await session.commit()

            created_entities = 0
            updated_entities = 0
            created_relationships = 0
            updated_relationships = 0
            entity_by_key: dict[str, SemanticEntity] = {}

            for entity_spec in plan.entities:
                entity = await session.scalar(
                    select(SemanticEntity)
                    .options(selectinload(SemanticEntity.aliases))
                    .where(
                        SemanticEntity.kind == entity_spec.kind,
                        SemanticEntity.normalized_name
                        == normalize_name(entity_spec.canonical_name),
                    )
                )
                if entity is None:
                    entity = SemanticEntity(
                        kind=entity_spec.kind,
                        canonical_name=entity_spec.canonical_name,
                        normalized_name=normalize_name(entity_spec.canonical_name),
                        description=entity_spec.description,
                        attributes=entity_spec.attributes,
                        confidence=entity_spec.confidence,
                    )
                    session.add(entity)
                    created_entities += 1
                    existing_aliases: set[str] = set()
                else:
                    entity.description = entity.description or entity_spec.description
                    entity.attributes = {**entity.attributes, **entity_spec.attributes}
                    entity.confidence = max(entity.confidence, entity_spec.confidence)
                    updated_entities += 1
                    existing_aliases = {alias.normalized_alias for alias in entity.aliases}
                await session.flush()

                for alias_value in entity_spec.aliases:
                    normalized_alias = normalize_name(alias_value)
                    if normalized_alias not in existing_aliases:
                        session.add(
                            EntityAlias(
                                entity_id=entity.id,
                                alias=alias_value,
                                normalized_alias=normalized_alias,
                                confidence=entity_spec.confidence,
                            )
                        )
                        existing_aliases.add(normalized_alias)
                entity_by_key[entity_spec.key] = entity

            episode_by_source_id = {
                episode.source_id: episode for episode in episodes if episode.source_id
            }
            for relationship_spec in plan.relationships:
                subject = entity_by_key[relationship_spec.subject_key]
                object_entity = entity_by_key[relationship_spec.object_key]
                relationship = await session.scalar(
                    select(SemanticRelationship)
                    .options(selectinload(SemanticRelationship.evidence))
                    .where(
                        SemanticRelationship.subject_id == subject.id,
                        SemanticRelationship.predicate == relationship_spec.predicate,
                        SemanticRelationship.object_id == object_entity.id,
                    )
                )
                if relationship is None:
                    relationship = SemanticRelationship(
                        subject_id=subject.id,
                        predicate=relationship_spec.predicate,
                        object_id=object_entity.id,
                        attributes=relationship_spec.attributes,
                        confidence=relationship_spec.confidence,
                    )
                    session.add(relationship)
                    created_relationships += 1
                    evidence_episode_ids: set[object] = set()
                else:
                    relationship.attributes = {
                        **relationship.attributes,
                        **relationship_spec.attributes,
                    }
                    relationship.confidence = max(
                        relationship.confidence, relationship_spec.confidence
                    )
                    updated_relationships += 1
                    evidence_episode_ids = {
                        item.episode_id for item in relationship.evidence
                    }
                await session.flush()

                for source_id in relationship_spec.evidence_source_ids:
                    episode = episode_by_source_id.get(source_id)
                    if episode and episode.id not in evidence_episode_ids:
                        session.add(
                            RelationshipEvidence(
                                relationship_id=relationship.id,
                                episode_id=episode.id,
                                excerpt=episode.content,
                                confidence=relationship_spec.confidence,
                            )
                        )
                        evidence_episode_ids.add(episode.id)

            run.stage = "committing"
            run.entity_count = len(plan.entities)
            run.relationship_count = len(plan.relationships)
            run.graph_diff = {
                "entities_created": created_entities,
                "entities_updated": updated_entities,
                "relationships_created": created_relationships,
                "relationships_updated": updated_relationships,
            }
            run.status = "succeeded"
            run.stage = "complete"
            run.completed_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(run)
            return run
        except Exception as error:
            await session.rollback()
            failed_run = await session.get(ConsolidationRun, run_id)
            if failed_run is not None:
                failed_run.status = "failed"
                failed_run.stage = "failed"
                failed_run.error = f"{type(error).__name__}: {error}"
                failed_run.completed_at = datetime.now(UTC)
                await session.commit()
            raise