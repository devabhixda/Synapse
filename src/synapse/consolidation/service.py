from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from synapse.config import Settings, get_settings
from synapse.consolidation.extractor import normalize_name
from synapse.consolidation.llm_extractor import LLMSemanticExtractor, SemanticExtractor
from synapse.db.models import (
    ConsolidationEpisode,
    ConsolidationRun,
    EntityAlias,
    Episode,
    EpisodeKind,
    ExternalAttribute,
    RelationshipEvidence,
    SemanticEntity,
    SemanticRelationship,
)
from synapse.enrichment.crustdata import CompanyEnricher, CrustdataEnricher

# Predicates where a later assertion replaces an earlier one instead of adding
# to it. Everything else accumulates: a blocker can block several things and a
# system can depend on many others at once, so closing those would delete
# history that is still true. Being wrong in this direction is expensive —
# a wrongly retired edge is a fact the graph can no longer state — so the sets
# stay small and only hold predicates that are functional by their own meaning.
SUBJECT_FUNCTIONAL = frozenset({"assigned_to"})
"""One object per subject: a task has a single assignee at a time."""

OBJECT_FUNCTIONAL = frozenset({"owns"})
"""One subject per object: a thing has a single owner at a time.

``owns`` points from person to thing, so the constraint is on the object. A
person owns many tasks; a task is owned by one person.
"""


class ConsolidationService:
    def __init__(
        self,
        extractor: SemanticExtractor | None = None,
        enricher: CompanyEnricher | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self.extractor = extractor or LLMSemanticExtractor(resolved)
        self.enricher = enricher or CrustdataEnricher(resolved)

    async def run(self, group_id: str, session: AsyncSession) -> ConsolidationRun:
        run = ConsolidationRun(
            group_id=group_id,
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
                    .where(
                        Episode.group_id == group_id,
                        Episode.kind == EpisodeKind.SOURCE,
                    )
                    .order_by(Episode.occurred_at, Episode.id)
                )
            )
            run.episode_count = len(episodes)
            session.add_all(
                [
                    ConsolidationEpisode(
                        group_id=group_id, run_id=run_id, episode_id=episode.id
                    )
                    for episode in episodes
                ]
            )
            run.stage = "extracting"
            await session.commit()

            extraction = await self.extractor.extract(episodes)
            plan = extraction.plan
            run.provider = extraction.provider
            if extraction.fallback_reason:
                run.graph_diff = {"extraction_fallback": extraction.fallback_reason}
            run.stage = "linking"
            await session.commit()

            created_entities = 0
            updated_entities = 0
            created_relationships = 0
            updated_relationships = 0
            created_enrichments = 0
            updated_enrichments = 0
            skipped_enrichments = 0
            cached_enrichments = 0
            failed_enrichments = 0
            enrichment_providers: set[str] = set()
            entity_by_key: dict[str, SemanticEntity] = {}

            for entity_spec in plan.entities:
                entity = await session.scalar(
                    select(SemanticEntity)
                    .options(selectinload(SemanticEntity.aliases))
                    .where(
                        SemanticEntity.group_id == group_id,
                        SemanticEntity.kind == entity_spec.kind,
                        SemanticEntity.normalized_name
                        == normalize_name(entity_spec.canonical_name),
                    )
                )
                if entity is None:
                    entity = SemanticEntity(
                        group_id=group_id,
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
                                group_id=group_id,
                                entity_id=entity.id,
                                alias=alias_value,
                                normalized_alias=normalized_alias,
                                confidence=entity_spec.confidence,
                            )
                        )
                        existing_aliases.add(normalized_alias)
                entity_by_key[entity_spec.key] = entity

            run.stage = "enriching"
            for entity_spec in plan.entities:
                if entity_spec.kind != "Company":
                    continue
                entity = entity_by_key[entity_spec.key]
                domain = entity_spec.attributes.get("domain")
                settings = get_settings()
                preferred_provider = "crustdata"
                cached_enrichment = await session.scalar(
                    select(ExternalAttribute).where(
                        ExternalAttribute.group_id == group_id,
                        ExternalAttribute.entity_id == entity.id,
                        ExternalAttribute.provider == preferred_provider,
                        ExternalAttribute.field_group == "company_profile",
                        ExternalAttribute.updated_at
                        >= datetime.now(UTC)
                        - timedelta(hours=settings.enrichment_cache_hours),
                    )
                )
                if cached_enrichment is not None:
                    cached_enrichments += 1
                    enrichment_providers.add(cached_enrichment.provider)
                    continue
                try:
                    enrichment = await self.enricher.enrich_company(
                        entity_spec.canonical_name, domain
                    )
                except Exception:
                    failed_enrichments += 1
                    continue
                if enrichment is None:
                    skipped_enrichments += 1
                    continue

                external_attribute = await session.scalar(
                    select(ExternalAttribute).where(
                        ExternalAttribute.group_id == group_id,
                        ExternalAttribute.entity_id == entity.id,
                        ExternalAttribute.provider == enrichment.provider,
                        ExternalAttribute.field_group == enrichment.field_group,
                    )
                )
                if external_attribute is None:
                    external_attribute = ExternalAttribute(
                        group_id=group_id,
                        entity_id=entity.id,
                        provider=enrichment.provider,
                        field_group=enrichment.field_group,
                        payload=enrichment.payload,
                        source_url=enrichment.source_url,
                        confidence=enrichment.confidence,
                    )
                    session.add(external_attribute)
                    created_enrichments += 1
                else:
                    external_attribute.payload = enrichment.payload
                    external_attribute.source_url = enrichment.source_url
                    external_attribute.confidence = enrichment.confidence
                    external_attribute.updated_at = datetime.now(UTC)
                    updated_enrichments += 1
                enrichment_providers.add(enrichment.provider)

            episode_by_source_id = {
                episode.source_id: episode for episode in episodes if episode.source_id
            }
            superseded_relationships = 0
            for relationship_spec in plan.relationships:
                subject = entity_by_key[relationship_spec.subject_key]
                object_entity = entity_by_key[relationship_spec.object_key]
                # When the edge became true, which is when the conversation
                # happened — not when consolidation got round to reading it.
                # Ordering supersession by write time would let a late import
                # of old messages retire a newer fact.
                evidence_times = [
                    episode.occurred_at
                    for source_id in relationship_spec.evidence_source_ids
                    if (episode := episode_by_source_id.get(source_id)) is not None
                ]
                asserted_at = max(evidence_times) if evidence_times else datetime.now(UTC)
                relationship = await session.scalar(
                    select(SemanticRelationship)
                    .options(selectinload(SemanticRelationship.evidence))
                    .where(
                        SemanticRelationship.group_id == group_id,
                        SemanticRelationship.subject_id == subject.id,
                        SemanticRelationship.predicate == relationship_spec.predicate,
                        SemanticRelationship.object_id == object_entity.id,
                    )
                )
                if relationship is None:
                    relationship = SemanticRelationship(
                        group_id=group_id,
                        subject_id=subject.id,
                        predicate=relationship_spec.predicate,
                        object_id=object_entity.id,
                        attributes=relationship_spec.attributes,
                        confidence=relationship_spec.confidence,
                        valid_at=asserted_at,
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
                    if relationship.invalid_at is not None:
                        # Asserted again after being retired: the edge is true
                        # once more, and true as of this assertion rather than
                        # the original one.
                        relationship.invalid_at = None
                        relationship.valid_at = asserted_at
                    updated_relationships += 1
                    evidence_episode_ids = {
                        item.episode_id for item in relationship.evidence
                    }
                await session.flush()

                superseded_relationships += await self._supersede_contradicted(
                    relationship, group_id, asserted_at, session
                )

                for source_id in relationship_spec.evidence_source_ids:
                    episode = episode_by_source_id.get(source_id)
                    if episode and episode.id not in evidence_episode_ids:
                        session.add(
                            RelationshipEvidence(
                                group_id=group_id,
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
                **run.graph_diff,
                "entities_created": created_entities,
                "entities_updated": updated_entities,
                "relationships_created": created_relationships,
                "relationships_updated": updated_relationships,
                "relationships_superseded": superseded_relationships,
                "enrichments_created": created_enrichments,
                "enrichments_updated": updated_enrichments,
                "enrichments_skipped": skipped_enrichments,
                "enrichments_cached": cached_enrichments,
                "enrichments_failed": failed_enrichments,
            }
            if enrichment_providers:
                run.provider = run.provider + "+" + "+".join(sorted(enrichment_providers))
            run.status = "succeeded"
            run.stage = "complete"
            run.completed_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(run)
            return run
        except Exception as error:
            await session.rollback()
            failed_run = await session.scalar(
                select(ConsolidationRun).where(
                    ConsolidationRun.id == run_id,
                    ConsolidationRun.group_id == group_id,
                )
            )
            if failed_run is not None:
                failed_run.status = "failed"
                failed_run.stage = "failed"
                failed_run.error = f"{type(error).__name__}: {error}"
                failed_run.completed_at = datetime.now(UTC)
                await session.commit()
            raise

    async def _supersede_contradicted(
        self,
        relationship: SemanticRelationship,
        group_id: str,
        asserted_at: datetime,
        session: AsyncSession,
    ) -> int:
        """Close open edges the new one contradicts. Returns how many.

        Only edges that were already true when this one was asserted are
        closed. An assertion cannot retire something that started later than
        it, which matters when an old export is imported after a newer one.
        """
        predicate = relationship.predicate
        if predicate in SUBJECT_FUNCTIONAL:
            conflict = (
                SemanticRelationship.subject_id == relationship.subject_id,
                SemanticRelationship.object_id != relationship.object_id,
            )
        elif predicate in OBJECT_FUNCTIONAL:
            conflict = (
                SemanticRelationship.object_id == relationship.object_id,
                SemanticRelationship.subject_id != relationship.subject_id,
            )
        else:
            return 0

        contradicted = list(
            await session.scalars(
                select(SemanticRelationship).where(
                    SemanticRelationship.group_id == group_id,
                    SemanticRelationship.predicate == predicate,
                    SemanticRelationship.invalid_at.is_(None),
                    SemanticRelationship.valid_at <= asserted_at,
                    *conflict,
                )
            )
        )
        for edge in contradicted:
            edge.invalid_at = asserted_at
        return len(contradicted)