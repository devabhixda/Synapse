import hashlib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from synapse.config import Settings, get_settings
from synapse.consolidation.extractor import normalize_name
from synapse.consolidation.llm_extractor import LLMSemanticExtractor, SemanticExtractor
from synapse.consolidation.models import (
    ConsolidationPlan,
    EntitySpec,
    RelationshipSpec,
    StatementSpec,
)
from synapse.db.models import (
    ConsolidationEpisode,
    ConsolidationRun,
    EntityAlias,
    Episode,
    EpisodeKind,
    ExternalAttribute,
    RelationshipEvidence,
    ReviewStatus,
    SemanticEntity,
    SemanticRelationship,
    SemanticStatement,
    StatementEvidence,
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


@dataclass(frozen=True)
class FunctionalPredicates:
    """Which predicates admit one answer at a time, and on which side.

    A host with its own vocabulary passes its own: "current_practice_for" is
    as functional as "owns", and only the host knows that.
    """

    subject: frozenset[str] = SUBJECT_FUNCTIONAL
    object: frozenset[str] = OBJECT_FUNCTIONAL


@dataclass
class RelationshipDiff:
    created: int = 0
    updated: int = 0
    superseded: int = 0


async def upsert_entities(
    specs: Sequence[EntitySpec], group_id: str, session: AsyncSession
) -> tuple[dict[str, SemanticEntity], int, int]:
    """Create or update each entity; returns them by spec key, and the counts."""
    created = 0
    updated = 0
    entity_by_key: dict[str, SemanticEntity] = {}
    for entity_spec in specs:
        entity = await session.scalar(
            select(SemanticEntity)
            .options(selectinload(SemanticEntity.aliases))
            .where(
                SemanticEntity.group_id == group_id,
                SemanticEntity.kind == entity_spec.kind,
                SemanticEntity.normalized_name == normalize_name(entity_spec.canonical_name),
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
            created += 1
            existing_aliases: set[str] = set()
        else:
            entity.description = entity.description or entity_spec.description
            entity.attributes = {**entity.attributes, **entity_spec.attributes}
            entity.confidence = max(entity.confidence, entity_spec.confidence)
            updated += 1
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
    return entity_by_key, created, updated


def _asserted_at(
    spec: RelationshipSpec | StatementSpec, episodes: Mapping[str, Episode]
) -> datetime:
    """When the edge became true: when it was said, not when it was read.

    Ordering supersession by write time would let a late import of old
    messages retire a newer fact.
    """
    if spec.asserted_at is not None:
        return spec.asserted_at
    times = [
        episode.occurred_at
        for source_id in spec.evidence_source_ids
        if (episode := episodes.get(source_id)) is not None
    ]
    return max(times) if times else datetime.now(UTC)


def _evidence_excerpt(
    spec: RelationshipSpec | StatementSpec, label: str, episode: Episode
) -> str:
    quote = spec.evidence_quotes.get(episode.source_id or "")
    if quote is None:
        return episode.content
    # Compared with whitespace collapsed: a quote is often lifted across a
    # line break. Anything else that differs means the words are not there,
    # and evidence that was not said is worse than none.
    if " ".join(quote.split()) not in " ".join(episode.content.split()):
        raise ValueError(
            f"evidence quote for {label} does not appear in episode {episode.source_id}"
        )
    return quote


async def apply_relationships(
    specs: Sequence[RelationshipSpec],
    entity_by_key: Mapping[str, SemanticEntity],
    episode_by_source_id: Mapping[str, Episode],
    group_id: str,
    session: AsyncSession,
    *,
    functional: FunctionalPredicates | None = None,
    status: ReviewStatus = ReviewStatus.ACTIVE,
) -> RelationshipDiff:
    """Merge edges into the graph, oldest assertion first.

    ``status`` is what new edges are written as. A proposed edge supersedes
    nothing until it is approved; an edge that is already active stays so.

    The order matters. Supersession compares when edges became true, and an
    edge can only close ones that were already true when it was asserted.
    Applied newest-first, the older edge arrives to find nothing it may close
    and is left open beside the newer one: two current owners. Plans come
    sorted by key, not by time, so this sorts them.
    """
    functional = functional or FunctionalPredicates()
    diff = RelationshipDiff()
    dated = sorted(
        ((_asserted_at(spec, episode_by_source_id), spec) for spec in specs),
        key=lambda pair: pair[0],
    )
    for asserted_at, relationship_spec in dated:
        subject = entity_by_key[relationship_spec.subject_key]
        object_entity = entity_by_key[relationship_spec.object_key]
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
                status=status,
            )
            session.add(relationship)
            diff.created += 1
            evidence_episode_ids: set[object] = set()
        else:
            relationship.attributes = {
                **relationship.attributes,
                **relationship_spec.attributes,
            }
            relationship.confidence = max(relationship.confidence, relationship_spec.confidence)
            if relationship.status == ReviewStatus.PROPOSED and status == ReviewStatus.ACTIVE:
                relationship.status = ReviewStatus.ACTIVE
            if relationship.invalid_at is not None and asserted_at > relationship.invalid_at:
                # Asserted again after being retired: true once more, as of
                # this assertion. Only a later assertion does that. A rebuild
                # re-asserts every edge from the evidence it always had, and
                # reopening on that would undo every retirement.
                relationship.invalid_at = None
                relationship.valid_at = asserted_at
            diff.updated += 1
            evidence_episode_ids = {item.episode_id for item in relationship.evidence}
        await session.flush()

        diff.superseded += await _supersede(
            relationship, group_id, asserted_at, session, functional
        )

        for source_id in relationship_spec.evidence_source_ids:
            episode = episode_by_source_id.get(source_id)
            if episode and episode.id not in evidence_episode_ids:
                session.add(
                    RelationshipEvidence(
                        group_id=group_id,
                        relationship_id=relationship.id,
                        episode_id=episode.id,
                        excerpt=_evidence_excerpt(
                            relationship_spec,
                            f"{relationship_spec.subject_key} {relationship_spec.predicate} "
                            f"{relationship_spec.object_key}",
                            episode,
                        ),
                        confidence=relationship_spec.confidence,
                    )
                )
                evidence_episode_ids.add(episode.id)
    return diff


async def apply_plan(
    plan: ConsolidationPlan,
    group_id: str,
    session: AsyncSession,
    *,
    functional: FunctionalPredicates | None = None,
    status: ReviewStatus = ReviewStatus.ACTIVE,
) -> RelationshipDiff:
    """Merge a plan's entities, edges and statements into the graph, without a full run.

    For a writer that works incrementally, such as an agent reading new
    conversations: the same merge a consolidation run does, over just these
    specs. Evidence episodes are looked up by source id. Edges and statements
    are written with ``status``, so an untrusted writer passes ``PROPOSED``.
    Entities carry no status: one that only proposals mention is inert.
    Flushes; the caller commits. The counts cover edges and statements.
    """
    source_ids = {sid for spec in plan.relationships for sid in spec.evidence_source_ids} | {
        sid for stated in plan.statements for sid in stated.evidence_source_ids
    }
    episodes = (
        {
            episode.source_id: episode
            for episode in await session.scalars(
                select(Episode).where(
                    Episode.group_id == group_id, Episode.source_id.in_(source_ids)
                )
            )
            if episode.source_id
        }
        if source_ids
        else {}
    )
    entity_by_key, _, _ = await upsert_entities(plan.entities, group_id, session)
    diff = await apply_relationships(
        plan.relationships,
        entity_by_key,
        episodes,
        group_id,
        session,
        functional=functional,
        status=status,
    )
    stated = await apply_statements(
        plan.statements, entity_by_key, episodes, group_id, session, status=status
    )
    return RelationshipDiff(
        diff.created + stated.created,
        diff.updated + stated.updated,
        diff.superseded + stated.superseded,
    )


def statement_fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


async def apply_statements(
    specs: Sequence[StatementSpec],
    entity_by_key: Mapping[str, SemanticEntity],
    episode_by_source_id: Mapping[str, Episode],
    group_id: str,
    session: AsyncSession,
    *,
    status: ReviewStatus = ReviewStatus.ACTIVE,
) -> RelationshipDiff:
    """Merge statements, oldest first, by the rules edges follow.

    An entity has one current statement per key, so a new text retires the
    previous one. The same text asserted again gains evidence instead.
    """
    diff = RelationshipDiff()
    dated = sorted(
        ((_asserted_at(spec, episode_by_source_id), spec) for spec in specs),
        key=lambda pair: pair[0],
    )
    for asserted_at, spec in dated:
        entity = entity_by_key[spec.entity_key]
        fingerprint = statement_fingerprint(spec.text)
        statement = await session.scalar(
            select(SemanticStatement)
            .options(selectinload(SemanticStatement.evidence))
            .where(
                SemanticStatement.group_id == group_id,
                SemanticStatement.entity_id == entity.id,
                SemanticStatement.key == spec.key,
                SemanticStatement.fingerprint == fingerprint,
            )
        )
        if statement is None:
            statement = SemanticStatement(
                group_id=group_id,
                entity_id=entity.id,
                key=spec.key,
                text=spec.text,
                fingerprint=fingerprint,
                attributes=spec.attributes,
                confidence=spec.confidence,
                valid_at=asserted_at,
                status=status,
            )
            session.add(statement)
            diff.created += 1
            evidence_episode_ids: set[object] = set()
        else:
            statement.attributes = {**statement.attributes, **spec.attributes}
            statement.confidence = max(statement.confidence, spec.confidence)
            if statement.status == ReviewStatus.PROPOSED and status == ReviewStatus.ACTIVE:
                statement.status = ReviewStatus.ACTIVE
            if statement.invalid_at is not None and asserted_at > statement.invalid_at:
                statement.invalid_at = None
                statement.valid_at = asserted_at
            diff.updated += 1
            evidence_episode_ids = {item.episode_id for item in statement.evidence}
        await session.flush()

        diff.superseded += await _supersede_statement(statement, asserted_at, session)

        for source_id in spec.evidence_source_ids:
            episode = episode_by_source_id.get(source_id)
            if episode and episode.id not in evidence_episode_ids:
                session.add(
                    StatementEvidence(
                        group_id=group_id,
                        statement_id=statement.id,
                        episode_id=episode.id,
                        excerpt=_evidence_excerpt(spec, f"{spec.entity_key} {spec.key}", episode),
                        confidence=spec.confidence,
                    )
                )
                evidence_episode_ids.add(episode.id)
    return diff


async def _supersede_statement(
    statement: SemanticStatement, asserted_at: datetime, session: AsyncSession
) -> int:
    """``_supersede`` for statements: one current text per entity and key."""
    if statement.invalid_at is not None or statement.status != ReviewStatus.ACTIVE:
        return 0
    held_at = max(statement.valid_at, asserted_at)
    others = (
        SemanticStatement.entity_id == statement.entity_id,
        SemanticStatement.key == statement.key,
        SemanticStatement.id != statement.id,
        SemanticStatement.status == ReviewStatus.ACTIVE,
    )
    closed = 0
    for other in await session.scalars(
        select(SemanticStatement).where(
            *others,
            SemanticStatement.invalid_at.is_(None),
            SemanticStatement.valid_at <= held_at,
        )
    ):
        other.invalid_at = held_at
        closed += 1
    successor = await session.scalar(
        select(func.min(SemanticStatement.valid_at)).where(
            *others, SemanticStatement.valid_at > held_at
        )
    )
    if successor is not None:
        statement.invalid_at = successor
    return closed


async def review_relationship(
    relationship_id: uuid.UUID,
    group_id: str,
    approve: bool,
    session: AsyncSession,
    *,
    functional: FunctionalPredicates | None = None,
) -> bool:
    """Accept or reject a proposed edge. False when there is no such proposal.

    Approval is when the edge takes effect: only now does it retire what it
    contradicts, dated from when it was said, not from when it was approved.
    """
    edge = await session.scalar(
        select(SemanticRelationship).where(
            SemanticRelationship.id == relationship_id,
            SemanticRelationship.group_id == group_id,
            SemanticRelationship.status == ReviewStatus.PROPOSED,
        )
    )
    if edge is None:
        return False
    edge.status = ReviewStatus.ACTIVE if approve else ReviewStatus.REJECTED
    await session.flush()
    if approve:
        await _supersede(
            edge, group_id, edge.valid_at, session, functional or FunctionalPredicates()
        )
    return True


async def review_statement(
    statement_id: uuid.UUID, group_id: str, approve: bool, session: AsyncSession
) -> bool:
    """Accept or reject a proposed statement; see ``review_relationship``."""
    statement = await session.scalar(
        select(SemanticStatement).where(
            SemanticStatement.id == statement_id,
            SemanticStatement.group_id == group_id,
            SemanticStatement.status == ReviewStatus.PROPOSED,
        )
    )
    if statement is None:
        return False
    statement.status = ReviewStatus.ACTIVE if approve else ReviewStatus.REJECTED
    await session.flush()
    if approve:
        await _supersede_statement(statement, statement.valid_at, session)
    return True


async def retire_statement(
    statement_id: uuid.UUID, group_id: str, at: datetime, session: AsyncSession
) -> bool:
    """Close one statement as of ``at``; see ``retire_relationship``."""
    statement = await session.scalar(
        select(SemanticStatement).where(
            SemanticStatement.id == statement_id,
            SemanticStatement.group_id == group_id,
            SemanticStatement.invalid_at.is_(None),
        )
    )
    if statement is None:
        return False
    statement.invalid_at = max(at, statement.valid_at)
    await session.flush()
    return True


async def retire_relationship(
    relationship_id: uuid.UUID, group_id: str, at: datetime, session: AsyncSession
) -> bool:
    """Close one edge as of ``at``. False when there is no such open edge.

    For a fact that stopped being true without anything replacing it, which
    supersession never sees. The edge is kept, so history still reads.
    """
    edge = await session.scalar(
        select(SemanticRelationship).where(
            SemanticRelationship.id == relationship_id,
            SemanticRelationship.group_id == group_id,
            SemanticRelationship.invalid_at.is_(None),
        )
    )
    if edge is None:
        return False
    edge.invalid_at = max(at, edge.valid_at)
    await session.flush()
    return True


async def _supersede(
    relationship: SemanticRelationship,
    group_id: str,
    asserted_at: datetime,
    session: AsyncSession,
    functional: FunctionalPredicates,
) -> int:
    """Settle a functional edge against the ones it contradicts. Returns how many closed.

    Open edges that were already true when this one was asserted are closed.
    An assertion cannot retire something that started later than it, which
    matters when an old export is imported after a newer one. In that case it
    is this edge that is out of date: it is closed as of the earliest
    contradicting edge that started after it, instead of being left current
    beside its successor.
    """
    predicate = relationship.predicate
    if predicate in functional.subject:
        conflict = (
            SemanticRelationship.subject_id == relationship.subject_id,
            SemanticRelationship.object_id != relationship.object_id,
        )
    elif predicate in functional.object:
        conflict = (
            SemanticRelationship.object_id == relationship.object_id,
            SemanticRelationship.subject_id != relationship.subject_id,
        )
    else:
        return 0
    same = (
        SemanticRelationship.group_id == group_id,
        SemanticRelationship.predicate == predicate,
        # Proposals neither retire accepted edges nor are retired by them.
        SemanticRelationship.status == ReviewStatus.ACTIVE,
        *conflict,
    )

    if relationship.invalid_at is not None or relationship.status != ReviewStatus.ACTIVE:
        return 0
    # The latest moment this edge is known to hold: an open edge affirmed
    # again later still dates from when it began, but it outranks anything
    # asserted before the affirmation.
    held_at = max(relationship.valid_at, asserted_at)
    closed = 0
    for edge in await session.scalars(
        select(SemanticRelationship).where(
            *same,
            SemanticRelationship.invalid_at.is_(None),
            SemanticRelationship.valid_at <= held_at,
        )
    ):
        edge.invalid_at = held_at
        closed += 1

    successor = await session.scalar(
        select(func.min(SemanticRelationship.valid_at)).where(
            *same, SemanticRelationship.valid_at > held_at
        )
    )
    if successor is not None:
        relationship.invalid_at = successor
    return closed


class ConsolidationService:
    def __init__(
        self,
        extractor: SemanticExtractor | None = None,
        enricher: CompanyEnricher | None = None,
        settings: Settings | None = None,
        functional: FunctionalPredicates | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self.extractor = extractor or LLMSemanticExtractor(resolved)
        self.enricher = enricher or CrustdataEnricher(resolved)
        self.functional = functional or FunctionalPredicates()

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

            created_enrichments = 0
            updated_enrichments = 0
            skipped_enrichments = 0
            cached_enrichments = 0
            failed_enrichments = 0
            enrichment_providers: set[str] = set()

            entity_by_key, created_entities, updated_entities = await upsert_entities(
                plan.entities, group_id, session
            )

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

            diff = await apply_relationships(
                plan.relationships,
                entity_by_key,
                {episode.source_id: episode for episode in episodes if episode.source_id},
                group_id,
                session,
                functional=self.functional,
            )
            created_relationships = diff.created
            updated_relationships = diff.updated
            superseded_relationships = diff.superseded

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
