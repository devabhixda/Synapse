import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from synapse.config import DEFAULT_RELATIONSHIP_PREDICATES, Settings, get_settings
from synapse.consolidation.extractor import normalize_name
from synapse.consolidation.llm_extractor import (
    KnownEntity,
    LLMSemanticExtractor,
    SemanticExtractor,
    UnusableExtraction,
    episode_key,
)
from synapse.consolidation.models import (
    ConsolidationBacklog,
    ConsolidationPlan,
    EntitySpec,
    ExtractionResult,
    RelationshipSpec,
    StatementSpec,
)
from synapse.db.models import (
    ConsolidationEpisode,
    ConsolidationEpisodeState,
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
from synapse.inference.chat import InferenceError, MalformedResponse

# Predicates where a later assertion replaces an earlier one instead of adding
# to it. Everything else accumulates: a blocker can block several things and a
# system can depend on many others at once, so closing those would delete
# history that is still true. Being wrong in this direction is expensive —
# a wrongly retired edge is a fact the graph can no longer state — so the sets
# stay small and only hold predicates that are functional by their own meaning.
# These are the defaults; ``Settings.relationship_predicates`` is the source.
SUBJECT_FUNCTIONAL = frozenset(
    p for p, c in DEFAULT_RELATIONSHIP_PREDICATES.items() if c == "one_per_subject"
)
"""One object per subject: a task has a single assignee at a time."""

OBJECT_FUNCTIONAL = frozenset(
    p for p, c in DEFAULT_RELATIONSHIP_PREDICATES.items() if c == "one_per_object"
)
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

    @classmethod
    def from_settings(cls, settings: Settings) -> "FunctionalPredicates":
        """The single-valued predicates ``Settings.relationship_predicates`` names."""
        predicates = settings.relationship_predicates
        return cls(
            subject=frozenset(p for p, c in predicates.items() if c == "one_per_subject"),
            object=frozenset(p for p, c in predicates.items() if c == "one_per_object"),
        )


def _resolve_functional(
    functional: FunctionalPredicates | None, settings: Settings | None
) -> FunctionalPredicates:
    """An explicit ``functional`` wins; otherwise the settings say."""
    if functional is not None:
        return functional
    return FunctionalPredicates.from_settings(settings or get_settings())


@dataclass
class RelationshipDiff:
    created: int = 0
    updated: int = 0
    superseded: int = 0


async def upsert_entities(
    specs: Sequence[EntitySpec], group_id: str, session: AsyncSession
) -> tuple[dict[str, SemanticEntity], int, int]:
    """Create or update each entity; returns them by spec key, and the counts.

    A spec whose name is an existing entity's alias, of the same kind, is that
    entity: incremental runs see a name long after the entity it refers to was
    created. An alias another entity already holds is not added again, since
    one name maps to one entity in a group.
    """
    created = 0
    updated = 0
    entity_by_key: dict[str, SemanticEntity] = {}
    for entity_spec in specs:
        normalized = normalize_name(entity_spec.canonical_name)
        entity = await session.scalar(
            select(SemanticEntity)
            .options(selectinload(SemanticEntity.aliases))
            .where(
                SemanticEntity.group_id == group_id,
                SemanticEntity.kind == entity_spec.kind,
                SemanticEntity.normalized_name == normalized,
            )
        )
        if entity is None:
            entity = await session.scalar(
                select(SemanticEntity)
                .options(selectinload(SemanticEntity.aliases))
                .join(EntityAlias, EntityAlias.entity_id == SemanticEntity.id)
                .where(
                    EntityAlias.group_id == group_id,
                    EntityAlias.normalized_alias == normalized,
                    SemanticEntity.group_id == group_id,
                    SemanticEntity.kind == entity_spec.kind,
                )
            )
        if entity is None:
            entity = SemanticEntity(
                group_id=group_id,
                kind=entity_spec.kind,
                canonical_name=entity_spec.canonical_name,
                normalized_name=normalized,
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
            if (
                not normalized_alias
                or normalized_alias in existing_aliases
                or normalized_alias == entity.normalized_name
            ):
                continue
            existing_aliases.add(normalized_alias)
            if await session.scalar(
                select(EntityAlias.id).where(
                    EntityAlias.group_id == group_id,
                    EntityAlias.normalized_alias == normalized_alias,
                )
            ):
                continue
            session.add(
                EntityAlias(
                    group_id=group_id,
                    entity_id=entity.id,
                    alias=alias_value,
                    normalized_alias=normalized_alias,
                    confidence=entity_spec.confidence,
                )
            )
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
    spec: RelationshipSpec | StatementSpec, label: str, episode: Episode, cited_as: str
) -> str:
    quote = spec.evidence_quotes.get(cited_as)
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
    settings: Settings | None = None,
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
    functional = _resolve_functional(functional, settings)
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
                            source_id,
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
    settings: Settings | None = None,
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
        functional=_resolve_functional(functional, settings),
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
                        excerpt=_evidence_excerpt(
                            spec, f"{spec.entity_key} {spec.key}", episode, source_id
                        ),
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
    settings: Settings | None = None,
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
            edge, group_id, edge.valid_at, session, _resolve_functional(functional, settings)
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


def normalize_text(value: str) -> str:
    """Words only, in lower case: how names are matched against conversations."""
    return " ".join(re.sub(r"[\W_]+", " ", value.casefold()).split())


def episode_fingerprint(
    content: str,
    participants: Sequence[str] | None,
    occurred_at: datetime,
    assertions: object,
) -> str:
    """A hash of what extraction reads from an episode; changes when it must be re-read.

    The time is hashed in UTC, as the column stores it: a writer's object can
    still hold the naive or offset value it was given.
    """
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    payload = json.dumps(
        [content, list(participants or []), occurred_at.astimezone(UTC).isoformat(), assertions],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _fingerprint_of(episode: Episode) -> str:
    return episode_fingerprint(
        episode.content,
        episode.participants,
        episode.occurred_at,
        (episode.details or {}).get("semantic_assertions"),
    )


def _stored_assertions(value: object) -> object:
    """``json_extract``'s result in the form the ORM would give for the same value."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


@dataclass
class _Due:
    episode_id: uuid.UUID
    chars: int


@dataclass
class _Scan:
    due: list[_Due]
    skipped: int
    total: int


class _BudgetExhausted(Exception):
    """The next extraction call would cross the run's token budget."""


class _Interrupted(Exception):
    """The model endpoint could not be used; the run stops and the rest stays due."""


class _Rejected(Exception):
    """The model could not handle this input; the episodes in it may be the cause."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause


# Statuses that describe the endpoint (auth, a wrong model name, rate limits),
# not the request. Other 4xx answers (too long, refused content) are about the
# input, and retrying it unchanged will fail the same way.
_ENDPOINT_STATUSES = frozenset({401, 403, 404, 408, 429})

# Episodes failed on their own, with no success yet in the run, after which a
# run takes the model, not the episodes, to be at fault. More than a batch's
# worth of splitting needs, so a few bad episodes at the head of the queue do
# not stop the run before it has tried the ones after them.
_UNPROVEN_FAILURE_LIMIT = 6


def _rejects_input(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return 400 <= status < 500 and status not in _ENDPOINT_STATUSES
    # No content at all: a content filter, or a completion budget spent on a
    # batch with more to say than it allows.
    return isinstance(error, InferenceError)


# Counters a run keeps in ``graph_diff``; see ``ConsolidationSummary``.
_COUNTERS = (
    "episodes_processed",
    "episodes_failed",
    "batches",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "entities_created",
    "entities_updated",
    "relationships_created",
    "relationships_updated",
    "relationships_superseded",
    "statements_created",
    "statements_updated",
    "statements_superseded",
    "enrichments_created",
    "enrichments_updated",
    "enrichments_skipped",
    "enrichments_cached",
    "enrichments_failed",
)


@dataclass
class _Progress:
    """What a run has done so far, shared by the batches it processes."""

    budget: "_Budget"
    stats: dict[str, Any]
    providers: set[str] = field(default_factory=set)
    fallbacks: list[str] = field(default_factory=list)
    given_up: int = 0
    vanished: int = 0
    succeeded: bool = False
    calls: int = 0
    # Episodes that failed on their own before anything in the run succeeded.
    # Counted against them once something does: until then the model, not
    # the episode, may be what is failing.
    unproven: list[tuple[Episode, str, Exception]] = field(default_factory=list)


@dataclass
class _KnownIndex:
    """The group's entities by every name they go by, for showing to extraction."""

    entities: dict[uuid.UUID, KnownEntity] = field(default_factory=dict)
    by_term: dict[str, set[uuid.UUID]] = field(default_factory=dict)
    longest: int = 1

    def add(self, entity_id: uuid.UUID, kind: str, name: str, aliases: Sequence[str]) -> None:
        current = self.entities.get(entity_id)
        merged = tuple(dict.fromkeys([*(current.aliases if current else ()), *aliases]))
        self.entities[entity_id] = KnownEntity(kind=kind, canonical_name=name, aliases=merged)
        for term in (name, *aliases):
            normalized = normalize_text(term)
            if not normalized:
                continue
            self.by_term.setdefault(normalized, set()).add(entity_id)
            self.longest = max(self.longest, min(len(normalized.split()), 6))

    def mentioned(self, episodes: Sequence[Episode], limit: int) -> list[KnownEntity]:
        """Up to ``limit`` entities the episodes name, most mentioned first."""
        if limit <= 0 or not self.by_term:
            return []
        grams: Counter[str] = Counter()
        texts = [episode.content for episode in episodes] + [
            participant for episode in episodes for participant in episode.participants or []
        ]
        for text in texts:
            words = normalize_text(text).split()
            for size in range(1, self.longest + 1):
                for start in range(len(words) - size + 1):
                    grams[" ".join(words[start : start + size])] += 1
        scores: Counter[uuid.UUID] = Counter()
        for term, ids in self.by_term.items():
            count = grams.get(term, 0)
            # A two-letter name ("Al", "HR") matches too much prose to count
            # on its own; it still counts as a participant.
            if count and (len(term) > 2 or any(
                normalize_text(p) == term for e in episodes for p in e.participants or []
            )):
                for entity_id in ids:
                    scores[entity_id] += count
        ranked = sorted(
            scores, key=lambda key: (-scores[key], self.entities[key].canonical_name)
        )
        return [self.entities[key] for key in ranked[:limit]]


class ConsolidationService:
    """System 2: merge what episodes assert into the semantic graph, incrementally.

    Each run reads only episodes it has not read in their current form, oldest
    first, in batches, and stops at the run's limits; what it did not reach is
    due next run. Progress is committed batch by batch, so a run that crashes
    or is interrupted loses at most the batch in flight. Run one at a time per
    group.
    """

    def __init__(
        self,
        extractor: SemanticExtractor | None = None,
        enricher: CompanyEnricher | None = None,
        settings: Settings | None = None,
        functional: FunctionalPredicates | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self.settings = resolved
        self.extractor = extractor or LLMSemanticExtractor(resolved)
        self.enricher = enricher or CrustdataEnricher(resolved)
        self.functional = _resolve_functional(functional, resolved)

    @property
    def enrichment_enabled(self) -> bool:
        """Whether runs enrich companies; never for an enricher with no credentials."""
        return bool(getattr(self.enricher, "configured", True))

    async def backlog(self, group_id: str, session: AsyncSession) -> ConsolidationBacklog:
        """How many episodes the next runs would read, without reading them."""
        scan = await self._scan(group_id, session)
        return ConsolidationBacklog(
            due=len(scan.due), skipped=scan.skipped, total=scan.total
        )

    async def reset(self, group_id: str, session: AsyncSession) -> int:
        """Make every episode due again, so the next runs rebuild the graph from all of them.

        For after changing the kinds, predicates or extraction model. The graph
        is kept; a rebuild re-asserts what it already holds without reopening
        anything it retired. Returns how many episodes are now due.
        """
        have = set(
            await session.scalars(
                select(ConsolidationEpisodeState.episode_id).where(
                    ConsolidationEpisodeState.group_id == group_id
                )
            )
        )
        for episode_id in await session.scalars(
            select(Episode.id).where(
                Episode.group_id == group_id, Episode.kind == EpisodeKind.SOURCE
            )
        ):
            if episode_id not in have:
                session.add(ConsolidationEpisodeState(group_id=group_id, episode_id=episode_id))
        await session.execute(
            update(ConsolidationEpisodeState)
            .where(ConsolidationEpisodeState.group_id == group_id)
            .values(fingerprint=None, failed_fingerprint=None, attempts=0, last_error=None)
        )
        await session.commit()
        return len((await self._scan(group_id, session)).due)

    async def run(
        self,
        group_id: str,
        session: AsyncSession,
        *,
        max_episodes: int | None = None,
        max_tokens: int | None = None,
    ) -> ConsolidationRun:
        """Consolidate the episodes that are due, up to this run's limits.

        ``max_episodes`` and ``max_tokens`` override the settings'
        ``consolidation_max_episodes_per_run`` and
        ``consolidation_max_tokens_per_run`` for this run. Read the outcome
        with ``ConsolidationSummary.from_run``; while its ``has_more`` is true
        there are episodes left for another run.

        If the model endpoint fails (a network error, an HTTP error, a
        configuration fault) the run stops, keeps the batches it finished, and
        returns with status ``failed``; nothing is counted against the episodes
        it did not reach, so the next run resumes there. Any other error rolls
        back the batch in flight, marks the run failed and is raised.
        """
        episode_limit = (
            max_episodes if max_episodes is not None
            else self.settings.consolidation_max_episodes_per_run
        )
        token_limit = (
            max_tokens if max_tokens is not None
            else self.settings.consolidation_max_tokens_per_run
        )
        model = getattr(self.extractor, "model", None)
        run = ConsolidationRun(
            group_id=group_id,
            status="running",
            stage="snapshotting",
            provider=(
                f"{model}-system2"
                if model and self.settings.consolidation_mode != "deterministic"
                else "deterministic-system2"
            ),
            started_at=datetime.now(UTC),
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        run_id = run.id
        stats: dict[str, Any] = dict.fromkeys(_COUNTERS, 0)
        stats["tokens_estimated"] = False
        stats["enrichment_enabled"] = self.enrichment_enabled
        progress = _Progress(budget=_Budget(token_limit), stats=stats)

        def record(stopped_reason: str, remaining: int, skipped: int) -> None:
            stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]
            run.episode_count = stats["episodes_processed"]
            run.graph_diff = {
                **stats,
                **(
                    {"extraction_fallback": progress.fallbacks[-1]}
                    if progress.fallbacks
                    else {}
                ),
                "stopped_reason": stopped_reason,
                "episodes_remaining": remaining,
                "episodes_skipped": skipped,
                # Worth running again now: work is left and nothing is broken.
                "has_more": remaining > 0 and stopped_reason != "error",
            }
            if progress.providers:
                run.provider = "+".join(sorted(progress.providers))[:80]

        try:
            scan = await self._scan(group_id, session)
            due = scan.due
            selected = due if episode_limit is None else due[:episode_limit]
            record("complete", len(due), scan.skipped)
            run.stage = "extracting"
            await session.commit()

            known = (
                await self._known_index(group_id, session)
                if selected and getattr(self.extractor, "uses_known_entities", False)
                else None
            )
            stopped_reason = "max_episodes" if len(selected) < len(due) else "complete"
            cursor = 0
            while cursor < len(selected):
                batch = self._next_batch(selected, cursor)
                episodes = await self._load(group_id, [item.episode_id for item in batch], session)
                progress.vanished += len(batch) - len(episodes)
                try:
                    await self._process(episodes, group_id, run_id, session, known, progress)
                except _BudgetExhausted:
                    stopped_reason = "max_tokens"
                    break
                except _Interrupted as error:
                    stopped_reason = "error"
                    run.status = "failed"
                    run.error = str(error)
                    break
                cursor += len(batch)

            if progress.unproven and stopped_reason != "error":
                # Every call this run made failed, too few to blame the model
                # but with nothing to show it works: counted against nobody.
                stopped_reason = "error"
                run.status = "failed"
                run.error = (
                    "the model failed on every episode it was given; nothing was counted "
                    f"against them: {progress.unproven[-1][2]}"
                )
            remaining = (
                len(due) - stats["episodes_processed"] - progress.given_up - progress.vanished
            )
            record(stopped_reason, max(remaining, 0), scan.skipped + progress.given_up)
            if run.status != "failed":
                run.status = "succeeded"
            run.stage = "complete" if run.status == "succeeded" else "failed"
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
                failed_run.graph_diff = {
                    **(failed_run.graph_diff or {}),
                    "stopped_reason": "error",
                }
                failed_run.completed_at = datetime.now(UTC)
                await session.commit()
            raise

    def _next_batch(self, selected: Sequence[_Due], cursor: int) -> list[_Due]:
        limit_episodes = self.settings.consolidation_batch_episodes
        limit_chars = self.settings.consolidation_batch_chars
        batch: list[_Due] = []
        chars = 0
        for item in selected[cursor:]:
            size = min(item.chars, self.settings.consolidation_episode_chars)
            if batch and (len(batch) >= limit_episodes or chars + size > limit_chars):
                break
            batch.append(item)
            chars += size
        return batch

    async def _scan(self, group_id: str, session: AsyncSession) -> _Scan:
        """Which source episodes are due, oldest first, and how many are skipped."""
        rows = (
            await session.execute(
                select(
                    Episode.id,
                    Episode.content,
                    Episode.participants,
                    Episode.occurred_at,
                    func.json_extract(Episode.details, "$.semantic_assertions"),
                )
                .where(
                    Episode.group_id == group_id,
                    Episode.kind == EpisodeKind.SOURCE,
                    Episode.stale_at.is_(None),
                )
                .order_by(Episode.occurred_at, Episode.id)
            )
        ).all()
        states = {
            state.episode_id: state
            for state in await session.scalars(
                select(ConsolidationEpisodeState).where(
                    ConsolidationEpisodeState.group_id == group_id
                )
            )
        }
        fingerprints = {
            row[0]: episode_fingerprint(row[1], row[2], row[3], _stored_assertions(row[4]))
            for row in rows
        }

        due: list[_Due] = []
        skipped = 0
        max_attempts = self.settings.consolidation_max_attempts
        for row in rows:
            fingerprint = fingerprints[row[0]]
            state = states.get(row[0])
            if state is not None and state.fingerprint == fingerprint:
                continue
            if (
                state is not None
                and state.failed_fingerprint == fingerprint
                and state.attempts >= max_attempts
            ):
                skipped += 1
                continue
            due.append(_Due(episode_id=row[0], chars=len(row[1])))
        return _Scan(due=due, skipped=skipped, total=len(rows))

    async def _load(
        self, group_id: str, episode_ids: Sequence[uuid.UUID], session: AsyncSession
    ) -> list[Episode]:
        loaded = {
            episode.id: episode
            for episode in await session.scalars(
                select(Episode).where(
                    Episode.group_id == group_id, Episode.id.in_(episode_ids)
                )
            )
        }
        # Deleted since the scan: nothing left to read.
        return [loaded[episode_id] for episode_id in episode_ids if episode_id in loaded]

    async def _known_index(self, group_id: str, session: AsyncSession) -> _KnownIndex:
        index = _KnownIndex()
        aliases: dict[uuid.UUID, list[str]] = {}
        for entity_id, alias in await session.execute(
            select(EntityAlias.entity_id, EntityAlias.alias).where(
                EntityAlias.group_id == group_id
            )
        ):
            aliases.setdefault(entity_id, []).append(alias)
        for entity_id, kind, name in await session.execute(
            select(SemanticEntity.id, SemanticEntity.kind, SemanticEntity.canonical_name).where(
                SemanticEntity.group_id == group_id
            )
        ):
            index.add(entity_id, kind, name, aliases.get(entity_id, []))
        return index

    async def _extract(
        self, episodes: list[Episode], known: _KnownIndex | None, progress: _Progress
    ) -> ExtractionResult:
        mentioned = (
            known.mentioned(episodes, self.settings.consolidation_known_entities)
            if known is not None
            else []
        )
        estimate_tokens = getattr(self.extractor, "estimate_tokens", None)
        estimate = estimate_tokens(episodes, mentioned) if estimate_tokens else 0
        # A run's first call may go over for one episode, or an episode larger
        # than the whole budget would stop every run before it.
        first = len(episodes) == 1 and progress.calls == 0
        if not first and not progress.budget.allows(estimate):
            raise _BudgetExhausted
        progress.calls += 1
        # Each batch commits before the next is extracted, so nothing holds
        # SQLite's write lock while the model thinks.
        try:
            if known is not None:
                return await self.extractor.extract(  # type: ignore[call-arg]
                    episodes, known_entities=mentioned
                )
            return await self.extractor.extract(episodes)
        except UnusableExtraction as error:
            progress.budget.spend(error.prompt_tokens + error.completion_tokens)
            progress.stats["llm_calls"] += 1
            progress.stats["prompt_tokens"] += error.prompt_tokens
            progress.stats["completion_tokens"] += error.completion_tokens
            raise _Rejected(error) from error
        except MalformedResponse as error:
            raise _Rejected(error) from error
        except (httpx.HTTPError, InferenceError) as error:
            if _rejects_input(error):
                raise _Rejected(error) from error
            raise _Interrupted(f"{type(error).__name__}: {error}") from error

    async def _process(
        self,
        episodes: list[Episode],
        group_id: str,
        run_id: uuid.UUID,
        session: AsyncSession,
        known: _KnownIndex | None,
        progress: _Progress,
    ) -> None:
        """Extract and apply one batch, splitting it when it cannot go whole.

        A batch the model answers unusably, or refuses, is split in two and
        each half tried on its own, down to single episodes; an episode that
        fails alone is counted a failed attempt and stays due until it has
        failed too often. A refusal, an empty or a cut-off answer is counted
        only once the run has shown the model works, and a run whose first six
        episodes all fail that way stops instead, counting nothing. A batch too
        large for what is left of the token budget is split the same way; when
        even one episode does not fit, the run stops there (unless it is the
        run's first call, which may go over for one episode).
        """
        if not episodes:
            return
        stats = progress.stats
        fingerprints = {episode.id: _fingerprint_of(episode) for episode in episodes}
        try:
            extraction = await self._extract(episodes, known, progress)
        except (_Rejected, _BudgetExhausted) as error:
            if len(episodes) > 1:
                middle = len(episodes) // 2
                for half in (episodes[:middle], episodes[middle:]):
                    await self._process(half, group_id, run_id, session, known, progress)
                return
            if isinstance(error, _BudgetExhausted):
                raise
            cause = error.cause if isinstance(error, _Rejected) else error
            failure = (episodes[0], fingerprints[episodes[0].id], cause)
            stats["episodes_failed"] += 1
            # A complete answer that is not a graph shows the model works and
            # this episode is what it cannot handle.
            answered = isinstance(cause, MalformedResponse) and not getattr(
                cause, "truncated", False
            )
            if progress.succeeded or answered:
                await self._count_failures([failure], run_id, session, progress)
                return
            progress.unproven.append(failure)
            if len(progress.unproven) >= _UNPROVEN_FAILURE_LIMIT:
                raise _Interrupted(
                    f"the first {len(progress.unproven)} episodes all failed, so the model "
                    f"is taken to be at fault and nothing was counted against them; "
                    f"last: {error}"
                ) from error
            return

        progress.budget.spend(extraction.prompt_tokens + extraction.completion_tokens)
        if not progress.succeeded:
            progress.succeeded = True
            if progress.unproven:
                await self._count_failures(progress.unproven, run_id, session, progress)
                progress.unproven = []
        stats["llm_calls"] += extraction.llm_calls
        stats["prompt_tokens"] += extraction.prompt_tokens
        stats["completion_tokens"] += extraction.completion_tokens
        stats["tokens_estimated"] = stats["tokens_estimated"] or extraction.tokens_estimated
        progress.providers.add(extraction.provider)
        if extraction.fallback_reason:
            progress.fallbacks.append(extraction.fallback_reason)

        plan = extraction.plan
        evidence = await self._evidence(plan, episodes, group_id, session)
        entity_by_key, created, updated = await upsert_entities(
            plan.entities, group_id, session
        )
        stats["entities_created"] += created
        stats["entities_updated"] += updated
        if self.enrichment_enabled:
            await self._enrich(plan, entity_by_key, group_id, session, stats)
        diff = await apply_relationships(
            plan.relationships,
            entity_by_key,
            evidence,
            group_id,
            session,
            functional=self.functional,
        )
        stats["relationships_created"] += diff.created
        stats["relationships_updated"] += diff.updated
        stats["relationships_superseded"] += diff.superseded
        stated = await apply_statements(
            plan.statements, entity_by_key, evidence, group_id, session
        )
        stats["statements_created"] += stated.created
        stats["statements_updated"] += stated.updated
        stats["statements_superseded"] += stated.superseded

        now = datetime.now(UTC)
        states = await self._states(group_id, [episode.id for episode in episodes], session)
        for episode in episodes:
            state = states[episode.id]
            state.fingerprint = fingerprints[episode.id]
            state.run_id = run_id
            state.consolidated_at = now
            state.failed_fingerprint = None
            state.attempts = 0
            state.last_error = None
            session.add(
                ConsolidationEpisode(group_id=group_id, run_id=run_id, episode_id=episode.id)
            )
        stats["episodes_processed"] += len(episodes)
        stats["batches"] += 1
        run = await session.get(ConsolidationRun, run_id)
        if run is not None:
            run.episode_count = stats["episodes_processed"]
            run.entity_count += len(plan.entities)
            run.relationship_count += len(plan.relationships)
            run.graph_diff = {
                **(run.graph_diff or {}),
                **stats,
                "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
            }
        await session.commit()

        if known is not None:
            for spec in plan.entities:
                entity = entity_by_key[spec.key]
                known.add(entity.id, entity.kind, entity.canonical_name, spec.aliases)

    async def _states(
        self, group_id: str, episode_ids: Sequence[uuid.UUID], session: AsyncSession
    ) -> dict[uuid.UUID, ConsolidationEpisodeState]:
        states = {
            state.episode_id: state
            for state in await session.scalars(
                select(ConsolidationEpisodeState).where(
                    ConsolidationEpisodeState.episode_id.in_(episode_ids)
                )
            )
        }
        for episode_id in episode_ids:
            if episode_id not in states:
                states[episode_id] = ConsolidationEpisodeState(
                    group_id=group_id, episode_id=episode_id, attempts=0
                )
                session.add(states[episode_id])
        return states

    async def _count_failures(
        self,
        failures: Sequence[tuple[Episode, str, Exception]],
        run_id: uuid.UUID,
        session: AsyncSession,
        progress: _Progress,
    ) -> None:
        """Count a failed attempt at each episode's current version."""
        for episode, fingerprint, error in failures:
            state = (await self._states(episode.group_id, [episode.id], session))[episode.id]
            if state.failed_fingerprint == fingerprint:
                state.attempts = (state.attempts or 0) + 1
            else:
                state.failed_fingerprint = fingerprint
                state.attempts = 1
            state.last_error = f"{type(error).__name__}: {error}"[:2000]
            state.run_id = run_id
            if state.attempts >= self.settings.consolidation_max_attempts:
                progress.given_up += 1
        await session.commit()

    async def _evidence(
        self,
        plan: ConsolidationPlan,
        episodes: Sequence[Episode],
        group_id: str,
        session: AsyncSession,
    ) -> dict[str, Episode]:
        """Evidence episodes by every key a plan may name them with.

        The batch's own by source id and by episode id, and any other episode
        of the group a plan cites by source id.
        """
        evidence: dict[str, Episode] = {}
        for episode in episodes:
            evidence[episode_key(episode)] = episode
            if episode.source_id:
                evidence[episode.source_id] = episode
        cited = (
            {sid for spec in plan.relationships for sid in spec.evidence_source_ids}
            | {sid for spec in plan.statements for sid in spec.evidence_source_ids}
        ) - set(evidence)
        if cited:
            for episode in await session.scalars(
                select(Episode).where(
                    Episode.group_id == group_id, Episode.source_id.in_(cited)
                )
            ):
                if episode.source_id:
                    evidence.setdefault(episode.source_id, episode)
        return evidence

    async def _enrich(
        self,
        plan: ConsolidationPlan,
        entity_by_key: Mapping[str, SemanticEntity],
        group_id: str,
        session: AsyncSession,
        stats: dict[str, Any],
    ) -> None:
        for entity_spec in plan.entities:
            if entity_spec.kind != "Company":
                continue
            entity = entity_by_key[entity_spec.key]
            domain = entity_spec.attributes.get("domain")
            cached_enrichment = await session.scalar(
                select(ExternalAttribute).where(
                    ExternalAttribute.group_id == group_id,
                    ExternalAttribute.entity_id == entity.id,
                    ExternalAttribute.provider == "crustdata",
                    ExternalAttribute.field_group == "company_profile",
                    ExternalAttribute.updated_at
                    >= datetime.now(UTC)
                    - timedelta(hours=self.settings.enrichment_cache_hours),
                )
            )
            if cached_enrichment is not None:
                stats["enrichments_cached"] += 1
                continue
            try:
                enrichment = await self.enricher.enrich_company(
                    entity_spec.canonical_name, domain
                )
            except Exception:
                stats["enrichments_failed"] += 1
                continue
            if enrichment is None:
                stats["enrichments_skipped"] += 1
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
                session.add(
                    ExternalAttribute(
                        group_id=group_id,
                        entity_id=entity.id,
                        provider=enrichment.provider,
                        field_group=enrichment.field_group,
                        payload=enrichment.payload,
                        source_url=enrichment.source_url,
                        confidence=enrichment.confidence,
                    )
                )
                stats["enrichments_created"] += 1
            else:
                external_attribute.payload = enrichment.payload
                external_attribute.source_url = enrichment.source_url
                external_attribute.confidence = enrichment.confidence
                external_attribute.updated_at = datetime.now(UTC)
                stats["enrichments_updated"] += 1


@dataclass
class _Budget:
    limit: int | None
    spent: int = 0

    def allows(self, estimate: int) -> bool:
        return self.limit is None or estimate <= 0 or self.spent + estimate <= self.limit

    def spend(self, tokens: int) -> None:
        self.spent += tokens
