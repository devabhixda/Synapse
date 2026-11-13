"""Proposals that wait for review, sentences about entities, and old databases."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from synapse.consolidation.models import ConsolidationPlan, EntitySpec, StatementSpec
from synapse.consolidation.models import RelationshipSpec as Spec
from synapse.consolidation.service import (
    apply_plan,
    retire_statement,
    review_relationship,
    review_statement,
)
from synapse.db.models import Episode, EpisodeKind, ReviewStatus, SourceType
from synapse.db.schema import create_schema
from synapse.semantic.repository import (
    load_relationships,
    load_statements,
    search_semantic_context,
)

TEAM = "team"
MARCH = datetime(2026, 3, 1, tzinfo=UTC)
JULY = datetime(2026, 7, 1, tzinfo=UTC)
THINGS = [
    EntitySpec(key="Team:cpe", kind="Team", canonical_name="CPE"),
    EntitySpec(key="Team:sre", kind="Team", canonical_name="SRE"),
    EntitySpec(key="Concept:quota", kind="Concept", canonical_name="GPU quota"),
]
OLD = "Ask in #ask-sre with the org ID."
NEW = "Ask in #ask-cpe with the org ID and region; about a day."


async def episode(source_id: str, at: datetime, session) -> None:
    session.add(Episode(
        group_id=TEAM, kind=EpisodeKind.SOURCE, source_type=SourceType.SLACK,
        source_id=source_id, conversation_id="C1", content=f"{source_id}: {OLD} {NEW}",
        participants=["Ann"], details={}, occurred_at=at,
    ))
    await session.commit()


def owns(team: str, source_id: str) -> Spec:
    return Spec(subject_key=f"Team:{team}", predicate="owns", object_key="Concept:quota",
                evidence_source_ids=[source_id])


def how_to(text: str, source_id: str) -> StatementSpec:
    return StatementSpec(entity_key="Concept:quota", key="how_to_request", text=text,
                         evidence_source_ids=[source_id], evidence_quotes={source_id: text})


def plan(*, edges=(), statements=()) -> ConsolidationPlan:
    return ConsolidationPlan(entities=THINGS, relationships=list(edges),
                             statements=list(statements))


async def test_a_proposal_changes_nothing_until_it_is_approved(session) -> None:
    await episode("march", MARCH, session)
    await episode("july", JULY, session)
    await apply_plan(plan(edges=[owns("sre", "march")]), TEAM, session)
    await apply_plan(plan(edges=[owns("cpe", "july")]), TEAM, session,
                     status=ReviewStatus.PROPOSED)
    await session.commit()

    assert [e.subject.canonical_name for e in await load_relationships(TEAM, session)] == ["SRE"]
    [proposal] = await load_relationships(TEAM, session, statuses=[ReviewStatus.PROPOSED])
    assert proposal.subject.canonical_name == "CPE"

    assert await review_relationship(proposal.id, TEAM, True, session)
    await session.commit()
    [current] = await load_relationships(TEAM, session)
    assert current.subject.canonical_name == "CPE"
    retired = [e for e in await load_relationships(TEAM, session, include_superseded=True)
               if e.subject.canonical_name == "SRE"]
    assert retired[0].invalid_at == JULY, "retired as of when it was said, not when approved"
    assert not await review_relationship(proposal.id, TEAM, True, session)


async def test_a_rejected_proposal_stays_out(session) -> None:
    await episode("march", MARCH, session)
    await apply_plan(plan(edges=[owns("cpe", "march")]), TEAM, session,
                     status=ReviewStatus.PROPOSED)
    [proposal] = await load_relationships(TEAM, session, statuses=[ReviewStatus.PROPOSED])
    assert await review_relationship(proposal.id, TEAM, False, session)
    # Proposed again from the same evidence: still rejected, not raised again.
    await apply_plan(plan(edges=[owns("cpe", "march")]), TEAM, session,
                     status=ReviewStatus.PROPOSED)
    await session.commit()
    assert await load_relationships(TEAM, session) == []
    assert await load_relationships(TEAM, session, statuses=[ReviewStatus.PROPOSED]) == []


async def test_a_new_text_under_the_same_key_versions_the_statement(session) -> None:
    await episode("march", MARCH, session)
    await episode("july", JULY, session)
    # Newest first, as a plan sorted by key might hand them over.
    await apply_plan(plan(statements=[how_to(NEW, "july"), how_to(OLD, "march")]), TEAM, session)
    await session.commit()

    [current] = await load_statements(TEAM, session)
    assert (current.text, current.valid_at) == (NEW, JULY)
    assert [item.excerpt for item in current.evidence] == [NEW]
    history = await load_statements(TEAM, session, include_superseded=True)
    old = next(s for s in history if s.text == OLD)
    assert (old.valid_at, old.invalid_at) == (MARCH, JULY)

    # The same sentence again is the same statement, not a third version.
    await apply_plan(plan(statements=[how_to(NEW.upper(), "july")]), TEAM, session)
    assert len(await load_statements(TEAM, session, include_superseded=True)) == 2


async def test_a_proposed_statement_is_reviewed_like_an_edge(session) -> None:
    await episode("march", MARCH, session)
    await episode("july", JULY, session)
    await apply_plan(plan(statements=[how_to(OLD, "march")]), TEAM, session)
    await apply_plan(plan(statements=[how_to(NEW, "july")]), TEAM, session,
                     status=ReviewStatus.PROPOSED)
    assert [s.text for s in await load_statements(TEAM, session)] == [OLD]

    [proposal] = await load_statements(TEAM, session, statuses=[ReviewStatus.PROPOSED])
    assert await review_statement(proposal.id, TEAM, True, session)
    await session.commit()
    [current] = await load_statements(TEAM, session)
    assert current.text == NEW

    assert await retire_statement(current.id, TEAM, JULY + timedelta(days=1), session)
    assert await load_statements(TEAM, session) == []


async def test_recall_reads_the_current_statement_with_its_quote(session) -> None:
    await episode("march", MARCH, session)
    await episode("july", JULY, session)
    await apply_plan(plan(statements=[how_to(OLD, "march"), how_to(NEW, "july")]), TEAM, session)
    await session.commit()

    result = await search_semantic_context("how do I get GPU quota?", TEAM, session)
    assert result.answer is not None
    assert NEW.rstrip(".") in result.answer and OLD.rstrip(".") not in result.answer
    assert [c.excerpt for c in result.citations] == [NEW]


async def test_create_schema_upgrades_a_database_from_before_the_columns(tmp_path: Path) -> None:
    """A host's existing file: edges with no status column, and no statement tables."""
    path = tmp_path / "old.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(create_schema)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE statement_evidence")
        db.execute("DROP TABLE semantic_statements")
        db.execute("DROP INDEX ix_semantic_relationships_group_current")
        db.execute("ALTER TABLE semantic_relationships DROP COLUMN status")
        db.execute("CREATE INDEX ix_semantic_relationships_group_current "
                   "ON semantic_relationships (group_id, invalid_at)")
        db.execute("INSERT INTO semantic_entities (id, group_id, kind, canonical_name,"
                   " normalized_name, attributes, confidence) VALUES"
                   " ('a', 'team', 'Team', 'CPE', 'cpe', '{}', 1), ('b', 'team', 'Concept',"
                   " 'GPU quota', 'gpu quota', '{}', 1)")
        db.execute("INSERT INTO semantic_relationships (id, group_id, subject_id, predicate,"
                   " object_id, attributes, confidence, valid_at) VALUES"
                   " ('r', 'team', 'a', 'owns', 'b', '{}', 1, '2026-03-01 00:00:00')")

    async with engine.begin() as conn:
        added = await conn.run_sync(create_schema)
        again = await conn.run_sync(create_schema)
    await engine.dispose()
    assert (added, again) == (["semantic_relationships.status"], [])
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM semantic_relationships").fetchall() == [("active",)]
        assert db.execute("SELECT count(*) FROM semantic_statements").fetchone() == (0,)
