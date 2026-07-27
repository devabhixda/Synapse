"""Retiring edges that a later conversation contradicts.

Consolidation is driven with a fixed plan rather than a model, so the input is
exactly the assertion under test. Episode timestamps carry the meaning here:
supersession is ordered by when things were said, not by when consolidation
happened to run.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from synapse.consolidation.models import ConsolidationPlan, EntitySpec, ExtractionResult
from synapse.consolidation.models import RelationshipSpec as Spec
from synapse.consolidation.service import ConsolidationService
from synapse.db.models import Episode, EpisodeKind, SemanticRelationship, SourceType
from synapse.semantic.repository import load_relationships

ACME = "acme"
GLOBEX = "globex"
MARCH = datetime(2026, 3, 1, tzinfo=UTC)
JULY = datetime(2026, 7, 1, tzinfo=UTC)

PEOPLE = [
    EntitySpec(key="Person:sam", kind="Person", canonical_name="Sam"),
    EntitySpec(key="Person:priya", kind="Person", canonical_name="Priya"),
    EntitySpec(key="Task:demo", kind="Task", canonical_name="Demo narrative"),
    EntitySpec(key="Task:spike", kind="Task", canonical_name="API spike"),
]


class FixedExtractor:
    def __init__(self, plan: ConsolidationPlan) -> None:
        self.plan = plan

    async def extract(self, episodes: list[Episode]) -> ExtractionResult:
        return ExtractionResult(plan=self.plan, provider="stub")


async def add_episode(group_id: str, source_id: str, occurred_at: datetime, session) -> None:
    session.add(
        Episode(
            group_id=group_id,
            kind=EpisodeKind.SOURCE,
            source_type=SourceType.SLACK,
            source_id=source_id,
            conversation_id="thread",
            content=f"{source_id} said something",
            participants=["Sam"],
            details={},
            occurred_at=occurred_at,
        )
    )
    await session.commit()


async def consolidate(group_id: str, relationships: list[Spec], session) -> None:
    plan = ConsolidationPlan(entities=PEOPLE, relationships=relationships)
    await ConsolidationService(extractor=FixedExtractor(plan), enricher=None).run(
        group_id, session
    )


async def edges(group_id: str, session, *, include_superseded: bool = False):
    return await load_relationships(group_id, session, include_superseded=include_superseded)


async def test_a_new_owner_retires_the_previous_one(session) -> None:
    """owns is object-functional: the task has one owner, so Sam's edge closes."""
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(ACME, [Spec(subject_key="Person:sam", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["march"])], session)

    await add_episode(ACME, "july", JULY, session)
    await consolidate(ACME, [Spec(subject_key="Person:priya", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["july"])], session)

    current = await edges(ACME, session)
    assert len(current) == 1
    assert current[0].subject.canonical_name == "Priya"
    assert current[0].valid_at == JULY

    everything = await edges(ACME, session, include_superseded=True)
    assert len(everything) == 2, "the old edge was deleted rather than retired"
    retired = next(edge for edge in everything if edge.subject.canonical_name == "Sam")
    assert retired.invalid_at == JULY
    assert retired.valid_at == MARCH


async def test_reassignment_retires_the_previous_assignee(session) -> None:
    """assigned_to is subject-functional: the task points at one person."""
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(ACME, [Spec(subject_key="Task:demo", predicate="assigned_to",
                                  object_key="Person:sam", evidence_source_ids=["march"])], session)

    await add_episode(ACME, "july", JULY, session)
    await consolidate(ACME, [Spec(subject_key="Task:demo", predicate="assigned_to",
                                  object_key="Person:priya",
                                  evidence_source_ids=["july"])], session)

    current = await edges(ACME, session)
    assert len(current) == 1
    assert current[0].object.canonical_name == "Priya"


async def test_predicates_that_accumulate_are_left_alone(session) -> None:
    """A task can block two things at once; closing either would lose a fact."""
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(ACME, [Spec(subject_key="Task:demo", predicate="blocks",
                                  object_key="Task:spike", evidence_source_ids=["march"])], session)

    await add_episode(ACME, "july", JULY, session)
    await consolidate(ACME, [Spec(subject_key="Task:spike", predicate="blocks",
                                  object_key="Task:demo", evidence_source_ids=["july"])], session)

    current = await edges(ACME, session)
    assert len(current) == 2
    assert all(edge.invalid_at is None for edge in current)


async def test_one_person_can_own_several_things(session) -> None:
    """owns constrains the object, not the subject."""
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(
        ACME,
        [
            Spec(subject_key="Person:sam", predicate="owns", object_key="Task:demo",
                 evidence_source_ids=["march"]),
            Spec(subject_key="Person:sam", predicate="owns", object_key="Task:spike",
                 evidence_source_ids=["march"]),
        ],
        session,
    )

    current = await edges(ACME, session)
    assert len(current) == 2


async def test_a_late_import_of_old_messages_cannot_retire_a_newer_fact(session) -> None:
    """Ordering is by when it was said, not by when it was consolidated."""
    await add_episode(ACME, "july", JULY, session)
    await consolidate(ACME, [Spec(subject_key="Person:priya", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["july"])], session)

    # An old thread surfaces afterwards. It must not unseat the newer owner.
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(ACME, [Spec(subject_key="Person:sam", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["march"])], session)

    current = await edges(ACME, session)
    owners = {edge.subject.canonical_name for edge in current}
    assert "Priya" in owners, "an older assertion retired a newer one"


async def test_an_edge_asserted_again_becomes_current(session) -> None:
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(ACME, [Spec(subject_key="Person:sam", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["march"])], session)

    await add_episode(ACME, "july", JULY, session)
    await consolidate(ACME, [Spec(subject_key="Person:priya", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["july"])], session)

    later = JULY + timedelta(days=30)
    await add_episode(ACME, "august", later, session)
    await consolidate(ACME, [Spec(subject_key="Person:sam", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["august"])], session)

    current = await edges(ACME, session)
    assert len(current) == 1
    assert current[0].subject.canonical_name == "Sam"
    assert current[0].valid_at == later
    assert current[0].invalid_at is None


async def test_supersession_does_not_reach_into_another_account(session) -> None:
    for group in (ACME, GLOBEX):
        await add_episode(group, f"march-{group}", MARCH, session)
        await consolidate(group, [Spec(subject_key="Person:sam", predicate="owns",
                                       object_key="Task:demo",
                                       evidence_source_ids=[f"march-{group}"])], session)

    await add_episode(ACME, "july", JULY, session)
    await consolidate(ACME, [Spec(subject_key="Person:priya", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["july"])], session)

    globex_edges = await edges(GLOBEX, session)
    assert len(globex_edges) == 1
    assert globex_edges[0].subject.canonical_name == "Sam"
    assert globex_edges[0].invalid_at is None


async def test_the_run_reports_what_it_retired(session) -> None:
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(ACME, [Spec(subject_key="Person:sam", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["march"])], session)

    await add_episode(ACME, "july", JULY, session)
    plan = ConsolidationPlan(
        entities=PEOPLE,
        relationships=[Spec(subject_key="Person:priya", predicate="owns",
                            object_key="Task:demo", evidence_source_ids=["july"])],
    )
    run = await ConsolidationService(extractor=FixedExtractor(plan), enricher=None).run(
        ACME, session
    )

    assert run.graph_diff["relationships_superseded"] == 1


async def test_retired_edges_do_not_reach_recall_answers(session) -> None:
    """The point of all this: a contradicted fact stops being stated."""
    await add_episode(ACME, "march", MARCH, session)
    await consolidate(ACME, [Spec(subject_key="Person:sam", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["march"])], session)
    await add_episode(ACME, "july", JULY, session)
    await consolidate(ACME, [Spec(subject_key="Person:priya", predicate="owns",
                                  object_key="Task:demo", evidence_source_ids=["july"])], session)

    from synapse.semantic.repository import search_semantic_context

    result = await search_semantic_context("who owns the demo narrative?", ACME, session)

    assert result.answer is not None
    assert "Priya owns Demo narrative" in result.answer
    assert "Sam owns Demo narrative" not in result.answer

    stored = list(await session.scalars(select(SemanticRelationship)))
    assert len(stored) == 2, "history should still be on disk, just not in the answer"
