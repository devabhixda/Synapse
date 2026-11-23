import uuid
from datetime import UTC, datetime

from synapse.db.models import Episode, EpisodeKind, SourceType
from synapse.recall.answerability import supports_query_shape
from synapse.recall.service import episode_citations, needs_substantive_update


def test_retrieval_episode_preserves_original_citations() -> None:
    original_id = str(uuid.uuid4())
    retrieval = Episode(
        id=uuid.uuid4(),
        kind=EpisodeKind.RETRIEVAL,
        source_type=SourceType.RECALL,
        source_id="recall-test",
        conversation_id="recall",
        content="What system is used?",
        summary="Prism supports Nova.",
        participants=["local-user"],
        details={
            "citations": [
                {
                    "kind": "episode",
                    "id": original_id,
                    "label": "slack episode",
                    "excerpt": "Prism should support Nova.",
                }
            ]
        },
        occurred_at=datetime.now(UTC),
    )

    citations = episode_citations(retrieval)

    assert len(citations) == 1
    assert citations[0].id == original_id


def test_substantive_status_queries_exclude_request_only_episodes() -> None:
    assert needs_substantive_update("Give me the current migration status and blockers")
    assert needs_substantive_update("What are the next actions and owners?")
    assert not needs_substantive_update("Who requested the latest status update?")


def test_compound_queries_require_the_answer_shape() -> None:
    assert not supports_query_shape(
        "What does Prism do and why is it optional?",
        "Prism indexes product metadata.",
    )
    assert supports_query_shape(
        "What does Prism do and why is it optional?",
        "Prism is optional because commerce is outside the core scope.",
    )
    assert not supports_query_shape(
        "How do Orbit Forge and Nova complement each other?",
        "Orbit Forge builds repository agents.",
    )
    assert supports_query_shape(
        "How do Orbit Forge and Nova complement each other?",
        "Orbit Forge and Nova are complementary tools.",
    )


async def test_mcp_lists_recall_and_episode_tools() -> None:
    from synapse.mcp.server import mcp

    tools = await mcp.list_tools()

    assert {tool.name for tool in tools} == {
        "recall",
        "get_episode",
        "list_skill_specs",
        "get_skill_spec",
    }

def _episode(content: str, summary: str) -> Episode:
    return Episode(
        id=uuid.uuid4(),
        group_id="acme",
        kind=EpisodeKind.SOURCE,
        source_type=SourceType.SLACK,
        source_id="slack-1",
        conversation_id="ext-acme",
        content=content,
        summary=summary,
        participants=["rishi"],
        details={},
        occurred_at=datetime.now(UTC),
    )


def test_matched_text_is_the_chunk_that_hit_not_the_summary() -> None:
    """A long thread is judged on the passage that matched, not its opening.

    The summary is the body's first characters, so for a ten-day thread it is
    whoever spoke first. Judging that is how a thread gets rejected for failing
    to answer a question its tenth message answers outright.
    """
    from synapse.recall.service import matched_text

    episode = _episode(
        content="caleb: getting a replica up tonight\nrishi: GB300 c=4 0.51 / 0.72",
        summary="caleb: getting a replica up tonight",
    )
    hit = {"memory": "rishi: GB300 c=4 0.51 / 0.72", "score": 0.8}

    assert matched_text(episode, hit) == "rishi: GB300 c=4 0.51 / 0.72"


def test_matched_text_falls_back_for_episodes_indexed_before_chunking() -> None:
    from synapse.recall.service import matched_text

    episode = _episode(content="body", summary="a summary")
    assert matched_text(episode, {}) == "a summary"
    assert matched_text(episode, {"memory": "   "}) == "a summary"


def test_citations_quote_the_matched_span_when_one_is_known() -> None:
    episode = _episode(content="the whole ten-day thread", summary="its opening")

    assert episode_citations(episode)[0].excerpt == "the whole ten-day thread"
    assert episode_citations(episode, "the passage")[0].excerpt == "the passage"


async def test_a_decline_is_confirmed_before_it_is_believed() -> None:
    """The two verdicts do not cost the same.

    Saying yes puts a cited conversation in front of someone who can read it.
    Saying no discards the episodic path and answers from the graph instead, so
    a spurious no is the expensive one and is worth asking about twice.
    """
    from synapse.recall.answerability import EpisodeAnswerabilityJudge
    from synapse.recall.models import EpisodeSelection

    verdicts = [
        EpisodeSelection(answerable=False, confidence=0.4),
        EpisodeSelection(answerable=True, selected_episode_id="e1", confidence=0.9),
    ]
    calls = []

    async def _attempt(self, query, candidates):
        calls.append(query)
        return verdicts[len(calls) - 1]

    judge = EpisodeAnswerabilityJudge.__new__(EpisodeAnswerabilityJudge)
    EpisodeAnswerabilityJudge._attempt = _attempt  # type: ignore[method-assign]
    try:
        selection = await judge.select("q", [{"episode_id": "e1", "text": "t"}])
    finally:
        del EpisodeAnswerabilityJudge._attempt

    assert len(calls) == 2
    assert selection.answerable and selection.selected_episode_id == "e1"


async def test_an_answered_query_is_judged_once() -> None:
    """The confirmation is only for declines; a yes must not cost a second call."""
    from synapse.recall.answerability import EpisodeAnswerabilityJudge
    from synapse.recall.models import EpisodeSelection

    calls = []

    async def _attempt(self, query, candidates):
        calls.append(query)
        return EpisodeSelection(answerable=True, selected_episode_id="e1", confidence=0.9)

    judge = EpisodeAnswerabilityJudge.__new__(EpisodeAnswerabilityJudge)
    EpisodeAnswerabilityJudge._attempt = _attempt  # type: ignore[method-assign]
    try:
        await judge.select("q", [{"episode_id": "e1", "text": "t"}])
    finally:
        del EpisodeAnswerabilityJudge._attempt

    assert len(calls) == 1


async def test_no_candidates_is_decided_without_asking_anyone() -> None:
    from synapse.recall.answerability import EpisodeAnswerabilityJudge

    calls = []

    async def _attempt(self, query, candidates):
        calls.append(query)
        raise AssertionError("should not be reached")

    judge = EpisodeAnswerabilityJudge.__new__(EpisodeAnswerabilityJudge)
    EpisodeAnswerabilityJudge._attempt = _attempt  # type: ignore[method-assign]
    try:
        selection = await judge.select("q", [])
    finally:
        del EpisodeAnswerabilityJudge._attempt

    assert not selection.answerable and calls == []


async def _graph_answer(session, write_back: bool):
    from datetime import UTC, datetime

    from sqlalchemy import select

    from synapse.config import Settings
    from synapse.consolidation.models import ConsolidationPlan, EntitySpec, RelationshipSpec
    from synapse.consolidation.service import apply_plan
    from synapse.db.models import Episode, EpisodeKind, SourceType
    from synapse.recall.models import EpisodeSelection
    from synapse.recall.service import RecallService

    session.add(Episode(group_id="owner", kind=EpisodeKind.SOURCE, source_type=SourceType.CALL,
                        source_id="c1", conversation_id="c1", content="Asha likes tea.",
                        participants=["Asha"], details={}, occurred_at=datetime.now(UTC)))
    await apply_plan(ConsolidationPlan(
        entities=[EntitySpec(key="Person:asha", kind="Person", canonical_name="Asha"),
                  EntitySpec(key="Concept:tea", kind="Concept", canonical_name="tea")],
        relationships=[RelationshipSpec(subject_key="Person:asha", predicate="likes",
                                        object_key="Concept:tea", evidence_source_ids=["c1"])],
    ), "owner", session)
    await session.commit()

    indexed: list[str] = []

    class Store:
        async def search(self, query: str, group_id: str) -> list[dict[str, object]]:
            return []

        async def add_retrieval_episode(self, **kwargs: object) -> str:
            indexed.append(str(kwargs["answer"]))
            return "vector-1"

    class Judge:
        async def select(self, query: str, candidates: list[object]) -> EpisodeSelection:
            return EpisodeSelection(answerable=False)

    recall = RecallService(
        memory_store=Store(),  # type: ignore[arg-type]
        judge=Judge(),  # type: ignore[arg-type]
        settings=Settings(_env_file=None, recall_write_back_semantic_fallback=write_back),
    )
    result = await recall.recall("what does Asha like?", "owner", session)
    retrievals = list(await session.scalars(
        select(Episode).where(Episode.kind == EpisodeKind.RETRIEVAL)
    ))
    return result, retrievals, indexed


async def test_a_graph_answer_is_not_written_back_by_default(session) -> None:
    """Indexing it would let a weak name match come back as if it had been said."""
    result, retrievals, indexed = await _graph_answer(session, write_back=False)

    assert result.answer_origin == "semantic_fallback"
    assert result.answer is not None and "Asha likes tea" in result.answer
    assert result.created_episode_id is None
    assert retrievals == [] and indexed == []


async def test_a_graph_answer_is_written_back_when_enabled(session) -> None:
    result, retrievals, indexed = await _graph_answer(session, write_back=True)

    assert result.answer_origin == "semantic_fallback"
    assert [str(episode.id) for episode in retrievals] == [result.created_episode_id]
    assert len(indexed) == 1
