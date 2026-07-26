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