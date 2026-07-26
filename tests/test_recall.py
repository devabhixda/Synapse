import uuid
from datetime import UTC, datetime

from synapse.db.models import Episode, EpisodeKind, SourceType
from synapse.recall.service import episode_citations


def test_retrieval_episode_preserves_original_citations() -> None:
    original_id = str(uuid.uuid4())
    retrieval = Episode(
        id=uuid.uuid4(),
        kind=EpisodeKind.RETRIEVAL,
        source_type=SourceType.RECALL,
        source_id="recall-test",
        conversation_id="recall",
        content="What system is used?",
        summary="Atlas API uses Titan.",
        participants=["local-user"],
        details={
            "citations": [
                {
                    "kind": "episode",
                    "id": original_id,
                    "label": "slack episode",
                    "excerpt": "Atlas API should use Titan.",
                }
            ]
        },
        occurred_at=datetime.now(UTC),
    )

    citations = episode_citations(retrieval)

    assert len(citations) == 1
    assert citations[0].id == original_id


async def test_mcp_lists_recall_and_episode_tools() -> None:
    from synapse.mcp.server import mcp

    tools = await mcp.list_tools()

    assert {tool.name for tool in tools} == {"recall", "get_episode"}