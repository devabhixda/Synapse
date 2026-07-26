import uuid

from mcp.server.fastmcp import FastMCP

from synapse.db.models import Episode
from synapse.db.session import session_factory
from synapse.recall.models import EpisodeDetail, RecallResult
from synapse.recall.service import RecallService

mcp = FastMCP(
    "synapse-memory",
    instructions=(
        "Use recall for organizational context. Synapse always searches episodic memory "
        "before semantic memory and records successful semantic fallbacks for future recall."
    ),
)


@mcp.tool(structured_output=True)
async def recall(query: str) -> RecallResult:
    """Recall organizational context using the episodic-first learning loop."""
    async with session_factory() as session:
        return await RecallService().recall(query, session)


@mcp.tool(structured_output=True)
async def get_episode(episode_id: str) -> EpisodeDetail:
    """Inspect one durable episode referenced by a recall result."""
    try:
        parsed_id = uuid.UUID(episode_id)
    except ValueError as error:
        raise ValueError("episode_id must be a valid UUID") from error

    async with session_factory() as session:
        episode = await session.get(Episode, parsed_id)
        if episode is None:
            raise ValueError(f"Episode {episode_id} was not found")
        return EpisodeDetail(
            id=str(episode.id),
            kind=episode.kind.value,
            source_type=episode.source_type.value,
            content=episode.content,
            summary=episode.summary,
            occurred_at=episode.occurred_at.isoformat(),
            participants=episode.participants,
            details=episode.details,
        )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()