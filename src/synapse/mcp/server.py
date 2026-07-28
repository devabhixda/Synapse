import uuid

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from synapse.db.models import Episode
from synapse.db.session import session_factory
from synapse.procedural.models import SkillSpecView
from synapse.procedural.service import get_skill_by_name, list_skills
from synapse.recall.models import EpisodeDetail, RecallResult
from synapse.recall.service import RecallService

mcp = FastMCP(
    "synapse-mem",
    instructions=(
        "Use recall for organizational context. Synapse always searches episodic memory "
        "before semantic memory and records successful semantic fallbacks for future recall."
    ),
)


@mcp.tool(structured_output=True)
async def recall(query: str, group_id: str) -> RecallResult:
    """Recall organizational context using the episodic-first learning loop.

    group_id names the account to search; memory is never shared across them.
    """
    async with session_factory() as session:
        return await RecallService().recall(query, group_id, session)


@mcp.tool(structured_output=True)
async def get_episode(episode_id: str, group_id: str) -> EpisodeDetail:
    """Inspect one durable episode referenced by a recall result.

    group_id must be the account the episode belongs to.
    """
    try:
        parsed_id = uuid.UUID(episode_id)
    except ValueError as error:
        raise ValueError("episode_id must be a valid UUID") from error

    async with session_factory() as session:
        episode = await session.scalar(
            select(Episode).where(Episode.id == parsed_id, Episode.group_id == group_id)
        )
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


@mcp.tool(structured_output=True)
async def list_skill_specs(group_id: str) -> list[SkillSpecView]:
    """List approved procedural workflows available to the coding agent.

    group_id names the account whose approved workflows to list.
    """
    async with session_factory() as session:
        return await list_skills(group_id, session, approved_only=True)


@mcp.tool(structured_output=True)
async def get_skill_spec(name: str, group_id: str) -> SkillSpecView:
    """Get one approved procedural workflow by its stable name.

    Names are unique per account, so group_id is required to resolve one.
    """
    async with session_factory() as session:
        return await get_skill_by_name(group_id, session, name, approved_only=True)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()