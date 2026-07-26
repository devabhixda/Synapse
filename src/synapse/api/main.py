import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from synapse.config import get_settings
from synapse.db.models import Episode
from synapse.db.session import engine, get_session
from synapse.episodic.service import SystemOneService
from synapse.inference.cactus import CactusInterpreter
from synapse.memory.mem0_store import Mem0EpisodeStore
from synapse.sources.mock import mock_events


class HealthResponse(BaseModel):
    status: str
    database: str
    cactus: str
    models: dict[str, str]


class EpisodeResponse(BaseModel):
    id: str
    kind: str
    source_type: str
    content: str
    summary: str | None
    participants: list[str]
    occurred_at: str
    mem0_id: str | None


class ReplayResponse(BaseModel):
    processed: int
    skipped: int
    indexed: int


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.dispose()


app = FastAPI(title="Synapse API", version="0.1.0", lifespan=lifespan)


async def cactus_status() -> str:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            responses = await asyncio.gather(
                client.get(f"{settings.cactus_chat_base_url.rstrip('/')}/models"),
                client.get(f"{settings.cactus_embedding_base_url.rstrip('/')}/models"),
            )
            for response in responses:
                response.raise_for_status()
    except (httpx.HTTPError, OSError):
        return "unavailable"
    return "available"


@app.get("/api/health", response_model=HealthResponse)
async def health(session: Annotated[AsyncSession, Depends(get_session)]) -> HealthResponse:
    await session.execute(text("SELECT 1"))
    settings = get_settings()
    cactus = await cactus_status()
    return HealthResponse(
        status="ok" if cactus == "available" else "degraded",
        database="available",
        cactus=cactus,
        models={
            "chat": settings.cactus_chat_model,
            "embedding": settings.cactus_embedding_model,
        },
    )


@app.get("/api/episodes", response_model=list[EpisodeResponse])
async def list_episodes(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[EpisodeResponse]:
    result = await session.scalars(select(Episode).order_by(Episode.occurred_at.desc()).limit(100))
    return [
        EpisodeResponse(
            id=str(episode.id),
            kind=episode.kind.value,
            source_type=episode.source_type.value,
            content=episode.content,
            summary=episode.summary,
            participants=episode.participants,
            occurred_at=episode.occurred_at.isoformat(),
            mem0_id=episode.mem0_id,
        )
        for episode in result
    ]


@app.get("/api/config")
async def public_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        "memory_hit_threshold": settings.memory_hit_threshold,
        "embedding_dimensions": settings.embedding_dimensions,
        "chat_model": settings.cactus_chat_model,
        "embedding_model": settings.cactus_embedding_model,
    }


@app.post("/api/demo/replay", response_model=ReplayResponse)
async def replay_demo(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ReplayResponse:
    service = SystemOneService(CactusInterpreter(), Mem0EpisodeStore())
    processed = 0
    skipped = 0
    indexed = 0
    for event in mock_events():
        episode, created = await service.process(event, session)
        if created:
            processed += 1
        else:
            skipped += 1
        indexed += int(episode.mem0_id is not None)
    return ReplayResponse(processed=processed, skipped=skipped, indexed=indexed)


def run() -> None:
    uvicorn.run("synapse.api.main:app", host="0.0.0.0", port=8000)
