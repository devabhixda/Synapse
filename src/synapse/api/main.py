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
from synapse.consolidation.service import ConsolidationService
from synapse.db.models import ConsolidationRun, Episode
from synapse.db.session import engine, get_session
from synapse.episodic.service import SystemOneService
from synapse.inference.cactus import CactusInterpreter
from synapse.memory.mem0_store import Mem0EpisodeStore
from synapse.semantic.repository import load_entities, load_relationships
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


class ConsolidationRunResponse(BaseModel):
    id: str
    status: str
    stage: str
    provider: str
    episode_count: int
    entity_count: int
    relationship_count: int
    graph_diff: dict[str, Any]
    error: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


class GraphNodeResponse(BaseModel):
    id: str
    kind: str
    name: str
    aliases: list[str]
    description: str | None
    attributes: dict[str, Any]
    confidence: float
    external_attributes: list["ExternalAttributeResponse"]


class ExternalAttributeResponse(BaseModel):
    provider: str
    field_group: str
    payload: dict[str, Any]
    source_url: str | None
    confidence: float
    observed_at: str


class GraphEvidenceResponse(BaseModel):
    episode_id: str
    source_type: str
    occurred_at: str
    excerpt: str
    confidence: float


class GraphEdgeResponse(BaseModel):
    id: str
    source: str
    target: str
    predicate: str
    attributes: dict[str, Any]
    confidence: float
    evidence: list[GraphEvidenceResponse]


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


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


def consolidation_response(run: ConsolidationRun) -> ConsolidationRunResponse:
    return ConsolidationRunResponse(
        id=str(run.id),
        status=run.status,
        stage=run.stage,
        provider=run.provider,
        episode_count=run.episode_count,
        entity_count=run.entity_count,
        relationship_count=run.relationship_count,
        graph_diff=run.graph_diff,
        error=run.error,
        created_at=run.created_at.isoformat(),
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


@app.post("/api/system2/consolidate", response_model=ConsolidationRunResponse)
async def consolidate(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConsolidationRunResponse:
    run = await ConsolidationService().run(session)
    return consolidation_response(run)


@app.get("/api/system2/runs", response_model=list[ConsolidationRunResponse])
async def list_consolidation_runs(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[ConsolidationRunResponse]:
    runs = await session.scalars(
        select(ConsolidationRun).order_by(ConsolidationRun.created_at.desc()).limit(25)
    )
    return [consolidation_response(run) for run in runs]


@app.get("/api/system2/graph", response_model=GraphResponse)
async def get_graph(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> GraphResponse:
    entities = await load_entities(session)
    relationships = await load_relationships(session)
    return GraphResponse(
        nodes=[
            GraphNodeResponse(
                id=str(entity.id),
                kind=entity.kind,
                name=entity.canonical_name,
                aliases=[alias.alias for alias in entity.aliases],
                description=entity.description,
                attributes=entity.attributes,
                confidence=entity.confidence,
                external_attributes=[
                    ExternalAttributeResponse(
                        provider=external.provider,
                        field_group=external.field_group,
                        payload=external.payload,
                        source_url=external.source_url,
                        confidence=external.confidence,
                        observed_at=external.observed_at.isoformat(),
                    )
                    for external in entity.external_attributes
                ],
            )
            for entity in entities
        ],
        edges=[
            GraphEdgeResponse(
                id=str(relationship.id),
                source=str(relationship.subject_id),
                target=str(relationship.object_id),
                predicate=relationship.predicate,
                attributes=relationship.attributes,
                confidence=relationship.confidence,
                evidence=[
                    GraphEvidenceResponse(
                        episode_id=str(evidence.episode_id),
                        source_type=evidence.episode.source_type.value,
                        occurred_at=evidence.episode.occurred_at.isoformat(),
                        excerpt=evidence.excerpt,
                        confidence=evidence.confidence,
                    )
                    for evidence in relationship.evidence
                ],
            )
            for relationship in relationships
        ],
    )


def run() -> None:
    uvicorn.run("synapse.api.main:app", host="0.0.0.0", port=8000)
