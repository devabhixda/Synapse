import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from synapse.config import get_settings
from synapse.consolidation.service import ConsolidationService
from synapse.db.models import GROUP_ID_LENGTH, ConsolidationRun, Episode
from synapse.db.session import engine, get_session
from synapse.episodic.service import SystemOneService
from synapse.inference.chat import LLMInterpreter
from synapse.memory.mem0_store import Mem0EpisodeStore
from synapse.procedural.models import (
    DetectionResult,
    PatternView,
    SkillSpecView,
    SkillStatusUpdate,
)
from synapse.procedural.service import ProceduralMemoryService, list_patterns, list_skills
from synapse.recall.models import RecallResult
from synapse.recall.service import RecallService
from synapse.semantic.repository import load_entities, load_relationships
from synapse.sources.mock import mock_events

GroupId = Annotated[
    str,
    Query(
        min_length=1,
        max_length=GROUP_ID_LENGTH,
        description="Account the request reads from and writes to.",
    ),
]
"""Required on every route that touches account data.

It has no default on purpose. A default would make the common case work without
naming an account, and the first time someone forgot to pass one their data
would quietly land in whichever account the default named.
"""


class HealthResponse(BaseModel):
    status: str
    database: str
    inference: str
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


class RecallRequest(BaseModel):
    query: str


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
    valid_at: str
    invalid_at: str | None
    evidence: list[GraphEvidenceResponse]


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.dispose()


app = FastAPI(title="Synapse API", version="0.1.0", lifespan=lifespan)


async def inference_status() -> str:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            responses = await asyncio.gather(
                client.get(f"{settings.chat_base_url.rstrip('/')}/models"),
                client.get(f"{settings.embedding_base_url.rstrip('/')}/models"),
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
    inference = await inference_status()
    return HealthResponse(
        status="ok" if inference == "available" else "degraded",
        database="available",
        inference=inference,
        models={
            "chat": settings.chat_model,
            "embedding": settings.embedding_model,
        },
    )


@app.get("/api/episodes", response_model=list[EpisodeResponse])
async def list_episodes(
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[EpisodeResponse]:
    result = await session.scalars(
        select(Episode)
        .where(Episode.group_id == group_id)
        .order_by(Episode.occurred_at.desc())
        .limit(100)
    )
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
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/api/demo/replay", response_model=ReplayResponse)
async def replay_demo(
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ReplayResponse:
    service = SystemOneService(LLMInterpreter(), Mem0EpisodeStore())
    processed = 0
    skipped = 0
    indexed = 0
    for event in mock_events():
        episode, created = await service.process(event, group_id, session)
        if created:
            processed += 1
        else:
            skipped += 1
        indexed += int(episode.mem0_id is not None)
    return ReplayResponse(processed=processed, skipped=skipped, indexed=indexed)


@app.post("/api/recall", response_model=RecallResult)
async def recall(
    request: RecallRequest,
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RecallResult:
    return await RecallService().recall(request.query, group_id, session)


@app.post("/api/procedural/detect", response_model=DetectionResult)
async def detect_procedural_patterns(
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DetectionResult:
    return await ProceduralMemoryService().detect(group_id, session)


@app.get("/api/procedural/patterns", response_model=list[PatternView])
async def get_procedural_patterns(
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[PatternView]:
    return await list_patterns(group_id, session)


@app.get("/api/procedural/skills", response_model=list[SkillSpecView])
async def get_procedural_skills(
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[SkillSpecView]:
    return await list_skills(group_id, session)


@app.post("/api/procedural/skills/{skill_id}/status", response_model=SkillSpecView)
async def update_procedural_skill_status(
    skill_id: str,
    update: SkillStatusUpdate,
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SkillSpecView:
    return await ProceduralMemoryService().set_status(
        skill_id, update.status, group_id, session
    )


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
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConsolidationRunResponse:
    run = await ConsolidationService().run(group_id, session)
    return consolidation_response(run)


@app.get("/api/system2/runs", response_model=list[ConsolidationRunResponse])
async def list_consolidation_runs(
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[ConsolidationRunResponse]:
    runs = await session.scalars(
        select(ConsolidationRun)
        .where(ConsolidationRun.group_id == group_id)
        .order_by(ConsolidationRun.created_at.desc())
        .limit(25)
    )
    return [consolidation_response(run) for run in runs]


@app.get("/api/system2/graph", response_model=GraphResponse)
async def get_graph(
    group_id: GroupId,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> GraphResponse:
    entities = await load_entities(group_id, session)
    relationships = await load_relationships(group_id, session)
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
                valid_at=relationship.valid_at.isoformat(),
                invalid_at=(
                    relationship.invalid_at.isoformat() if relationship.invalid_at else None
                ),
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
