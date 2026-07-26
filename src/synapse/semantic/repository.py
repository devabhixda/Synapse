import json
import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from synapse.db.models import RelationshipEvidence, SemanticEntity, SemanticRelationship
from synapse.recall.models import RecallCitation


@dataclass
class SemanticContextResult:
    answer: str | None
    citations: list[RecallCitation]
    selected_ids: list[str]


async def load_entities(session: AsyncSession) -> list[SemanticEntity]:
    return list(
        await session.scalars(
            select(SemanticEntity)
            .options(
                selectinload(SemanticEntity.aliases),
                selectinload(SemanticEntity.external_attributes),
            )
            .order_by(SemanticEntity.kind, SemanticEntity.canonical_name)
        )
    )


async def load_relationships(session: AsyncSession) -> list[SemanticRelationship]:
    return list(
        await session.scalars(
            select(SemanticRelationship)
            .options(
                selectinload(SemanticRelationship.subject),
                selectinload(SemanticRelationship.object),
                selectinload(SemanticRelationship.evidence).selectinload(
                    RelationshipEvidence.episode
                ),
            )
            .order_by(SemanticRelationship.predicate, SemanticRelationship.id)
        )
    )


async def search_semantic_context(
    query: str, session: AsyncSession
) -> SemanticContextResult:
    entities = await load_entities(session)
    relationships = await load_relationships(session)
    normalized_query = normalize_text(query)
    query_tokens = meaningful_tokens(normalized_query)

    ranked: list[tuple[int, SemanticEntity]] = []
    for entity in entities:
        names = [entity.canonical_name, *(alias.alias for alias in entity.aliases)]
        score = 0
        for name in names:
            normalized_name = normalize_text(name)
            if normalized_name and normalized_name in normalized_query:
                score = max(score, 10)
            score = max(score, len(query_tokens & meaningful_tokens(normalized_name)) * 2)
        if score > 0:
            ranked.append((score, entity))

    if not ranked:
        return SemanticContextResult(answer=None, citations=[], selected_ids=[])

    ranked.sort(key=lambda item: (-item[0], item[1].canonical_name))
    matched_entities = [entity for _, entity in ranked[:4]]
    matched_ids = {entity.id for entity in matched_entities}
    specific_matched_ids = {
        entity.id
        for entity in matched_entities
        if entity.kind not in {"Project", "Concept"}
    }
    relationship_anchor_ids = specific_matched_ids or matched_ids
    connected = [
        relationship
        for relationship in relationships
        if relationship.subject_id in relationship_anchor_ids
        or relationship.object_id in relationship_anchor_ids
    ]

    answer_parts: list[str] = []
    citations: list[RecallCitation] = []
    selected_ids = [str(entity.id) for entity in matched_entities]
    for entity in matched_entities:
        detail_parts = [f"{entity.kind} {entity.canonical_name}"]
        if entity.description:
            detail_parts.append(entity.description)
        if entity.aliases:
            detail_parts.append(
                "Aliases: " + ", ".join(alias.alias for alias in entity.aliases)
            )
        for external in entity.external_attributes:
            detail_parts.extend(format_external_profile(external.payload))
            citations.append(
                RecallCitation(
                    kind="external",
                    id=str(external.id),
                    label=f"{external.provider} {external.field_group}",
                    url=external.source_url,
                )
            )
            selected_ids.append(str(external.id))
        answer_parts.append(". ".join(part.rstrip(".") for part in detail_parts) + ".")

    for relationship in connected:
        answer_parts.append(
            f"{relationship.subject.canonical_name} "
            f"{relationship.predicate.replace('_', ' ')} "
            f"{relationship.object.canonical_name}."
        )
        selected_ids.append(str(relationship.id))
        for evidence in relationship.evidence:
            citation = RecallCitation(
                kind="episode",
                id=str(evidence.episode_id),
                label=f"{evidence.episode.source_type.value} episode",
                excerpt=evidence.excerpt,
            )
            if citation.id not in {item.id for item in citations}:
                citations.append(citation)

    return SemanticContextResult(
        answer=" ".join(answer_parts),
        citations=citations,
        selected_ids=list(dict.fromkeys(selected_ids)),
    )


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def meaningful_tokens(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "does",
        "how",
        "in",
        "is",
        "it",
        "of",
        "the",
        "to",
        "what",
        "which",
        "who",
    }
    return {token for token in value.split() if token not in stopwords and len(token) > 1}


def format_external_profile(payload: dict[str, object]) -> list[str]:
    details: list[str] = []
    basic_info = payload.get("basic_info")
    if isinstance(basic_info, dict):
        industries = basic_info.get("industries")
        if isinstance(industries, list) and industries:
            details.append("Industries: " + ", ".join(str(value) for value in industries))
        employee_range = basic_info.get("employee_count_range")
        if employee_range:
            details.append(f"Employee range: {employee_range}")
        description = basic_info.get("description")
        if description:
            details.append(str(description))
    headcount = payload.get("headcount")
    if isinstance(headcount, dict) and headcount.get("total") is not None:
        details.append(f"Headcount: {headcount['total']}")
    funding = payload.get("funding")
    if isinstance(funding, dict) and funding.get("total_investment_usd") is not None:
        details.append(f"Total funding USD: {funding['total_investment_usd']}")
    if not details:
        details.append("External profile: " + json.dumps(payload, sort_keys=True))
    return details