import json
import re
from typing import Protocol

import httpx
from pydantic import ValidationError

from synapse.config import Settings, get_settings
from synapse.consolidation.extractor import (
    derive_semantic_plan,
    entity_key,
    normalize_name,
)
from synapse.consolidation.models import (
    ConsolidationPlan,
    EntitySpec,
    ExtractionResult,
    GraphAdjudication,
    RawConsolidationPlan,
    RelationshipSpec,
)
from synapse.db.models import Episode
from synapse.inference.chat import chat_completion, message_content, parse_json_object

ALLOWED_ENTITY_KINDS = {
    "Person",
    "Team",
    "Company",
    "Product",
    "Project",
    "System",
    "Repository",
    "Decision",
    "Task",
    "Blocker",
    "Concept",
}

SYSTEM_PROMPT = """You are Synapse System 2 resolving entity aliases during overnight
memory consolidation. Review candidate aliases against source episode excerpts. Return
exactly one JSON object with: `approved` (boolean), `alias_map` (object mapping aliases to
canonical names), and `confidence` (0 to 1). Include only aliases directly supported by the
evidence. Do not add facts or comments. Output JSON only."""


class SemanticExtractor(Protocol):
    async def extract(self, episodes: list[Episode]) -> ExtractionResult: ...


class LLMSemanticExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def extract(self, episodes: list[Episode]) -> ExtractionResult:
        candidate = derive_semantic_plan(episodes)
        if self.settings.consolidation_mode == "deterministic":
            return ExtractionResult(plan=candidate, provider="deterministic-system2")

        try:
            adjudication = await self._request_adjudication(episodes, candidate)
            if not adjudication.approved:
                raise ValueError("The model rejected the evidence-backed candidate graph")
            apply_alias_map(candidate, adjudication.alias_map)
            return ExtractionResult(
                plan=candidate, provider=f"{self.settings.chat_model}-system2"
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, ValidationError) as error:
            return ExtractionResult(
                plan=candidate,
                provider="deterministic-system2-fallback",
                fallback_reason=f"{type(error).__name__}: {error}",
            )

    async def _request_adjudication(
        self, episodes: list[Episode], candidate: ConsolidationPlan
    ) -> GraphAdjudication:
        candidate_aliases = {
            alias: entity.canonical_name
            for entity in candidate.entities
            for alias in entity.aliases
        }
        if not candidate_aliases:
            return GraphAdjudication(approved=True, confidence=1.0)
        alias_terms = {
            normalize_name(value)
            for pair in candidate_aliases.items()
            for value in pair
        }
        episode_payload = [
            {
                "source_id": episode.source_id,
                "content": episode.content,
            }
            for episode in episodes
            if any(term in normalize_name(episode.content) for term in alias_terms)
        ]
        body = await chat_completion(
            self.settings,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Evidence excerpts:\n"
                    + json.dumps(episode_payload, separators=(",", ":"))
                    + "\nCandidate aliases (alias to canonical):\n"
                    + json.dumps(candidate_aliases, separators=(",", ":")),
                },
            ],
            max_tokens=120,
            timeout=45,
        )
        return GraphAdjudication.model_validate(
            parse_json_object(message_content(body))
        )


def apply_alias_map(plan: ConsolidationPlan, alias_map: dict[str, str]) -> None:
    entities_by_name = {
        normalize_name(entity.canonical_name): entity for entity in plan.entities
    }
    for left, right in alias_map.items():
        left_entity = entities_by_name.get(normalize_name(left))
        right_entity = entities_by_name.get(normalize_name(right))
        if left_entity is not None and right_entity is None:
            alias = " ".join(right.split())
            if alias and normalize_name(alias) != normalize_name(left_entity.canonical_name):
                left_entity.aliases = sorted(set(left_entity.aliases) | {alias})
        elif right_entity is not None and left_entity is None:
            alias = " ".join(left.split())
            if alias and normalize_name(alias) != normalize_name(right_entity.canonical_name):
                right_entity.aliases = sorted(set(right_entity.aliases) | {alias})


def normalize_predicate(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        raise ValueError("Relationship predicate cannot be empty")
    return normalized


def normalize_raw_plan(
    raw_plan: RawConsolidationPlan, episodes: list[Episode]
) -> ConsolidationPlan:
    valid_source_ids = {episode.source_id for episode in episodes if episode.source_id}
    entities: dict[str, EntitySpec] = {}

    for raw_entity in raw_plan.entities:
        kind = raw_entity.kind.strip().title()
        if kind not in ALLOWED_ENTITY_KINDS:
            kind = "Concept"
        name = " ".join(raw_entity.canonical_name.split())
        if not name:
            continue
        key = entity_key(kind, name)
        existing = entities.get(key)
        if existing is None:
            entities[key] = EntitySpec(
                key=key,
                kind=kind,
                canonical_name=name,
                aliases=sorted(
                    {
                        " ".join(alias.split())
                        for alias in raw_entity.aliases
                        if normalize_name(alias) != normalize_name(name)
                    }
                ),
                description=raw_entity.description,
                attributes=raw_entity.attributes,
                confidence=raw_entity.confidence,
            )
        else:
            existing.aliases = sorted(set(existing.aliases) | set(raw_entity.aliases))
            existing.attributes.update(raw_entity.attributes)
            existing.confidence = max(existing.confidence, raw_entity.confidence)

    relationships: dict[tuple[str, str, str], RelationshipSpec] = {}
    for raw_relationship in raw_plan.relationships:
        subject_kind = raw_relationship.subject_kind.strip().title()
        object_kind = raw_relationship.object_kind.strip().title()
        subject_key = entity_key(subject_kind, raw_relationship.subject_name)
        object_key = entity_key(object_kind, raw_relationship.object_name)
        if subject_key not in entities or object_key not in entities:
            continue
        predicate = normalize_predicate(raw_relationship.predicate)
        evidence = sorted(set(raw_relationship.evidence_source_ids) & valid_source_ids)
        if not evidence:
            continue
        identity = (subject_key, predicate, object_key)
        existing_relationship = relationships.get(identity)
        if existing_relationship is None:
            relationships[identity] = RelationshipSpec(
                subject_key=subject_key,
                predicate=predicate,
                object_key=object_key,
                evidence_source_ids=evidence,
                attributes=raw_relationship.attributes,
                confidence=raw_relationship.confidence,
            )
        else:
            existing_relationship.evidence_source_ids = sorted(
                set(existing_relationship.evidence_source_ids) | set(evidence)
            )
            existing_relationship.attributes.update(raw_relationship.attributes)
            existing_relationship.confidence = max(
                existing_relationship.confidence, raw_relationship.confidence
            )

    return ConsolidationPlan(
        entities=sorted(entities.values(), key=lambda item: item.key),
        relationships=sorted(
            relationships.values(),
            key=lambda item: (item.subject_key, item.predicate, item.object_key),
        ),
    )