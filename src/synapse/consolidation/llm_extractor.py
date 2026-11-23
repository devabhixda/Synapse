import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from json_repair import repair_json
from pydantic import ValidationError

from synapse.config import DEFAULT_ENTITY_KINDS, Settings, get_settings
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
    RawEntitySpec,
    RawRelationshipSpec,
    RelationshipSpec,
)
from synapse.db.models import Episode
from synapse.domain.models import semantic_assertions
from synapse.inference.assertions import normalize_kind
from synapse.inference.assertions import normalize_predicate as configured_predicate
from synapse.inference.chat import (
    InferenceError,
    MalformedResponse,
    chat_completion,
    message_content,
    parse_json_object,
)

# The default kinds, kept under the old name for code that imported it. The
# kinds actually allowed are ``Settings.entity_kinds``.
ALLOWED_ENTITY_KINDS = set(DEFAULT_ENTITY_KINDS)

SYSTEM_PROMPT = """You are Synapse System 2 resolving entity aliases during overnight
memory consolidation. Review candidate aliases against source episode excerpts. Return
exactly one JSON object with: `approved` (boolean), `alias_map` (object mapping aliases to
canonical names), and `confidence` (0 to 1). Include only aliases directly supported by the
evidence. Do not add facts or comments. Output JSON only."""


CONTENT_PROMPT = """You are Synapse System 2, consolidating conversations into a knowledge
graph of durable facts. Return exactly one JSON object:
{{"entities":[{{"kind":"","canonical_name":"","aliases":[],"description":""}}],
"relationships":[{{"subject_kind":"","subject_name":"","predicate":"","object_kind":"",
"object_name":"","evidence":["e1"],"quote":""}}]}}
Rules:
- Only facts a conversation states explicitly. No guesses, no small talk.
- Entity kinds: {kinds}. Predicates: {predicates}. Use related_to only when none fits.
- Single-valued predicates hold one current value: {functional}. When a conversation
  says the value changed, emit the new value; it replaces the old one.
- Known entities already exist. When a conversation refers to one, by name or alias,
  use its exact canonical_name and kind; list a new alias only for a new name.
- evidence: the ids of the conversations that state the relationship.
- quote: the shortest verbatim span, under 200 characters, from that conversation that
  states it.
Output JSON only."""

# Prompt overhead per call beyond the conversations and known entities, in
# tokens, for budgeting before a call is made.
PROMPT_OVERHEAD_TOKENS = 600
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class KnownEntity:
    """An entity already in the graph, shown to extraction so it links to it."""

    kind: str
    canonical_name: str
    aliases: tuple[str, ...] = ()


class UnusableExtraction(MalformedResponse):
    """The model answered, but not with a usable graph; carries what it cost."""

    def __init__(
        self,
        message: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        truncated: bool = False,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # Not a complete answer (cut off at the completion budget, or empty):
        # possibly too much to say, possibly a model that cannot answer at all.
        self.truncated = truncated


class SemanticExtractor(Protocol):
    async def extract(self, episodes: list[Episode]) -> ExtractionResult: ...


def has_assertions(episode: Episode) -> bool:
    assertions = semantic_assertions(episode.details or {})
    return bool(assertions.entities or assertions.relationships)


def excerpt(content: str, limit: int) -> str:
    """At most ``limit`` characters: the opening and, mostly, the end.

    A conversation that grows (a day's thread) has its newest messages last,
    and those are the ones that change facts.
    """
    if len(content) <= limit:
        return content
    head = limit // 4
    tail = limit - head
    return content[:head] + "\n[...]\n" + content[-tail:]


def episode_key(episode: Episode) -> str:
    """How a content-extracted plan names its evidence episodes."""
    return str(episode.id)


class LLMSemanticExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def model(self) -> str:
        return self.settings.consolidation_chat_model or self.settings.chat_model

    @property
    def reads_content(self) -> bool:
        """Whether episodes without attached assertions are read by the model."""
        return (
            self.settings.consolidation_content_extraction
            and self.settings.consolidation_mode != "deterministic"
        )

    @property
    def uses_known_entities(self) -> bool:
        return self.reads_content

    def _chat_settings(self) -> Settings:
        if self.model == self.settings.chat_model:
            return self.settings
        return self.settings.model_copy(update={"chat_model": self.model})

    def unread(self, episodes: Sequence[Episode]) -> list[Episode]:
        """The episodes content extraction would send to the model."""
        if not self.reads_content:
            return []
        return [episode for episode in episodes if not has_assertions(episode)]

    def estimate_tokens(
        self, episodes: Sequence[Episode], known_entities: Sequence[KnownEntity] = ()
    ) -> int:
        """An upper estimate of what extracting ``episodes`` will cost, before calling."""
        unread = self.unread(episodes)
        if not unread:
            return 0
        chars = sum(
            min(len(episode.content), self.settings.consolidation_episode_chars) + 120
            for episode in unread
        ) + sum(
            len(entity.canonical_name) + sum(len(alias) for alias in entity.aliases) + 40
            for entity in known_entities
        )
        return (
            PROMPT_OVERHEAD_TOKENS
            + chars // CHARS_PER_TOKEN
            + self.settings.consolidation_max_output_tokens
            + self.settings.chat_reasoning_tokens
        )

    async def extract(
        self, episodes: list[Episode], known_entities: Sequence[KnownEntity] = ()
    ) -> ExtractionResult:
        """The graph ``episodes`` assert.

        Attached assertions and participants are read without a model, and a
        model only checks the aliases among them, falling back to the
        unchecked plan if it cannot. With content extraction on, the model
        also reads the episodes that carry no assertions; that failing raises,
        so the episodes stay due rather than being marked read with nothing
        taken from them.
        """
        result = await self._from_assertions(episodes)
        unread = self.unread(episodes)
        if not unread:
            return result
        content = await self._from_content(unread, known_entities)
        return ExtractionResult(
            plan=merge_plans(result.plan, content.plan),
            provider=content.provider,
            fallback_reason=result.fallback_reason,
            llm_calls=result.llm_calls + content.llm_calls,
            prompt_tokens=result.prompt_tokens + content.prompt_tokens,
            completion_tokens=result.completion_tokens + content.completion_tokens,
            tokens_estimated=result.tokens_estimated or content.tokens_estimated,
        )

    async def _from_assertions(self, episodes: list[Episode]) -> ExtractionResult:
        candidate = derive_semantic_plan(episodes)
        if self.settings.consolidation_mode == "deterministic":
            return ExtractionResult(plan=candidate, provider="deterministic-system2")
        if not any(entity.aliases for entity in candidate.entities):
            # Nothing to adjudicate, so nothing to ask.
            return ExtractionResult(plan=candidate, provider=f"{self.model}-system2")

        try:
            adjudication, usage = await self._request_adjudication(episodes, candidate)
            if not adjudication.approved:
                raise ValueError("The model rejected the evidence-backed candidate graph")
            apply_alias_map(candidate, adjudication.alias_map)
            return ExtractionResult(
                plan=candidate,
                provider=f"{self.model}-system2",
                llm_calls=1,
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
                tokens_estimated=usage[2],
            )
        except (
            httpx.HTTPError,
            InferenceError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            return ExtractionResult(
                plan=candidate,
                provider="deterministic-system2-fallback",
                fallback_reason=f"{type(error).__name__}: {error}",
            )

    async def _from_content(
        self, episodes: list[Episode], known_entities: Sequence[KnownEntity]
    ) -> ExtractionResult:
        labelled = {f"e{index}": episode for index, episode in enumerate(episodes, start=1)}
        conversations = [
            {
                "id": label,
                "date": episode.occurred_at.date().isoformat(),
                "participants": episode.participants,
                "content": excerpt(episode.content, self.settings.consolidation_episode_chars),
            }
            for label, episode in labelled.items()
        ]
        known = [
            {"kind": entity.kind, "name": entity.canonical_name, "aliases": list(entity.aliases)}
            if entity.aliases
            else {"kind": entity.kind, "name": entity.canonical_name}
            for entity in known_entities
        ]
        predicates = self.settings.relationship_predicates
        functional = [name for name, cardinality in predicates.items() if cardinality != "many"]
        messages = [
            {
                "role": "system",
                "content": CONTENT_PROMPT.format(
                    kinds=", ".join(self.settings.entity_kinds),
                    predicates=", ".join(predicates),
                    functional=", ".join(functional) or "none",
                ),
            },
            {
                "role": "user",
                "content": "Known entities:\n"
                + json.dumps(known, separators=(",", ":"), ensure_ascii=False)
                + "\nConversations:\n"
                + json.dumps(conversations, separators=(",", ":"), ensure_ascii=False),
            },
        ]
        body = await chat_completion(
            self._chat_settings(),
            messages,
            max_tokens=self.settings.consolidation_max_output_tokens,
            timeout=120,
        )
        prompt_tokens, completion_tokens, estimated = token_usage(
            body, sum(len(message["content"]) for message in messages)
        )
        try:
            content = message_content(body)
        except InferenceError as error:
            # Billed but empty: a content filter, or a budget spent thinking.
            raise UnusableExtraction(
                str(error), prompt_tokens, completion_tokens, truncated=True
            ) from error
        truncated = _finish_reason(body) == "length"
        try:
            payload = parse_json_object(content)
        except (ValueError, json.JSONDecodeError):
            repaired: object = None if truncated else repair_json(content, return_objects=True)
            if not isinstance(repaired, dict):
                raise UnusableExtraction(
                    "the extraction response is not a JSON object"
                    + (" (cut off at the completion budget)" if truncated else ""),
                    prompt_tokens,
                    completion_tokens,
                    truncated=truncated,
                ) from None
            payload = {str(key): value for key, value in repaired.items()}
        if (
            ("entities" not in payload and "relationships" not in payload)
            or not isinstance(payload.get("entities", []), list)
            or not isinstance(payload.get("relationships", []), list)
        ):
            raise UnusableExtraction(
                "the extraction response has no entity or relationship list",
                prompt_tokens,
                completion_tokens,
            )
        plan = content_plan(payload, labelled, self.settings)
        return ExtractionResult(
            plan=plan,
            provider=f"{self.model}-system2-content",
            llm_calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_estimated=estimated,
        )

    async def _request_adjudication(
        self, episodes: list[Episode], candidate: ConsolidationPlan
    ) -> tuple[GraphAdjudication, tuple[int, int, bool]]:
        candidate_aliases = {
            alias: entity.canonical_name
            for entity in candidate.entities
            for alias in entity.aliases
        }
        if not candidate_aliases:
            return GraphAdjudication(approved=True, confidence=1.0), (0, 0, False)
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Evidence excerpts:\n"
                + json.dumps(episode_payload, separators=(",", ":"))
                + "\nCandidate aliases (alias to canonical):\n"
                + json.dumps(candidate_aliases, separators=(",", ":")),
            },
        ]
        body = await chat_completion(
            self._chat_settings(),
            messages,
            max_tokens=120,
            timeout=45,
        )
        usage = token_usage(body, sum(len(message["content"]) for message in messages))
        return (
            GraphAdjudication.model_validate(parse_json_object(message_content(body))),
            usage,
        )


def _finish_reason(body: Mapping[str, Any]) -> str | None:
    try:
        reason = body["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return reason if isinstance(reason, str) else None


def token_usage(body: Mapping[str, Any], prompt_chars: int) -> tuple[int, int, bool]:
    """Prompt and completion tokens the endpoint reports, or an estimate if it doesn't."""
    usage = body.get("usage")
    if isinstance(usage, Mapping):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion, False
    try:
        completion_chars = len(str(body["choices"][0]["message"].get("content") or ""))
    except (KeyError, IndexError, TypeError, AttributeError):
        completion_chars = 0
    return prompt_chars // CHARS_PER_TOKEN, completion_chars // CHARS_PER_TOKEN, True


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if isinstance(item, str | int | float | bool)
    }


def content_plan(
    payload: Mapping[str, Any], labelled: Mapping[str, Episode], settings: Settings
) -> ConsolidationPlan:
    """The plan a content-extraction response describes, keeping what is usable.

    A malformed entity or relationship is dropped rather than failing the batch,
    an entity named only in a relationship is added, predicates are mapped onto
    the configured ones, and evidence ids are mapped back to episodes.
    """
    entities: list[RawEntitySpec] = []
    for item in payload.get("entities") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            entities.append(
                RawEntitySpec.model_validate(
                    {
                        "kind": item.get("kind") or "",
                        "canonical_name": item.get("canonical_name") or item.get("name") or "",
                        "aliases": [
                            str(alias)
                            for alias in item.get("aliases") or []
                            if isinstance(alias, str) and alias.strip()
                        ],
                        "description": item.get("description") or None,
                        "attributes": _string_map(item.get("attributes")),
                    }
                )
            )
        except ValidationError:
            continue

    relationships: list[RawRelationshipSpec] = []
    for item in payload.get("relationships") or []:
        if not isinstance(item, Mapping):
            continue
        evidence = item.get("evidence", item.get("evidence_source_ids")) or []
        if isinstance(evidence, str):
            evidence = [evidence]
        try:
            relationship = RawRelationshipSpec.model_validate(
                {
                    "subject_kind": item.get("subject_kind") or "",
                    "subject_name": item.get("subject_name") or "",
                    "predicate": item.get("predicate") or "",
                    "object_kind": item.get("object_kind") or "",
                    "object_name": item.get("object_name") or "",
                    "evidence_source_ids": [
                        episode_key(labelled[label])
                        for label in (str(value) for value in evidence)
                        if label in labelled
                    ],
                    "quote": item.get("quote") if isinstance(item.get("quote"), str) else None,
                }
            )
            relationship.predicate = configured_predicate(
                relationship.predicate, settings.relationship_predicates
            )
        except (ValidationError, ValueError):
            continue
        if not relationship.subject_name.strip() or not relationship.object_name.strip():
            continue
        relationships.append(relationship)

    kinds = settings.entity_kinds
    named = {
        entity_key(normalize_kind(entity.kind, kinds), entity.canonical_name)
        for entity in entities
    }
    for relationship in relationships:
        for kind, name in (
            (relationship.subject_kind, relationship.subject_name),
            (relationship.object_kind, relationship.object_name),
        ):
            key = entity_key(normalize_kind(kind, kinds), name)
            if key not in named:
                named.add(key)
                entities.append(RawEntitySpec(kind=kind, canonical_name=name))

    return normalize_raw_plan(
        RawConsolidationPlan(entities=entities, relationships=relationships),
        list(labelled.values()),
        kinds,
        evidence_key=episode_key,
    )


def merge_plans(*plans: ConsolidationPlan) -> ConsolidationPlan:
    """One plan holding every entity, edge and statement of ``plans``, merged by key."""
    entities: dict[str, EntitySpec] = {}
    relationships: dict[tuple[str, str, str], RelationshipSpec] = {}
    statements = []
    for plan in plans:
        for entity in plan.entities:
            current = entities.get(entity.key)
            if current is None:
                entities[entity.key] = entity.model_copy(deep=True)
                continue
            current.aliases = sorted(set(current.aliases) | set(entity.aliases))
            current.attributes.update(entity.attributes)
            current.description = current.description or entity.description
            current.confidence = max(current.confidence, entity.confidence)
        for relationship in plan.relationships:
            identity = (relationship.subject_key, relationship.predicate, relationship.object_key)
            existing = relationships.get(identity)
            if existing is None:
                relationships[identity] = relationship.model_copy(deep=True)
                continue
            existing.evidence_source_ids = sorted(
                set(existing.evidence_source_ids) | set(relationship.evidence_source_ids)
            )
            existing.evidence_quotes = {
                **relationship.evidence_quotes,
                **existing.evidence_quotes,
            }
            existing.attributes.update(relationship.attributes)
            existing.confidence = max(existing.confidence, relationship.confidence)
        statements.extend(plan.statements)
    return ConsolidationPlan(
        entities=sorted(entities.values(), key=lambda item: item.key),
        relationships=sorted(
            relationships.values(),
            key=lambda item: (item.subject_key, item.predicate, item.object_key),
        ),
        statements=statements,
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


def _source_id(episode: Episode) -> str | None:
    return episode.source_id


def _collapse(value: str) -> str:
    return " ".join(value.split())


def normalize_raw_plan(
    raw_plan: RawConsolidationPlan,
    episodes: list[Episode],
    kinds: Sequence[str] = DEFAULT_ENTITY_KINDS,
    *,
    evidence_key: Callable[[Episode], str | None] = _source_id,
) -> ConsolidationPlan:
    """Normalize a model's plan, keeping only evidence among ``episodes``.

    ``evidence_key`` is how the raw plan names its evidence episodes; by
    default their source ids. A relationship's quote is kept for the evidence
    episodes it appears in, and if it appears in some but not others, only
    those are kept as evidence.
    """
    by_key = {
        key: episode for episode in episodes if (key := evidence_key(episode)) is not None
    }
    valid_source_ids = set(by_key)
    entities: dict[str, EntitySpec] = {}

    for raw_entity in raw_plan.entities:
        kind = normalize_kind(raw_entity.kind, kinds)
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
        # Normalized the same way as the entities, or a relationship naming a
        # kind in different case would miss the entity it refers to.
        subject_kind = normalize_kind(raw_relationship.subject_kind, kinds)
        object_kind = normalize_kind(raw_relationship.object_kind, kinds)
        subject_key = entity_key(subject_kind, raw_relationship.subject_name)
        object_key = entity_key(object_kind, raw_relationship.object_name)
        if subject_key not in entities or object_key not in entities:
            continue
        predicate = normalize_predicate(raw_relationship.predicate)
        evidence = sorted(set(raw_relationship.evidence_source_ids) & valid_source_ids)
        if not evidence:
            continue
        quotes: dict[str, str] = {}
        quote = _collapse(raw_relationship.quote or "")
        if quote:
            quotes = {
                source_id: quote
                for source_id in evidence
                if quote in _collapse(by_key[source_id].content)
            }
            if quotes:
                evidence = sorted(quotes)
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
                evidence_quotes=quotes,
            )
        else:
            existing_relationship.evidence_source_ids = sorted(
                set(existing_relationship.evidence_source_ids) | set(evidence)
            )
            existing_relationship.evidence_quotes = {
                **quotes,
                **existing_relationship.evidence_quotes,
            }
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