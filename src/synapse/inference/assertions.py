import json
import re
from collections.abc import Sequence

import httpx
from json_repair import repair_json
from pydantic import BaseModel, Field, ValidationError

from synapse.config import (
    DEFAULT_ENTITY_KINDS,
    FALLBACK_ENTITY_KIND,
    Settings,
    get_settings,
)
from synapse.domain.models import (
    SemanticAssertions,
    SemanticEntityAssertion,
    SemanticRelationshipAssertion,
    SourceEvent,
)
from synapse.inference.chat import (
    InferenceError,
    chat_completion,
    message_content,
    parse_json_object,
)

SYSTEM_PROMPT_TEMPLATE = """Extract the single strongest explicit organizational relationship from
one work conversation. Return exactly one flat JSON object with: supported (boolean),
subject_kind, subject_name, predicate, object_kind, object_name, confidence. Allowed kinds:
{kinds}. Use a short snake_case predicate. If no explicit relationship exists, return
supported false and empty strings. Predicate must be one of: owns, assigned_to, blocks,
resolves, decides, proposes, requests, updates, implements, uses, part_of, affects,
depends_on, related_to. Do not infer facts or add commentary. Output JSON only."""

# The default kinds, kept under the old name for code that imported it.
ALLOWED_KINDS = set(DEFAULT_ENTITY_KINDS)


def system_prompt(kinds: Sequence[str] = DEFAULT_ENTITY_KINDS) -> str:
    """The extraction prompt, naming the kinds the model may use."""
    return SYSTEM_PROMPT_TEMPLATE.format(kinds=", ".join(kinds))


SYSTEM_PROMPT = system_prompt()

ALLOWED_PREDICATES = {
    "owns",
    "assigned_to",
    "blocks",
    "resolves",
    "decides",
    "proposes",
    "requests",
    "updates",
    "implements",
    "uses",
    "part_of",
    "affects",
    "depends_on",
    "related_to",
}


class CompactAssertion(BaseModel):
    supported: bool
    subject_kind: str = ""
    subject_name: str = ""
    predicate: str = ""
    object_kind: str = ""
    object_name: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)


class LLMAssertionInterpreter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def extract(self, event: SourceEvent) -> SemanticAssertions:
        try:
            body = await chat_completion(
                self.settings,
                [
                    {
                        "role": "system",
                        "content": system_prompt(self.settings.entity_kinds),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Participants: "
                            + json.dumps(event.participants, separators=(",", ":"))
                            + "\nConversation:\n"
                            + event.content
                        ),
                    },
                ],
                max_tokens=140,
                timeout=30,
            )
            content = message_content(body)
            try:
                payload = parse_json_object(content)
            except (ValueError, json.JSONDecodeError):
                repaired: object = repair_json(content, return_objects=True)
                if not isinstance(repaired, dict):
                    raise ValueError(
                        "Repaired assertion response is not an object"
                    ) from None
                payload = {str(key): value for key, value in repaired.items()}
            assertion = CompactAssertion.model_validate(payload)
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise InferenceError(
                f"Semantic assertion extraction failed for {event.source_id}: {error}"
            ) from error
        if not assertion.supported:
            return SemanticAssertions()
        required_values = (
            assertion.subject_kind,
            assertion.subject_name,
            assertion.predicate,
            assertion.object_kind,
            assertion.object_name,
        )
        if not all(value.strip() for value in required_values):
            raise InferenceError(
                f"Semantic assertion extraction returned incomplete data for {event.source_id}"
            )
        kinds = self.settings.entity_kinds
        subject_kind = normalize_kind(assertion.subject_kind, kinds)
        object_kind = normalize_kind(assertion.object_kind, kinds)
        predicate = normalize_predicate(assertion.predicate)
        return SemanticAssertions(
            entities=[
                SemanticEntityAssertion(
                    kind=subject_kind,
                    name=" ".join(assertion.subject_name.split()),
                    confidence=assertion.confidence,
                ),
                SemanticEntityAssertion(
                    kind=object_kind,
                    name=" ".join(assertion.object_name.split()),
                    confidence=assertion.confidence,
                ),
            ],
            relationships=[
                SemanticRelationshipAssertion(
                    subject_kind=subject_kind,
                    subject_name=" ".join(assertion.subject_name.split()),
                    predicate=predicate,
                    object_kind=object_kind,
                    object_name=" ".join(assertion.object_name.split()),
                    confidence=assertion.confidence,
                )
            ],
        )


def normalize_kind(value: str, kinds: Sequence[str] = DEFAULT_ENTITY_KINDS) -> str:
    """The configured spelling of ``value``'s kind, or ``Concept`` if it has none.

    Matched without regard to case or spacing, since the model's capitalisation
    is not something to depend on.
    """
    wanted = " ".join(value.split()).casefold()
    return next(
        (kind for kind in kinds if " ".join(kind.split()).casefold() == wanted),
        FALLBACK_ENTITY_KIND,
    )


def normalize_predicate(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not normalized:
        raise ValueError("Assertion predicate cannot be empty")
    if normalized in ALLOWED_PREDICATES:
        return normalized
    mappings = (
        (("owner",), "owns"),
        (("assign",), "assigned_to"),
        (("block", "fail", "broken"), "blocks"),
        (("resolve", "fix"), "resolves"),
        (("decid", "set_", "remove"), "decides"),
        (("propos", "suggest", "recommend"), "proposes"),
        (("request", "ask", "need", "require"), "requests"),
        (("update", "change", "adjust"), "updates"),
        (("implement", "develop", "create", "make", "add", "populate", "generate"), "implements"),
        (("use", "log_on"), "uses"),
        (("depend",), "depends_on"),
    )
    return next(
        (
            predicate
            for markers, predicate in mappings
            if any(marker in normalized for marker in markers)
        ),
        "related_to",
    )
