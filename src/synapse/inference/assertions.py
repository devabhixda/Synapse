import json
from collections.abc import Iterable, Sequence

import httpx
from json_repair import repair_json
from pydantic import BaseModel, Field, ValidationError

from synapse.config import (
    DEFAULT_ENTITY_KINDS,
    DEFAULT_RELATIONSHIP_PREDICATES,
    FALLBACK_ENTITY_KIND,
    FALLBACK_PREDICATE,
    Settings,
    get_settings,
    snake_case,
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

SYSTEM_PROMPT_TEMPLATE = """Extract the single strongest explicit {scope} from
{source}. Return exactly one flat JSON object with: supported (boolean),
subject_kind, subject_name, predicate, object_kind, object_name, confidence. Allowed kinds:
{kinds}. Use a short snake_case predicate. If no explicit relationship exists, return
supported false and empty strings. Predicate must be one of: {predicates}.
Do not infer facts or add commentary. Output JSON only."""

# The defaults, kept under their old names for code that imported them. The
# ones actually allowed are ``Settings.entity_kinds`` and
# ``Settings.relationship_predicates``.
ALLOWED_KINDS = set(DEFAULT_ENTITY_KINDS)
ALLOWED_PREDICATES = set(DEFAULT_RELATIONSHIP_PREDICATES)


def system_prompt(
    kinds: Sequence[str] = DEFAULT_ENTITY_KINDS,
    predicates: Iterable[str] = DEFAULT_RELATIONSHIP_PREDICATES,
) -> str:
    """The extraction prompt, naming the kinds and predicates the model may use.

    The default vocabulary is about work, and so is the default wording; a
    host that configures its own (a personal assistant, say) gets prompt
    wording that does not tell the model its conversations are about work.
    """
    predicate_list = list(predicates)
    default = list(kinds) == list(DEFAULT_ENTITY_KINDS) and predicate_list == list(
        DEFAULT_RELATIONSHIP_PREDICATES
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        scope="organizational relationship" if default else "relationship",
        source="one work conversation" if default else "one conversation",
        kinds=", ".join(kinds),
        predicates=", ".join(predicate_list),
    )


SYSTEM_PROMPT = system_prompt()


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
                        "content": system_prompt(
                            self.settings.entity_kinds,
                            self.settings.relationship_predicates,
                        ),
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
        predicate = normalize_predicate(
            assertion.predicate, self.settings.relationship_predicates
        )
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


def normalize_predicate(
    value: str, predicates: Iterable[str] = DEFAULT_RELATIONSHIP_PREDICATES
) -> str:
    """The configured predicate ``value`` means, or ``related_to`` if none.

    An exact match (after snake-casing) wins. Failing that, a few common
    phrasings are mapped onto the workplace predicates, but only onto ones the
    configuration actually allows.
    """
    allowed = set(predicates)
    normalized = snake_case(value)
    if not normalized:
        raise ValueError("Assertion predicate cannot be empty")
    if normalized in allowed:
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
            if predicate in allowed and any(marker in normalized for marker in markers)
        ),
        FALLBACK_PREDICATE,
    )
