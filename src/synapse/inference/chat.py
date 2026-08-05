"""The one place that knows how to reach a chat model.

Four callers need a JSON-returning completion — System 1 classification, the
compact assertion extractor, the answerability judge, and System 2 alias
adjudication. They differ in prompt, token budget, and how long they are willing
to wait, and in nothing else. Keeping the request in one function is what makes
the model server a configuration detail: there is a single place where the URL,
the key, and the model name are read, so pointing Synapse at a different
provider is a settings change rather than an edit in four files.
"""

import json
import logging
import re
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from synapse.config import Settings, get_settings
from synapse.domain.models import FastObservation, SourceEvent, semantic_assertions

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Classify one work event. Return exactly one JSON object with `intent`
and `salience`. Intent must be one of: record_decision, report_blocker, request_status,
assign_owner, propose_integration, share_update. Salience must be a number from 0 to 1.
Example: {"intent":"record_decision","salience":0.9}. Output JSON only."""


type FastIntent = Literal[
    "record_decision",
    "report_blocker",
    "request_status",
    "assign_owner",
    "propose_integration",
    "share_update",
]


class FastClassification(BaseModel):
    intent: FastIntent
    salience: float = Field(ge=0, le=1)


class InferenceError(RuntimeError):
    """Raised when a call to the chat model cannot be turned into a usable answer."""


class MalformedResponse(InferenceError):
    """The endpoint answered, but not with the object that was asked for.

    Separate from its parent because the two want opposite handling. A model
    that returns a broken object on one event and a good one on the next is
    noise to absorb; an endpoint that cannot answer at all — no content, or a
    budget spent entirely on thinking — is a configuration fault, and absorbing
    that would degrade every episode in the corpus quietly.
    """


async def chat_completion(
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    # noqa below: handed to httpx, which times out connect/read/write
    # separately. Wrapping the call in asyncio.timeout as the rule suggests
    # would be coarser, not safer.
    timeout: float,  # noqa: ASYNC109
    temperature: float = 0.1,
) -> dict[str, Any]:
    """One JSON-mode completion against the configured chat endpoint.

    ``max_tokens`` is what the caller needs for the answer; a reasoning model
    also bills its thinking against it, so ``chat_reasoning_tokens`` is added on
    top rather than the caller having to know which kind of model it is talking
    to.

    Returns the raw response body; callers pull the message content out
    themselves because they disagree about what a malformed one means.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.chat_base_url.rstrip('/')}/chat/completions",
            json={
                "model": settings.chat_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens + settings.chat_reasoning_tokens,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {settings.chat_api_key}"},
        )
        response.raise_for_status()
    body: dict[str, Any] = response.json()
    return body


def message_content(body: dict[str, Any]) -> str:
    """The assistant's text, or an explanation of why there isn't any.

    A reasoning model that runs out of budget mid-thought returns a well-formed
    response whose content is ``null``, which otherwise surfaces several frames
    later as an ``AttributeError`` on ``NoneType`` and sends whoever reads the
    log looking in the wrong place.
    """
    choice = body["choices"][0]
    content = choice["message"].get("content")
    if content:
        return str(content)
    reason = choice.get("finish_reason")
    if reason == "length":
        raise InferenceError(
            "the model spent its whole completion budget without answering; "
            "raise chat_reasoning_tokens if it is a reasoning model"
        )
    raise InferenceError(f"the model returned no content (finish_reason={reason})")


class LLMInterpreter:
    """System 1: label one event with an intent and a salience."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def observe(self, event: SourceEvent) -> FastObservation:
        """Label the event, falling back to the rule-based labeller if asked twice.

        What this produces is an intent, a salience and a summary — metadata
        about the conversation, not the conversation. The content is stored
        either way, and the deterministic labeller can supply all three without
        a model.

        So a model that mumbles must not cost the episode. Raising here aborts
        the write, and the caller's watermark then moves past messages that were
        never stored: for a thread the next refresh re-reads it, but nothing
        goes back for a day's batch, so the failure is permanent. A worse label
        is a far smaller loss than the conversation.

        Only a malformed answer is absorbed. An endpoint that cannot answer at
        all is a configuration fault and still raises, because degrading every
        episode quietly is worse than stopping — and the caller now holds its
        watermark back on failure, so stopping costs nothing.

        The call is retried once first — the failures seen in practice are
        a malformed object from an otherwise healthy endpoint, and a second ask
        usually returns a good one.
        """
        if self.settings.inference_mode == "fake":
            return deterministic_observation(event)

        for attempt in (1, 2):
            try:
                return await self._classify(event)
            except MalformedResponse as error:
                if attempt == 1:
                    continue
                logger.warning(
                    "System 1 classification failed for %s (%s); labelling it "
                    "deterministically so the episode is still stored",
                    event.source_id,
                    error,
                )
                return deterministic_observation(event)
        raise AssertionError("unreachable")

    async def _classify(self, event: SourceEvent) -> FastObservation:
        started = time.perf_counter()
        try:
            body = await chat_completion(
                self.settings,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": event.content},
                ],
                max_tokens=48,
                timeout=8,
            )
            classification = FastClassification.model_validate(
                parse_json_object(message_content(body))
            )
            return assemble_observation(
                event,
                classification,
                provider=self.settings.chat_model,
                confidence=body.get("confidence"),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except InferenceError:
            # message_content's own diagnosis — no content, or a budget spent
            # thinking. Systematic, and it names the setting that fixes it.
            raise
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise MalformedResponse(
                f"System 1 classification failed for {event.source_id}: {error}"
            ) from error


def parse_json_object(value: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", value.strip(), flags=re.MULTILINE).strip()
    parsed: object = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("System 1 response must be a JSON object")
    return {str(key): item for key, item in parsed.items()}


def deterministic_observation(event: SourceEvent) -> FastObservation:
    lowered = event.content.lower()
    intent: FastIntent = "share_update"
    salience = 0.45
    if any(term in lowered for term in ("decided", "decision", "will use")):
        intent = "record_decision"
        salience = 0.9
    elif any(term in lowered for term in ("blocked", "blocker", "failing")):
        intent = "report_blocker"
        salience = 0.8
    elif "status" in lowered or "update" in lowered:
        intent = "request_status"
        salience = 0.65

    classification = FastClassification(intent=intent, salience=salience)
    return assemble_observation(
        event,
        classification,
        provider="deterministic-fallback",
        confidence=1.0,
    )


def assemble_observation(
    event: SourceEvent,
    classification: FastClassification,
    *,
    provider: str,
    confidence: float | None,
    latency_ms: float | None = None,
) -> FastObservation:
    assertions = semantic_assertions(event.metadata)
    entity_mentions = list(
        dict.fromkeys(
            assertion.name
            for assertion in assertions.entities
            if assertion.name.casefold() in event.content.casefold()
        )
    )

    summary = " ".join(event.content.split())
    if len(summary) > 180:
        summary = f"{summary[:177]}..."
    return FastObservation(
        summary=summary,
        intent=classification.intent,
        salience=classification.salience,
        entity_mentions=entity_mentions,
        tags=[event.source_type.value, classification.intent],
        provider=provider,
        confidence=confidence,
        latency_ms=latency_ms,
    )
