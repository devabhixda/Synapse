import json
import re
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from synapse.config import Settings, get_settings
from synapse.domain.models import FastObservation, SourceEvent

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


class CactusInferenceError(RuntimeError):
    """Raised when System 1 cannot obtain a valid local-model classification."""


class CactusInterpreter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def observe(self, event: SourceEvent) -> FastObservation:
        if self.settings.inference_mode == "fake":
            return deterministic_observation(event)

        started = time.perf_counter()
        try:
            payload = {
                "model": self.settings.cactus_chat_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": event.content},
                ],
                "temperature": 0.1,
                "max_tokens": 48,
                "response_format": {"type": "json_object"},
            }
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.post(
                    f"{self.settings.cactus_chat_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer cactus-local"},
                )
                response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            classification = FastClassification.model_validate(parse_json_object(content))
            return assemble_observation(
                event,
                classification,
                provider="cactus-gemma",
                confidence=body.get("confidence"),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise CactusInferenceError(
                f"System 1 Gemma classification failed for {event.source_id}: {error}"
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
    lowered = event.content.lower()
    known_entities = [
        entity
        for entity in ("Titan", "Crustdata", "Synapse", "Atlas API")
        if entity.lower() in lowered
    ]
    if "auth database" in lowered:
        known_entities.append("auth database")

    summary = " ".join(event.content.split())
    if len(summary) > 180:
        summary = f"{summary[:177]}..."
    return FastObservation(
        summary=summary,
        intent=classification.intent,
        salience=classification.salience,
        entity_mentions=known_entities,
        tags=[event.source_type.value, classification.intent],
        provider=provider,
        confidence=confidence,
        latency_ms=latency_ms,
    )
