import json
import re
import time
from typing import Any

import httpx

from synapse.config import Settings, get_settings
from synapse.domain.models import FastObservation, SourceEvent

SYSTEM_PROMPT = """You are Synapse System 1. Interpret one work event quickly.
Return JSON with: summary (one sentence), intent (short snake_case), salience (0-1),
entity_mentions (strings), and tags (strings). Do not infer cross-event relationships.
Use non-thinking mode and output JSON only."""


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
                    {"role": "user", "content": f"{event.source_type.value}: {event.content}"},
                ],
                "temperature": 0.1,
                "max_tokens": 220,
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
            parsed = parse_json_object(content)
            return FastObservation(
                **parsed,
                provider="cactus-gemma",
                confidence=body.get("confidence"),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            fallback = deterministic_observation(event)
            return fallback.model_copy(
                update={"latency_ms": (time.perf_counter() - started) * 1000}
            )


def parse_json_object(value: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", value.strip(), flags=re.MULTILINE).strip()
    parsed: object = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("System 1 response must be a JSON object")
    return {str(key): item for key, item in parsed.items()}


def deterministic_observation(event: SourceEvent) -> FastObservation:
    lowered = event.content.lower()
    intent = "share_update"
    tags = [event.source_type.value]
    salience = 0.45
    if any(term in lowered for term in ("decided", "decision", "will use")):
        intent = "record_decision"
        tags.append("decision")
        salience = 0.9
    elif any(term in lowered for term in ("blocked", "blocker", "failing")):
        intent = "report_blocker"
        tags.append("blocker")
        salience = 0.8
    elif "status" in lowered or "update" in lowered:
        intent = "request_status"
        tags.append("status")
        salience = 0.65

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
        intent=intent,
        salience=salience,
        entity_mentions=known_entities,
        tags=tags,
        provider="deterministic-fallback",
        confidence=1.0,
    )
