import json

import httpx
from pydantic import ValidationError

from synapse.config import Settings, get_settings
from synapse.inference.cactus import parse_json_object
from synapse.recall.models import EpisodeSelection

SYSTEM_PROMPT = """Decide whether one supplied episode directly answers the user's query.
Return exactly one JSON object with `answerable` (boolean), `selected_episode_id` (string or
null), and `confidence` (0 to 1). Select an episode only when its text contains the answer,
not merely a related topic. Do not answer the query and do not add facts. Output JSON only."""


class EpisodeAnswerabilityJudge:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def select(
        self, query: str, candidates: list[dict[str, object]]
    ) -> EpisodeSelection:
        if not candidates:
            return EpisodeSelection(answerable=False, confidence=1.0)
        request = {
            "model": self.settings.cactus_chat_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Query: "
                    + query
                    + "\nEpisodes: "
                    + json.dumps(candidates, separators=(",", ":")),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 100,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self.settings.cactus_chat_base_url.rstrip('/')}/chat/completions",
                    json=request,
                    headers={"Authorization": "Bearer cactus-local"},
                )
                response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            selection = EpisodeSelection.model_validate(parse_json_object(content))
        except (httpx.HTTPError, KeyError, TypeError, ValueError, ValidationError):
            return EpisodeSelection(answerable=False, confidence=0.0)

        candidate_ids = {str(candidate["episode_id"]) for candidate in candidates}
        if selection.selected_episode_id not in candidate_ids:
            return EpisodeSelection(answerable=False, confidence=selection.confidence)
        return selection