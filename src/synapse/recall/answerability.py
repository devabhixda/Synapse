import json

import httpx
from pydantic import ValidationError

from synapse.config import Settings, get_settings
from synapse.inference.chat import chat_completion, message_content, parse_json_object
from synapse.recall.models import EpisodeSelection

SYSTEM_PROMPT = """Decide whether one supplied episode directly answers the user's query.
Return exactly one JSON object with `answerable` (boolean), `selected_episode_id` (string or
null), and `confidence` (0 to 1). Select an episode only when its text contains the answer,
not merely a related topic. For a compound query, the same episode must answer every requested
clause. A query asking what happened and why requires both the fact and its rationale; reject
an episode that supplies only one. Do not answer the query and do not add facts. Output JSON
only."""


def supports_query_shape(query: str, episode_text: str) -> bool:
    normalized_query = " ".join(query.casefold().split())
    normalized_text = " ".join(episode_text.casefold().split())
    if "why" in normalized_query and not any(
        marker in normalized_text
        for marker in (
            "because",
            "reason",
            "so ",
            "therefore",
            "outside scope",
            "outside the",
            "not required",
            "non-critical",
            "deferred",
        )
    ):
        return False
    if "how" in normalized_query and "complement" in normalized_query:
        return "complement" in normalized_text
    return True


class EpisodeAnswerabilityJudge:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def select(
        self, query: str, candidates: list[dict[str, object]]
    ) -> EpisodeSelection:
        if not candidates:
            return EpisodeSelection(answerable=False, confidence=1.0)
        try:
            body = await chat_completion(
                self.settings,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Query: "
                        + query
                        + "\nEpisodes: "
                        + json.dumps(candidates, separators=(",", ":")),
                    },
                ],
                max_tokens=100,
                timeout=15,
            )
            selection = EpisodeSelection.model_validate(
                parse_json_object(message_content(body))
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, ValidationError):
            return EpisodeSelection(answerable=False, confidence=0.0)

        candidate_ids = {str(candidate["episode_id"]) for candidate in candidates}
        if selection.selected_episode_id not in candidate_ids:
            return EpisodeSelection(answerable=False, confidence=selection.confidence)
        selected_candidate = next(
            candidate
            for candidate in candidates
            if str(candidate["episode_id"]) == selection.selected_episode_id
        )
        if not supports_query_shape(query, str(selected_candidate.get("text") or "")):
            return EpisodeSelection(answerable=False, confidence=selection.confidence)
        return selection