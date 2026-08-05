import json

import httpx
from pydantic import ValidationError

from synapse.config import Settings, get_settings
from synapse.inference.chat import (
    InferenceError,
    chat_completion,
    message_content,
    parse_json_object,
)
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
        """Decide which candidate, if any, answers the query.

        A decline is confirmed by asking a second time. The two verdicts do not
        cost the same: saying yes puts a cited conversation in front of someone
        who can read it and disagree, while saying no discards the episodic path
        and answers from the graph's abstraction instead. Temperature is already
        zero, but the same prompt against the same evidence still lands
        differently now and then, and asking twice makes a spurious no rarer
        without making a considered one any harder to reach.

        The second call happens only on a decline, so a query that is answered
        pays nothing for it.
        """
        if not candidates:
            return EpisodeSelection(answerable=False, confidence=1.0)

        selection = await self._attempt(query, candidates)
        if selection.answerable:
            return selection
        return await self._attempt(query, candidates)

    async def _attempt(
        self, query: str, candidates: list[dict[str, object]]
    ) -> EpisodeSelection:
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
                # A yes/no on fixed evidence is a decision, not a composition:
                # there is nothing here that variety improves. At 0.1 the same
                # query against the same episodes lands on different verdicts
                # between runs, and a spurious "no" is expensive — it discards
                # the episodic path for that query entirely.
                temperature=0.0,
            )
            selection = EpisodeSelection.model_validate(
                parse_json_object(message_content(body))
            )
        except (
            httpx.HTTPError,
            InferenceError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ):
            # A judge that cannot answer is not a judge that says no: recall
            # falls through to the graph, which is the right move when the
            # episodic path could not be evaluated at all.
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