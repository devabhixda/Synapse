"""A reasoning model gets room to think, and says so when it runs out.

Every call site sizes ``max_tokens`` for the answer it expects, which is the
whole cost on a model that writes its answer directly. A reasoning model bills
its private thinking against the same budget, and the failure is quiet: the
request succeeds, ``finish_reason`` is ``length``, and ``content`` is ``null``.
Nothing about that says "raise the budget", so these tests pin both halves —
that the extra allowance reaches the wire, and that running out produces an
error naming the setting that fixes it.
"""

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from synapse.config import Settings
from synapse.consolidation.llm_extractor import LLMSemanticExtractor
from synapse.db.models import Episode, EpisodeKind, SourceType
from synapse.domain.models import SourceEvent
from synapse.inference.chat import InferenceError, LLMInterpreter
from synapse.recall.answerability import EpisodeAnswerabilityJudge

THINKS = Settings(
    chat_base_url="https://inference.example.com/v1",
    chat_model="openai/gpt-oss-120b",
    chat_api_key="sk-secret",
    chat_reasoning_tokens=2048,
)
DIRECT = Settings(
    chat_base_url="https://inference.example.com/v1",
    chat_model="google/gemma-4-E2B-it",
)


class Endpoint:
    """A chat endpoint that answers, or thinks until the budget is gone."""

    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.request: httpx.Request | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": self.content},
                        "finish_reason": self.finish_reason,
                    }
                ]
            },
        )

    @property
    def budget(self) -> int:
        assert self.request is not None
        return int(json.loads(self.request.content)["max_tokens"])


def serving(monkeypatch: pytest.MonkeyPatch, endpoint: Endpoint) -> Endpoint:
    original = httpx.AsyncClient.__init__

    def patched(self: httpx.AsyncClient, **kwargs: Any) -> None:
        original(self, **{**kwargs, "transport": httpx.MockTransport(endpoint.handle)})

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    return endpoint


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Endpoint:
    return serving(monkeypatch, Endpoint('{"intent":"share_update","salience":0.5}'))


@pytest.fixture
def out_of_budget(monkeypatch: pytest.MonkeyPatch) -> Endpoint:
    return serving(monkeypatch, Endpoint(None, finish_reason="length"))


def event() -> SourceEvent:
    return SourceEvent(
        source_id="evt-1",
        source_type=SourceType.SLACK,
        conversation_id="thread-1",
        content="Priya took over the checkout service.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
    )


async def test_the_reasoning_allowance_is_added_to_the_request(answers: Endpoint) -> None:
    await LLMInterpreter(THINKS).observe(event())

    # 48 for the answer the call site wants, plus room to think.
    assert answers.budget == 48 + 2048


async def test_a_model_that_does_not_reason_is_billed_nothing_extra(
    answers: Endpoint,
) -> None:
    await LLMInterpreter(DIRECT).observe(event())

    assert answers.budget == 48


async def test_running_out_of_budget_names_the_setting_that_fixes_it(
    out_of_budget: Endpoint,
) -> None:
    with pytest.raises(InferenceError, match="chat_reasoning_tokens"):
        await LLMInterpreter(DIRECT).observe(event())


async def test_a_judge_that_cannot_think_falls_through_to_the_graph(
    out_of_budget: Endpoint,
) -> None:
    # Not an error: recall still has the semantic path to try.
    selection = await EpisodeAnswerabilityJudge(DIRECT).select(
        "who owns checkout?", [{"episode_id": "e1", "text": "Priya owns checkout."}]
    )

    assert selection.answerable is False


async def test_adjudication_that_runs_out_keeps_the_deterministic_graph(
    out_of_budget: Endpoint,
) -> None:
    episode = Episode(
        group_id="acme",
        kind=EpisodeKind.SOURCE,
        source_type=SourceType.SLACK,
        source_id="evt-1",
        conversation_id="thread-1",
        content="Priya owns the checkout service, also called checkout-svc.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
        # An alias to adjudicate is what makes this reach the model at all.
        details={
            "semantic_assertions": {
                "entities": [
                    {
                        "kind": "System",
                        "name": "checkout service",
                        "aliases": ["checkout-svc"],
                    }
                ]
            }
        },
    )

    result = await LLMSemanticExtractor(DIRECT).extract([episode])

    assert result.provider == "deterministic-system2-fallback"
    assert "chat_reasoning_tokens" in (result.fallback_reason or "")
