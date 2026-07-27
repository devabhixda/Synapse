"""Every chat call goes to the configured endpoint, with the configured key.

Which model server answers is meant to be a settings change and nothing more.
That is easy to claim and easy to break: a second base URL read from somewhere
else, or a bearer token left as a literal, both keep the local setup working and
silently strip the credential from a hosted one. These tests pin the request
itself rather than the settings object, so a call that bypasses the shared
client is visible.
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
from synapse.inference.assertions import LLMAssertionInterpreter
from synapse.inference.chat import LLMInterpreter
from synapse.recall.answerability import EpisodeAnswerabilityJudge

REMOTE = Settings(
    chat_base_url="https://inference.example.com/v1/",
    chat_model="openai/gpt-oss-120b",
    chat_api_key="sk-secret",
    embedding_base_url="https://embeddings.example.com/v1",
    embedding_model="baai/bge-m3",
    embedding_api_key="sk-embed",
    embedding_dimensions=1024,
)


class Recorder:
    """Stands in for the network, remembering the one request it was given."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.request: httpx.Request | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": self.content}}]},
        )


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Route every AsyncClient through a transport that records the request."""
    recording = Recorder('{"intent":"share_update","salience":0.5}')
    original = httpx.AsyncClient.__init__

    def patched(self: httpx.AsyncClient, **kwargs: Any) -> None:
        original(self, **{**kwargs, "transport": httpx.MockTransport(recording.handle)})

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    return recording


def event() -> SourceEvent:
    return SourceEvent(
        source_id="evt-1",
        source_type=SourceType.SLACK,
        conversation_id="thread-1",
        content="Priya took over the checkout service.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
    )


async def test_classification_calls_the_configured_endpoint(recorder: Recorder) -> None:
    await LLMInterpreter(REMOTE).observe(event())

    assert recorder.request is not None
    assert str(recorder.request.url) == "https://inference.example.com/v1/chat/completions"
    assert recorder.request.headers["authorization"] == "Bearer sk-secret"


async def test_the_configured_model_name_is_the_one_requested(recorder: Recorder) -> None:
    await LLMInterpreter(REMOTE).observe(event())

    assert recorder.request is not None
    assert json.loads(recorder.request.content)["model"] == "openai/gpt-oss-120b"


async def test_the_classification_provider_records_the_model_that_answered(
    recorder: Recorder,
) -> None:
    observation = await LLMInterpreter(REMOTE).observe(event())

    assert observation.provider == "openai/gpt-oss-120b"


async def test_assertion_extraction_uses_the_same_endpoint(recorder: Recorder) -> None:
    recorder.content = '{"supported":false}'

    await LLMAssertionInterpreter(REMOTE).extract(event())

    assert recorder.request is not None
    assert str(recorder.request.url).startswith("https://inference.example.com/v1")
    assert recorder.request.headers["authorization"] == "Bearer sk-secret"


async def test_the_answerability_judge_uses_the_same_endpoint(recorder: Recorder) -> None:
    recorder.content = '{"answerable":false,"selected_episode_id":null,"confidence":0.1}'

    await EpisodeAnswerabilityJudge(REMOTE).select(
        "who owns checkout?", [{"episode_id": "e1", "text": "Priya owns checkout."}]
    )

    assert recorder.request is not None
    assert recorder.request.headers["authorization"] == "Bearer sk-secret"


async def test_alias_adjudication_uses_the_same_endpoint(recorder: Recorder) -> None:
    recorder.content = '{"approved":true,"alias_map":{},"confidence":1.0}'
    episode = Episode(
        group_id="acme",
        kind=EpisodeKind.SOURCE,
        source_type=SourceType.SLACK,
        source_id="evt-1",
        conversation_id="thread-1",
        content="Priya owns the checkout service, also called checkout-svc.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
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

    result = await LLMSemanticExtractor(REMOTE).extract([episode])

    assert recorder.request is not None
    assert str(recorder.request.url) == "https://inference.example.com/v1/chat/completions"
    assert recorder.request.headers["authorization"] == "Bearer sk-secret"
    assert result.provider == "openai/gpt-oss-120b-system2"


def test_embeddings_are_configured_apart_from_chat() -> None:
    assert REMOTE.embedding_base_url != REMOTE.chat_base_url
    assert REMOTE.embedding_api_key != REMOTE.chat_api_key
