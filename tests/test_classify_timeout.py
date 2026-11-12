"""System 1 waits as long as it is configured to, and says why when it gives up.

A reasoning model labelling a long conversation can take longer than the
fixed eight seconds this call used to allow. Each timeout cost the episode its
model label, and the log line said nothing: a timeout's message is empty.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from synapse.config import Settings
from synapse.db.models import SourceType
from synapse.domain.models import SourceEvent
from synapse.inference.chat import LLMInterpreter

HOSTED = Settings(
    chat_base_url="https://inference.example.com/v1",
    chat_model="deepseek-ai/DeepSeek-V4-Flash",
    chat_api_key="sk-secret",
)


def event() -> SourceEvent:
    return SourceEvent(
        source_id="evt-1",
        source_type=SourceType.SLACK,
        conversation_id="thread-1",
        content="Priya took over the checkout service.",
        participants=["Priya"],
        occurred_at=datetime.now(UTC),
    )


def client_timeouts(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[Any]:
    seen: list[Any] = []
    original = httpx.AsyncClient.__init__

    def patched(self: httpx.AsyncClient, **kwargs: Any) -> None:
        seen.append(kwargs.get("timeout"))
        original(self, **{**kwargs, "transport": httpx.MockTransport(handler)})

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    return seen


async def answer(request: httpx.Request) -> httpx.Response:
    content = '{"intent":"share_update","salience":0.5}'
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def time_out(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("", request=request)


async def test_the_default_wait_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = client_timeouts(monkeypatch, answer)
    await LLMInterpreter(HOSTED).observe(event())
    assert seen == [8]


async def test_the_wait_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = client_timeouts(monkeypatch, answer)
    patient = HOSTED.model_copy(update={"chat_classify_timeout": 60})
    await LLMInterpreter(patient).observe(event())
    assert seen == [60]


async def test_a_timeout_is_named_in_the_fallback_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client_timeouts(monkeypatch, time_out)
    with caplog.at_level(logging.WARNING):
        observation = await LLMInterpreter(HOSTED).observe(event())
    assert observation.provider != HOSTED.chat_model  # the rule-based label
    assert "ReadTimeout" in caplog.text
