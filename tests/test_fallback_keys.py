"""A fallback API key takes over only when the primary is out of quota.

The point is to spend a free-tier key first and a paid one only once the free
one runs dry, so these pin both halves: a quota refusal moves the call to the
fallback (and keeps later calls there for the cooldown), and nothing else does.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from synapse.config import Settings
from synapse.db.models import SourceType
from synapse.domain.models import SourceEvent
from synapse.inference.chat import chat_completion
from synapse.inference.keys import KeyPool, is_quota_error
from synapse.memory.mem0_store import Mem0EpisodeStore

FREE, PAID = "free-key", "paid-key"
QUOTA = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}}

SETTINGS = Settings(
    chat_base_url="https://generativelanguage.example.com/v1beta/openai/",
    chat_model="gemini-flash",
    chat_api_key=FREE,
    chat_fallback_api_key=PAID,
    embedding_base_url="https://generativelanguage.example.com/v1beta/openai/",
    embedding_model="gemini-embedding",
    embedding_api_key=FREE,
    embedding_fallback_api_key=PAID,
    embedding_dimensions=8,
)


class QuotaServer:
    """Refuses the keys in ``exhausted`` with a quota error; answers the rest."""

    def __init__(self, exhausted: set[str], status: int = 429, body: object = QUOTA) -> None:
        self.exhausted = exhausted
        self.status = status
        self.body = body
        self.keys: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"].removeprefix("Bearer ")
        self.keys.append(key)
        if key in self.exhausted:
            # x-should-retry stops the OpenAI SDK retrying with backoff, which
            # would only slow the test down.
            return httpx.Response(self.status, json=self.body, headers={"x-should-retry": "false"})
        if request.url.path.endswith("/embeddings"):
            count = len(json.loads(request.content)["input"])
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "gemini-embedding",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.1] * 8}
                        for i in range(count)
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        return self.handle(request)


def route_chat(monkeypatch: pytest.MonkeyPatch, server: QuotaServer) -> None:
    original = httpx.AsyncClient.__init__

    def patched(self: httpx.AsyncClient, **kwargs: Any) -> None:
        original(self, **{**kwargs, "transport": httpx.MockTransport(server.handle_async)})

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


async def ask(settings: Settings = SETTINGS) -> dict[str, Any]:
    return await chat_completion(
        settings, [{"role": "user", "content": "hi"}], max_tokens=8, timeout=5
    )


async def test_a_quota_refusal_moves_the_call_to_the_fallback_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = QuotaServer(exhausted={FREE})
    route_chat(monkeypatch, server)

    await ask()
    await ask()

    # The second call skips the resting free key rather than paying for a 429.
    assert server.keys == [FREE, PAID, PAID]


async def test_the_primary_key_is_used_while_it_has_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = QuotaServer(exhausted=set())
    route_chat(monkeypatch, server)

    await ask()
    await ask()

    assert server.keys == [FREE, FREE]


async def test_without_a_fallback_a_quota_refusal_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = QuotaServer(exhausted={FREE})
    route_chat(monkeypatch, server)

    with pytest.raises(httpx.HTTPStatusError):
        await ask(SETTINGS.model_copy(update={"chat_fallback_api_key": None}))
    assert server.keys == [FREE]


async def test_other_failures_do_not_spend_the_fallback_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = QuotaServer(exhausted={FREE}, status=500, body={"error": "boom"})
    route_chat(monkeypatch, server)

    with pytest.raises(httpx.HTTPStatusError):
        await ask()
    assert server.keys == [FREE]


def test_quota_is_recognised_by_status_or_by_body() -> None:
    request = httpx.Request("POST", "https://example.com")

    def error(status: int, body: object) -> httpx.HTTPStatusError:
        response = httpx.Response(status, json=body, request=request)
        return httpx.HTTPStatusError("failed", request=request, response=response)

    assert is_quota_error(error(429, {}))
    assert is_quota_error(error(403, [QUOTA]))
    assert not is_quota_error(error(500, {"error": "boom"}))
    assert not is_quota_error(ValueError("RESOURCE_EXHAUSTED"))


def test_the_primary_key_is_tried_again_after_the_cooldown() -> None:
    now = [0.0]
    pool = KeyPool(clock=lambda: now[0])

    pool.rest("chat", FREE, 60)
    assert pool.order("chat", FREE, PAID) == [PAID, FREE]
    # Quota is per model: chat running dry leaves embeddings on the free key.
    assert pool.order("embedding", FREE, PAID) == [FREE, PAID]
    now[0] = 61
    assert pool.order("chat", FREE, PAID) == [FREE, PAID]
    assert pool.order("chat", FREE, None) == [FREE]


async def test_embeddings_fall_back_to_the_second_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import mem0.memory.main as mem0_main  # type: ignore[import-untyped]

    monkeypatch.setattr(mem0_main, "MEM0_TELEMETRY", False)
    monkeypatch.setattr(mem0_main, "capture_event", lambda *args, **kwargs: None)
    server = QuotaServer(exhausted={FREE})
    original = httpx.Client.__init__

    def patched(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **{**kwargs, "transport": httpx.MockTransport(server.handle)})

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    settings = SETTINGS.model_copy(
        update={"database_url": f"sqlite+aiosqlite:///{tmp_path / 'synapse.db'}"}
    )
    store = Mem0EpisodeStore(settings)

    ids = await store.add_episode(
        episode_id="e1",
        group_id="owner",
        summary="Dinner with Asha",
        event=SourceEvent(
            source_id="whatsapp-2026-11-21",
            source_type=SourceType.WHATSAPP,
            conversation_id="owner",
            content="Asha: dinner at 8 on Friday?",
            participants=["Asha"],
            occurred_at=datetime.now(UTC),
        ),
    )

    assert len(ids) == 1
    assert server.keys == [FREE, PAID]
