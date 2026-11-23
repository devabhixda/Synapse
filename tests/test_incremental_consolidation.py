"""Consolidation that reads each episode once, in bounded runs.

A host runs consolidation on a schedule over a memory that only grows, so a run
must read what is new or changed, stop at its limits, and leave the rest for the
next run, without losing or double-counting anything when a run fails.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from synapse.config import Settings
from synapse.consolidation import llm_extractor
from synapse.consolidation.llm_extractor import LLMSemanticExtractor, UnusableExtraction
from synapse.consolidation.models import (
    ConsolidationPlan,
    ConsolidationSummary,
    EntitySpec,
    ExtractionResult,
    RelationshipSpec,
)
from synapse.consolidation.service import ConsolidationService
from synapse.db.models import (
    EntityAlias,
    Episode,
    EpisodeKind,
    SemanticEntity,
    SourceType,
)
from synapse.enrichment.crustdata import CrustdataEnricher
from synapse.semantic.repository import load_relationships, search_semantic_context

OWNER = "owner"
DAY = datetime(2026, 3, 1, tzinfo=UTC)
PERSONAL = {"works_at": "one_per_subject", "likes": "many"}


async def add(
    session, source_id: str, content: str, days: int = 0, details: dict[str, Any] | None = None
) -> Episode:
    episode = Episode(
        group_id=OWNER,
        kind=EpisodeKind.SOURCE,
        source_type=SourceType.WHATSAPP,
        source_id=source_id,
        conversation_id=source_id,
        content=content,
        participants=["Abhi"],
        details=details or {},
        occurred_at=DAY + timedelta(days=days),
    )
    session.add(episode)
    await session.commit()
    return episode


class Recorder:
    """Returns an empty plan and remembers what each call was asked to read."""

    def __init__(self, fail: set[str] | None = None, tokens_per_episode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail or set()
        self.tokens_per_episode = tokens_per_episode

    def estimate_tokens(self, episodes: list[Episode], known: object = ()) -> int:
        return self.tokens_per_episode * len(episodes)

    async def extract(self, episodes: list[Episode]) -> ExtractionResult:
        ids = [episode.source_id or "" for episode in episodes]
        self.calls.append(ids)
        if self.fail & set(ids):
            raise UnusableExtraction("garbled", 5, 5)
        spent = self.tokens_per_episode * len(episodes)
        return ExtractionResult(
            plan=ConsolidationPlan(entities=[], relationships=[]),
            provider="stub",
            llm_calls=1,
            prompt_tokens=spent,
        )


def service(extractor: Any, **settings: Any) -> ConsolidationService:
    return ConsolidationService(
        extractor=extractor, enricher=None, settings=Settings(_env_file=None, **settings)
    )


async def test_a_run_reads_only_what_is_new_or_changed(session) -> None:
    first = await add(session, "a", "Lunch at noon.")
    asserted = {"semantic_assertions": {"entities": [{"kind": "Person", "name": "Asha",
                                                      "confidence": 0.9}]}}
    await add(session, "b", "Call Asha.", days=1, details=asserted)
    recorder = Recorder()
    consolidation = service(recorder)

    await consolidation.run(OWNER, session)
    await consolidation.run(OWNER, session)
    assert recorder.calls == [["a", "b"]], "the second run read episodes again"

    first.content = "Lunch at noon. Moved to 1pm."
    await session.commit()
    await add(session, "c", "Gym at 6.", days=2)
    run = await consolidation.run(OWNER, session)

    assert recorder.calls[-1] == ["a", "c"]
    assert ConsolidationSummary.from_run(run).episodes_processed == 2


async def test_a_backlog_is_spread_over_runs_and_reported(session) -> None:
    for day in range(5):
        await add(session, f"e{day}", f"Day {day}.", days=day)
    recorder = Recorder()
    consolidation = service(recorder, consolidation_batch_episodes=2)

    summary = ConsolidationSummary.from_run(
        await consolidation.run(OWNER, session, max_episodes=3)
    )
    assert recorder.calls == [["e0", "e1"], ["e2"]]
    assert (summary.episodes_processed, summary.episodes_remaining) == (3, 2)
    assert (summary.has_more, summary.stopped_reason, summary.batches) == (
        True, "max_episodes", 2,
    )
    assert (await consolidation.backlog(OWNER, session)).due == 2

    summary = ConsolidationSummary.from_run(
        await consolidation.run(OWNER, session, max_episodes=3)
    )
    assert recorder.calls[-1] == ["e3", "e4"]
    assert (summary.has_more, summary.stopped_reason) == (False, "complete")


async def test_a_token_budget_shrinks_batches_then_stops(session) -> None:
    for day in range(4):
        await add(session, f"e{day}", f"Day {day}.", days=day)
    recorder = Recorder(tokens_per_episode=100)

    run = await service(recorder).run(OWNER, session, max_tokens=250)
    summary = ConsolidationSummary.from_run(run)

    assert recorder.calls == [["e0", "e1"]]
    assert (summary.stopped_reason, summary.prompt_tokens, summary.episodes_remaining) == (
        "max_tokens", 200, 2,
    )


async def test_an_unusable_answer_isolates_the_episode_that_caused_it(session) -> None:
    for day in range(4):
        await add(session, f"e{day}", f"Day {day}.", days=day)
    recorder = Recorder(fail={"e2"})
    consolidation = service(recorder, consolidation_max_attempts=2)

    summary = ConsolidationSummary.from_run(await consolidation.run(OWNER, session))
    assert recorder.calls == [["e0", "e1", "e2", "e3"], ["e0", "e1"], ["e2", "e3"], ["e2"],
                              ["e3"]]
    assert (summary.episodes_processed, summary.episodes_failed) == (3, 1)
    assert (summary.episodes_remaining, summary.has_more) == (1, True)

    summary = ConsolidationSummary.from_run(await consolidation.run(OWNER, session))
    assert recorder.calls[-1] == ["e2"]
    assert (summary.episodes_skipped, summary.has_more) == (1, False)
    assert recorder.calls[-1] == ["e2"] and len(recorder.calls) == 6

    await consolidation.run(OWNER, session)
    assert len(recorder.calls) == 6, "an episode that keeps failing was retried forever"


async def test_an_unreachable_model_stops_the_run_and_the_next_resumes(session) -> None:
    await add(session, "a", "Lunch at noon.")

    class Down:
        async def extract(self, episodes: list[Episode]) -> ExtractionResult:
            raise httpx.ConnectError("no route")

    run = await service(Down()).run(OWNER, session)
    summary = ConsolidationSummary.from_run(run)
    assert (run.status, summary.stopped_reason, summary.episodes_remaining) == (
        "failed", "error", 1,
    )
    assert "ConnectError" in (run.error or "")

    recorder = Recorder()
    await service(recorder).run(OWNER, session)
    assert recorder.calls == [["a"]]


def stub_chat(monkeypatch: pytest.MonkeyPatch, answers: list[dict[str, Any]]) -> list[str]:
    """Serve ``answers`` in order as the chat model; returns the prompts it was sent."""
    prompts: list[str] = []

    async def chat_completion(settings: Settings, messages: list[dict[str, str]], **_: Any):
        prompts.append(f"{settings.chat_model}\n{messages[-1]['content']}")
        return {
            "choices": [{"message": {"content": json.dumps(answers.pop(0))}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 120},
        }

    monkeypatch.setattr(llm_extractor, "chat_completion", chat_completion)
    return prompts


def works_at(company: str, label: str, quote: str) -> dict[str, Any]:
    return {
        "entities": [{"kind": "Organization", "canonical_name": company}],
        "relationships": [{
            "subject_kind": "Person", "subject_name": "Abhi", "predicate": "works at",
            "object_kind": "Organization", "object_name": company,
            "evidence": [label], "quote": quote,
        }],
    }


async def test_content_extraction_links_known_entities_and_retires_old_facts(
    session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        chat_model="big-model",
        consolidation_chat_model="cheap-model",
        consolidation_content_extraction=True,
        entity_kinds=["Person", "Organization"],
        relationship_predicates=PERSONAL,
    )
    consolidation = ConsolidationService(
        extractor=LLMSemanticExtractor(settings), enricher=None, settings=settings
    )
    prompts = stub_chat(monkeypatch, [
        works_at("Infilect", "e1", "I joined Infilect"),
        works_at("Docwise", "e1", "starting at Docwise"),
    ])

    await add(session, "march", "I joined Infilect this week.")
    first = ConsolidationSummary.from_run(await consolidation.run(OWNER, session))
    await add(session, "july", "Big news: starting at Docwise on Monday, Infilect is done.", 120)
    await consolidation.run(OWNER, session)

    assert prompts[0].startswith("cheap-model")
    assert '"name":"Infilect"' in prompts[1], "the second batch was not shown what exists"
    assert (first.llm_calls, first.prompt_tokens, first.completion_tokens) == (1, 900, 120)
    assert first.provider == "cheap-model-system2-content"

    current = await load_relationships(OWNER, session)
    assert [(edge.predicate, edge.object.canonical_name) for edge in current] == [
        ("works_at", "Docwise")
    ]
    assert [item.excerpt for item in current[0].evidence] == ["starting at Docwise"]
    everything = await load_relationships(OWNER, session, include_superseded=True)
    assert len(everything) == 2


async def test_a_name_that_is_an_alias_links_to_its_entity(session) -> None:
    await add(session, "a", "Mom called.")
    plan = ConsolidationPlan(
        entities=[EntitySpec(key="Person:asha", kind="Person", canonical_name="Asha",
                             aliases=["Mom"])],
        relationships=[],
    )

    class Fixed:
        async def extract(self, episodes: list[Episode]) -> ExtractionResult:
            return ExtractionResult(plan=plan, provider="stub")

    await service(Fixed()).run(OWNER, session)
    await add(session, "b", "Mom likes tea.", days=1)
    plan = ConsolidationPlan(
        entities=[
            EntitySpec(key="Person:mom", kind="Person", canonical_name="Mom"),
            EntitySpec(key="Person:priya", kind="Person", canonical_name="Priya",
                       aliases=["Mom"]),
        ],
        relationships=[RelationshipSpec(subject_key="Person:mom", predicate="knows",
                                        object_key="Person:priya", evidence_source_ids=["b"])],
    )
    await service(Fixed()).run(OWNER, session)

    names = sorted(await session.scalars(select(SemanticEntity.canonical_name)))
    assert names == ["Asha", "Priya"], "an alias became a second entity"
    assert len(list(await session.scalars(select(EntityAlias)))) == 1
    [edge] = await load_relationships(OWNER, session)
    assert edge.subject.canonical_name == "Asha"


async def test_enrichment_without_a_key_is_skipped_without_a_request(
    session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("enrichment made a request without a key")

    monkeypatch.setattr(httpx, "AsyncClient", no_network)
    await add(session, "a", "Acme hired me.")
    plan = ConsolidationPlan(
        entities=[EntitySpec(key="Company:acme", kind="Company", canonical_name="Acme",
                             attributes={"domain": "acme.example"})],
        relationships=[],
    )

    class Fixed:
        async def extract(self, episodes: list[Episode]) -> ExtractionResult:
            return ExtractionResult(plan=plan, provider="stub")

    settings = Settings(_env_file=None, crustdata_api_key=None)
    run = await ConsolidationService(
        extractor=Fixed(), enricher=CrustdataEnricher(settings), settings=settings
    ).run(OWNER, session)

    summary = ConsolidationSummary.from_run(run)
    assert (summary.enrichment_enabled, summary.enrichments_skipped) == (False, 0)


async def test_reset_makes_every_episode_due_again(session) -> None:
    await add(session, "old", "Lunch at noon.")
    await add(session, "new", "Call Asha.", days=1)
    recorder = Recorder()
    consolidation = service(recorder)

    await consolidation.run(OWNER, session)
    assert await consolidation.reset(OWNER, session) == 2
    await consolidation.run(OWNER, session)

    assert recorder.calls == [["old", "new"], ["old", "new"]]


async def test_a_model_that_refuses_everything_is_not_blamed_on_the_episodes(
    session,
) -> None:
    for day in range(4):
        await add(session, f"e{day}", f"Day {day}.", days=day)
    calls: list[int] = []

    class Refusing:
        async def extract(self, episodes: list[Episode]) -> ExtractionResult:
            calls.append(len(episodes))
            request = httpx.Request("POST", "https://model.example/v1/chat/completions")
            response = httpx.Response(400, request=request, text="API key not valid")
            raise httpx.HTTPStatusError("400", request=request, response=response)

    run = await service(Refusing()).run(OWNER, session)
    summary = ConsolidationSummary.from_run(run)

    assert (run.status, summary.stopped_reason, summary.has_more) == ("failed", "error", False)
    assert summary.episodes_remaining == 4 and summary.episodes_skipped == 0
    assert len(calls) < 10, "a broken endpoint was called for every split of the batch"
    assert (await service(Recorder()).backlog(OWNER, session)).due == 4


async def test_a_graph_answer_states_a_bounded_number_of_edges(session) -> None:
    await add(session, "a", "Abhi likes many things.")
    things = [f"Thing {index}" for index in range(30)]
    plan = ConsolidationPlan(
        entities=[EntitySpec(key="Person:abhi", kind="Person", canonical_name="Abhi")]
        + [EntitySpec(key=f"Concept:{t.lower()}", kind="Concept", canonical_name=t)
           for t in things],
        relationships=[
            RelationshipSpec(subject_key="Person:abhi", predicate="likes",
                             object_key=f"Concept:{t.lower()}", evidence_source_ids=["a"])
            for t in things
        ],
    )

    class Fixed:
        async def extract(self, episodes: list[Episode]) -> ExtractionResult:
            return ExtractionResult(plan=plan, provider="stub")

    await service(Fixed()).run(OWNER, session)
    result = await search_semantic_context("what does Abhi like?", OWNER, session)

    assert result.answer is not None
    assert result.answer.count("Abhi likes") == 12


async def test_episodes_the_model_refuses_do_not_hold_up_the_ones_after_them(
    session,
) -> None:
    for day in range(3):
        await add(session, f"bad{day}", "Refused.", days=day)
    for day in range(3, 7):
        await add(session, f"good{day}", "Fine.", days=day)

    class Picky:
        async def extract(self, episodes: list[Episode]) -> ExtractionResult:
            if any((episode.source_id or "").startswith("bad") for episode in episodes):
                request = httpx.Request("POST", "https://model.example/v1/chat/completions")
                response = httpx.Response(400, request=request, text="too long")
                raise httpx.HTTPStatusError("400", request=request, response=response)
            return ExtractionResult(
                plan=ConsolidationPlan(entities=[], relationships=[]), provider="stub"
            )

    summary = ConsolidationSummary.from_run(
        await service(Picky(), consolidation_max_attempts=1).run(OWNER, session)
    )

    assert (summary.status, summary.episodes_processed, summary.episodes_failed) == (
        "succeeded", 4, 3,
    )
    assert (summary.episodes_skipped, summary.has_more) == (3, False)


async def test_an_answer_without_a_graph_is_not_taken_for_an_empty_one(
    session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(_env_file=None, chat_model="m", consolidation_content_extraction=True)
    stub_chat(monkeypatch, [{"facts": ["Abhi likes tea"]}])
    await add(session, "a", "I like tea.")

    run = await ConsolidationService(
        extractor=LLMSemanticExtractor(settings), enricher=None, settings=settings
    ).run(OWNER, session)
    summary = ConsolidationSummary.from_run(run)

    assert (summary.episodes_processed, summary.episodes_failed) == (0, 1)
    assert summary.prompt_tokens == 900, "the unusable call was not charged"
