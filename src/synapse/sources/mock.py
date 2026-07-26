from datetime import UTC, datetime
from typing import Any

from synapse.db.models import SourceType
from synapse.domain.models import SourceEvent


def entity(
    kind: str,
    name: str,
    *,
    description: str | None = None,
    attributes: dict[str, str] | None = None,
    aliases: list[str] | None = None,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "description": description,
        "attributes": attributes or {},
        "aliases": aliases or [],
        "confidence": confidence,
    }


def relationship(
    subject_kind: str,
    subject_name: str,
    predicate: str,
    object_kind: str,
    object_name: str,
    *,
    attributes: dict[str, str] | None = None,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "subject_kind": subject_kind,
        "subject_name": subject_name,
        "predicate": predicate,
        "object_kind": object_kind,
        "object_name": object_name,
        "attributes": attributes or {},
        "confidence": confidence,
    }


SYNAPSE = entity(
    "Project",
    "Synapse",
    description="Episodic-first organizational memory with semantic consolidation.",
)

SPONSOR_PROFILES = {
    "Channel3": (
        entity(
            "Company",
            "Channel3",
            description="Agentic commerce infrastructure for AI product discovery.",
            attributes={
                "domain": "trychannel3.com",
                "official_source": "https://trychannel3.com/",
            },
        ),
        entity(
            "Product",
            "Channel3 API",
            description=(
                "Product-data API with real-time prices, metadata, and monetization."
            ),
        ),
    ),
    "Allus AI": (
        entity(
            "Company",
            "Allus AI",
            description="Manufacturing vision AI for inspection and process monitoring.",
            attributes={
                "domain": "allus.ai",
                "official_source": "https://www.allus.ai/",
            },
        ),
        entity(
            "Product",
            "Allus AI vision platform",
            description=(
                "Vision foundation model deployable to cameras, industrial PCs, and robots."
            ),
        ),
    ),
    "Random Labs": (
        entity(
            "Company",
            "Random Labs",
            description="Builder of general software agents for repository-scale work.",
            attributes={
                "domain": "randomlabs.ai",
                "official_source": "https://randomlabs.ai/",
            },
        ),
        entity(
            "Product",
            "Slate",
            description="Generalist software agent for long-running repository work.",
        ),
    ),
    "Hexclave": (
        entity(
            "Company",
            "Hexclave",
            aliases=["Stack Auth"],
            description="Open-source authentication and user-management platform.",
            attributes={
                "domain": "hexclave.com",
                "official_source": "https://www.hexclave.com/",
            },
        ),
        entity(
            "Product",
            "Hexclave platform",
            description="Authentication, teams, RBAC, payments, and product analytics.",
        ),
    ),
}


def sponsor_research(company_name: str, product_name: str) -> dict[str, Any]:
    company, product = SPONSOR_PROFILES[company_name]
    return {
        "entities": [company, product, SYNAPSE],
        "relationships": [
            relationship("Company", company_name, "offers", "Product", product_name),
            relationship("Company", company_name, "evaluated_for", "Project", "Synapse"),
        ],
    }


SEMANTIC_ASSERTIONS: dict[str, dict[str, Any]] = {
    "meet-hackathon-kickoff-001": {
        "entities": [
            SYNAPSE,
            entity("Task", "Sponsor evaluation"),
            entity("Task", "Demo narrative"),
            entity("Task", "VS Code MCP flow"),
        ],
        "relationships": [
            relationship("Person", "Sam", "owns", "Task", "Sponsor evaluation"),
            relationship("Person", "Priya", "owns", "Task", "Demo narrative"),
            relationship("Person", "Jon", "owns", "Task", "VS Code MCP flow"),
        ],
    },
    "slack-channel3-research-001": sponsor_research("Channel3", "Channel3 API"),
    "slack-channel3-fit-001": {
        "entities": [
            SPONSOR_PROFILES["Channel3"][0],
            SYNAPSE,
            entity("Task", "Channel3 API spike"),
        ],
        "relationships": [
            relationship("Company", "Channel3", "optional_for", "Project", "Synapse"),
            relationship("Person", "Sam", "owns", "Task", "Channel3 API spike"),
        ],
    },
    "slack-allus-research-001": sponsor_research(
        "Allus AI", "Allus AI vision platform"
    ),
    "slack-allus-fit-001": {
        "entities": [SPONSOR_PROFILES["Allus AI"][0], SYNAPSE],
        "relationships": [
            relationship(
                "Company",
                "Allus AI",
                "deferred_for",
                "Project",
                "Synapse",
                attributes={"reason": "manufacturing vision is outside the memory POC"},
            )
        ],
    },
    "slack-random-labs-research-001": sponsor_research("Random Labs", "Slate"),
    "slack-random-labs-fit-001": {
        "entities": [
            SPONSOR_PROFILES["Random Labs"][0],
            SYNAPSE,
            entity(
                "Decision",
                "Keep Synapse MCP read-only",
                description=(
                    "Synapse supplies context and procedures; coding agents own execution."
                ),
            ),
        ],
        "relationships": [
            relationship("Company", "Random Labs", "complements", "Project", "Synapse"),
            relationship(
                "Decision",
                "Keep Synapse MCP read-only",
                "governs",
                "Project",
                "Synapse",
            ),
        ],
    },
    "slack-hexclave-research-001": sponsor_research(
        "Hexclave", "Hexclave platform"
    ),
    "slack-hexclave-fit-001": {
        "entities": [SPONSOR_PROFILES["Hexclave"][0], SYNAPSE],
        "relationships": [
            relationship(
                "Company",
                "Hexclave",
                "identity_candidate_for",
                "Project",
                "Synapse",
                attributes={"timing": "after the single-user POC"},
            )
        ],
    },
    "meet-sponsor-review-001": {
        "entities": [profile[0] for profile in SPONSOR_PROFILES.values()] + [SYNAPSE],
        "relationships": [
            relationship("Company", "Channel3", "optional_for", "Project", "Synapse"),
            relationship("Company", "Allus AI", "deferred_for", "Project", "Synapse"),
            relationship("Company", "Random Labs", "complements", "Project", "Synapse"),
            relationship(
                "Company", "Hexclave", "identity_candidate_for", "Project", "Synapse"
            ),
        ],
    },
    "slack-channel3-blocker-001": {
        "entities": [
            SPONSOR_PROFILES["Channel3"][0],
            entity("Task", "Channel3 API spike"),
            entity(
                "Blocker",
                "Channel3 developer API key unavailable",
                attributes={"impact": "non-critical"},
            ),
        ],
        "relationships": [
            relationship(
                "Blocker",
                "Channel3 developer API key unavailable",
                "blocks",
                "Task",
                "Channel3 API spike",
            )
        ],
    },
    "slack-status-response-001": {
        "entities": [SYNAPSE, entity("Task", "Sponsor evaluation")],
        "relationships": [
            relationship("Person", "Sam", "owns", "Task", "Sponsor evaluation"),
            relationship("Person", "Priya", "owns", "Task", "Demo narrative"),
        ],
    },
    "meet-recall-rehearsal-001": {
        "entities": [SYNAPSE, entity("Task", "Recall rehearsal")],
        "relationships": [
            relationship("Task", "Recall rehearsal", "validates", "Project", "Synapse")
        ],
    },
    "slack-channel3-resolution-001": {
        "entities": [
            SPONSOR_PROFILES["Channel3"][0],
            entity("Task", "Channel3 API spike", attributes={"status": "closed"}),
            entity(
                "Blocker",
                "Channel3 developer API key unavailable",
                attributes={"impact": "accepted limitation"},
            ),
        ],
        "relationships": [
            relationship(
                "Blocker",
                "Channel3 developer API key unavailable",
                "blocks",
                "Task",
                "Channel3 API spike",
            )
        ],
    },
    "meet-demo-readiness-001": {
        "entities": [
            SYNAPSE,
            entity("Decision", "Approve the sponsor demo narrative"),
        ],
        "relationships": [
            relationship(
                "Decision",
                "Approve the sponsor demo narrative",
                "approves",
                "Project",
                "Synapse",
            )
        ],
    },
}


def mock_events() -> list[SourceEvent]:
    events = [
        SourceEvent(
            source_id="meet-hackathon-kickoff-001",
            source_type=SourceType.MEETING,
            conversation_id="hackathon-kickoff/2026-07-21",
            content=(
                "Hackathon kickoff: the team will demo Synapse as an episodic-first memory "
                "system. Priya owns the story, Maya owns the memory architecture, Jon owns "
                "the VS Code MCP flow, and Sam will evaluate Channel3, Allus AI, Random "
                "Labs, and Hexclave without forcing a sponsor into the core path."
            ),
            participants=["Priya", "Maya", "Jon", "Sam"],
            occurred_at=datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
            metadata={"meeting": "Hackathon kickoff", "project": "Synapse"},
        ),
        SourceEvent(
            source_id="slack-channel3-research-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/channel3",
            content=(
                "Sam: Verified from Channel3's official site: Channel3 provides agentic "
                "commerce infrastructure. Its API gives AI shopping experiences access to "
                "more than 100 million products with real-time prices, deep metadata, and "
                "built-in monetization."
            ),
            participants=["Sam", "Priya"],
            occurred_at=datetime(2026, 7, 21, 10, 10, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "channel3",
                "company": "Channel3",
                "domain": "trychannel3.com",
                "fact_source": "https://trychannel3.com/",
                "fact_scope": "public product claim",
            },
        ),
        SourceEvent(
            source_id="slack-channel3-fit-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/channel3",
            content=(
                "Priya: Channel3 is a strong optional demo extension: a shopping-agent query "
                "could become a rich episode, but commerce is not required to prove the "
                "memory loop. Sam owns an API-key check by Tuesday noon."
            ),
            participants=["Priya", "Sam"],
            occurred_at=datetime(2026, 7, 21, 10, 24, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "channel3",
                "company": "Channel3",
                "evaluation": "optional extension",
                "deadline": "2026-07-22T12:00:00Z",
            },
        ),
        SourceEvent(
            source_id="slack-allus-research-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/allus-ai",
            content=(
                "Sam: Verified from Allus AI's official site: Allus AI offers a vision "
                "foundation model for manufacturing quality inspection, process monitoring, "
                "and anomaly detection. Deployments can run on cameras, industrial PCs, or "
                "robots at the edge, with cloud analytics available."
            ),
            participants=["Sam", "Maya"],
            occurred_at=datetime(2026, 7, 21, 11, 5, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "allus-ai",
                "company": "Allus AI",
                "domain": "allus.ai",
                "fact_source": "https://www.allus.ai/",
                "fact_scope": "public product claim",
            },
        ),
        SourceEvent(
            source_id="slack-allus-fit-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/allus-ai",
            content=(
                "Maya: Allus AI validates our local-first talking point, but manufacturing "
                "vision is outside the Synapse POC. We decided not to add an artificial "
                "integration; the demo will recall why it was evaluated and deferred."
            ),
            participants=["Maya", "Sam", "Priya"],
            occurred_at=datetime(2026, 7, 21, 11, 18, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "allus-ai",
                "company": "Allus AI",
                "evaluation": "deferred",
                "decision_reason": "outside the memory POC",
            },
        ),
        SourceEvent(
            source_id="slack-random-labs-research-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/random-labs",
            content=(
                "Jon: Verified from Random Labs documentation: Random Labs builds Slate, "
                "an open-beta generalist software agent for long-running repository work. "
                "Slate coordinates parallel agents and manages context deliberately."
            ),
            participants=["Jon", "Maya"],
            occurred_at=datetime(2026, 7, 21, 13, 20, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "random-labs",
                "company": "Random Labs",
                "domain": "randomlabs.ai",
                "fact_source": "https://randomlabs.ai/",
                "fact_scope": "public product claim",
            },
        ),
        SourceEvent(
            source_id="slack-random-labs-fit-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/random-labs",
            content=(
                "Jon: Slate and Synapse are complementary. Slate performs repository work; "
                "Synapse supplies recalled organizational context and approved procedures. "
                "We decided the demo handoff ends at MCP guidance, never code execution."
            ),
            participants=["Jon", "Maya", "Priya"],
            occurred_at=datetime(2026, 7, 21, 13, 42, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "random-labs",
                "company": "Random Labs",
                "evaluation": "complementary",
                "decision": "context handoff only",
            },
        ),
        SourceEvent(
            source_id="slack-hexclave-research-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/hexclave",
            content=(
                "Sam: Verified from Hexclave's official documentation: Hexclave is an "
                "open-source authentication and user-management platform, formerly Stack "
                "Auth. It supports prebuilt auth UI, teams, RBAC, API keys, payments, "
                "webhooks, analytics, and framework SDKs."
            ),
            participants=["Sam", "Jon"],
            occurred_at=datetime(2026, 7, 21, 14, 15, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "hexclave",
                "company": "Hexclave",
                "domain": "hexclave.com",
                "fact_source": "https://www.hexclave.com/",
                "fact_scope": "public product claim",
            },
        ),
        SourceEvent(
            source_id="slack-hexclave-fit-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/hexclave",
            content=(
                "Jon: Hexclave is the best fit for a future multi-user Synapse, especially "
                "teams and RBAC. We decided to defer authentication for the single-user POC "
                "and record Hexclave as the preferred identity candidate after the demo."
            ),
            participants=["Jon", "Maya", "Priya"],
            occurred_at=datetime(2026, 7, 21, 14, 32, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "hexclave",
                "company": "Hexclave",
                "evaluation": "preferred future integration",
                "decision_reason": "single-user POC has no auth requirement",
            },
        ),
        SourceEvent(
            source_id="meet-sponsor-review-001",
            source_type=SourceType.MEETING,
            conversation_id="sponsor-review/2026-07-22",
            content=(
                "Sponsor review: keep the core architecture honest. Random Labs is the "
                "closest workflow complement, Hexclave is the preferred future identity "
                "layer, Channel3 remains an optional commerce demo, and Allus AI is deferred "
                "because manufacturing vision is outside scope."
            ),
            participants=["Priya", "Maya", "Jon", "Sam"],
            occurred_at=datetime(2026, 7, 22, 9, 30, tzinfo=UTC),
            metadata={"meeting": "Sponsor review", "project": "Synapse"},
        ),
        SourceEvent(
            source_id="slack-channel3-blocker-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/channel3",
            content=(
                "Sam: The optional Channel3 spike is blocked because we do not have a "
                "developer API key yet. Public documentation is enough for sponsor context, "
                "so this does not block the Synapse demo."
            ),
            participants=["Sam", "Priya"],
            occurred_at=datetime(2026, 7, 22, 10, 2, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "channel3",
                "company": "Channel3",
                "blocker": "developer API key unavailable",
                "impact": "optional spike only",
            },
        ),
        SourceEvent(
            source_id="slack-status-request-001",
            source_type=SourceType.SLACK,
            conversation_id="hackathon-status/thread-8",
            content=(
                "Priya: Can I get a sponsor-integration status update with confirmed facts, "
                "decisions, open blockers, owners, and the next demo actions?"
            ),
            participants=["Priya", "Maya", "Jon", "Sam"],
            occurred_at=datetime(2026, 7, 22, 11, 0, tzinfo=UTC),
            metadata={"channel": "hackathon-status", "thread_ts": "8"},
        ),
        SourceEvent(
            source_id="slack-status-response-001",
            source_type=SourceType.SLACK,
            conversation_id="hackathon-status/thread-8",
            content=(
                "Maya: Sponsor research is complete for all four companies. Jon owns the "
                "Random Labs and Hexclave talking points, Sam owns Channel3 and Allus AI, "
                "and Priya owns the final narrative. The only blocker is Channel3 API access, "
                "which is non-critical. Next action is a recall rehearsal Wednesday at 2 PM."
            ),
            participants=["Maya", "Priya", "Jon", "Sam"],
            occurred_at=datetime(2026, 7, 22, 11, 8, tzinfo=UTC),
            metadata={
                "channel": "hackathon-status",
                "thread_ts": "8",
                "status": "on track",
                "next_action_at": "2026-07-23T14:00:00Z",
            },
        ),
        SourceEvent(
            source_id="meet-recall-rehearsal-001",
            source_type=SourceType.MEETING,
            conversation_id="recall-rehearsal/2026-07-23",
            content=(
                "Recall rehearsal: ask what each sponsor does, why it fits or was deferred, "
                "who owns follow-up, and which claims came from official sources. The first "
                "query should use semantic fallback; a paraphrase should then hit the new "
                "retrieval episode."
            ),
            participants=["Priya", "Maya", "Jon", "Sam"],
            occurred_at=datetime(2026, 7, 23, 14, 0, tzinfo=UTC),
            metadata={
                "meeting": "Recall rehearsal",
                "project": "Synapse",
                "acceptance_test": "semantic fallback then episodic hit",
            },
        ),
        SourceEvent(
            source_id="slack-status-request-002",
            source_type=SourceType.SLACK,
            conversation_id="hackathon-status/thread-11",
            content=(
                "Priya: Please send the final sponsor status, including integration choices, "
                "the Channel3 blocker, demo owners, and next actions."
            ),
            participants=["Priya", "Maya", "Jon", "Sam"],
            occurred_at=datetime(2026, 7, 24, 9, 10, tzinfo=UTC),
            metadata={"channel": "hackathon-status", "thread_ts": "11"},
        ),
        SourceEvent(
            source_id="slack-status-response-002",
            source_type=SourceType.SLACK,
            conversation_id="hackathon-status/thread-11",
            content=(
                "Jon: Final status is ready. Random Labs and Hexclave have primary talking "
                "points; Channel3 is optional and its missing API key remains non-critical; "
                "Allus AI is an evidence-backed defer. No core blockers remain. Priya will "
                "run the demo and Sam will verify public-source citations before kickoff."
            ),
            participants=["Jon", "Priya", "Sam"],
            occurred_at=datetime(2026, 7, 24, 9, 18, tzinfo=UTC),
            metadata={
                "channel": "hackathon-status",
                "thread_ts": "11",
                "status": "demo ready",
            },
        ),
        SourceEvent(
            source_id="slack-channel3-resolution-001",
            source_type=SourceType.SLACK,
            conversation_id="sponsor-research/channel3",
            content=(
                "Sam: We closed the Channel3 spike for this demo. The API-key blocker remains "
                "unresolved, but the decision is to use verified public product context and "
                "avoid a fake integration. This has no impact on demo readiness."
            ),
            participants=["Sam", "Priya"],
            occurred_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
            metadata={
                "channel": "sponsor-research",
                "thread_ts": "channel3",
                "company": "Channel3",
                "evaluation": "public context only",
                "blocker_status": "accepted limitation",
            },
        ),
        SourceEvent(
            source_id="meet-demo-readiness-001",
            source_type=SourceType.MEETING,
            conversation_id="demo-readiness/2026-07-24",
            content=(
                "Demo readiness review approved the sponsor narrative. Every public product "
                "claim has an official source, internal choices are labeled as team decisions, "
                "and missing provider coverage or API access will be shown as unknown rather "
                "than invented. Synapse is ready for the final walkthrough."
            ),
            participants=["Priya", "Maya", "Jon", "Sam"],
            occurred_at=datetime(2026, 7, 24, 15, 30, tzinfo=UTC),
            metadata={
                "meeting": "Demo readiness review",
                "project": "Synapse",
                "status": "approved",
            },
        ),
    ]
    for event in events:
        assertions = SEMANTIC_ASSERTIONS.get(event.source_id)
        if assertions is not None:
            event.metadata["semantic_assertions"] = assertions
    return events
