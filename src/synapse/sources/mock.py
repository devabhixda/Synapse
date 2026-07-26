from datetime import UTC, datetime

from synapse.db.models import SourceType
from synapse.domain.models import SourceEvent


def mock_events() -> list[SourceEvent]:
    return [
        SourceEvent(
            source_id="slack-eng-001",
            source_type=SourceType.SLACK,
            conversation_id="eng-platform/thread-42",
            content=(
                "Maya: We decided the auth database will be called Titan. "
                "Atlas API should use it for all new sessions."
            ),
            participants=["Maya", "Jon"],
            occurred_at=datetime(2026, 7, 21, 9, 5, tzinfo=UTC),
            metadata={"channel": "eng-platform", "thread_ts": "42"},
        ),
        SourceEvent(
            source_id="meet-architecture-001",
            source_type=SourceType.MEETING,
            conversation_id="architecture-sync/2026-07-21",
            content=(
                "Architecture sync: Titan is the auth database. The team will migrate "
                "Atlas API reads first, with Maya owning the change by Friday."
            ),
            participants=["Maya", "Jon", "Priya"],
            occurred_at=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            metadata={"meeting": "Architecture sync"},
        ),
        SourceEvent(
            source_id="slack-status-001",
            source_type=SourceType.SLACK,
            conversation_id="project-synapse/thread-7",
            content="Priya: Can I get a status update on the Titan migration and current blockers?",
            participants=["Priya", "Maya"],
            occurred_at=datetime(2026, 7, 22, 10, 15, tzinfo=UTC),
            metadata={"channel": "project-synapse", "thread_ts": "7"},
        ),
        SourceEvent(
            source_id="slack-blocker-001",
            source_type=SourceType.SLACK,
            conversation_id="project-synapse/thread-7",
            content="Maya: Atlas API staging is blocked because the session backfill is failing.",
            participants=["Maya", "Priya"],
            occurred_at=datetime(2026, 7, 22, 10, 18, tzinfo=UTC),
            metadata={"channel": "project-synapse", "thread_ts": "7"},
        ),
        SourceEvent(
            source_id="meet-partnerships-001",
            source_type=SourceType.MEETING,
            conversation_id="partnerships/2026-07-23",
            content=(
                "Partnerships discussed using Crustdata at crustdata.com to enrich company "
                "and people nodes during the nightly Synapse sleep cycle."
            ),
            participants=["Priya", "Sam"],
            occurred_at=datetime(2026, 7, 23, 16, 0, tzinfo=UTC),
            metadata={"meeting": "Partnership review", "domain": "crustdata.com"},
        ),
    ]
