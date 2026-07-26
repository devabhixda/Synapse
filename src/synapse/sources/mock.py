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
            source_id="slack-security-001",
            source_type=SourceType.SLACK,
            conversation_id="eng-security/thread-18",
            content=(
                "Nia: Security approved Titan encryption with KMS key auth-prod. "
                "We decided session tokens will keep a 30-minute TTL during migration."
            ),
            participants=["Nia", "Maya", "Jon"],
            occurred_at=datetime(2026, 7, 21, 15, 20, tzinfo=UTC),
            metadata={"channel": "eng-security", "thread_ts": "18"},
        ),
        SourceEvent(
            source_id="slack-repository-001",
            source_type=SourceType.SLACK,
            conversation_id="eng-platform/thread-51",
            content=(
                "Maya: The implementation lives in the services/atlas-api repository. "
                "Atlas API will dual-write sessions to Titan behind titan_sessions_v2."
            ),
            participants=["Maya", "Jon"],
            occurred_at=datetime(2026, 7, 21, 16, 10, tzinfo=UTC),
            metadata={
                "channel": "eng-platform",
                "thread_ts": "51",
                "repository": "services/atlas-api",
                "feature_flag": "titan_sessions_v2",
            },
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
            source_id="meet-incident-001",
            source_type=SourceType.MEETING,
            conversation_id="incident-review/2026-07-22",
            content=(
                "Incident review found the session backfill failing because legacy user_id "
                "values do not match Titan UUIDs. Jon owns the repair script by 3 PM; the "
                "staging deadline moves from Friday to Monday."
            ),
            participants=["Jon", "Maya", "Priya", "Nia"],
            occurred_at=datetime(2026, 7, 22, 11, 30, tzinfo=UTC),
            metadata={
                "meeting": "Incident review",
                "root_cause": "legacy user_id to Titan UUID mismatch",
                "new_deadline": "2026-07-27",
            },
        ),
        SourceEvent(
            source_id="slack-unblocked-001",
            source_type=SourceType.SLACK,
            conversation_id="project-synapse/thread-7",
            content=(
                "Jon: The backfill repair completed and the blocker is resolved. Atlas API "
                "staging now reads 95% of sessions from Titan with no auth error increase."
            ),
            participants=["Jon", "Maya", "Priya"],
            occurred_at=datetime(2026, 7, 22, 15, 42, tzinfo=UTC),
            metadata={
                "channel": "project-synapse",
                "thread_ts": "7",
                "staging_read_percent": 95,
                "blocker_status": "resolved",
            },
        ),
        SourceEvent(
            source_id="meet-rollout-001",
            source_type=SourceType.MEETING,
            conversation_id="launch-review/2026-07-23",
            content=(
                "Launch review decided the Titan production rollout will be 10% Monday, "
                "50% Wednesday, and 100% Friday. Priya owns rollout coordination; rollback "
                "starts if authentication errors exceed 1%."
            ),
            participants=["Priya", "Maya", "Jon", "Nia"],
            occurred_at=datetime(2026, 7, 23, 9, 0, tzinfo=UTC),
            metadata={
                "meeting": "Launch review",
                "rollout": {"2026-07-27": 10, "2026-07-29": 50, "2026-07-31": 100},
                "rollback_error_threshold_percent": 1,
            },
        ),
        SourceEvent(
            source_id="slack-status-002",
            source_type=SourceType.SLACK,
            conversation_id="project-synapse/thread-12",
            content=(
                "Priya: Please send today's Titan migration status with rollout readiness, "
                "remaining blockers, owners, and next actions."
            ),
            participants=["Priya", "Maya", "Jon"],
            occurred_at=datetime(2026, 7, 23, 11, 5, tzinfo=UTC),
            metadata={"channel": "project-synapse", "thread_ts": "12"},
        ),
        SourceEvent(
            source_id="slack-status-response-002",
            source_type=SourceType.SLACK,
            conversation_id="project-synapse/thread-12",
            content=(
                "Maya: Titan rollout is ready for Monday at 10%. Backfill is complete, Jon "
                "owns production monitoring, Priya owns coordination, and there are no open "
                "blockers. Next action is the go/no-go check at 09:00 Monday."
            ),
            participants=["Maya", "Priya", "Jon"],
            occurred_at=datetime(2026, 7, 23, 11, 12, tzinfo=UTC),
            metadata={
                "channel": "project-synapse",
                "thread_ts": "12",
                "status": "ready",
            },
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
        SourceEvent(
            source_id="slack-partnership-002",
            source_type=SourceType.SLACK,
            conversation_id="partnerships/thread-9",
            content=(
                "Sam: Crustdata identify and enrich are approved for the Synapse demo. "
                "External company attributes must stay labeled as enrichment and may only "
                "run during the System 2 sleep cycle."
            ),
            participants=["Sam", "Priya"],
            occurred_at=datetime(2026, 7, 24, 9, 25, tzinfo=UTC),
            metadata={
                "channel": "partnerships",
                "thread_ts": "9",
                "domain": "crustdata.com",
            },
        ),
        SourceEvent(
            source_id="meet-retrospective-001",
            source_type=SourceType.MEETING,
            conversation_id="titan-retrospective/2026-07-24",
            content=(
                "Titan migration retrospective: the staged read migration and session "
                "backfill completed without increased auth errors. The team kept the "
                "30-minute token TTL, documented services/atlas-api ownership, and approved "
                "the phased production rollout plan."
            ),
            participants=["Maya", "Jon", "Priya", "Nia"],
            occurred_at=datetime(2026, 7, 24, 15, 0, tzinfo=UTC),
            metadata={"meeting": "Titan migration retrospective", "status": "complete"},
        ),
    ]
