from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from synapse.db.models import ConsolidationRun


class EntitySpec(BaseModel):
    key: str
    kind: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)


class RelationshipSpec(BaseModel):
    subject_key: str
    predicate: str
    object_key: str
    evidence_source_ids: list[str]
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)
    # The words in each evidence episode that support the edge, by source id.
    # A quote must appear in its episode, and is what the evidence row keeps;
    # without one the row keeps the whole episode, as it always has.
    evidence_quotes: dict[str, str] = Field(default_factory=dict)
    # When the edge became true. Defaults to when its newest evidence episode
    # began, which is wrong for a long thread: a reply today on a thread from
    # August dates the edge to August. A caller that knows when the deciding
    # message was said passes it here.
    asserted_at: datetime | None = None


class StatementSpec(BaseModel):
    """A sentence about an entity; see ``SemanticStatement``."""

    entity_key: str
    key: str
    text: str
    evidence_source_ids: list[str]
    evidence_quotes: dict[str, str] = Field(default_factory=dict)
    asserted_at: datetime | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)


class ConsolidationPlan(BaseModel):
    entities: list[EntitySpec]
    relationships: list[RelationshipSpec]
    statements: list[StatementSpec] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    plan: ConsolidationPlan
    provider: str
    fallback_reason: str | None = None
    # What producing the plan cost, for the run's budget and summary. A model
    # endpoint that reports no usage leaves the extractor to estimate it, and
    # says so with ``tokens_estimated``.
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False


class GraphAdjudication(BaseModel):
    approved: bool
    alias_map: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class RawEntitySpec(BaseModel):
    kind: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class RawRelationshipSpec(BaseModel):
    subject_kind: str
    subject_name: str
    predicate: str
    object_kind: str
    object_name: str
    evidence_source_ids: list[str]
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)
    # Verbatim words from the evidence that state the relationship. Kept as the
    # evidence excerpt for each episode it appears in; dropped if it appears in
    # none, since evidence that was not said is worse than none.
    quote: str | None = None


class RawConsolidationPlan(BaseModel):
    entities: list[RawEntitySpec]
    relationships: list[RawRelationshipSpec]

class ConsolidationBacklog(BaseModel):
    """Episodes still to consolidate in a group."""

    due: int
    skipped: int
    total: int


StopReason = Literal["complete", "max_episodes", "max_tokens", "error"]


class ConsolidationSummary(BaseModel):
    """What one consolidation run did, and what it left for the next.

    ``episodes_remaining`` counts episodes still due after this run, not
    counting ones skipped after repeated failures (``episodes_skipped``).
    ``has_more`` is true when some remain and the run did not stop on an
    error, so running again now would make progress; each run picks up where
    the last stopped. ``episodes_failed`` counts episodes the model could not
    handle this run; they stay due until they have failed too often.
    """

    run_id: str
    status: str
    stopped_reason: StopReason
    error: str | None = None
    provider: str
    episodes_processed: int = 0
    episodes_failed: int = 0
    episodes_skipped: int = 0
    episodes_remaining: int = 0
    has_more: bool = False
    batches: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_estimated: bool = False
    entities_created: int = 0
    entities_updated: int = 0
    relationships_created: int = 0
    relationships_updated: int = 0
    relationships_superseded: int = 0
    statements_created: int = 0
    statements_updated: int = 0
    statements_superseded: int = 0
    enrichments_created: int = 0
    enrichments_updated: int = 0
    enrichments_skipped: int = 0
    enrichments_cached: int = 0
    enrichments_failed: int = 0
    enrichment_enabled: bool = False
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @classmethod
    def from_run(cls, run: "ConsolidationRun") -> "ConsolidationSummary":
        diff: dict[str, Any] = dict(run.graph_diff or {})
        fields = {name for name in cls.model_fields} - {
            "run_id", "status", "error", "provider", "started_at", "completed_at"
        }
        values = {name: diff[name] for name in fields if name in diff}
        values.setdefault("stopped_reason", "error" if run.status == "failed" else "complete")
        return cls(
            run_id=str(run.id),
            status=run.status,
            error=run.error,
            provider=run.provider,
            started_at=run.started_at,
            completed_at=run.completed_at,
            **values,
        )
