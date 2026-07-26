import uuid
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from synapse.config import get_settings
from synapse.db.models import RecallEvent, SkillSpec, SkillSpecRevision, WorkflowPattern
from synapse.procedural.models import (
    DetectionResult,
    PatternView,
    SkillSpecSnapshot,
    SkillSpecView,
    SkillTemplate,
)
from synapse.procedural.templates import TEMPLATES


def classify_recall_family(query: str) -> str:
    normalized = " ".join(query.casefold().split())
    if any(
        phrase in normalized
        for phrase in ("industry", "company size", "sector", "headcount", "funding")
    ):
        return "company_research"
    if any(phrase in normalized for phrase in ("status", "blocker", "next action")):
        return "project_status_briefing"
    if any(
        phrase in normalized
        for phrase in (
            "architecture",
            "database",
            "decision",
            "migration",
            "integration",
            "system design",
        )
    ):
        return "architecture_context"
    return "knowledge_lookup"


def is_procedural_evidence(answer_origin: str) -> bool:
    return answer_origin in {"episodic", "semantic_fallback"}


class ProceduralMemoryService:
    async def detect(self, session: AsyncSession) -> DetectionResult:
        events = list(
            await session.scalars(select(RecallEvent).order_by(RecallEvent.created_at))
        )
        grouped: dict[str, list[RecallEvent]] = defaultdict(list)
        for event in events:
            if not is_procedural_evidence(event.answer_origin):
                continue
            family = classify_recall_family(event.query)
            event.normalized_intent = family
            grouped[family].append(event)

        settings = get_settings()
        candidate_groups = {
            family: family_events
            for family, family_events in grouped.items()
            if family in TEMPLATES
            and len(family_events) >= settings.procedural_min_occurrences
        }
        patterns_created = 0
        patterns_updated = 0
        revisions_created = 0

        for family, family_events in candidate_groups.items():
            template = TEMPLATES[family]
            pattern = await session.scalar(
                select(WorkflowPattern).where(WorkflowPattern.key == family)
            )
            confidence = min(0.98, 0.55 + len(family_events) * 0.1)
            evidence_ids = [str(event.id) for event in family_events]
            if pattern is None:
                pattern = WorkflowPattern(
                    key=family,
                    name=template.name.replace("_", " ").title(),
                    description=template.purpose,
                    status="candidate",
                    occurrence_count=len(family_events),
                    confidence=confidence,
                    evidence_recall_ids=evidence_ids,
                    first_seen_at=family_events[0].created_at,
                    last_seen_at=family_events[-1].created_at,
                )
                session.add(pattern)
                await session.flush()
                patterns_created += 1
            else:
                pattern.occurrence_count = len(family_events)
                pattern.confidence = confidence
                pattern.evidence_recall_ids = evidence_ids
                pattern.first_seen_at = family_events[0].created_at
                pattern.last_seen_at = family_events[-1].created_at
                patterns_updated += 1

            revisions_created += await self._upsert_skill(
                pattern, template, confidence, session
            )

        await session.commit()
        return DetectionResult(
            recall_events_scanned=len(events),
            candidate_families=len(candidate_groups),
            patterns_created=patterns_created,
            patterns_updated=patterns_updated,
            revisions_created=revisions_created,
        )

    async def _upsert_skill(
        self,
        pattern: WorkflowPattern,
        template: SkillTemplate,
        confidence: float,
        session: AsyncSession,
    ) -> int:
        snapshot = SkillSpecSnapshot(
            name=template.name,
            purpose=template.purpose,
            triggers=template.triggers,
            required_queries=template.required_queries,
            steps=template.steps,
            guardrails=template.guardrails,
            expected_output=template.expected_output,
            confidence=confidence,
        )
        skill = await session.scalar(
            select(SkillSpec).where(SkillSpec.pattern_id == pattern.id)
        )
        if skill is None:
            skill = SkillSpec(
                pattern_id=pattern.id,
                version=1,
                status="candidate",
                **snapshot.model_dump(),
            )
            session.add(skill)
            await session.flush()
            session.add(
                SkillSpecRevision(
                    skill_spec_id=skill.id,
                    version=1,
                    snapshot=snapshot.model_dump(mode="json"),
                )
            )
            return 1

        latest = await session.scalar(
            select(SkillSpecRevision)
            .where(SkillSpecRevision.skill_spec_id == skill.id)
            .order_by(SkillSpecRevision.version.desc())
            .limit(1)
        )
        serialized = snapshot.model_dump(mode="json")
        if latest is not None and latest.snapshot == serialized:
            return 0

        skill.version += 1
        skill.status = "candidate"
        pattern.status = "candidate"
        skill.name = snapshot.name
        skill.purpose = snapshot.purpose
        skill.triggers = snapshot.triggers
        skill.required_queries = snapshot.required_queries
        skill.steps = snapshot.steps
        skill.guardrails = snapshot.guardrails
        skill.expected_output = snapshot.expected_output
        skill.confidence = snapshot.confidence
        session.add(
            SkillSpecRevision(
                skill_spec_id=skill.id,
                version=skill.version,
                snapshot=serialized,
            )
        )
        return 1

    async def set_status(
        self, skill_id: str, status: str, session: AsyncSession
    ) -> SkillSpecView:
        try:
            parsed_id = uuid.UUID(skill_id)
        except ValueError as error:
            raise ValueError("skill_id must be a valid UUID") from error
        skill = await session.get(SkillSpec, parsed_id)
        if skill is None:
            raise ValueError(f"Skill spec {skill_id} was not found")
        if status not in {"candidate", "approved", "rejected"}:
            raise ValueError("Invalid skill status")
        skill.status = status
        pattern = await session.get(WorkflowPattern, skill.pattern_id)
        if pattern is not None:
            pattern.status = status
        await session.commit()
        await session.refresh(skill)
        return skill_view(skill)


async def list_patterns(session: AsyncSession) -> list[PatternView]:
    patterns = await session.scalars(
        select(WorkflowPattern).order_by(
            WorkflowPattern.occurrence_count.desc(), WorkflowPattern.name
        )
    )
    return [pattern_view(pattern) for pattern in patterns]


async def list_skills(
    session: AsyncSession, *, approved_only: bool = False
) -> list[SkillSpecView]:
    statement = select(SkillSpec).order_by(SkillSpec.name)
    if approved_only:
        statement = statement.where(SkillSpec.status == "approved")
    skills = await session.scalars(statement)
    return [skill_view(skill) for skill in skills]


async def get_skill_by_name(
    session: AsyncSession, name: str, *, approved_only: bool = True
) -> SkillSpecView:
    statement = select(SkillSpec).where(SkillSpec.name == name)
    if approved_only:
        statement = statement.where(SkillSpec.status == "approved")
    skill = await session.scalar(statement)
    if skill is None:
        qualifier = "approved " if approved_only else ""
        raise ValueError(f"No {qualifier}skill spec named {name}")
    return skill_view(skill)


def pattern_view(pattern: WorkflowPattern) -> PatternView:
    return PatternView(
        id=str(pattern.id),
        key=pattern.key,
        name=pattern.name,
        description=pattern.description,
        status=pattern.status,
        occurrence_count=pattern.occurrence_count,
        confidence=pattern.confidence,
        evidence_recall_ids=pattern.evidence_recall_ids,
        first_seen_at=pattern.first_seen_at.isoformat(),
        last_seen_at=pattern.last_seen_at.isoformat(),
    )


def skill_view(skill: SkillSpec) -> SkillSpecView:
    return SkillSpecView(
        id=str(skill.id),
        pattern_id=str(skill.pattern_id),
        name=skill.name,
        version=skill.version,
        status=skill.status,
        purpose=skill.purpose,
        triggers=skill.triggers,
        required_queries=skill.required_queries,
        steps=skill.steps,
        guardrails=skill.guardrails,
        expected_output=skill.expected_output,
        confidence=skill.confidence,
    )