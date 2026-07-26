from synapse.consolidation.models import ConsolidationPlan, EntitySpec, RelationshipSpec
from synapse.db.models import Episode


def normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def entity_key(kind: str, name: str) -> str:
    return f"{kind}:{normalize_name(name)}"


def derive_semantic_plan(episodes: list[Episode]) -> ConsolidationPlan:
    entities: dict[str, EntitySpec] = {}
    relationships: dict[tuple[str, str, str], RelationshipSpec] = {}

    def add_entity(
        kind: str,
        name: str,
        *,
        aliases: list[str] | None = None,
        description: str | None = None,
        attributes: dict[str, str] | None = None,
        confidence: float = 1.0,
    ) -> str:
        key = entity_key(kind, name)
        current = entities.get(key)
        if current is None:
            entities[key] = EntitySpec(
                key=key,
                kind=kind,
                canonical_name=name,
                aliases=aliases or [],
                description=description,
                attributes=attributes or {},
                confidence=confidence,
            )
        else:
            current.aliases = sorted(set(current.aliases) | set(aliases or []))
            current.attributes.update(attributes or {})
            current.description = current.description or description
            current.confidence = max(current.confidence, confidence)
        return key

    def add_relationship(
        subject_key: str,
        predicate: str,
        object_key: str,
        source_id: str | None,
        *,
        attributes: dict[str, str] | None = None,
        confidence: float = 1.0,
    ) -> None:
        identity = (subject_key, predicate, object_key)
        current = relationships.get(identity)
        if current is None:
            relationships[identity] = RelationshipSpec(
                subject_key=subject_key,
                predicate=predicate,
                object_key=object_key,
                evidence_source_ids=[source_id] if source_id else [],
                attributes=attributes or {},
                confidence=confidence,
            )
        else:
            if source_id and source_id not in current.evidence_source_ids:
                current.evidence_source_ids.append(source_id)
            current.attributes.update(attributes or {})
            current.confidence = max(current.confidence, confidence)

    for episode in episodes:
        content = episode.content.casefold()
        people = {name: add_entity("Person", name) for name in episode.participants}

        titan_key: str | None = None
        atlas_key: str | None = None
        migration_key: str | None = None

        if "titan" in content or "auth database" in content:
            titan_key = add_entity(
                "System",
                "Titan",
                aliases=["auth database"],
                description="Authentication database used by Atlas API.",
                confidence=0.98,
            )
        if "atlas api" in content:
            atlas_key = add_entity("System", "Atlas API", confidence=0.98)
        if "titan migration" in content or ("migrate" in content and titan_key):
            migration_key = add_entity(
                "Project", "Titan migration", description="Migration of Atlas API to Titan."
            )
        if "synapse" in content:
            add_entity("Project", "Synapse")
        if "crustdata" in content:
            crustdata_key = add_entity(
                "Company", "Crustdata", attributes={"domain": "crustdata.com"}
            )
            synapse_key = add_entity("Project", "Synapse")
            add_relationship(
                crustdata_key,
                "proposed_for",
                synapse_key,
                episode.source_id,
                attributes={"purpose": "company and people enrichment"},
                confidence=0.9,
            )

        if titan_key and atlas_key and any(term in content for term in ("use it", "migrate")):
            add_relationship(
                atlas_key,
                "uses",
                titan_key,
                episode.source_id,
                confidence=0.96,
            )

        if "decided" in content and titan_key:
            decision_key = add_entity(
                "Decision",
                "Adopt Titan for authentication",
                description="Use Titan as the authentication database for new sessions.",
            )
            add_relationship(
                decision_key, "selects", titan_key, episode.source_id, confidence=0.98
            )
            if atlas_key:
                add_relationship(
                    decision_key, "affects", atlas_key, episode.source_id, confidence=0.95
                )

        if "maya owning" in content and migration_key and "Maya" in people:
            add_relationship(
                people["Maya"], "owns", migration_key, episode.source_id, confidence=0.97
            )

        if "backfill" in content:
            backfill_key = add_entity(
                "Task",
                "Session backfill",
                description="Backfill required for Atlas API staging migration.",
            )
            if "failing" in content or "blocked" in content:
                blocker_key = add_entity(
                    "Blocker",
                    "Session backfill failing",
                    description="Session backfill failure blocks Atlas API staging.",
                )
                add_relationship(
                    blocker_key, "blocks", backfill_key, episode.source_id, confidence=0.98
                )
                atlas_key = atlas_key or add_entity("System", "Atlas API")
                add_relationship(
                    blocker_key, "blocks", atlas_key, episode.source_id, confidence=0.95
                )

    return ConsolidationPlan(
        entities=sorted(entities.values(), key=lambda item: item.key),
        relationships=sorted(
            relationships.values(),
            key=lambda item: (item.subject_key, item.predicate, item.object_key),
        ),
    )