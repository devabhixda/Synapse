from synapse.consolidation.models import ConsolidationPlan, EntitySpec, RelationshipSpec
from synapse.db.models import Episode
from synapse.domain.models import semantic_assertions


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
            return
        if source_id and source_id not in current.evidence_source_ids:
            current.evidence_source_ids.append(source_id)
        current.attributes.update(attributes or {})
        current.confidence = max(current.confidence, confidence)

    for episode in episodes:
        for participant in episode.participants:
            add_entity("Person", participant)

        assertions = semantic_assertions(episode.details)
        for entity_assertion in assertions.entities:
            add_entity(
                entity_assertion.kind,
                entity_assertion.name,
                aliases=entity_assertion.aliases,
                description=entity_assertion.description,
                attributes=entity_assertion.attributes,
                confidence=entity_assertion.confidence,
            )

        for relationship_assertion in assertions.relationships:
            subject_key = add_entity(
                relationship_assertion.subject_kind,
                relationship_assertion.subject_name,
            )
            object_key = add_entity(
                relationship_assertion.object_kind,
                relationship_assertion.object_name,
            )
            add_relationship(
                subject_key,
                relationship_assertion.predicate,
                object_key,
                episode.source_id,
                attributes=relationship_assertion.attributes,
                confidence=relationship_assertion.confidence,
            )

    return ConsolidationPlan(
        entities=sorted(entities.values(), key=lambda item: item.key),
        relationships=sorted(
            relationships.values(),
            key=lambda item: (item.subject_key, item.predicate, item.object_key),
        ),
    )