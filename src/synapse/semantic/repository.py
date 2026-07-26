from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from synapse.db.models import RelationshipEvidence, SemanticEntity, SemanticRelationship


async def load_entities(session: AsyncSession) -> list[SemanticEntity]:
    return list(
        await session.scalars(
            select(SemanticEntity)
            .options(
                selectinload(SemanticEntity.aliases),
                selectinload(SemanticEntity.external_attributes),
            )
            .order_by(SemanticEntity.kind, SemanticEntity.canonical_name)
        )
    )


async def load_relationships(session: AsyncSession) -> list[SemanticRelationship]:
    return list(
        await session.scalars(
            select(SemanticRelationship)
            .options(
                selectinload(SemanticRelationship.subject),
                selectinload(SemanticRelationship.object),
                selectinload(SemanticRelationship.evidence).selectinload(
                    RelationshipEvidence.episode
                ),
            )
            .order_by(SemanticRelationship.predicate, SemanticRelationship.id)
        )
    )