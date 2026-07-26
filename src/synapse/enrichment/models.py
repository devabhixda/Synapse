from typing import Any

from pydantic import BaseModel, Field


class EnrichmentResult(BaseModel):
    provider: str
    field_group: str
    payload: dict[str, Any]
    source_url: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)