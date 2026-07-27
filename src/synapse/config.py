from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Synapse"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://synapse:synapse@localhost:5432/synapse"
    cactus_chat_base_url: str = "http://localhost:8080/v1"
    cactus_embedding_base_url: str = "http://localhost:8080/v1"
    cactus_chat_model: str = "google/gemma-4-E2B-it"
    cactus_embedding_model: str = "google/gemma-4-E2B-it"
    embedding_dimensions: int = 1536
    inference_mode: str = "auto"
    consolidation_mode: str = "auto"
    crustdata_api_key: str | None = None
    crustdata_base_url: str = "https://api.crustdata.com"
    crustdata_api_version: str = "2025-11-01"
    enrichment_cache_hours: int = Field(default=24, ge=1)
    procedural_min_occurrences: int = Field(default=2, ge=2)
    memory_hit_threshold: float = Field(default=0.62, ge=0, le=1)
    mem0_collection: str = "synapse_episodes_gemma_1536"

    @property
    def mem0_database_url(self) -> str:
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
