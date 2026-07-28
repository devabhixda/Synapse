"""Configuration, including which model server answers each kind of call.

Inference is addressed as two OpenAI-compatible endpoints — one for chat, one
for embeddings — rather than as a named provider. The defaults point at a local
Cactus serving Gemma, which is how the project is meant to be run, but nothing
below the settings knows that: pointing the URLs at a hosted gateway and
supplying its key is the whole of what switching providers involves.

Chat and embeddings are configured separately because they are not always the
same service. A gateway that serves chat may serve no embedding model at all,
leaving embeddings to a second deployment with its own URL, key, and dimension.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Synapse"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://synapse:synapse@localhost:5432/synapse"
    chat_base_url: str = "http://localhost:8080/v1"
    embedding_base_url: str = "http://localhost:8080/v1"
    chat_model: str = "google/gemma-4-E2B-it"
    embedding_model: str = "google/gemma-4-E2B-it"
    # A local Cactus ignores the bearer token, so the default is a placeholder
    # rather than a secret. A hosted endpoint checks it.
    chat_api_key: str = "cactus-local"
    embedding_api_key: str = "cactus-local"
    embedding_dimensions: int = 1536
    # Extra completion budget for models that think before answering. Every
    # call site sizes its own budget for the answer it expects, which is all a
    # model like Gemma spends tokens on. A reasoning model bills its private
    # reasoning against the same budget and can exhaust it before writing a
    # single character, so those budgets need to be raised — by a lot, and
    # without changing what each call site asks for. Set this to a few thousand
    # when pointing at one; it costs nothing on a model that doesn't reason,
    # which stops when it is done rather than when it hits the cap.
    chat_reasoning_tokens: int = Field(default=0, ge=0)
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
