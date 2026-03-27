"""Pydantic Settings — loads config.yaml then overrides with environment variables."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


# ── Nested sub-models (plain BaseModel, not BaseSettings) ───────────────────


class EmbeddingsSettings(BaseModel):
    model: str = "BAAI/bge-m3"
    batch_size: int = 32
    dimensions: int = 1024


class ChunkingSettings(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50


class HybridSettings(BaseModel):
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    fusion_type: Literal["ranked", "relative_score"] = "ranked"


class RetrievalSettings(BaseModel):
    top_k: int = 10
    collection: str = "RegulatoryChunks"
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)
    search_strategy: Literal["vector", "keyword", "hybrid"] = "hybrid"
    hybrid: HybridSettings = HybridSettings()


class RerankingSettings(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class LLMSettings(BaseModel):
    model: str = "llama3.2:3b"
    temperature: float = 0
    top_p: float = 1.0
    num_ctx: int = 4096
    num_predict: int = 1024


class RAGSettings(BaseModel):
    max_context_tokens: int = 4096


class WeaviateSettings(BaseModel):
    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051


class OllamaSettings(BaseModel):
    host: str = "http://localhost:11434"


class PhoenixSettings(BaseModel):
    host: str = "localhost"
    port: int = 6006


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "agent"
    password: str = ""
    name: str = "agent_memory"
    echo: bool = False

    @property
    def url(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class MemorySettings(BaseModel):
    max_history_turns: int = 20
    summary_threshold: int = 15
    semantic_top_k: int = 5
    procedural_top_k: int = 3


class AgentSettings(BaseModel):
    max_iterations: int = 5
    tool_timeout_s: float = 30.0


class IngestionSettings(BaseModel):
    sources: list[dict[str, Any]] = Field(default_factory=list)
    web_delay: float = 1.0


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: Literal["text", "json"] = "text"


# ── YAML config source ──────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file. Returns empty dict if file is missing or invalid."""
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ── Root Settings ────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Project-wide configuration.

    Priority (highest → lowest):
        1. Environment variables (e.g. WEAVIATE_HOST, LOG_LEVEL)
        2. config/config.yaml values
        3. Field defaults defined here
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )

    embeddings: EmbeddingsSettings = EmbeddingsSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    reranking: RerankingSettings = RerankingSettings()
    llm: LLMSettings = LLMSettings()
    rag: RAGSettings = RAGSettings()
    weaviate: WeaviateSettings = WeaviateSettings()
    ollama: OllamaSettings = OllamaSettings()
    phoenix: PhoenixSettings = PhoenixSettings()
    database: DatabaseSettings = DatabaseSettings()
    memory: MemorySettings = MemorySettings()
    agent: AgentSettings = AgentSettings()
    ingestion: IngestionSettings = IngestionSettings()
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def from_yaml(cls, path: Path | None = None, **env_overrides: Any) -> Settings:
        """Build Settings from YAML file + environment variables.

        This is the canonical constructor. It reads the YAML file first,
        then lets pydantic-settings layer environment variables on top.
        """
        yaml_data = _load_yaml(path or _CONFIG_PATH)
        # Merge env_overrides into yaml_data (for testing)
        yaml_data.update(env_overrides)
        return cls(**yaml_data)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Call ``get_settings.cache_clear()`` in tests to reset.
    """
    return Settings.from_yaml()
