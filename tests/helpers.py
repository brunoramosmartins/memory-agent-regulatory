"""Shared test helpers for building Settings instances."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import (
    DatabaseSettings,
    HybridSettings,
    MemorySettings,
    RerankingSettings,
    RetrievalSettings,
    Settings,
)
from src.memory.models import Base


def make_test_settings(
    search_strategy: str = "vector",
    min_similarity: float = 0.0,
    reranking_enabled: bool = False,
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranking_top_n: int = 5,
    hybrid_alpha: float = 0.5,
    hybrid_fusion_type: str = "ranked",
    max_context_tokens: int = 4096,
    **kwargs,
) -> Settings:
    """Build a Settings object with test-friendly defaults.

    Only overrides values relevant to retriever/RAG tests — everything
    else keeps its model default.
    """
    return Settings(
        retrieval=RetrievalSettings(
            search_strategy=search_strategy,
            min_similarity=min_similarity,
            hybrid=HybridSettings(
                alpha=hybrid_alpha,
                fusion_type=hybrid_fusion_type,
            ),
        ),
        reranking=RerankingSettings(
            enabled=reranking_enabled,
            model=reranking_model,
            top_n=reranking_top_n,
        ),
        database=DatabaseSettings(host="localhost", password="test"),
        memory=MemorySettings(),
        **kwargs,
    )


def make_sqlite_engine() -> Engine:
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def make_test_session(engine: Engine | None = None) -> Session:
    """Create a test SQLAlchemy session backed by SQLite."""
    if engine is None:
        engine = make_sqlite_engine()
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return factory()
