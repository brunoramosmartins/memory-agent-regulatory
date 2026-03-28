"""Database engine and session factory."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Return a cached SQLAlchemy engine.

    When called without arguments, reads the URL from centralized settings.
    """
    global _engine
    if _engine is None:
        if url is None:
            from src.config import get_settings

            db = get_settings().database
            url = db.url
            echo = db.echo
        _engine = create_engine(url, echo=echo, pool_pre_ping=True)
    return _engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a cached session factory."""
    global _session_factory
    if _session_factory is None:
        if engine is None:
            engine = get_engine()
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return _session_factory


def get_session() -> Session:
    """Create and return a new database session.

    Convenience wrapper around get_session_factory() for callers that
    need a ready-to-use session without managing the factory directly.
    """
    factory = get_session_factory()
    return factory()


def reset_engine() -> None:
    """Reset cached engine and session factory. Useful for tests."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
