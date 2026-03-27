"""Shared pytest fixtures for the memory-agent-regulatory test suite."""

import pytest
from sqlalchemy import Engine

from tests.helpers import make_sqlite_engine, make_test_session, make_test_settings


@pytest.fixture
def test_settings():
    """Default test settings with reranking disabled."""
    return make_test_settings()


@pytest.fixture
def db_engine() -> Engine:
    """In-memory SQLite engine with all tables created."""
    return make_sqlite_engine()


@pytest.fixture
def db_session(db_engine):
    """SQLAlchemy session backed by in-memory SQLite. Rolls back after test."""
    session = make_test_session(db_engine)
    yield session
    session.close()
