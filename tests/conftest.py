"""Shared pytest fixtures for the memory-agent-regulatory test suite."""

import pytest

from tests.helpers import make_test_settings


@pytest.fixture
def test_settings():
    """Default test settings with reranking disabled."""
    return make_test_settings()
