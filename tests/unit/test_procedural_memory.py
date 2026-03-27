"""Unit tests for procedural memory (mocked Weaviate client)."""

import uuid
from unittest.mock import MagicMock

from src.memory.procedural import (
    PROCEDURAL_COLLECTION,
    ProceduralPattern,
    find_patterns,
    init_procedural_collection,
    store_pattern,
)


def _make_mock_client(collection_exists: bool = True):
    client = MagicMock()
    client.collections.exists.return_value = collection_exists
    return client


def _make_fake_embedding(dims: int = 1024) -> list[float]:
    return [0.1] * dims


class TestInitProceduralCollection:
    def test_creates_when_not_exists(self):
        client = _make_mock_client(collection_exists=False)
        init_procedural_collection(client)

        client.collections.create.assert_called_once()
        call_kwargs = client.collections.create.call_args
        assert call_kwargs.kwargs["name"] == PROCEDURAL_COLLECTION

    def test_skips_when_exists(self):
        client = _make_mock_client(collection_exists=True)
        init_procedural_collection(client)

        client.collections.create.assert_not_called()

    def test_recreate_deletes_first(self):
        client = _make_mock_client(collection_exists=True)
        client.collections.exists.side_effect = [True, False]
        init_procedural_collection(client, recreate=True)

        client.collections.delete.assert_called_once_with(PROCEDURAL_COLLECTION)
        client.collections.create.assert_called_once()


class TestStorePattern:
    def test_store_returns_uuid(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock

        result = store_pattern(
            client,
            trigger="user asks about fees",
            action="check fee schedule document first",
            embedding=_make_fake_embedding(),
            metadata={"priority": "high"},
        )

        assert isinstance(result, str)
        uuid.UUID(result)
        collection_mock.data.insert.assert_called_once()

    def test_store_without_metadata(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock

        result = store_pattern(
            client,
            trigger="test trigger",
            action="test action",
            embedding=_make_fake_embedding(),
        )

        assert isinstance(result, str)


class TestFindPatterns:
    def test_find_returns_results(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock

        mock_obj = MagicMock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {
            "trigger": "user asks about deadlines",
            "action": "check compliance calendar",
        }
        mock_obj.metadata.distance = 0.15
        collection_mock.query.near_vector.return_value.objects = [mock_obj]

        results = find_patterns(client, _make_fake_embedding(), limit=3)

        assert len(results) == 1
        assert isinstance(results[0], ProceduralPattern)
        assert results[0].trigger == "user asks about deadlines"
        assert results[0].action == "check compliance calendar"
        assert results[0].score == 0.85  # 1.0 - 0.15

    def test_find_empty_results(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock
        collection_mock.query.near_vector.return_value.objects = []

        results = find_patterns(client, _make_fake_embedding())
        assert results == []
