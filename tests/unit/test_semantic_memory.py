"""Unit tests for semantic memory (mocked Weaviate client)."""

import uuid
from unittest.mock import MagicMock

from src.memory.semantic import (
    SEMANTIC_COLLECTION,
    SemanticResult,
    delete,
    init_semantic_collection,
    search,
    store,
)


def _make_mock_client(collection_exists: bool = True):
    """Create a mock Weaviate client."""
    client = MagicMock()
    client.collections.exists.return_value = collection_exists
    return client


def _make_fake_embedding(dims: int = 1024) -> list[float]:
    return [0.1] * dims


class TestInitSemanticCollection:
    def test_creates_collection_when_not_exists(self):
        client = _make_mock_client(collection_exists=False)
        init_semantic_collection(client)

        client.collections.create.assert_called_once()
        call_kwargs = client.collections.create.call_args
        assert call_kwargs.kwargs["name"] == SEMANTIC_COLLECTION

    def test_skips_creation_when_exists(self):
        client = _make_mock_client(collection_exists=True)
        init_semantic_collection(client)

        client.collections.create.assert_not_called()

    def test_recreate_deletes_then_creates(self):
        client = _make_mock_client(collection_exists=True)
        # After delete, exists returns False for the create check
        client.collections.exists.side_effect = [True, False]
        init_semantic_collection(client, recreate=True)

        client.collections.delete.assert_called_once_with(SEMANTIC_COLLECTION)
        client.collections.create.assert_called_once()


class TestStore:
    def test_store_returns_uuid(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock

        result = store(
            client,
            content="PIX has no fees",
            embedding=_make_fake_embedding(),
            source="agent_response",
            thread_id="thread-123",
        )

        assert isinstance(result, str)
        # Verify it's a valid UUID
        uuid.UUID(result)
        collection_mock.data.insert.assert_called_once()


class TestSearch:
    def test_search_returns_results(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock

        # Mock Weaviate response
        mock_obj = MagicMock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {
            "content": "PIX is free",
            "source": "agent",
            "thread_id": "t1",
        }
        mock_obj.metadata.distance = 0.2
        collection_mock.query.near_vector.return_value.objects = [mock_obj]

        results = search(client, _make_fake_embedding(), limit=5)

        assert len(results) == 1
        assert isinstance(results[0], SemanticResult)
        assert results[0].content == "PIX is free"
        assert results[0].score == 0.8  # 1.0 - 0.2

    def test_search_with_thread_filter(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock
        collection_mock.query.near_vector.return_value.objects = []

        search(client, _make_fake_embedding(), thread_id="specific-thread")

        call_kwargs = collection_mock.query.near_vector.call_args.kwargs
        assert call_kwargs["filters"] is not None

    def test_search_without_filter(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock
        collection_mock.query.near_vector.return_value.objects = []

        search(client, _make_fake_embedding())

        call_kwargs = collection_mock.query.near_vector.call_args.kwargs
        assert call_kwargs["filters"] is None


class TestDelete:
    def test_delete_returns_true_on_success(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock

        assert delete(client, "some-uuid") is True
        collection_mock.data.delete_by_id.assert_called_once_with("some-uuid")

    def test_delete_returns_false_on_error(self):
        client = _make_mock_client()
        collection_mock = MagicMock()
        client.collections.get.return_value = collection_mock
        collection_mock.data.delete_by_id.side_effect = Exception("not found")

        assert delete(client, "bad-uuid") is False
