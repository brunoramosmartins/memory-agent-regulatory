"""Unit tests for the refactored Context Builder."""

from dataclasses import dataclass

from src.memory.manager import MemoryContext
from src.memory.procedural import ProceduralPattern
from src.memory.semantic import SemanticResult
from src.rag.context_builder import (
    _estimate_tokens,
    build_context,
)


@dataclass
class FakeTurn:
    role: str
    content: str


@dataclass
class FakeRetrievalResult:
    text: str
    chunk_id: str
    document_id: str
    page_number: int
    section_title: str | None
    similarity_score: float | None
    source_file: str | None = None


class TestBuildContextEmpty:
    def test_empty_everything(self):
        result = build_context()
        assert result == ""

    def test_empty_memory_no_chunks(self):
        ctx = MemoryContext()
        result = build_context(memory_context=ctx)
        assert result == ""


class TestBuildContextWithMemory:
    def test_history_only(self):
        ctx = MemoryContext(
            history=[FakeTurn("user", "What are fees?"), FakeTurn("assistant", "No fees.")]
        )
        result = build_context(memory_context=ctx)

        assert "### Conversation History" in result
        assert "User: What are fees?" in result
        assert "Assistant: No fees." in result

    def test_summary_included(self):
        ctx = MemoryContext(summary="User discussed regulatory deadlines.")
        result = build_context(memory_context=ctx)

        assert "### Session Summary" in result
        assert "User discussed regulatory deadlines." in result

    def test_semantic_results(self):
        ctx = MemoryContext(
            semantic_results=[
                SemanticResult("id1", "PIX is free", "agent", "t1", 0.9),
                SemanticResult("id2", "Fees apply for business", "agent", "t1", 0.8),
            ]
        )
        result = build_context(memory_context=ctx)

        assert "### Related Knowledge" in result
        assert "PIX is free" in result
        assert "score: 0.90" in result

    def test_procedural_patterns(self):
        ctx = MemoryContext(
            patterns=[
                ProceduralPattern("id1", "asks about fees", "check fee doc", 0.85),
            ]
        )
        result = build_context(memory_context=ctx)

        assert "### Workflow Patterns" in result
        assert "asks about fees" in result
        assert "check fee doc" in result

    def test_all_sections_present(self):
        ctx = MemoryContext(
            history=[FakeTurn("user", "Hello")],
            semantic_results=[SemanticResult("id1", "Fact", "src", "t1", 0.9)],
            patterns=[ProceduralPattern("id1", "trigger", "action", 0.8)],
            summary="Summary text",
        )
        result = build_context(memory_context=ctx)

        assert "### Session Summary" in result
        assert "### Conversation History" in result
        assert "### Related Knowledge" in result
        assert "### Workflow Patterns" in result


class TestBuildContextWithRetrieval:
    def test_chunks_formatted(self):
        chunks = [
            FakeRetrievalResult("Regulation text", "c1", "DOC-001", 5, None, 0.9),
        ]
        result = build_context(chunks=chunks)

        assert "### Retrieved Documents" in result
        assert "[DOC-001, p. 5]" in result
        assert "Regulation text" in result

    def test_memory_and_retrieval_combined(self):
        ctx = MemoryContext(history=[FakeTurn("user", "Question")])
        chunks = [FakeRetrievalResult("Answer text", "c1", "DOC", 1, None, 0.9)]

        result = build_context(chunks=chunks, memory_context=ctx)

        assert "### Conversation History" in result
        assert "### Retrieved Documents" in result


class TestBuildContextTokenBudget:
    def test_budget_drops_low_priority(self):
        ctx = MemoryContext(
            summary="Short summary",
            history=[FakeTurn("user", "Q" * 500)],  # large history
        )
        chunks = [FakeRetrievalResult("X" * 500, "c1", "DOC", 1, None, 0.9)]

        # Very small budget: should keep summary, maybe history, drop retrieval
        result = build_context(chunks=chunks, memory_context=ctx, max_tokens=50)

        assert "### Session Summary" in result
        # Retrieval is lowest priority and likely dropped
        assert "### Retrieved Documents" not in result

    def test_no_budget_includes_all(self):
        ctx = MemoryContext(summary="Summary")
        chunks = [FakeRetrievalResult("Text", "c1", "DOC", 1, None, 0.9)]

        result = build_context(chunks=chunks, memory_context=ctx, max_tokens=None)

        assert "### Session Summary" in result
        assert "### Retrieved Documents" in result


class TestEstimateTokens:
    def test_basic_estimate(self):
        assert _estimate_tokens("hello world") == max(1, len("hello world") // 4)

    def test_empty_string(self):
        assert _estimate_tokens("") == 1  # min 1
