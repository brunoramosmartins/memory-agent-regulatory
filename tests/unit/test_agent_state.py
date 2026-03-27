"""Unit tests for AgentState."""

from src.agent.state import AgentState
from src.memory.manager import MemoryContext


class TestAgentStateDefaults:
    def test_default_values(self):
        state = AgentState()
        assert state.query == ""
        assert state.thread_id == ""
        assert state.messages == []
        assert state.memory_context is None
        assert state.tool_results == []
        assert state.response == ""
        assert state.tool_request is None
        assert state.iteration_count == 0
        assert state.max_iterations == 5
        assert state.metadata == {}

    def test_custom_values(self):
        state = AgentState(query="What is PIX?", thread_id="t1", max_iterations=10)
        assert state.query == "What is PIX?"
        assert state.thread_id == "t1"
        assert state.max_iterations == 10

    def test_memory_context_assignment(self):
        ctx = MemoryContext(summary="Test summary")
        state = AgentState(memory_context=ctx)
        assert state.memory_context is not None
        assert state.memory_context.summary == "Test summary"

    def test_mutable_defaults_independent(self):
        s1 = AgentState()
        s2 = AgentState()
        s1.messages.append({"role": "user", "content": "hello"})
        assert len(s2.messages) == 0
