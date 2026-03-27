"""Unit tests for agent node implementations."""

from src.agent.nodes import (
    _parse_tool_request,
    reason,
    retrieve_memory,
    should_use_tool,
    tool_call,
    write_memory,
)
from src.agent.state import AgentState
from src.memory.manager import MemoryContext

# ---------------------------------------------------------------------------
# _parse_tool_request
# ---------------------------------------------------------------------------


class TestParseToolRequest:
    def test_valid_tool_json(self):
        text = '{"tool": "calculate", "args": {"expression": "2+2"}}'
        result = _parse_tool_request(text)
        assert result is not None
        assert result["tool"] == "calculate"
        assert result["args"]["expression"] == "2+2"

    def test_no_tool_json(self):
        assert _parse_tool_request("The answer is 42.") is None

    def test_tool_json_embedded_in_text(self):
        text = 'Let me calculate: {"tool": "calculate", "args": {"expression": "5*3"}}'
        result = _parse_tool_request(text)
        assert result is not None
        assert result["tool"] == "calculate"

    def test_malformed_json(self):
        assert _parse_tool_request('{"tool": "calc", "args":') is None

    def test_missing_tool_key(self):
        assert _parse_tool_request('{"name": "calc"}') is None

    def test_empty_args(self):
        text = '{"tool": "search_documents"}'
        result = _parse_tool_request(text)
        assert result is not None
        assert result["args"] == {}


# ---------------------------------------------------------------------------
# retrieve_memory node
# ---------------------------------------------------------------------------


class TestRetrieveMemoryNode:
    def test_skips_without_deps(self):
        state = AgentState(query="test", thread_id="t1")
        result = retrieve_memory(state)
        assert result.memory_context is None

    def test_calls_memory_manager(self):
        mock_ctx = MemoryContext(summary="remembered")
        calls = []

        class FakeManager:
            def read_context(self, thread_id, embedding):
                calls.append((thread_id, embedding))
                return mock_ctx

        state = AgentState(query="test", thread_id="t1")
        result = retrieve_memory(
            state,
            memory_manager=FakeManager(),
            embed_fn=lambda q: [0.1, 0.2],
        )

        assert result.memory_context is mock_ctx
        assert len(calls) == 1
        assert calls[0][0] == "t1"


# ---------------------------------------------------------------------------
# reason node
# ---------------------------------------------------------------------------


class TestReasonNode:
    def test_plain_answer(self):
        state = AgentState(query="What is 2+2?", thread_id="t1")
        result = reason(state, llm_fn=lambda msgs: "The answer is 4.")
        assert result.response == "The answer is 4."
        assert result.tool_request is None
        assert result.iteration_count == 1

    def test_tool_request_parsed(self):
        llm_output = '{"tool": "calculate", "args": {"expression": "2+2"}}'
        state = AgentState(query="Calculate 2+2", thread_id="t1")
        result = reason(state, llm_fn=lambda msgs: llm_output)
        assert result.tool_request is not None
        assert result.tool_request["tool"] == "calculate"
        assert result.response == ""

    def test_error_without_llm(self):
        state = AgentState(query="test")
        result = reason(state)
        assert "Error" in result.response

    def test_iteration_increments(self):
        state = AgentState(query="test", iteration_count=2)
        reason(state, llm_fn=lambda msgs: "ok")
        assert state.iteration_count == 3


# ---------------------------------------------------------------------------
# tool_call node
# ---------------------------------------------------------------------------


class TestToolCallNode:
    def test_executes_calculate(self):
        state = AgentState(
            query="test",
            tool_request={"tool": "calculate", "args": {"expression": "3+4"}},
        )
        result = tool_call(state)
        assert len(result.tool_results) == 1
        assert result.tool_results[0]["output"] == "7"
        assert result.tool_results[0]["success"] is True
        assert result.tool_request is None  # cleared

    def test_no_tool_request(self):
        state = AgentState(query="test")
        result = tool_call(state)
        assert len(result.tool_results) == 0


# ---------------------------------------------------------------------------
# write_memory node
# ---------------------------------------------------------------------------


class TestWriteMemoryNode:
    def test_skips_without_manager(self):
        state = AgentState(query="test", response="answer")
        result = write_memory(state)
        assert result.response == "answer"

    def test_writes_both_turns(self):
        written = []

        class FakeManager:
            def write_turn(self, thread_id, role, content, metadata=None):
                written.append((thread_id, role, content))

        state = AgentState(query="question", thread_id="t1", response="answer")
        write_memory(state, memory_manager=FakeManager())
        assert len(written) == 2
        assert written[0] == ("t1", "user", "question")
        assert written[1] == ("t1", "assistant", "answer")


# ---------------------------------------------------------------------------
# should_use_tool routing
# ---------------------------------------------------------------------------


class TestShouldUseTool:
    def test_routes_to_tool(self):
        state = AgentState(
            tool_request={"tool": "calculate", "args": {}},
            iteration_count=1,
            max_iterations=5,
        )
        assert should_use_tool(state) == "tool_call"

    def test_routes_to_write_no_request(self):
        state = AgentState(tool_request=None)
        assert should_use_tool(state) == "write_memory"

    def test_routes_to_write_max_iterations(self):
        state = AgentState(
            tool_request={"tool": "calculate", "args": {}},
            iteration_count=5,
            max_iterations=5,
        )
        assert should_use_tool(state) == "write_memory"
