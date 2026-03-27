"""Unit tests for agent graph construction and execution."""

from src.agent.graph import build_graph, run_agent


class TestBuildGraph:
    def test_graph_compiles(self):
        compiled = build_graph()
        assert compiled is not None

    def test_graph_with_deps(self):
        deps = {"llm_fn": lambda msgs: "test"}
        compiled = build_graph(deps)
        assert compiled is not None


class TestRunAgent:
    def test_simple_query(self):
        """End-to-end with mocked LLM — no memory, no tools."""
        deps = {
            "llm_fn": lambda msgs: "The answer is 42.",
            "build_context_fn": lambda **kw: "",
        }
        response = run_agent("What is the meaning?", thread_id="test-1", deps=deps)
        assert response == "The answer is 42."

    def test_auto_generates_thread_id(self):
        deps = {"llm_fn": lambda msgs: "ok"}
        response = run_agent("hi", deps=deps)
        assert response == "ok"

    def test_tool_call_loop(self):
        """LLM requests a tool, then answers after seeing the result."""
        call_count = [0]

        def fake_llm(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"tool": "calculate", "args": {"expression": "10 * 5"}}'
            return "The result is 50."

        deps = {"llm_fn": fake_llm, "build_context_fn": lambda **kw: ""}
        response = run_agent("What is 10 times 5?", thread_id="t1", deps=deps)
        assert "50" in response
        assert call_count[0] == 2

    def test_max_iterations_stops_loop(self):
        """LLM always requests tools — should stop at max_iterations."""
        def always_tool(msgs):
            return '{"tool": "calculate", "args": {"expression": "1+1"}}'

        deps = {"llm_fn": always_tool, "build_context_fn": lambda **kw: ""}
        # run_agent uses default max_iterations=5
        response = run_agent("loop forever", thread_id="t2", deps=deps)
        # Should terminate (not hang) — response will be empty since LLM never gave plain text
        assert isinstance(response, str)
