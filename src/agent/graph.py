"""LangGraph state graph — compiles the agent loop."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    reason,
    retrieve_memory,
    should_use_tool,
    tool_call,
    write_memory,
)
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph(deps: dict[str, Any] | None = None) -> StateGraph:
    """Build and return the compiled agent StateGraph.

    Args:
        deps: Dependency dict injected into every node. Expected keys:
            - memory_manager: MemoryManager instance
            - embed_fn: callable(str) -> list[float]
            - llm_fn: callable(messages) -> str
            - build_context_fn: callable(memory_context=...) -> str

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    deps = deps or {}

    # Wrap nodes so they receive dependencies
    def _retrieve(state: AgentState) -> AgentState:
        return retrieve_memory(state, **deps)

    def _reason(state: AgentState) -> AgentState:
        return reason(state, **deps)

    def _tool_call(state: AgentState) -> AgentState:
        return tool_call(state, **deps)

    def _write(state: AgentState) -> AgentState:
        return write_memory(state, **deps)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("retrieve_memory", _retrieve)
    graph.add_node("reason", _reason)
    graph.add_node("tool_call", _tool_call)
    graph.add_node("write_memory", _write)

    # Edges
    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "reason")
    graph.add_conditional_edges("reason", should_use_tool, {
        "tool_call": "tool_call",
        "write_memory": "write_memory",
    })
    graph.add_edge("tool_call", "reason")
    graph.add_edge("write_memory", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_agent(
    query: str,
    thread_id: str | None = None,
    deps: dict[str, Any] | None = None,
) -> str:
    """Run the agent loop for a single query and return the response.

    Args:
        query: User question.
        thread_id: Conversation identifier (auto-generated if None).
        deps: Dependency dict for nodes.

    Returns:
        The agent's text response.
    """
    thread_id = thread_id or str(uuid.uuid4())

    initial_state = AgentState(query=query, thread_id=thread_id)

    compiled = build_graph(deps)
    final_state = compiled.invoke(initial_state)

    # LangGraph may return a dict or the dataclass depending on version
    if isinstance(final_state, dict):
        return final_state.get("response", "")
    return final_state.response
