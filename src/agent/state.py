"""AgentState — typed state definition for the LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.memory.manager import MemoryContext


@dataclass
class AgentState:
    """State that flows through the LangGraph agent loop.

    Each node reads and updates this state. The graph passes it
    between nodes automatically.
    """

    # Current user query
    query: str = ""

    # Conversation thread identifier
    thread_id: str = ""

    # Message history for this turn (role, content pairs)
    messages: list[dict[str, str]] = field(default_factory=list)

    # Aggregated memory context from MemoryManager.read_context()
    memory_context: MemoryContext | None = None

    # Results from tool calls in this turn
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    # LLM response text (set by reason node)
    response: str = ""

    # Whether the LLM requested a tool call
    tool_request: dict[str, Any] | None = None

    # Number of reasoning iterations in this turn
    iteration_count: int = 0

    # Maximum allowed iterations before forced stop
    max_iterations: int = 5

    # Metadata for logging and tracing
    metadata: dict[str, Any] = field(default_factory=dict)
