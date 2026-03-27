"""Node implementations for the LangGraph agent loop."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.agent.state import AgentState
from src.agent.tools import ToolResult, execute_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a regulatory Q&A assistant with access to memory from previous \
conversations and a set of tools. Answer the user's question using the \
provided context. If you need to perform a calculation or search for \
documents, request a tool call.

To call a tool, respond EXACTLY in this JSON format (no other text):
{{"tool": "<tool_name>", "args": {{<arguments>}}}}

Available tools:
- calculate: Evaluate a math expression. Args: {{"expression": "2 + 2"}}
- search_documents: Search regulatory documents. Args: {{"query": "your query"}}

If you do NOT need a tool, respond with your answer in plain text.

{context}"""


# ---------------------------------------------------------------------------
# Helper: parse tool request from LLM output
# ---------------------------------------------------------------------------

_TOOL_JSON_RE = re.compile(r'\{\s*"tool"\s*:', re.DOTALL)


def _parse_tool_request(text: str) -> dict[str, Any] | None:
    """Try to extract a tool-call JSON from the LLM response.

    Returns dict with 'tool' and 'args' keys, or None.
    """
    text = text.strip()
    match = _TOOL_JSON_RE.search(text)
    if not match:
        return None

    json_str = text[match.start() :]
    # Find the closing brace (simple greedy approach)
    depth = 0
    end = 0
    for i, ch in enumerate(json_str):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == 0:
        return None

    try:
        data = json.loads(json_str[:end])
        if "tool" in data:
            return {"tool": data["tool"], "args": data.get("args", {})}
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# Node: retrieve_memory
# ---------------------------------------------------------------------------


def retrieve_memory(state: AgentState, **deps: Any) -> AgentState:
    """Load memory context for the current thread.

    Dependencies (passed via deps):
        memory_manager: MemoryManager instance
        embed_fn: callable(str) -> list[float]
    """
    memory_manager = deps.get("memory_manager")
    embed_fn = deps.get("embed_fn")

    if memory_manager is None or embed_fn is None:
        logger.warning("retrieve_memory: missing dependencies, skipping")
        return state

    query_embedding = embed_fn(state.query)
    state.memory_context = memory_manager.read_context(state.thread_id, query_embedding)

    logger.info(
        "retrieve_memory thread=%s has_context=%s",
        state.thread_id,
        state.memory_context is not None,
    )
    return state


# ---------------------------------------------------------------------------
# Node: reason
# ---------------------------------------------------------------------------


def reason(state: AgentState, **deps: Any) -> AgentState:
    """Call the LLM to reason about the query and decide on an action.

    Dependencies (passed via deps):
        llm_fn: callable(messages: list[dict]) -> str
        build_context_fn: callable(memory_context, ...) -> str
    """
    llm_fn = deps.get("llm_fn")
    build_context_fn = deps.get("build_context_fn")

    if llm_fn is None:
        logger.error("reason: llm_fn dependency missing")
        state.response = "Error: LLM not configured."
        return state

    # Build context string from memory
    context_str = ""
    if build_context_fn and state.memory_context:
        context_str = build_context_fn(memory_context=state.memory_context)

    system_msg = SYSTEM_PROMPT.format(context=context_str)

    # Assemble messages for the LLM
    messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]

    # Add conversation history from this turn
    for msg in state.messages:
        messages.append(msg)

    # Add tool results if any
    for tr in state.tool_results:
        messages.append(
            {
                "role": "system",
                "content": f"Tool '{tr['name']}' returned: {tr['output']}",
            }
        )

    # Add the current query
    messages.append({"role": "user", "content": state.query})

    # Call LLM
    llm_output = llm_fn(messages)
    state.iteration_count += 1

    # Check if LLM wants to call a tool
    tool_req = _parse_tool_request(llm_output)
    if tool_req:
        state.tool_request = tool_req
        state.response = ""
        logger.info(
            "reason: tool_request=%s iteration=%d",
            tool_req["tool"],
            state.iteration_count,
        )
    else:
        state.tool_request = None
        state.response = llm_output
        logger.info("reason: final answer iteration=%d", state.iteration_count)

    return state


# ---------------------------------------------------------------------------
# Node: tool_call
# ---------------------------------------------------------------------------


def tool_call(state: AgentState, **deps: Any) -> AgentState:
    """Execute the requested tool and store the result."""
    if state.tool_request is None:
        logger.warning("tool_call: no tool_request in state")
        return state

    name = state.tool_request["tool"]
    args = state.tool_request.get("args", {})

    logger.info("tool_call: executing %s with args=%s", name, args)

    result: ToolResult = execute_tool(name, args)

    state.tool_results.append(
        {
            "name": result.name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
        }
    )

    # Clear the tool request so reason can re-evaluate
    state.tool_request = None

    return state


# ---------------------------------------------------------------------------
# Node: write_memory
# ---------------------------------------------------------------------------


def write_memory(state: AgentState, **deps: Any) -> AgentState:
    """Persist the conversation turn to memory.

    Dependencies (passed via deps):
        memory_manager: MemoryManager instance
    """
    memory_manager = deps.get("memory_manager")

    if memory_manager is None:
        logger.warning("write_memory: memory_manager not available, skipping")
        return state

    # Write user turn
    memory_manager.write_turn(
        thread_id=state.thread_id,
        role="user",
        content=state.query,
    )

    # Write assistant turn
    if state.response:
        memory_manager.write_turn(
            thread_id=state.thread_id,
            role="assistant",
            content=state.response,
        )

    logger.info("write_memory: persisted turns for thread=%s", state.thread_id)
    return state


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def should_use_tool(state: AgentState) -> str:
    """Conditional edge: route to tool_call or write_memory.

    Returns:
        "tool_call" if a tool was requested and iterations are within budget.
        "write_memory" otherwise.
    """
    if state.tool_request is not None and state.iteration_count < state.max_iterations:
        return "tool_call"
    return "write_memory"
