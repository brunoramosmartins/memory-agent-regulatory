"""Tool definitions and router for the agent."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    name: str
    success: bool
    output: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------

_ALLOWED_CALC_CHARS = re.compile(r"^[\d\s\+\-\*/\.\(\)]+$")


def _calculate(expression: str) -> ToolResult:
    """Evaluate a safe mathematical expression.

    Only allows digits, basic operators, parentheses, and decimals.
    No eval() on arbitrary code.
    """
    expression = expression.strip()
    if not expression:
        return ToolResult(name="calculate", success=False, output="", error="Empty expression")

    if not _ALLOWED_CALC_CHARS.match(expression):
        return ToolResult(
            name="calculate",
            success=False,
            output="",
            error=f"Unsafe characters in expression: {expression}",
        )

    try:
        # Use compile + eval with restricted builtins for safety
        code = compile(expression, "<calc>", "eval")
        allowed_names = {"__builtins__": {}}
        result = eval(code, allowed_names)  # noqa: S307
        if isinstance(result, int | float) and not math.isnan(result) and not math.isinf(result):
            return ToolResult(name="calculate", success=True, output=str(result))
        return ToolResult(
            name="calculate", success=False, output="", error="Result is not a finite number"
        )
    except Exception as e:
        return ToolResult(name="calculate", success=False, output="", error=str(e))


# ---------------------------------------------------------------------------
# Search documents (placeholder — wraps retrieval in Phase 4+)
# ---------------------------------------------------------------------------


def _search_documents(query: str) -> ToolResult:
    """Search regulatory documents.

    This is a placeholder that will be connected to the retrieval
    pipeline in later phases. For now it returns a structured message.
    """
    logger.info("search_documents called with query=%s", query)
    return ToolResult(
        name="search_documents",
        success=True,
        output=f"Search results for: {query} (retrieval pipeline not yet connected)",
    )


# ---------------------------------------------------------------------------
# Tool router
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, Any] = {
    "calculate": _calculate,
    "search_documents": _search_documents,
}

AVAILABLE_TOOLS = list(_TOOL_REGISTRY.keys())


def execute_tool(name: str, args: dict[str, Any] | None = None) -> ToolResult:
    """Route a tool request to the appropriate implementation.

    Args:
        name: Tool name (must be in AVAILABLE_TOOLS).
        args: Tool arguments as a dict.

    Returns:
        ToolResult with success/failure and output.
    """
    if name not in _TOOL_REGISTRY:
        return ToolResult(
            name=name,
            success=False,
            output="",
            error=f"Unknown tool: {name}. Available: {AVAILABLE_TOOLS}",
        )

    func = _TOOL_REGISTRY[name]
    args = args or {}

    try:
        if name == "calculate":
            return func(args.get("expression", ""))
        elif name == "search_documents":
            return func(args.get("query", ""))
        else:
            return ToolResult(name=name, success=False, output="", error="Unhandled tool")
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return ToolResult(name=name, success=False, output="", error=str(e))
