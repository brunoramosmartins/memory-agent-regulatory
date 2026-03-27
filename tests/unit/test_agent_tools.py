"""Unit tests for agent tools."""

from src.agent.tools import AVAILABLE_TOOLS, ToolResult, execute_tool


class TestCalculateTool:
    def test_basic_addition(self):
        result = execute_tool("calculate", {"expression": "2 + 3"})
        assert result.success is True
        assert result.output == "5"

    def test_complex_expression(self):
        result = execute_tool("calculate", {"expression": "(10 + 5) * 2"})
        assert result.success is True
        assert result.output == "30"

    def test_float_result(self):
        result = execute_tool("calculate", {"expression": "7 / 2"})
        assert result.success is True
        assert result.output == "3.5"

    def test_empty_expression(self):
        result = execute_tool("calculate", {"expression": ""})
        assert result.success is False
        assert result.error is not None

    def test_unsafe_characters_rejected(self):
        result = execute_tool("calculate", {"expression": "__import__('os')"})
        assert result.success is False

    def test_division_by_zero(self):
        result = execute_tool("calculate", {"expression": "1 / 0"})
        assert result.success is False


class TestSearchDocumentsTool:
    def test_returns_placeholder(self):
        result = execute_tool("search_documents", {"query": "PIX fees"})
        assert result.success is True
        assert "PIX fees" in result.output


class TestToolRouter:
    def test_available_tools(self):
        assert "calculate" in AVAILABLE_TOOLS
        assert "search_documents" in AVAILABLE_TOOLS

    def test_unknown_tool(self):
        result = execute_tool("nonexistent", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_missing_args_defaults(self):
        result = execute_tool("calculate")
        assert result.success is False  # empty expression

    def test_result_is_tool_result(self):
        result = execute_tool("calculate", {"expression": "1"})
        assert isinstance(result, ToolResult)
        assert result.name == "calculate"
