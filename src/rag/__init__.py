"""RAG module - Context building and prompt generation."""

from .context_builder import build_context
from .prompt_template import build_prompt

__all__ = [
    "build_context",
    "build_prompt",
]
