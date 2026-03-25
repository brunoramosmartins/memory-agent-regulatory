"""Shared tokenizer utilities for token counting across the project."""

from transformers import AutoTokenizer

DEFAULT_MODEL = "BAAI/bge-m3"

_tokenizer_cache: dict[str, AutoTokenizer] = {}


def get_tokenizer(model_name: str = DEFAULT_MODEL) -> AutoTokenizer:
    """Return cached tokenizer instance."""
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def count_tokens(text: str, model_name: str = DEFAULT_MODEL) -> int:
    """Count tokens in text using the actual tokenizer."""
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))
