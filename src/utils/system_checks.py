"""Dependency health checks for Weaviate and Ollama."""

import logging

logger = logging.getLogger(__name__)


def is_weaviate_ready(host: str = "localhost", port: int = 8080) -> bool:
    """Check if Weaviate is reachable."""
    try:
        from src.vectorstore.weaviate_client import is_weaviate_ready as _check

        return _check(host=host, port=port)
    except Exception as e:
        logger.debug("Weaviate not ready: %s", e)
        return False


def is_ollama_ready(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is reachable."""
    try:
        import urllib.request

        req = urllib.request.Request(host, method="GET")
        with urllib.request.urlopen(req, timeout=2) as _:
            return True
    except Exception as e:
        logger.debug("Ollama not ready: %s", e)
        return False


def check_evaluation_dependencies() -> tuple[bool, str]:
    """
    Check if dependencies for evaluation are ready.

    Returns (ready, message). For retrieval-only evaluation, Weaviate is sufficient.
    """
    if not is_weaviate_ready():
        return (
            False,
            "Weaviate is not running. Run: docker compose up -d",
        )
    return True, "OK"


def check_rag_dependencies() -> tuple[bool, str]:
    """
    Check if dependencies for RAG pipeline are ready.

    Returns (ready, message). Requires Weaviate and Ollama.
    """
    if not is_weaviate_ready():
        return (
            False,
            "Weaviate is not running. Run: docker compose up -d",
        )
    if not is_ollama_ready():
        return False, "Ollama is not running. Run: ollama serve"
    return True, "OK"
