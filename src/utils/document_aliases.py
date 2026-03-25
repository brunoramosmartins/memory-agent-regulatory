"""Document alias lookup — maps raw document_id to human-readable display names."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ALIASES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "document_aliases.yaml"
_ALIASES: dict[str, str] | None = None


def _load_aliases() -> dict[str, str]:
    """Load aliases from YAML config file. Returns empty dict on failure."""
    try:
        import yaml

        with open(_ALIASES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        logger.debug("document_aliases.yaml not found at %s", _ALIASES_PATH)
        return {}
    except Exception as e:
        logger.warning("Failed to load document aliases: %s", e)
        return {}


def get_document_alias(document_id: str) -> str:
    """
    Return display alias for a document_id.

    Falls back to document_id itself when no alias is registered.
    """
    global _ALIASES
    if _ALIASES is None:
        _ALIASES = _load_aliases()
    return _ALIASES.get(document_id, document_id)
