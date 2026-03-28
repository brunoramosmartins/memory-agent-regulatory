"""Document alias resolver — maps filenames/IDs to human-readable names."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_ALIASES: dict[str, str] | None = None
_ALIASES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "document_aliases.yaml"


def _load_aliases() -> dict[str, str]:
    global _ALIASES
    if _ALIASES is not None:
        return _ALIASES

    _ALIASES = {}
    if _ALIASES_PATH.exists():
        with open(_ALIASES_PATH) as f:
            _ALIASES = yaml.safe_load(f) or {}
        logger.debug("Loaded %d document aliases", len(_ALIASES))
    else:
        logger.warning("Document aliases file not found: %s", _ALIASES_PATH)
    return _ALIASES


def get_document_alias(source_file: str | None, document_id: str | None = None) -> str:
    """Resolve a human-readable name for a document.

    Tries matching by filename from source_file path, then falls back to document_id.
    """
    aliases = _load_aliases()

    # Try source_file filename match
    if source_file:
        filename = Path(source_file).name
        if filename in aliases:
            return aliases[filename]
        # Try stem match
        stem = Path(source_file).stem
        if stem in aliases:
            return aliases[stem]

    # Try document_id match
    if document_id and document_id in aliases:
        return aliases[document_id]

    # Fallback: use filename or document_id as-is
    if source_file:
        return Path(source_file).stem.replace("_", " ").title()
    if document_id:
        return document_id
    return "Documento desconhecido"
