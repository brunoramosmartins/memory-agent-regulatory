"""Query embedding generation for semantic retrieval."""

from src.embeddings.embedding_generator import DEFAULT_MODEL, get_embedding_model


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
) -> list[float]:
    """
    Generate embedding for a single query.

    Uses the same model as corpus indexing (BGE-M3) for embedding space consistency.
    Deterministic for identical input.
    """
    model = get_embedding_model(model_name)
    vector = model.encode([query], normalize_embeddings=normalize)[0]
    return vector.tolist()
