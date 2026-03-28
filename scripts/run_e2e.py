"""End-to-end test script — full pipeline with Phoenix observability.

Runs the agent with Ollama + memory + retrieval and traces everything in Phoenix.

Prerequisites:
    - Docker running (PostgreSQL + Weaviate)
    - Ollama running with model pulled
    - Documents ingested (python scripts/run_ingestion.py)
    - Migrations applied (alembic -c migrations/alembic.ini upgrade head)
    - Phoenix running in a separate terminal: python -m phoenix.server.main serve

Usage:
    python scripts/run_e2e.py                          # traces to Phoenix at localhost:6006
    python scripts/run_e2e.py --no-phoenix             # skip tracing
    python scripts/run_e2e.py --query "What are PIX fees?"
    python scripts/run_e2e.py --phoenix-url http://localhost:6006
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("e2e")


def _setup_phoenix(phoenix_url: str = "http://localhost:6006") -> bool:
    """Connect OpenTelemetry tracing to an external Phoenix instance.

    Phoenix must already be running in a separate terminal:
        python -m phoenix.server.main serve

    Returns True if OTEL exporter was successfully registered.
    """
    try:
        from opentelemetry import trace
        from phoenix.otel import register

        tracer_provider = register(endpoint=f"{phoenix_url}/v1/traces")
        trace.set_tracer_provider(tracer_provider)

        logger.info("OpenTelemetry tracing connected to Phoenix at %s", phoenix_url)
        return True
    except ImportError as e:
        logger.warning("Phoenix OTEL not available: %s", e)
        logger.warning("Install with: pip install arize-phoenix openinference-instrumentation")
        return False
    except Exception as e:
        logger.warning("Phoenix tracing setup failed: %s", e)
        return False


def _build_full_deps() -> dict:
    """Build dependency dict with all services connected."""
    deps = {}

    # 1. Ollama LLM
    try:
        import ollama

        from src.config.settings import get_settings

        settings = get_settings()
        model_name = settings.llm.model

        def llm_fn(messages: list[dict]) -> str:
            resp = ollama.chat(model=model_name, messages=messages)
            return resp["message"]["content"]

        deps["llm_fn"] = llm_fn
        logger.info("LLM: Ollama (%s)", model_name)
    except Exception as e:
        logger.error("Ollama not available: %s", e)
        sys.exit(1)

    # 2. Embedding function
    try:
        from src.embeddings.embedding_generator import get_embedding_model

        embed_model = get_embedding_model()

        def embed_fn(text: str) -> list[float]:
            return embed_model.encode(text).tolist()

        deps["embed_fn"] = embed_fn
        logger.info("Embeddings: BGE-M3 loaded")
    except Exception as e:
        logger.warning("Embedding model not available: %s", e)

    # 3. Memory Manager (PostgreSQL + Weaviate)
    try:
        from src.memory.database import get_session
        from src.memory.manager import MemoryManager
        from src.memory.procedural import init_procedural_collection
        from src.memory.semantic import init_semantic_collection
        from src.vectorstore.weaviate_client import get_weaviate_client

        db_sess = get_session()
        wc = get_weaviate_client()

        # Ensure memory collections exist in Weaviate
        init_semantic_collection(wc)
        init_procedural_collection(wc)

        mm = MemoryManager(db_session=db_sess, weaviate_client=wc)
        deps["memory_manager"] = mm
        deps["_db_session"] = db_sess  # keep ref for cleanup
        logger.info("Memory: PostgreSQL + Weaviate connected")
    except Exception as e:
        logger.warning("Memory services not available: %s -- continuing without memory", e)

    # 4. Context builder with retrieval
    try:
        from src.rag.context_builder import build_context
        from src.retrieval.retriever import retrieve

        def build_context_fn(memory_context=None, **kwargs) -> str:
            # Retrieve relevant chunks from Weaviate
            query = kwargs.get("query", "")
            chunks = []
            if query:
                try:
                    results = retrieve(query, top_k=5)
                    chunks = [
                        {"text": r.text, "source": r.source_file or r.document_id}
                        for r in results
                    ]
                    logger.info("Retrieved %d chunks for context", len(chunks))
                except Exception as e:
                    logger.warning("Retrieval failed: %s", e)

            return build_context(chunks=chunks, memory_context=memory_context)

        deps["build_context_fn"] = build_context_fn
        logger.info("Context builder: retrieval + memory merge")
    except Exception as e:
        logger.warning("Context builder not available: %s", e)
        deps["build_context_fn"] = lambda **kw: ""

    return deps


def _run_conversation(deps: dict, queries: list[str], thread_id: str) -> None:
    """Run a multi-turn conversation through the agent."""
    from src.agent.graph import run_agent
    from src.observability.tracing import span_set_input, span_set_output, trace_span

    for i, query in enumerate(queries, 1):
        logger.info("\n--- Turn %d ---", i)
        logger.info("User: %s", query)

        with trace_span(
            f"agent_turn_{i}",
            attributes={"turn": i, "thread_id": thread_id},
            openinference_span_kind="CHAIN",
        ) as span:
            if span and span.is_recording():
                span_set_input(span, query)

            response = run_agent(
                query=query,
                thread_id=thread_id,
                deps=deps,
            )

            if span and span.is_recording():
                span_set_output(span, response or "")

        logger.info("Agent: %s", response or "(empty response)")
        print(f"\nUser:  {query}")
        print(f"Agent: {response or '(empty response)'}\n")


def _cleanup(deps: dict) -> None:
    """Close all connections gracefully."""
    # Close Weaviate client
    try:
        from src.vectorstore.weaviate_client import close_weaviate_client

        close_weaviate_client()
        logger.info("Weaviate connection closed")
    except Exception as e:
        logger.debug("Weaviate cleanup: %s", e)

    # Close PostgreSQL session + dispose engine pool
    try:
        if session := deps.get("_db_session"):
            session.close()
        from src.memory.database import reset_engine

        reset_engine()
        logger.info("PostgreSQL connections closed")
    except Exception as e:
        logger.debug("PostgreSQL cleanup: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end test with Phoenix tracing")
    parser.add_argument(
        "--no-phoenix",
        action="store_true",
        help="Skip Phoenix tracing (Phoenix must be running separately otherwise)",
    )
    parser.add_argument(
        "--phoenix-url",
        type=str,
        default="http://localhost:6006",
        help="URL of external Phoenix instance (default: http://localhost:6006)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run (default: multi-turn demo)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="e2e-demo",
        help="Thread ID for conversation",
    )
    args = parser.parse_args()

    # Phoenix setup (connects to external instance)
    phoenix_active = False
    if not args.no_phoenix:
        phoenix_active = _setup_phoenix(args.phoenix_url)

    # Build dependencies
    logger.info("\n=== Building dependencies ===")
    deps = _build_full_deps()

    try:
        # Define queries
        if args.query:
            queries = [args.query]
        else:
            queries = [
                "Quais são os limites de transferência do PIX?",
                "E para pessoa jurídica, os limites são diferentes?",
                "Como funciona o Mecanismo Especial de Devolução (MED)?",
                "Qual o prazo para contestar uma transação PIX?",
            ]

        # Run conversation
        logger.info("\n=== Running conversation (%d turns) ===", len(queries))
        _run_conversation(deps, queries, thread_id=args.thread_id)

        # Summary
        print("\n" + "=" * 60)
        print("E2E test complete!")
        if phoenix_active:
            print(f"Phoenix dashboard: {args.phoenix_url}")
            print("Traces are available in your external Phoenix instance.")
        print("=" * 60)
    finally:
        _cleanup(deps)


if __name__ == "__main__":
    main()
