# Memory-Aware Regulatory Agent

Memory-aware Q&A agent for regulatory documentation, built with LangGraph, PostgreSQL, Weaviate, and Ollama.

## Quick Start

```bash
cp .env.example .env
# Fill in POSTGRES_PASSWORD in .env
make up          # Start Weaviate + PostgreSQL
make test        # Run unit tests
```

## Stack

- **Orchestration:** LangGraph
- **LLM:** Llama (Ollama, local)
- **Vector DB:** Weaviate
- **Relational DB:** PostgreSQL
- **Observability:** OpenTelemetry + Phoenix
- **UI:** Streamlit
