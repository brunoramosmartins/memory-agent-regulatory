# Architecture Decision Records

This document captures key architectural decisions made during the development of the Memory-Aware Regulatory Agent.

---

## ADR-001: LangGraph for Agent Orchestration

**Status**: Accepted

**Context**: The agent needs a cyclic reasoning loop (retrieve → reason → tool → reason → write) that traditional linear chains cannot express. We need conditional routing, state management, and the ability to loop back.

**Decision**: Use LangGraph's StateGraph with typed state (`AgentState` dataclass), conditional edges, and compile-to-execute pattern.

**Consequences**:
- Clear separation of concerns: each node is an independent function
- Easy to test: nodes are pure functions that transform state
- Built-in support for conditional routing and iteration limits
- Tight coupling to LangGraph API (acceptable for a portfolio project)

---

## ADR-002: PostgreSQL for Conversational + Summary Memory

**Status**: Accepted

**Context**: Conversational memory requires ordered retrieval by thread_id and timestamp. Summary memory needs upsert semantics. Both benefit from ACID transactions and structured queries.

**Decision**: Use PostgreSQL with SQLAlchemy ORM and Alembic migrations. Store conversation turns and session summaries in relational tables.

**Alternatives considered**:
- **SQLite**: Simpler setup but no concurrent access, harder to deploy
- **Redis**: Fast but not durable by default, no complex queries
- **MongoDB**: Would work but adds unnecessary complexity for structured data

**Consequences**:
- Reliable ordered retrieval with indexes on (thread_id, timestamp)
- Schema migrations via Alembic
- Requires Docker service running for full functionality

---

## ADR-003: Weaviate for Semantic + Procedural Memory

**Status**: Accepted

**Context**: Semantic and procedural memories require vector similarity search. The system needs to store embeddings alongside metadata and retrieve by cosine similarity.

**Decision**: Use Weaviate with separate collections for semantic and procedural memory. BGE-M3 (1024 dimensions) as the embedding model.

**Alternatives considered**:
- **Pinecone**: Managed service, requires API key and internet
- **ChromaDB**: Simpler but less mature for production use
- **pgvector**: Could consolidate into PostgreSQL, but separating vector and relational concerns is cleaner for demonstration

**Consequences**:
- Clean separation of vector and relational storage
- Weaviate provides built-in hybrid search capabilities
- BGE-M3 offers strong multilingual support (relevant for Portuguese regulatory content)

---

## ADR-004: Ollama for Local LLM Inference

**Status**: Accepted

**Context**: The project must run entirely locally without API keys or internet access. We need a local LLM that supports chat-style interaction.

**Decision**: Use Ollama with Llama 3.2 (3B parameters) as the default model. Configurable via settings.

**Alternatives considered**:
- **OpenAI API**: Better quality but requires API key and costs money
- **vLLM**: More performant but heavier setup
- **llama.cpp directly**: Lower-level, Ollama provides a cleaner API

**Consequences**:
- Zero cost, fully local, no API keys
- Model quality limited by hardware (3B model on consumer hardware)
- Easy to swap models via configuration

---

## ADR-005: MemoryManager Facade Pattern

**Status**: Accepted

**Context**: Four memory types (conversational, semantic, procedural, summary) each have their own storage backend and access patterns. The agent nodes should not need to know about individual memory systems.

**Decision**: Implement a `MemoryManager` class that provides a unified interface: `read_context()` returns all memory types aggregated into a `MemoryContext`, and `write_turn()` handles persistence and auto-summarization.

**Consequences**:
- Agent nodes interact with a single interface
- Easy to add new memory types without changing agent code
- Auto-summarization logic centralized in one place
- Testing is simplified: mock one object instead of four

---

## ADR-006: Character-Based Token Estimation

**Status**: Accepted

**Context**: The Context Builder needs to fit content within a token budget. Using a real tokenizer (tiktoken, sentencepiece) would add a heavy dependency.

**Decision**: Use a simple heuristic: 4 characters per token. This is a reasonable approximation for English/Portuguese text.

**Alternatives considered**:
- **tiktoken**: Accurate but adds dependency and is model-specific
- **sentencepiece**: Accurate but heavy and slow to load

**Consequences**:
- Lightweight, no additional dependencies
- Approximately 10-15% error margin (acceptable for budget allocation)
- Can be swapped for a real tokenizer in production

---

## ADR-007: Synthetic Sessions for Evaluation

**Status**: Accepted

**Context**: We need multi-turn conversation data to evaluate the memory system. Real user data is not available for a portfolio project.

**Decision**: Build a session generator that creates synthetic multi-turn conversations from topic templates. Sessions include follow-up questions, context references, and topic switches that test memory capabilities.

**Consequences**:
- Reproducible evaluation (seeded generation)
- Covers key memory scenarios by design
- Results show relative performance, not absolute quality
- Limited diversity compared to real users

---

## ADR-008: Streamlit for Portfolio UI

**Status**: Accepted

**Context**: The project needs a visual demo for the portfolio. The UI should show chat, memory activity, and evaluation results.

**Decision**: Use Streamlit with multi-page layout: main chat, evaluation dashboard, and memory explorer.

**Alternatives considered**:
- **Gradio**: Good for demos but less flexible for multi-page apps
- **FastAPI + React**: More professional but significantly more development time
- **Chainlit**: Chat-focused but less flexible for dashboards

**Consequences**:
- Rapid development with minimal frontend code
- Built-in support for charts, tables, and session state
- Easy to deploy (Streamlit Cloud or Docker)
- Limited customization compared to a full frontend framework
