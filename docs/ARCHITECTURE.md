# Architecture

## System Overview

The Memory-Aware Regulatory Agent is a multi-turn Q&A system that augments traditional RAG with a structured memory layer. The system is designed to maintain context across sessions, learn from interactions, and provide consistent answers to regulatory questions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app/main.py)                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Chat Panel    │  │ Memory Sidebar   │  │ Eval Dashboard   │  │
│  └──────┬───────┘  └──────────────────┘  └──────────────────┘  │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                Agent Orchestrator (LangGraph StateGraph)         │
│                                                                  │
│  retrieve_memory ──► reason ──► tool_call ──► write_memory      │
│        │               │           │              │              │
│        │          ┌────┘           │              │              │
│        │          ▼                │              │              │
│        │    LLM (Ollama)           │              │              │
│        │                           │              │              │
│        ▼                           ▼              ▼              │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Memory Manager (Facade)                  │       │
│  │  read_context() | write_turn() | write_semantic()     │       │
│  └──────────┬──────────────┬──────────────┬─────────────┘       │
└─────────────┼──────────────┼──────────────┼─────────────────────┘
              │              │              │
    ┌─────────┼──────────────┼──────────────┼──────────┐
    │         ▼              ▼              ▼          │
    │  ┌────────────┐ ┌───────────┐ ┌──────────────┐  │
    │  │Conversation│ │ Semantic  │ │ Procedural   │  │
    │  │  (SQL)     │ │ (Vector)  │ │ (Vector)     │  │
    │  ├────────────┤ └─────┬─────┘ └──────┬───────┘  │
    │  │ Summary    │       │              │          │
    │  │  (SQL)     │       │              │          │
    │  └─────┬──────┘       │              │          │
    │        │              │              │          │
    │        ▼              ▼              ▼          │
    │   PostgreSQL       Weaviate                    │
    │                  (BGE-M3 1024d)                 │
    └─────────────────────────────────────────────────┘
```

## Component Details

### 1. Streamlit UI (`app/`)

The user-facing interface with three views:

- **Chat Panel** (`main.py`): Conversational interface with memory activity sidebar
- **Evaluation Dashboard** (`pages/evaluation.py`): Baseline vs memory-aware comparison with charts
- **Memory Explorer** (`pages/memory_explorer.py`): Browse stored memories by type and thread

### 2. Agent Orchestrator (`src/agent/`)

A LangGraph StateGraph with four nodes:

| Node | Responsibility |
|------|---------------|
| `retrieve_memory` | Load conversational history, semantic context, procedural patterns, and summaries |
| `reason` | Call LLM with context, decide whether to answer or call a tool |
| `tool_call` | Execute the requested tool (calculate, search_documents) |
| `write_memory` | Persist the turn to conversational memory, trigger auto-summarize |

**Routing logic**: After `reason`, a conditional edge checks if the LLM requested a tool. If yes, route to `tool_call` then back to `reason`. If no, route to `write_memory` then END.

**Stop conditions**: Max iterations reached, answer complete, or no tool request.

### 3. Memory Layer (`src/memory/`)

Four memory types unified behind a `MemoryManager` facade:

| Memory Type | Storage | Purpose | Read Pattern |
|-------------|---------|---------|-------------|
| Conversational | PostgreSQL | Turn-by-turn history | Last N turns by thread_id |
| Semantic | Weaviate | Content indexed by meaning | Top-K by cosine similarity |
| Procedural | Weaviate | Trigger-action patterns | Top-K by query similarity |
| Summary | PostgreSQL | Compressed session history | Latest summary by thread_id |

**Auto-summarization**: When turn count exceeds threshold (default: 15), the MemoryManager compresses history into a summary.

### 4. Context Builder (`src/rag/context_builder.py`)

Merges memory context + retrieved documents into a single prompt string with priority-based token budgeting:

```
Priority: summary > history > semantic > procedural > retrieval
```

Token estimation uses 4 chars per token (avoids heavy tokenizer dependency).

### 5. Ingestion Pipeline (`src/ingestion/`)

Multi-source document loading:

- **PDF Loader**: PyMuPDF-based extraction, one Document per page
- **Web Scraper**: BeautifulSoup with heading-based splitting
- **Text Chunker**: Configurable size/overlap with sentence-boundary preference
- **Source Registry**: Factory pattern routing source types to loaders

### 6. Evaluation Engine (`src/evaluation/`)

Five metrics comparing baseline vs memory-aware pipelines:

1. **Consistency Score** — Cross-session response similarity
2. **Context Reuse Rate** — Memory utilization frequency
3. **Token Efficiency** — Baseline tokens / memory tokens ratio
4. **Retrieval Precision** — Relevant / retrieved documents
5. **Latency Impact** — Avg, P50, P95 latency comparison

### 7. Simulation Engine (`src/simulation/`)

Generates synthetic multi-turn sessions for evaluation, covering regulatory topics with follow-up questions, context references, and topic switches.

## Data Flow (per request)

1. User sends query via Streamlit
2. `retrieve_memory` loads context from PostgreSQL + Weaviate
3. `Context Builder` merges memory into prompt with token budget
4. `reason` calls LLM (Ollama) with enriched context
5. If tool needed: `tool_call` executes, result fed back to `reason`
6. LLM produces final response
7. `write_memory` persists turn + updates vector memory
8. If turn count > threshold: auto-summarize compresses history
9. Response displayed to user with memory activity in sidebar

## Infrastructure

All services run locally via Docker Compose:

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Conversational + summary memory |
| Weaviate | 8080 / 50051 | Semantic + procedural memory |
| Ollama | 11434 | Local LLM inference |
