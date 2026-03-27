#!/usr/bin/env bash
# ============================================================================
# setup_github.sh — Create labels, milestones, and issues for
# memory-agent-regulatory on GitHub via gh CLI.
#
# Usage (run from WSL inside the repo directory):
#   bash scripts/setup_github.sh
#
# Prerequisites:
#   - gh auth login (already authenticated)
#   - Repository exists on GitHub
# ============================================================================

set -euo pipefail

REPO="brunoramosmartins/memory-agent-regulatory"

echo "=== Setting up GitHub project: $REPO ==="

# --------------------------------------------------------------------------
# 1. LABELS
# --------------------------------------------------------------------------
echo ""
echo "--- Creating labels ---"

# Delete GitHub default labels that we don't use
for label in "enhancement" "help wanted" "invalid" "question" "wontfix" "duplicate"; do
  gh label delete "$label" --repo "$REPO" --yes 2>/dev/null || true
done

# Create project labels (--force overwrites if exists)
gh label create "feature"          --color "0E8A16" --description "New functionality"                    --repo "$REPO" --force
gh label create "bug"              --color "D73A4A" --description "Something is broken"                  --repo "$REPO" --force
gh label create "experiment"       --color "7057FF" --description "Exploratory work, may not merge"      --repo "$REPO" --force
gh label create "documentation"    --color "0075CA" --description "Documentation improvements"           --repo "$REPO" --force
gh label create "infrastructure"   --color "FBCA04" --description "CI, Docker, tooling"                  --repo "$REPO" --force
gh label create "memory"           --color "1D76DB" --description "Memory layer related"                 --repo "$REPO" --force
gh label create "agent"            --color "E4E669" --description "Agent orchestration related"           --repo "$REPO" --force
gh label create "evaluation"       --color "D4C5F9" --description "Evaluation and metrics related"       --repo "$REPO" --force
gh label create "good-first-issue" --color "7057FF" --description "Easy entry point for contributors"    --repo "$REPO" --force
gh label create "blocked"          --color "B60205" --description "Waiting on dependency"                --repo "$REPO" --force

echo "Labels created."

# --------------------------------------------------------------------------
# 2. MILESTONES
# --------------------------------------------------------------------------
echo ""
echo "--- Creating milestones ---"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 0 — Baseline RAG Hardening" \
  -f description="Bootstrap repo from rag-pix-regulation, copy components, Docker Compose (Weaviate + PostgreSQL), CI pipeline, Makefile, structured logging" \
  -f state="open" 2>/dev/null || echo "  Phase 0 milestone may already exist"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 1 — Memory Layer" \
  -f description="PostgreSQL episodic memory, Weaviate semantic memory, procedural memory, summary memory, Alembic migrations, unit + integration tests" \
  -f state="open" 2>/dev/null || echo "  Phase 1 milestone may already exist"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 2 — Memory Manager" \
  -f description="Unified MemoryManager facade, MemoryContext dataclass, Context Builder refactor, auto-summarization trigger" \
  -f state="open" 2>/dev/null || echo "  Phase 2 milestone may already exist"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 3 — Agent Loop" \
  -f description="LangGraph StateGraph, AgentState, node implementations (retrieve, reason, tool_call, write_memory), conditional edges, stop conditions" \
  -f state="open" 2>/dev/null || echo "  Phase 3 milestone may already exist"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 4 — Multi-Source Ingestion" \
  -f description="PDF loader adaptation, web scraper (BeautifulSoup), FAQ parser, source registry, ingestion script" \
  -f state="open" 2>/dev/null || echo "  Phase 4 milestone may already exist"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 5 — Simulation Engine" \
  -f description="Synthetic multi-turn session generation (50-100 sessions), user simulator with personas, reproducible with seed" \
  -f state="open" 2>/dev/null || echo "  Phase 5 milestone may already exist"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 6 — Evaluation Engine" \
  -f description="Five metrics (consistency, reuse rate, token efficiency, retrieval precision, latency), baseline vs memory-aware comparison, report generation" \
  -f state="open" 2>/dev/null || echo "  Phase 6 milestone may already exist"

gh api repos/$REPO/milestones --method POST \
  -f title="Phase 7 — Productization" \
  -f description="Streamlit chat UI, memory visualization panel, evaluation dashboard, memory explorer, ARCHITECTURE.md, DECISIONS.md, portfolio-grade README" \
  -f state="open" 2>/dev/null || echo "  Phase 7 milestone may already exist"

echo "Milestones created."

# --------------------------------------------------------------------------
# Helper: get milestone number by title
# --------------------------------------------------------------------------
get_milestone() {
  gh api repos/$REPO/milestones --paginate -q ".[] | select(.title == \"$1\") | .number"
}

echo ""
echo "--- Fetching milestone numbers ---"
M0=$(get_milestone "Phase 0 — Baseline RAG Hardening")
M1=$(get_milestone "Phase 1 — Memory Layer")
M2=$(get_milestone "Phase 2 — Memory Manager")
M3=$(get_milestone "Phase 3 — Agent Loop")
M4=$(get_milestone "Phase 4 — Multi-Source Ingestion")
M5=$(get_milestone "Phase 5 — Simulation Engine")
M6=$(get_milestone "Phase 6 — Evaluation Engine")
M7=$(get_milestone "Phase 7 — Productization")

echo "  Phase 0: #$M0 | Phase 1: #$M1 | Phase 2: #$M2 | Phase 3: #$M3"
echo "  Phase 4: #$M4 | Phase 5: #$M5 | Phase 6: #$M6 | Phase 7: #$M7"

# --------------------------------------------------------------------------
# 3. ISSUES
# --------------------------------------------------------------------------
echo ""
echo "--- Creating issues ---"

# ======================== PHASE 0 ========================

gh issue create --repo "$REPO" \
  --title "Repository Bootstrap and Component Copy" \
  --milestone "$M0" \
  --label "feature,infrastructure" \
  --body "$(cat <<'EOF'
## Context

The memory-agent-regulatory project reuses proven components from rag-pix-regulation (config, retrieval, observability, vectorstore). This issue covers the initial copy, adaptation, and verification of those components in the new repository.

## Tasks

- [ ] Create repository and initialize Git
- [ ] Copy directories: `src/config`, `src/retrieval`, `src/observability`, `src/vectorstore`, `src/rag`
- [ ] Copy files: `tests/helpers.py`, `config/config.yaml`, `docker-compose.yml`, `pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml`
- [ ] Create placeholder directories: `src/agent`, `src/memory`, `src/ingestion`, `src/simulation`, `src/evaluation`, `app`, `scripts`, `docs`, `migrations`, `data/raw`, `data/processed`, `data/evaluation`
- [ ] Add `__init__.py` to all new packages
- [ ] Update `pyproject.toml`: name=`memory-agent-regulatory`, update description and author
- [ ] Remove PIX-specific defaults from `config/config.yaml` and `src/config/settings.py`
- [ ] Update Phoenix project name in `src/observability/tracing.py`

## Definition of Done

- `python -c "from src.config import get_settings; print(get_settings())"` executes without error
- `pytest -m "not slow and not integration" -v` passes (no import errors, no PIX-specific failures)
- Directory structure matches the repository structure specification

## References

- Source project: `rag-pix-regulation`
EOF
)"
echo "  Issue #1 created"

gh issue create --repo "$REPO" \
  --title "Infrastructure Setup (Docker Compose)" \
  --milestone "$M0" \
  --label "feature,infrastructure" \
  --body "$(cat <<'EOF'
## Context

The project requires PostgreSQL for conversational memory and Weaviate for vector operations. The existing docker-compose.yml from rag-pix includes Weaviate; this issue adds PostgreSQL.

## Tasks

- [ ] Add PostgreSQL 16 Alpine service to `docker-compose.yml`
- [ ] Configure: database=`agent_memory`, user=`agent`, password from `.env`
- [ ] Add named volume `postgres_data` for persistence
- [ ] Expose port 5432
- [ ] Add healthcheck for PostgreSQL
- [ ] Create `.env.example` with `POSTGRES_PASSWORD`, `WEAVIATE_URL`, `OLLAMA_BASE_URL`
- [ ] Verify `docker compose up -d` starts both services without errors
- [ ] Verify PostgreSQL accepts connections: `psql -h localhost -U agent -d agent_memory`

## Definition of Done

- `docker compose up -d` starts Weaviate and PostgreSQL with healthy status
- `docker compose ps` shows both services as healthy
- PostgreSQL accepts connections and `agent_memory` database exists

## References

- PostgreSQL Docker: https://hub.docker.com/_/postgres
- Current docker-compose.yml from rag-pix-regulation
EOF
)"
echo "  Issue #2 created"

gh issue create --repo "$REPO" \
  --title "Logging Layer and Observability Validation" \
  --milestone "$M0" \
  --label "feature,infrastructure" \
  --body "$(cat <<'EOF'
## Context

Without structured logging, the evaluation engine (Phase 6) cannot compute metrics. This issue ensures every retrieval call emits a structured log entry with the fields required for later analysis.

## Tasks

- [ ] Verify `src/observability/tracing.py` initializes OpenTelemetry correctly with new project name
- [ ] Add structured log fields to retrieval pipeline: `query`, `retrieved_doc_ids`, `scores`, `latency_ms`, `timestamp`
- [ ] Ensure logs output as JSON (for parsing by evaluation engine)
- [ ] Add a manual validation script: `scripts/validate_logging.py` that runs a sample query and asserts log fields are present
- [ ] Run 5 sample queries, inspect logs, confirm all fields are present

## Definition of Done

- Every retrieval call produces a JSON log entry with: `query`, `retrieved_doc_ids`, `scores`, `latency_ms`
- `scripts/validate_logging.py` passes
- Phoenix dashboard shows traces for the new project

## References

- `src/observability/tracing.py` (copied from rag-pix)
- Phoenix documentation: https://docs.arize.com/phoenix
EOF
)"
echo "  Issue #3 created"

gh issue create --repo "$REPO" \
  --title "Developer Tooling (Makefile, Linting, CI)" \
  --milestone "$M0" \
  --label "feature,infrastructure" \
  --body "$(cat <<'EOF'
## Context

Consistent developer experience requires a Makefile for common commands, linting with ruff, type checking with mypy, and CI that runs on every push.

## Tasks

- [ ] Create `Makefile` with targets: `up`, `down`, `test`, `test-unit`, `test-integration`, `lint`, `format`, `typecheck`
- [ ] Create `requirements-dev.txt`: ruff, mypy, pytest, pytest-cov, pytest-asyncio, httpx (for test client)
- [ ] Configure ruff in `pyproject.toml`: line-length=99, target Python 3.12, select=["E", "F", "I", "UP", "B", "SIM"]
- [ ] Configure mypy in `pyproject.toml`: strict=false (initially), warn_return_any=true
- [ ] Update `.github/workflows/ci.yml` for new project name, add lint step before tests
- [ ] Verify `make lint` passes on current codebase
- [ ] Verify CI runs and passes on push to feature branch

## Definition of Done

- `make lint` and `make format` work correctly
- `make test-unit` runs only unit tests
- CI pipeline: lint → unit tests → integration tests (marker-based)
- CI green on `main` after merge

## References

- ruff configuration: https://docs.astral.sh/ruff/configuration/
- CI workflow from rag-pix-regulation
EOF
)"
echo "  Issue #4 created"

# ======================== PHASE 1 ========================

gh issue create --repo "$REPO" \
  --title "PostgreSQL Schema and Conversational Memory" \
  --milestone "$M1" \
  --label "feature,memory" \
  --body "$(cat <<'EOF'
## Context

The agent needs episodic memory to recall what was discussed in a conversation thread. PostgreSQL stores conversation turns with thread-level isolation, enabling fast retrieval of recent history.

## Tasks

- [ ] Install: `pip install psycopg[binary] sqlalchemy alembic`
- [ ] Initialize Alembic: `alembic init migrations`
- [ ] Configure `alembic.ini` and `migrations/env.py` to read `POSTGRES_URL` from settings
- [ ] Define SQLAlchemy model: `ConversationTurn(id, thread_id, role, content, timestamp, metadata_json)`
- [ ] Create migration: `alembic revision --autogenerate -m "create conversations table"`
- [ ] Add index on `(thread_id, timestamp)` for fast history retrieval
- [ ] Implement `conversational.py`:
  - `save_turn(thread_id, role, content, metadata=None) → ConversationTurn`
  - `get_history(thread_id, limit=20) → list[ConversationTurn]`
  - `count_turns(thread_id) → int`
- [ ] Write unit tests with SQLite in-memory backend
- [ ] Write integration test with real PostgreSQL (`@pytest.mark.integration`)

## Definition of Done

- `alembic upgrade head` creates the table without errors
- `save_turn` + `get_history` round-trip test passes
- Retrieval by thread_id returns turns ordered by timestamp descending
- Index exists on `(thread_id, timestamp)`

## References

- SQLAlchemy 2.0 docs: https://docs.sqlalchemy.org/en/20/
- Alembic tutorial: https://alembic.sqlalchemy.org/en/latest/tutorial.html
EOF
)"
echo "  Issue #5 created"

gh issue create --repo "$REPO" \
  --title "Weaviate Semantic Memory Collection" \
  --milestone "$M1" \
  --label "feature,memory" \
  --body "$(cat <<'EOF'
## Context

Semantic memory enables the agent to find contextually relevant information across threads using vector similarity. This extends the existing Weaviate setup from rag-pix with a dedicated collection for agent memory.

## Tasks

- [ ] Define Weaviate collection schema `SemanticMemory`: properties=[content, source, thread_id, timestamp], vectorizer config
- [ ] Implement `semantic.py`:
  - `store(content, source, thread_id, metadata=None) → str` (returns object ID)
  - `search(query, limit=5, filters=None) → list[SemanticResult]`
  - `delete(object_id) → bool`
- [ ] Use the existing embedding pipeline from `src/retrieval/query_embedding.py`
- [ ] Write unit tests with mocked Weaviate client
- [ ] Write integration test with real Weaviate (`@pytest.mark.integration`)

## Definition of Done

- Collection created in Weaviate with correct schema
- `store` + `search` round-trip returns the stored item in top results
- Filters by `thread_id` work correctly
- Unit and integration tests pass

## References

- Weaviate Python client: https://weaviate.io/developers/weaviate/client-libraries/python
- Existing vectorstore client: `src/vectorstore/client.py`
EOF
)"
echo "  Issue #6 created"

gh issue create --repo "$REPO" \
  --title "Procedural and Summary Memory" \
  --milestone "$M1" \
  --label "feature,memory" \
  --body "$(cat <<'EOF'
## Context

Procedural memory stores recurring workflow patterns (e.g., "when user asks about fees, always check the fee schedule document first"). Summary memory compresses long conversation histories to reduce token usage. Both are needed before the Memory Manager can offer a unified API.

## Tasks

- [ ] Implement `procedural.py`:
  - `store_pattern(trigger, action, metadata) → str`
  - `find_patterns(query, limit=3) → list[ProceduralPattern]`
  - Storage: Weaviate collection `ProceduralMemory` with structured metadata
- [ ] Implement `summary.py`:
  - `summarize_session(thread_id) → str` (calls LLM to compress history)
  - `get_summary(thread_id) → Optional[str]`
  - `should_summarize(thread_id) → bool` (checks turn count against threshold)
  - Storage: PostgreSQL table `session_summaries(thread_id, summary, created_at)`
- [ ] Create Alembic migration for `session_summaries` table
- [ ] Write unit tests for both modules
- [ ] Write integration test for summary generation with mocked LLM

## Definition of Done

- Procedural patterns can be stored and retrieved by similarity
- Summary generation produces a compressed version of conversation history
- `should_summarize` returns True when turn count exceeds `summary_threshold`
- All tests pass

## References

- LangGraph state management: https://langchain-ai.github.io/langgraph/
- Summarization patterns: https://python.langchain.com/docs/tutorials/summarization/
EOF
)"
echo "  Issue #7 created"

# ======================== PHASE 2 ========================

gh issue create --repo "$REPO" \
  --title "MemoryManager Unified API" \
  --milestone "$M2" \
  --label "feature,memory" \
  --body "$(cat <<'EOF'
## Context

Individual memory types (conversational, semantic, procedural, summary) should not be accessed directly by the agent. A unified MemoryManager prevents fragmentation, enforces read/write order, and provides a single point of control for auto-summarization and future optimizations.

## Tasks

- [ ] Create `MemoryManager.__init__(conversational, semantic, procedural, summary, settings)` with dependency injection
- [ ] Implement `read_context(thread_id, query)`:
  1. Fetch conversational history (up to `max_history_turns`)
  2. Search semantic memory with `query`
  3. Find matching procedural patterns
  4. Fetch session summary (if exists)
  5. Return `MemoryContext(history, semantic_results, patterns, summary)`
- [ ] Implement `write_turn(thread_id, role, content, metadata)`:
  1. Save to conversational memory
  2. Check `should_summarize(thread_id)`
  3. If threshold met, call `summarize_session(thread_id)`
- [ ] Implement `write_semantic(content, source, thread_id)`
- [ ] Implement `write_pattern(trigger, action, metadata)`
- [ ] Add structured logging to every read/write operation
- [ ] Write unit tests: mock all four backends, verify call order and arguments
- [ ] Write integration test: end-to-end read/write through MemoryManager

## Definition of Done

- `read_context` returns a populated `MemoryContext` with data from all four sources
- `write_turn` persists and auto-summarizes when threshold is met
- All memory operations are logged with `thread_id`, `operation`, `latency_ms`
- Unit and integration tests pass

## References

- Facade pattern: https://refactoring.guru/design-patterns/facade
- `src/memory/conversational.py`, `src/memory/semantic.py`, `src/memory/procedural.py`, `src/memory/summary.py`
EOF
)"
echo "  Issue #8 created"

gh issue create --repo "$REPO" \
  --title "Context Builder Refactor" \
  --milestone "$M2" \
  --label "feature,memory" \
  --body "$(cat <<'EOF'
## Context

The RAG pipeline from rag-pix builds context from retrieval only. In this project, context must merge multiple memory sources into a single LLM prompt. This issue refactors the pipeline into a Context Builder that depends on MemoryManager.

## Tasks

- [ ] Rename `src/rag/pipeline.py` → `src/rag/context_builder.py`
- [ ] Implement `ContextBuilder.build(query, memory_context: MemoryContext) → str`:
  1. Format conversational history (recent turns)
  2. Append relevant semantic memories
  3. Append matching procedural patterns
  4. Append session summary (if available)
  5. Append retrieval results (from existing hybrid search)
  6. Enforce token budget (truncate from lowest-priority sections first)
- [ ] Define priority order: summary > history > semantic > procedural > retrieval
- [ ] Add token counting (using `tiktoken` or character approximation)
- [ ] Write unit tests with fixture memory contexts
- [ ] Write test verifying token budget enforcement

## Definition of Done

- `ContextBuilder.build()` produces a structured prompt string with all memory sections
- Token budget is respected (longest sections truncated first)
- Output is deterministic given the same inputs
- Tests pass

## References

- Token counting: https://github.com/openai/tiktoken
- `src/rag/context_builder.py` (original from rag-pix)
EOF
)"
echo "  Issue #9 created"

# ======================== PHASE 3 ========================

gh issue create --repo "$REPO" \
  --title "AgentState and Graph Definition" \
  --milestone "$M3" \
  --label "feature,agent" \
  --body "$(cat <<'EOF'
## Context

LangGraph requires a typed state definition and a compiled state graph. This issue creates the foundational graph structure that all nodes plug into.

## Tasks

- [ ] Define `AgentState` in `state.py`:
  ```python
  class AgentState(TypedDict):
      messages: Annotated[list, add_messages]
      memory_context: Optional[MemoryContext]
      tool_results: list[dict]
      metadata: dict
      iteration_count: int
  ```
- [ ] Define `StateGraph` in `graph.py`:
  - Nodes: `retrieve_memory`, `reason`, `tool_call`, `write_memory`
  - Entry: `retrieve_memory`
  - Conditional edge after `reason`: check if tool needed
  - Edge from `tool_call` → `reason`
  - Edge from `write_memory` → `END`
- [ ] Add `compile()` function that returns the compiled graph
- [ ] Add `run_agent(query, thread_id) → str` entry point
- [ ] Write test that the graph compiles without errors
- [ ] Write test that a simple query flows through all expected nodes

## Definition of Done

- `compile()` returns a valid LangGraph `CompiledGraph`
- `run_agent("What is PIX?")` returns a non-empty response
- Node execution order is logged
- Tests pass

## References

- LangGraph quickstart: https://langchain-ai.github.io/langgraph/tutorials/introduction/
- `src/agent/state.py`, `src/agent/graph.py`
EOF
)"
echo "  Issue #10 created"

gh issue create --repo "$REPO" \
  --title "Node Implementations (Retrieve, Reason, Tool, Write)" \
  --milestone "$M3" \
  --label "feature,agent" \
  --body "$(cat <<'EOF'
## Context

Each node in the agent graph performs a specific function. This issue implements the four core nodes with proper error handling, logging, and state updates.

## Tasks

- [ ] `retrieve_memory(state) → state`:
  - Extract query from last message
  - Call `MemoryManager.read_context(thread_id, query)`
  - Store result in `state["memory_context"]`
- [ ] `reason(state) → state`:
  - Build prompt via `ContextBuilder.build(query, memory_context)`
  - Call LLM (Ollama) with the prompt
  - Parse response: extract answer and optional tool request
  - Append AIMessage to `state["messages"]`
  - Increment `state["iteration_count"]`
- [ ] `tool_call(state) → state`:
  - Extract tool name and arguments from LLM response
  - Execute tool via `tools.py`
  - Append tool result to `state["tool_results"]`
- [ ] `write_memory(state) → state`:
  - Call `MemoryManager.write_turn()` for the current exchange
  - Optionally write semantic memory if response contains reusable knowledge
- [ ] Add structured logging to every node: `node_name`, `thread_id`, `latency_ms`
- [ ] Write unit tests for each node with mocked dependencies

## Definition of Done

- Each node updates state correctly and returns it
- Logging confirms node entry/exit with timing
- Unit tests cover happy path and error cases for each node
- `reason` node respects `max_steps` stop condition

## References

- LangGraph nodes: https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes
- `src/memory/manager.py`, `src/rag/context_builder.py`
EOF
)"
echo "  Issue #11 created"

gh issue create --repo "$REPO" \
  --title "Tool Definitions and Router" \
  --milestone "$M3" \
  --label "feature,agent" \
  --body "$(cat <<'EOF'
## Context

The agent needs tools for cases where memory and retrieval are insufficient. Initial tools include document search and a simple calculator. The tool router maps LLM tool requests to implementations.

## Tasks

- [ ] Implement `search_documents` tool: wraps hybrid search from `src/retrieval/`
- [ ] Implement `calculate` tool: evaluates safe mathematical expressions
- [ ] Implement tool router: `execute_tool(name, args) → ToolResult`
- [ ] Add tool timeout (configurable via `AgentSettings.tool_timeout_s`)
- [ ] Add error handling: tool failure returns error message, does not crash the graph
- [ ] Write unit tests for each tool
- [ ] Write test for timeout behavior

## Definition of Done

- `search_documents` returns relevant chunks from Weaviate
- `calculate` handles basic arithmetic safely (no `eval` on arbitrary code)
- Tool timeout is enforced
- Failed tool calls return structured error, not exceptions
- Tests pass

## References

- LangGraph tools: https://langchain-ai.github.io/langgraph/concepts/low_level/#tools
- `src/retrieval/hybrid_search.py`
EOF
)"
echo "  Issue #12 created"

# ======================== PHASE 4 ========================

gh issue create --repo "$REPO" \
  --title "Web Scraper for FAQ and Regulatory Sites" \
  --milestone "$M4" \
  --label "feature" \
  --body "$(cat <<'EOF'
## Context

Beyond PDFs, the agent needs to ingest FAQ pages and regulatory website content. Web scraping adds a new source type that must integrate with the existing chunking and indexing pipeline.

## Tasks

- [ ] Implement `web_scraper.py`:
  - `scrape_url(url) → list[Document]`
  - Extract main content, strip navigation/headers/footers
  - Split by headings (h1, h2, h3) into chunks
  - Preserve source URL and heading hierarchy in metadata
- [ ] Handle common edge cases: JavaScript-rendered pages (log warning, skip), rate limiting, timeouts
- [ ] Add configuration in `config.yaml`: `ingestion.web_urls` list
- [ ] Write unit tests with saved HTML fixtures
- [ ] Write integration test against a real URL (`@pytest.mark.integration`)

## Definition of Done

- FAQ pages are correctly parsed into chunks with heading metadata
- Navigation elements are stripped
- Rate limiting is respected (configurable delay between requests)
- Unit tests pass with HTML fixtures
- Integration test succeeds against at least one real URL

## References

- BeautifulSoup docs: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- httpx docs: https://www.python-httpx.org/
EOF
)"
echo "  Issue #13 created"

gh issue create --repo "$REPO" \
  --title "Source Registry and Ingestion Script" \
  --milestone "$M4" \
  --label "feature" \
  --body "$(cat <<'EOF'
## Context

Multiple source types (PDF, web, FAQ) need a routing mechanism and a single entry point for batch ingestion. The source registry maps each configured source to its loader.

## Tasks

- [ ] Implement `source_registry.py`:
  - `SourceRegistry.register(source_type, loader_class)`
  - `SourceRegistry.ingest(source_config) → list[Document]`
  - Built-in registrations: `"pdf" → PdfLoader`, `"web" → WebScraper`
- [ ] Implement `scripts/run_ingestion.py`:
  - Read `config.yaml` → `ingestion.sources`
  - For each source, resolve loader via registry
  - Ingest, chunk, embed, store in Weaviate
  - Log: source type, document count, chunk count, duration
- [ ] Add `--source` CLI flag to run a single source type
- [ ] Write unit tests for registry routing
- [ ] Write integration test for full pipeline

## Definition of Done

- `python scripts/run_ingestion.py` ingests all configured sources
- `python scripts/run_ingestion.py --source web` ingests only web sources
- Registry raises clear error for unknown source types
- Logs show per-source statistics
- Tests pass

## References

- Registry pattern: https://refactoring.guru/design-patterns/factory-method
- `src/ingestion/pdf_loader.py`, `src/ingestion/web_scraper.py`
EOF
)"
echo "  Issue #14 created"

# ======================== PHASE 5 ========================

gh issue create --repo "$REPO" \
  --title "Session Generator" \
  --milestone "$M5" \
  --label "feature,evaluation" \
  --body "$(cat <<'EOF'
## Context

The evaluation engine needs realistic multi-turn sessions to compare baseline vs memory-aware performance. Manually creating these is impractical; an LLM-assisted generator produces them at scale.

## Tasks

- [ ] Implement `session_generator.py`:
  - `generate_sessions(topics, count_per_topic, turns_range) → list[Session]`
  - Use LLM (Ollama) to generate realistic user queries for each topic
  - Include follow-ups that depend on prior turns
  - Include implicit references ("what about the fees for that?")
- [ ] Define `Session` schema:
  ```python
  @dataclass
  class Session:
      session_id: str
      topic: str
      turns: list[Turn]  # Turn(role, content, expected_behavior)
  ```
- [ ] Define topics: regulatory fees, deadlines, compliance, comparisons, cross-topic
- [ ] Generate 10 sessions per topic (50 total minimum)
- [ ] Save to `data/evaluation/synthetic_sessions.json`
- [ ] Write validation: each session has 3-8 turns, at least one implicit reference, topic consistency

## Definition of Done

- 50+ sessions generated and saved
- Each session passes validation checks
- Sessions include diversity: direct questions, follow-ups, implicit references, comparisons
- Generator is reproducible with seed parameter

## References

- Synthetic data generation patterns: https://arxiv.org/abs/2305.13169
- `src/simulation/session_generator.py`
EOF
)"
echo "  Issue #15 created"

gh issue create --repo "$REPO" \
  --title "User Simulator" \
  --milestone "$M5" \
  --label "feature,evaluation" \
  --body "$(cat <<'EOF'
## Context

Beyond static sessions, the user simulator enables dynamic interaction with the agent during evaluation — it generates the next user message based on the agent's response, simulating realistic conversation flow.

## Tasks

- [ ] Implement `user_simulator.py`:
  - `UserSimulator.__init__(persona, topic, session_plan)`
  - `generate_next_turn(agent_response) → str`
  - Personas: "confused_user" (asks clarifications), "expert_user" (asks deep questions), "comparative_user" (compares regulations)
- [ ] Each persona has a behavior profile affecting follow-up style
- [ ] Add randomness control via seed for reproducibility
- [ ] Write unit tests with mocked LLM
- [ ] Write integration test: simulate a 5-turn conversation

## Definition of Done

- `UserSimulator` generates contextually appropriate follow-up turns
- Different personas produce meaningfully different interaction styles
- Reproducible with same seed
- Tests pass

## References

- User simulation in dialogue: https://arxiv.org/abs/2308.01423
- `src/simulation/user_simulator.py`
EOF
)"
echo "  Issue #16 created"

# ======================== PHASE 6 ========================

gh issue create --repo "$REPO" \
  --title "Metric Implementations" \
  --milestone "$M6" \
  --label "feature,evaluation" \
  --body "$(cat <<'EOF'
## Context

Each metric must be independently computable from execution logs. This issue implements the five core metrics as pure functions that take structured logs as input and return scores.

## Tasks

- [ ] Implement `consistency_score(sessions_log) → float`:
  - Group responses by topic
  - Compute pairwise semantic similarity (embedding cosine)
  - Threshold: cosine > 0.85 = consistent
- [ ] Implement `context_reuse_rate(sessions_log) → float`:
  - Count responses where `memory_reads > 0` AND response references retrieved content
- [ ] Implement `token_efficiency(baseline_log, memory_log) → float`:
  - `sum(baseline_tokens) / sum(memory_tokens)`
- [ ] Implement `retrieval_precision(sessions_log, relevance_labels) → float`:
  - `relevant_docs / retrieved_docs` per query, averaged
- [ ] Implement `latency_impact(baseline_log, memory_log) → dict`:
  - `avg`, `p50`, `p95` latency for both pipelines
- [ ] Write unit tests with fixture logs for each metric

## Definition of Done

- Each metric function returns a float or dict
- Edge cases handled: empty logs, zero denominators, missing fields
- Unit tests cover happy path and edge cases
- Functions are stateless and side-effect-free

## References

- Evaluation framework specification in ROADMAP
- Embedding similarity: `src/retrieval/query_embedding.py`
EOF
)"
echo "  Issue #17 created"

gh issue create --repo "$REPO" \
  --title "Evaluation Runner and Report Generator" \
  --milestone "$M6" \
  --label "feature,evaluation" \
  --body "$(cat <<'EOF'
## Context

The runner orchestrates running synthetic sessions through both pipelines and collecting metrics. The report generator produces human-readable output for the portfolio.

## Tasks

- [ ] Implement `runner.py`:
  - `EvaluationRunner.__init__(agent, baseline, sessions, metrics)`
  - `run() → EvaluationResult`:
    1. For each session: run through baseline, collect logs
    2. For each session: run through memory-aware agent, collect logs
    3. Compute all metrics
    4. Return `EvaluationResult(baseline_metrics, memory_metrics, per_session)`
- [ ] Implement `report.py`:
  - `generate_markdown(result) → str`: Markdown table with all metrics
  - `generate_json(result) → dict`: structured export
  - `save_report(result, path)`: write both formats to `data/evaluation/`
- [ ] Implement `scripts/run_evaluation.py`:
  - Load sessions from `data/evaluation/synthetic_sessions.json`
  - Initialize both pipelines
  - Run evaluation
  - Generate and save report
- [ ] Write unit tests for runner with mocked pipelines
- [ ] Write test for report format validation

## Definition of Done

- `python scripts/run_evaluation.py` runs end-to-end without errors
- Report includes all five metrics for both pipelines
- Markdown report is portfolio-quality (clear formatting, no raw JSON dumps)
- JSON export is valid and parseable
- Tests pass

## References

- `src/evaluation/metrics.py`
- Synthetic sessions: `data/evaluation/synthetic_sessions.json`
EOF
)"
echo "  Issue #18 created"

# ======================== PHASE 7 ========================

gh issue create --repo "$REPO" \
  --title "Streamlit Chat Interface" \
  --milestone "$M7" \
  --label "feature" \
  --body "$(cat <<'EOF'
## Context

The chat interface is the primary demo surface for the portfolio. It must show the agent's responses alongside the memory operations that powered them, making the system's intelligence visible.

## Tasks

- [ ] Implement `app/main.py`:
  - Chat input/output with `st.chat_message`
  - Conversation history persisted in `st.session_state`
  - Thread management: new conversation, continue existing
- [ ] Add sidebar: memory visualization panel
  - Show memories read per turn (source, content snippet, relevance score)
  - Show memories written per turn (type, content)
  - Color-code by memory type (conversational=blue, semantic=green, procedural=orange, summary=purple)
- [ ] Add Phoenix trace link per turn (if tracing is active)
- [ ] Handle loading states and errors gracefully
- [ ] Test manually: run 5-turn conversation, verify memory panel updates

## Definition of Done

- Chat interface renders conversation with proper formatting
- Memory panel shows read/write operations per turn
- New conversations create new thread_id
- App starts cleanly with `streamlit run app/main.py`

## References

- Streamlit chat: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
- `src/agent/graph.py` for agent entry point
EOF
)"
echo "  Issue #19 created"

gh issue create --repo "$REPO" \
  --title "Evaluation Dashboard and Documentation" \
  --milestone "$M7" \
  --label "feature,documentation" \
  --body "$(cat <<'EOF'
## Context

The evaluation dashboard presents benchmark results visually. Combined with polished documentation, this makes the project portfolio-ready for v1.0.0.

## Tasks

- [ ] Implement `app/pages/evaluation.py`:
  - Load evaluation results from `data/evaluation/`
  - Display comparison table: baseline vs memory-aware
  - Bar charts for each metric (using Streamlit built-in charting)
  - Highlight improvements in green, regressions in red
- [ ] Implement `app/pages/memory_explorer.py`:
  - Browse memories by type (conversational, semantic, procedural, summary)
  - Filter by thread_id
  - Show content, metadata, timestamps
- [ ] Write `docs/ARCHITECTURE.md`: system design with diagrams
- [ ] Write `docs/DECISIONS.md`: ADRs for key choices (LangGraph, PostgreSQL, Weaviate, Ollama)
- [ ] Polish `README.md`: problem, approach, architecture diagram, experiments, results, lessons learned
- [ ] Final review: all links work, all diagrams render, no placeholder text

## Definition of Done

- Evaluation dashboard displays all five metrics with visualizations
- Memory explorer shows stored memories with filtering
- All documentation written and reviewed
- README is portfolio-grade
- No placeholder or TODO text remaining

## References

- Streamlit charts: https://docs.streamlit.io/develop/api-reference/charts
- Evaluation results: `data/evaluation/`
EOF
)"
echo "  Issue #20 created"

# --------------------------------------------------------------------------
# DONE
# --------------------------------------------------------------------------
echo ""
echo "=== Setup complete! ==="
echo ""
echo "Created:"
echo "  - 10 labels"
echo "  - 8 milestones"
echo "  - 20 issues"
echo ""
echo "Verify at: https://github.com/$REPO/issues"
