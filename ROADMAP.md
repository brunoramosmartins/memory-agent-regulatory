# Memory-Aware Regulatory Agent — Project Roadmap

> **Repository:** `memory-agent-regulatory`
> **Domain:** Regulatory Q&A (PIX as demo use case)
> **Stack:** Python 3.12 · LangGraph · PostgreSQL · Weaviate · Ollama (Llama) · Streamlit
> **Goal:** Demonstrate senior-level AI system design with evaluation rigor

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [GitHub Semantic Guide](#github-semantic-guide)
4. [System Prompt Specification](#system-prompt-specification)
5. [Agent Governance](#agent-governance)
6. [Evaluation Framework](#evaluation-framework)
7. [Phased Roadmap](#phased-roadmap)
   - [Phase 0 — Baseline RAG Hardening](#phase-0--baseline-rag-hardening)
   - [Phase 1 — Memory Layer](#phase-1--memory-layer)
   - [Phase 2 — Memory Manager](#phase-2--memory-manager)
   - [Phase 3 — Agent Loop](#phase-3--agent-loop)
   - [Phase 4 — Multi-Source Ingestion](#phase-4--multi-source-ingestion)
   - [Phase 5 — Simulation Engine](#phase-5--simulation-engine)
   - [Phase 6 — Evaluation Engine](#phase-6--evaluation-engine)
   - [Phase 7 — Productization](#phase-7--productization)
8. [GitHub Workflow Standards](#github-workflow-standards)
9. [MCP & External Integrations](#mcp--external-integrations)
10. [README Structure (Portfolio)](#readme-structure-portfolio)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        User / Simulator                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ query
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Streamlit Chat Interface                          │
│              (app/main.py — conversation UI + memory panel)          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Agent Orchestrator (LangGraph)                      │
│  ┌────────────┐  ┌──────────┐  ┌───────────┐  ┌─────────────────┐  │
│  │ retrieve   │→ │  reason  │→ │ tool_call │→ │ write_memory    │  │
│  │ _memory    │  │          │  │           │  │                 │  │
│  └────────────┘  └──────────┘  └───────────┘  └─────────────────┘  │
│         │              │             │                │              │
│         │              ▼             │                │              │
│         │     ┌──────────────┐       │                │              │
│         │     │  LLM (Ollama)│       │                │              │
│         │     │  Llama local │       │                │              │
│         │     └──────────────┘       │                │              │
└─────────┼────────────────────────────┼────────────────┼──────────────┘
          │                            │                │
          ▼                            ▼                ▼
┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│    Context Builder   │  │   Tool Router    │  │   Memory Write-back │
│  ┌─────────────────┐ │  │  ┌────────────┐  │  │  ┌───────────────┐  │
│  │ Conversational  │ │  │  │ search     │  │  │  │ persist turn  │  │
│  │ Memory (SQL)    │ │  │  │ calculate  │  │  │  │ update vector │  │
│  ├─────────────────┤ │  │  │ lookup     │  │  │  │ trigger       │  │
│  │ Semantic Memory │ │  │  └────────────┘  │  │  │ summarize     │  │
│  │ (Weaviate)      │ │  └──────────────────┘  │  └───────────────┘  │
│  ├─────────────────┤ │                        └─────────────────────┘
│  │ Workflow Memory │ │
│  │ (Vector + meta) │ │
│  ├─────────────────┤ │
│  │ Summary Memory  │ │
│  └─────────────────┘ │
└──────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    Observability Layer                               │
│        OpenTelemetry + Phoenix · Structured Logs · Metrics          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    Infrastructure                                    │
│     Docker Compose: Weaviate · PostgreSQL · Ollama                  │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow (per request)

```
1. User sends query
2. Agent retrieves conversational history (SQL) + semantic context (Weaviate)
3. Context Builder merges memory into prompt
4. LLM reasons over context → decides: answer directly OR call tool
5. If tool needed → Tool Router executes → result fed back to LLM
6. LLM produces final response
7. Memory Write-back persists turn + updates vector memory
8. If turn count exceeds threshold → Summary Memory compresses history
9. Response returned to user
```

---

## Repository Structure

```
memory-agent-regulatory/
├── src/
│   ├── __init__.py
│   ├── agent/                          # LangGraph orchestration
│   │   ├── __init__.py
│   │   ├── graph.py                    # StateGraph definition, compile, entry point
│   │   ├── nodes.py                    # Node functions: retrieve, reason, tool_call, write_memory
│   │   ├── state.py                    # AgentState TypedDict (messages, memory, metadata)
│   │   └── tools.py                    # Tool definitions: search, calculate, lookup
│   │
│   ├── memory/                         # Structured memory layer
│   │   ├── __init__.py
│   │   ├── manager.py                  # MemoryManager — unified read/write facade
│   │   ├── conversational.py           # PostgreSQL episodic store (thread_id, role, content, ts)
│   │   ├── semantic.py                 # Weaviate vector memory (embed + store + query)
│   │   ├── procedural.py              # Workflow/task patterns (vector + metadata)
│   │   └── summary.py                 # Compressed session summaries (triggered by threshold)
│   │
│   ├── ingestion/                      # Multi-source document ingestion
│   │   ├── __init__.py
│   │   ├── pdf_loader.py              # PDF parsing + chunking (adapted from rag-pix)
│   │   ├── web_scraper.py             # FAQ + regulatory site scraping (BeautifulSoup)
│   │   └── source_registry.py         # Route source type → appropriate loader
│   │
│   ├── retrieval/                      # COPIED from rag-pix — hybrid search + reranking
│   │   ├── __init__.py
│   │   ├── hybrid_search.py
│   │   ├── reranker.py
│   │   ├── dedup.py
│   │   └── query_embedding.py
│   │
│   ├── config/                         # COPIED from rag-pix — Pydantic Settings
│   │   ├── __init__.py
│   │   ├── settings.py                # Extended with MemorySettings, AgentSettings, IngestionSettings
│   │   └── logging.py                 # setup_logging() with structured JSON output
│   │
│   ├── observability/                  # COPIED from rag-pix — OpenTelemetry + Phoenix
│   │   ├── __init__.py
│   │   └── tracing.py
│   │
│   ├── vectorstore/                    # COPIED from rag-pix — Weaviate client
│   │   ├── __init__.py
│   │   └── client.py
│   │
│   ├── rag/                            # ADAPTED — becomes Context Builder
│   │   ├── __init__.py
│   │   └── context_builder.py         # Merges memory + retrieval into LLM prompt
│   │
│   ├── simulation/                     # Synthetic evaluation data
│   │   ├── __init__.py
│   │   ├── session_generator.py       # Generate multi-turn sessions (50–100)
│   │   └── user_simulator.py          # Simulated user behavior patterns
│   │
│   └── evaluation/                     # Metrics engine
│       ├── __init__.py
│       ├── metrics.py                 # consistency, reuse_rate, token_efficiency, precision, latency
│       ├── runner.py                  # Orchestrates baseline vs memory-aware runs
│       └── report.py                  # Export results as JSON + Markdown tables
│
├── app/                                # Streamlit UI
│   ├── main.py                        # Chat interface + memory visualization panel
│   └── pages/
│       ├── evaluation.py              # Evaluation dashboard
│       └── memory_explorer.py         # Browse stored memories
│
├── config/
│   └── config.yaml                    # Runtime configuration (domains, thresholds, sources)
│
├── migrations/                         # Alembic database migrations
│   ├── alembic.ini
│   ├── env.py
│   └── versions/                      # Auto-generated migration scripts
│
├── data/
│   ├── raw/                           # Source documents (PDFs, scraped HTML)
│   ├── processed/                     # Chunked + indexed data
│   └── evaluation/                    # Synthetic sessions + evaluation results
│
├── scripts/
│   ├── bootstrap.py                   # One-command project setup (DB + Weaviate + seed data)
│   ├── run_ingestion.py               # Ingest documents from all configured sources
│   ├── run_evaluation.py              # Execute full evaluation pipeline
│   └── run_agent.py                   # CLI entry point for the agent
│
├── tests/
│   ├── conftest.py                    # Shared fixtures (db session, weaviate client, settings)
│   ├── helpers.py                     # make_test_settings() factory (copied from rag-pix)
│   ├── unit/
│   │   ├── test_memory_manager.py
│   │   ├── test_conversational_memory.py
│   │   ├── test_semantic_memory.py
│   │   ├── test_summary.py
│   │   ├── test_context_builder.py
│   │   ├── test_agent_state.py
│   │   ├── test_nodes.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_agent_graph.py
│   │   ├── test_memory_roundtrip.py
│   │   ├── test_ingestion_pipeline.py
│   │   └── test_evaluation_runner.py
│   └── e2e/
│       └── test_full_session.py       # End-to-end multi-turn session test
│
├── docs/
│   ├── ARCHITECTURE.md                # Detailed system design document
│   ├── DECISIONS.md                   # Architecture Decision Records (ADRs)
│   └── EVALUATION.md                  # Evaluation methodology + results
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                     # CI: lint + unit tests + integration (marker-based)
│   ├── ISSUE_TEMPLATE/
│   │   ├── task.md
│   │   └── bug.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docker-compose.yml                 # Weaviate + PostgreSQL + Ollama
├── pyproject.toml                     # Project metadata, deps, coverage, markers
├── requirements.txt                   # Pinned production dependencies
├── requirements-dev.txt               # Dev/test dependencies (pytest, ruff, mypy)
├── Makefile                           # Common commands: make test, make lint, make up
├── .env.example                       # Environment variable template
├── .gitignore
├── LICENSE
└── README.md                          # Portfolio-grade project overview
```

---

## GitHub Semantic Guide

This section explains **tags, releases, milestones, and issues** — what they are, when to create each, and how they relate in this project.

### Tags

A **tag** is a Git pointer to a specific commit. In this project, tags follow semantic versioning and mark the completion of each phase.

**Format:** `v{major}.{minor}.{patch}-{phase-name}`

**When to create:** After merging the final PR of a phase into `main`, when all Definition of Done criteria are met.

```bash
# Example: Phase 0 complete
git tag -a v0.1.0-baseline-hardening -m "Phase 0: Baseline RAG hardening complete"
git push origin v0.1.0-baseline-hardening

# Example: Phase 3 complete
git tag -a v0.4.0-agent-loop -m "Phase 3: Agent loop with LangGraph operational"
git push origin v0.4.0-agent-loop
```

**Versioning rules:**

| Version | Meaning | Example |
|---------|---------|---------|
| `v0.x.0` | Internal milestone (infrastructure, plumbing) | `v0.1.0-baseline-hardening` |
| `v0.x.y` | Patch within a phase (bug fix after tagging) | `v0.3.1-memory-manager` |
| `v1.0.0` | Public portfolio release (all phases complete) | `v1.0.0` |

### Releases

A **release** is a GitHub Release built from a tag. It includes release notes, a changelog, and optionally downloadable artifacts.

**When to create:** Only when there is **external value** — someone outside the project (a recruiter, a peer reviewer, a potential collaborator) would benefit from a packaged snapshot.

| Phase | Tag | Release? | Reasoning |
|-------|-----|----------|-----------|
| Phase 0 | `v0.1.0-baseline-hardening` | No | Internal infrastructure — no external audience |
| Phase 1 | `v0.2.0-memory-layer` | No | Internal plumbing — no runnable demo |
| Phase 2 | `v0.3.0-memory-manager` | No | API layer only — not user-facing |
| Phase 3 | `v0.4.0-agent-loop` | **Yes** | First runnable agent — demonstrates core capability |
| Phase 4 | `v0.5.0-multi-source-ingestion` | No | Data pipeline improvement — incremental |
| Phase 5 | `v0.6.0-simulation-engine` | No | Internal evaluation tooling |
| Phase 6 | `v0.7.0-evaluation-engine` | **Yes** | Publishable results — demonstrates rigor |
| Phase 7 | `v0.8.0-productization` | **Yes** | Streamlit UI — interactive demo |
| Final | `v1.0.0` | **Yes** | Portfolio launch with full evaluation results |

```bash
# Creating a release (Phase 3 example)
# 1. Tag the commit
git tag -a v0.4.0-agent-loop -m "Phase 3: Agent loop operational"
git push origin v0.4.0-agent-loop

# 2. Create release via GitHub CLI
gh release create v0.4.0-agent-loop \
  --title "v0.4.0 — Agent Loop" \
  --notes-file docs/releases/v0.4.0.md
```

### Milestones

A **milestone** groups related issues and tracks progress toward a phase completion. Each phase maps to exactly one milestone.

**When to create:** At the start of each phase, before creating the phase's issues.

```bash
# Create milestone via GitHub CLI
gh api repos/{owner}/{repo}/milestones \
  --method POST \
  -f title="Phase 1 — Memory Layer" \
  -f description="PostgreSQL episodic memory, Weaviate semantic memory, and foundational memory tests" \
  -f due_on="2026-04-15T00:00:00Z"
```

**Milestone closure checklist:**

1. All issues in the milestone are closed
2. CI is green on `main`
3. Phase-specific verification commands pass
4. Tag is created

### Issues

An **issue** represents a single unit of work. Every issue belongs to a milestone and carries labels.

**When to create:** At the start of a phase — create all issues for that phase before beginning implementation.

**Issue body format** (every issue must include all four sections):

```markdown
## Context
Why this issue exists, what problem it solves, and how it fits into the phase.

## Tasks
- [ ] Specific, actionable checklist item 1
- [ ] Specific, actionable checklist item 2
- [ ] Specific, actionable checklist item 3

## Definition of Done
- Verifiable criterion 1
- Verifiable criterion 2
- CI green, no regressions

## References
- Link to relevant documentation
- Link to related code or prior issue
```

### How They Relate

```
Milestone (Phase 1 — Memory Layer)
  ├── Issue #5: Conversational schema
  ├── Issue #6: Vector memory collection
  ├── Issue #7: Memory Manager API
  └── Issue #8: Memory layer tests
         │
         ▼ (all issues closed)
      Tag: v0.2.0-memory-layer
         │
         ▼ (external value? → No)
      No Release
```

---

## System Prompt Specification

The agent's system prompt governs LLM behavior. This spec defines **what to include** and explicitly states **what NOT to include** to avoid duplicating framework defaults (LangGraph already handles message formatting, tool schemas, and conversation threading).

### Workflow Steps

```
1. RETRIEVE — Query conversational memory (last N turns) and semantic memory (top-k relevant)
2. ASSESS — Determine if retrieved context is sufficient to answer
3. TOOL — If context is insufficient, select and invoke the appropriate tool
4. REASON — Synthesize retrieved context + tool results into a grounded answer
5. RESPOND — Produce structured output (answer + optional reasoning trace)
6. PERSIST — Write relevant outputs to memory (conversational + semantic)
```

### Analytical Principles

- **Memory-first:** Always check memory before generating new content. If the answer exists in memory, reuse it.
- **Retrieval over generation:** Prefer citing retrieved documents over synthesizing from parametric knowledge.
- **Minimal token usage:** Do not repeat the user's question. Do not re-explain the system. Do not pad responses.
- **Factual grounding:** Every claim about a regulation must trace to a retrieved document or an explicit "I don't have this information" statement.
- **Consistency enforcement:** If a prior answer on the same topic exists in memory, the new answer must align or explicitly acknowledge the correction.

### Stop Conditions

| Condition | Action |
|-----------|--------|
| Answer is complete and grounded | Return response |
| Maximum iterations reached (`max_steps`) | Return best-effort response with disclaimer |
| No improvement between iterations | Break loop, return current response |
| Tool call fails after retry | Return partial response, log error |

### Output Format

```
Direct answer (1–3 sentences)
---
[Optional] Reasoning trace: step-by-step logic (only when user requests or confidence is low)
[Optional] Sources: document IDs used
```

### Explicitly Excluded (do NOT include in prompt)

These are handled by LangGraph or the application layer — including them in the prompt wastes tokens and risks conflicts:

- Message formatting instructions (LangGraph manages `HumanMessage`/`AIMessage` structure)
- Tool JSON schemas (LangGraph injects tool definitions automatically)
- Conversation history formatting (managed by the `Context Builder`)
- System identity preamble ("You are a helpful assistant…") — replaced by domain-specific instructions above
- Retry logic instructions (handled by `graph.py` stop conditions)

---

## Agent Governance

### Observability Strategy

| Layer | Tool | What It Captures |
|-------|------|-----------------|
| Tracing | OpenTelemetry + Phoenix | Per-request spans: retrieval latency, LLM call duration, tool execution time |
| Logging | Python `structlog` (JSON) | Every node entry/exit, memory read/write operations, error context |
| Metrics | Custom `evaluation/metrics.py` | Aggregated: consistency score, reuse rate, token efficiency, precision, latency |

### Health Metrics to Monitor

| Metric | Healthy Range | Alert Threshold |
|--------|---------------|-----------------|
| Retrieval latency (p95) | < 500ms | > 1000ms |
| LLM call latency (p95) | < 3s | > 8s |
| Tool call success rate | > 95% | < 80% |
| Memory write success rate | 100% | < 99% |
| Agent loop iterations (avg) | 1–2 | > 4 consistently |
| Token usage per response | < baseline | > 1.5x baseline |

### Expected vs. Problematic Tool Call Patterns

**Expected patterns:**

```
retrieve_memory → reason → respond                     # Direct answer from memory
retrieve_memory → reason → tool_call(search) → respond # Needed external info
retrieve_memory → reason → tool_call(search) → reason → respond  # Two-step reasoning
```

**Problematic patterns (investigate if seen):**

```
retrieve_memory → tool_call → tool_call → tool_call → max_steps  # Tool loop — broken stop condition
retrieve_memory → reason → respond [memory never written]        # Memory write-back skipped
reason → respond [no retrieve_memory]                            # Memory bypass — context builder issue
tool_call(same_tool, same_args) repeated                         # Infinite retry — missing dedup
```

### Interpreting Execution Logs

Each request produces a structured log trace. Key fields:

```json
{
  "request_id": "uuid",
  "thread_id": "session-uuid",
  "nodes_visited": ["retrieve_memory", "reason", "respond"],
  "memory_reads": {"conversational": 5, "semantic": 3},
  "memory_writes": {"conversational": 1, "semantic": 0},
  "tool_calls": [],
  "total_tokens": 1847,
  "latency_ms": 2340,
  "error": null
}
```

**Interpretation guide:**

- `nodes_visited` shows the execution path — compare against expected patterns above
- `memory_reads.conversational = 0` when `thread_id` exists → conversational memory retrieval is broken
- `memory_writes.semantic = 0` over many turns → semantic memory is not learning
- `total_tokens` consistently higher than baseline → memory is inflating context, not reducing it
- `error` is non-null → check the Phoenix trace for the failing span

---

## Evaluation Framework

### Metrics

#### 1. Consistency Score

Measures whether the agent gives aligned answers across sessions when asked about the same topic.

```
consistency = (number of consistent answers) / (total related queries)
```

**Measurement:** Run the same topic-group queries across multiple sessions. Compare answers pairwise using semantic similarity (embedding cosine > 0.85 = consistent).

#### 2. Context Reuse Rate

Measures how often the agent leverages stored memory rather than generating from scratch.

```
reuse_rate = (responses using memory) / (total responses)
```

**Measurement:** Count responses where `memory_reads > 0` AND the response references retrieved content.

#### 3. Token Efficiency

Measures whether memory reduces total token consumption.

```
token_efficiency = baseline_tokens / memory_tokens
```

**Target:** > 1.0 (memory-aware uses fewer tokens). Values < 1.0 indicate memory is adding overhead.

#### 4. Retrieval Precision

Measures the relevance of retrieved documents.

```
precision = relevant_docs / retrieved_docs
```

**Measurement:** Human-annotated relevance labels on a sample of 50 queries.

#### 5. Latency Impact

Measures the latency cost of the memory layer.

```
delta_latency = avg_latency_memory - avg_latency_baseline
```

**Target:** < 200ms additional latency from memory operations.

### Evaluation Protocol

1. Generate 50–100 synthetic multi-turn sessions (Phase 5)
2. Run each session through the **baseline** (stateless RAG) pipeline
3. Run each session through the **memory-aware** agent pipeline
4. Compute all five metrics for both pipelines
5. Generate comparison report with tables and charts

---

## Phased Roadmap

---

### Phase 0 — Baseline RAG Hardening

**Objective:** Bootstrap the new repository from `rag-pix-regulation`, verify that the copied retrieval pipeline works in isolation, and establish the logging/observability foundation required for all subsequent evaluation.

**Duration:** 1–2 days

#### Tasks

- [ ] Create repository and initialize Git
- [ ] Copy reusable components from `rag-pix-regulation` (config, retrieval, observability, vectorstore, rag, tests/helpers, docker-compose, pyproject.toml, CI)
- [ ] Create placeholder directories: `src/agent`, `src/memory`, `src/ingestion`, `src/simulation`, `src/evaluation`, `app`
- [ ] Add PostgreSQL service to `docker-compose.yml`
- [ ] Update `pyproject.toml`: project name, description, authors, markers
- [ ] Remove PIX-specific content from config defaults; generalize document aliases
- [ ] Update Phoenix registration to use new project name
- [ ] Add structured logging to every retrieval call (query, retrieved_doc_ids, scores, latency_ms)
- [ ] Add `Makefile` with targets: `up`, `down`, `test`, `test-unit`, `test-integration`, `lint`, `format`
- [ ] Add `.env.example` with all required environment variables
- [ ] Add `requirements-dev.txt` with ruff, mypy, pytest, pytest-cov, pytest-asyncio
- [ ] Verify: `docker compose up -d` starts Weaviate + PostgreSQL
- [ ] Verify: `pytest -m "not slow and not integration" -v` passes
- [ ] Verify: `python -c "from src.config import get_settings; print(get_settings())"` works

#### Deliverables

- [ ] Repository bootstrapped with all copied components
- [ ] `docker compose up -d` starts cleanly
- [ ] CI green with marker-based test execution
- [ ] Structured logs emitted per retrieval request
- [ ] `Makefile` operational

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-0-bootstrap` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: bootstrap project from rag-pix-regulation` |
| Milestone | Phase 0 — Baseline RAG Hardening |
| Tag | `v0.1.0-baseline-hardening` |
| Release | **No** — internal infrastructure, no external audience |

#### Issues

---

**Issue #1: Repository Bootstrap and Component Copy**

## Context

The memory-agent-regulatory project reuses proven components from rag-pix-regulation (config, retrieval, observability, vectorstore). This issue covers the initial copy, adaptation, and verification of those components in the new repository.

## Tasks

- [ ] `mkdir memory-agent-regulatory && cd memory-agent-regulatory && git init`
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
- Copy table in roadmap Step 1

---

**Issue #2: Infrastructure Setup (Docker Compose)**

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

---

**Issue #3: Logging Layer and Observability Validation**

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

---

**Issue #4: Developer Tooling (Makefile, Linting, CI)**

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

---

### Phase 1 — Memory Layer

**Objective:** Implement the foundational persistence layer — PostgreSQL schema for conversational memory and a Weaviate collection for semantic memory. Each memory type must be independently testable with clear read/write operations.

**Duration:** 3–5 days

#### Tasks

- [ ] Design and create PostgreSQL schema: `conversations` table (id, thread_id, role, content, timestamp, metadata)
- [ ] Set up Alembic for database migrations
- [ ] Create initial migration script
- [ ] Implement `src/memory/conversational.py`: `save_turn()`, `get_history()`, `get_by_thread()`
- [ ] Design Weaviate collection schema: `SemanticMemory` (content, embedding, source, timestamp, metadata)
- [ ] Implement `src/memory/semantic.py`: `store()`, `search()`, `delete()`
- [ ] Implement `src/memory/procedural.py`: workflow pattern storage (vector + structured metadata)
- [ ] Implement `src/memory/summary.py`: `summarize_session()`, `get_summary()`
- [ ] Write unit tests for each memory type (mocked storage)
- [ ] Write integration tests that run against real PostgreSQL and Weaviate (marked `@pytest.mark.integration`)

#### Deliverables

- [ ] PostgreSQL schema created via Alembic migration
- [ ] All four memory types implemented with read/write operations
- [ ] Unit tests passing for all memory types
- [ ] Integration tests passing against Docker services

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-1-memory-layer` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: implement memory layer (conversational + semantic + procedural + summary)` |
| Milestone | Phase 1 — Memory Layer |
| Tag | `v0.2.0-memory-layer` |
| Release | **No** — internal plumbing, not runnable |

#### Issues

---

**Issue #5: PostgreSQL Schema and Conversational Memory**

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

---

**Issue #6: Weaviate Semantic Memory Collection**

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

---

**Issue #7: Procedural and Summary Memory**

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

---

### Phase 2 — Memory Manager

**Objective:** Create a unified `MemoryManager` facade that abstracts all four memory types behind a single API. The Context Builder will depend only on this interface, never on individual memory implementations.

**Duration:** 2–3 days

#### Tasks

- [ ] Implement `src/memory/manager.py`: `MemoryManager` class with:
  - `read_context(thread_id, query) → MemoryContext` (reads from all memory types)
  - `write_turn(thread_id, role, content, metadata=None)` (writes conversational + conditionally triggers summary)
  - `write_semantic(content, source, thread_id)` (writes semantic memory)
  - `write_pattern(trigger, action, metadata)` (writes procedural memory)
- [ ] Define `MemoryContext` dataclass: `history`, `semantic_results`, `patterns`, `summary`
- [ ] Implement auto-summarization trigger: after `write_turn`, check `should_summarize()` and compress if needed
- [ ] Refactor `src/rag/pipeline.py` → `src/rag/context_builder.py`: merge memory context into LLM prompt
- [ ] Write unit tests for MemoryManager with mocked memory backends
- [ ] Write integration test: full read/write cycle through MemoryManager

#### Deliverables

- [ ] `MemoryManager` with unified API
- [ ] `MemoryContext` dataclass
- [ ] Context Builder refactored to use MemoryManager
- [ ] Auto-summarization working
- [ ] All tests passing

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-2-memory-manager` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: unified MemoryManager API and Context Builder` |
| Milestone | Phase 2 — Memory Manager |
| Tag | `v0.3.0-memory-manager` |
| Release | **No** — API layer only, not user-facing |

#### Issues

---

**Issue #8: MemoryManager Unified API**

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

---

**Issue #9: Context Builder Refactor**

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
- `src/rag/pipeline.py` (original from rag-pix)

---

### Phase 3 — Agent Loop

**Objective:** Implement the LangGraph-based agent loop that connects retrieval, reasoning, tool execution, and memory write-back into a cyclic state graph. This is the first phase where the agent can actually respond to queries.

**Duration:** 3–5 days

#### Tasks

- [ ] Install LangGraph: `pip install langgraph`
- [ ] Define `AgentState` TypedDict in `src/agent/state.py`: messages, memory_context, tool_results, metadata, iteration_count
- [ ] Implement graph nodes in `src/agent/nodes.py`:
  - `retrieve_memory`: read from MemoryManager
  - `reason`: call LLM with context
  - `tool_call`: execute selected tool
  - `write_memory`: persist turn and semantic memory
- [ ] Implement conditional edges:
  - After `reason`: route to `tool_call` if tool needed, else to `write_memory`
  - After `tool_call`: route back to `reason`
  - After `write_memory`: END
- [ ] Implement stop conditions: `max_steps`, answer complete, no improvement
- [ ] Define tools in `src/agent/tools.py`: `search_documents`, `calculate`
- [ ] Compile and test the graph with a simple query
- [ ] Write unit tests for each node function
- [ ] Write integration test: full agent turn with mocked LLM

#### Deliverables

- [ ] LangGraph state graph compiled and operational
- [ ] Agent responds to queries using memory + retrieval
- [ ] Stop conditions prevent infinite loops
- [ ] All tests passing

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-3-agent-loop` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: LangGraph agent loop with memory integration` |
| Milestone | Phase 3 — Agent Loop |
| Tag | `v0.4.0-agent-loop` |
| Release | **Yes** — first runnable agent, demonstrates core capability |

#### Issues

---

**Issue #10: AgentState and Graph Definition**

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

---

**Issue #11: Node Implementations (Retrieve, Reason, Tool, Write)**

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

---

**Issue #12: Tool Definitions and Router**

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

---

### Phase 4 — Multi-Source Ingestion

**Objective:** Extend the ingestion pipeline to handle PDF documents, FAQ pages, and regulatory websites. A source registry routes each source type to the appropriate loader.

**Duration:** 2–3 days

#### Tasks

- [ ] Install: `pip install beautifulsoup4 httpx`
- [ ] Adapt `pdf_loader.py` from rag-pix for generic regulatory documents
- [ ] Implement `web_scraper.py`: fetch and parse FAQ pages + regulatory sites
- [ ] Implement `source_registry.py`: route source type (pdf, web, faq) to the correct loader
- [ ] Implement chunking strategy for web content (heading-based splitting)
- [ ] Create `scripts/run_ingestion.py`: reads sources from `config.yaml`, ingests all
- [ ] Write unit tests for each loader
- [ ] Write integration test for full ingestion pipeline

#### Deliverables

- [ ] PDF, FAQ, and web sources can be ingested into Weaviate
- [ ] Source registry correctly routes all configured sources
- [ ] `scripts/run_ingestion.py` runs end-to-end
- [ ] All tests passing

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-4-multi-source-ingestion` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: multi-source ingestion (PDF + web + FAQ)` |
| Milestone | Phase 4 — Multi-Source Ingestion |
| Tag | `v0.5.0-multi-source-ingestion` |
| Release | **No** — data pipeline improvement, incremental |

#### Issues

---

**Issue #13: Web Scraper for FAQ and Regulatory Sites**

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

---

**Issue #14: Source Registry and Ingestion Script**

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

---

### Phase 5 — Simulation Engine

**Objective:** Generate 50–100 synthetic multi-turn sessions that simulate realistic user interactions. These sessions provide the test data for the evaluation engine.

**Duration:** 2–3 days

#### Tasks

- [ ] Implement `session_generator.py`: generate sessions grouped by topic
- [ ] Implement `user_simulator.py`: simulate user behavior patterns (follow-up, implicit reference, topic switch)
- [ ] Define session format (JSON): session_id, topic, turns (role, content, expected_behavior)
- [ ] Generate sessions across regulatory topics: fees, deadlines, compliance, comparisons
- [ ] Include dependency patterns: turn 3 references turn 1 implicitly
- [ ] Create `data/evaluation/synthetic_sessions.json` with 50+ sessions
- [ ] Write unit tests for session generator
- [ ] Validate sessions: each has 3–8 turns, topic consistency, at least one implicit reference

#### Deliverables

- [ ] 50–100 synthetic multi-turn sessions in JSON format
- [ ] Sessions cover diverse topics and interaction patterns
- [ ] Validation script confirms session quality
- [ ] All tests passing

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-5-simulation-engine` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: simulation engine with synthetic multi-turn sessions` |
| Milestone | Phase 5 — Simulation Engine |
| Tag | `v0.6.0-simulation-engine` |
| Release | **No** — internal evaluation tooling |

#### Issues

---

**Issue #15: Session Generator**

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
- [ ] Define topics: ["pix_fees", "pix_deadlines", "pix_compliance", "pix_comparisons", "cross_topic"]
- [ ] Generate 10 sessions per topic (50 total minimum)
- [ ] Save to `data/evaluation/synthetic_sessions.json`
- [ ] Write validation: each session has 3–8 turns, at least one implicit reference, topic consistency

## Definition of Done

- 50+ sessions generated and saved
- Each session passes validation checks
- Sessions include diversity: direct questions, follow-ups, implicit references, comparisons
- Generator is reproducible with seed parameter

## References

- Synthetic data generation patterns: https://arxiv.org/abs/2305.13169
- `src/simulation/session_generator.py`

---

**Issue #16: User Simulator**

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

---

### Phase 6 — Evaluation Engine

**Objective:** Implement the five evaluation metrics, run baseline vs memory-aware comparisons, and generate a publication-grade report with tables and charts.

**Duration:** 2–3 days

#### Tasks

- [ ] Implement `metrics.py`: consistency_score, context_reuse_rate, token_efficiency, retrieval_precision, latency_impact
- [ ] Implement `runner.py`:
  - Run synthetic sessions through baseline (stateless) pipeline
  - Run same sessions through memory-aware agent pipeline
  - Collect per-session metrics
- [ ] Implement `report.py`:
  - Generate Markdown table comparing baseline vs memory-aware
  - Generate JSON export for programmatic analysis
  - Include per-metric breakdown and aggregate statistics
- [ ] Create `scripts/run_evaluation.py`: orchestrates the full evaluation
- [ ] Run evaluation and verify results are reasonable
- [ ] Write unit tests for each metric function
- [ ] Document methodology in `docs/EVALUATION.md`

#### Deliverables

- [ ] All five metrics implemented and tested
- [ ] Evaluation runner executes both pipelines
- [ ] Comparison report generated (Markdown + JSON)
- [ ] `docs/EVALUATION.md` documents methodology

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-6-evaluation-engine` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: evaluation engine with baseline vs memory-aware comparison` |
| Milestone | Phase 6 — Evaluation Engine |
| Tag | `v0.7.0-evaluation-engine` |
| Release | **Yes** — publishable results, demonstrates evaluation rigor |

#### Issues

---

**Issue #17: Metric Implementations**

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

- Evaluation framework specification in this roadmap
- Embedding similarity: `src/retrieval/query_embedding.py`

---

**Issue #18: Evaluation Runner and Report Generator**

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

---

### Phase 7 — Productization

**Objective:** Build a Streamlit UI that serves as the portfolio demo: interactive chat, memory visualization panel, evaluation dashboard, and Phoenix tracing integration.

**Duration:** 2–3 days

#### Tasks

- [ ] Install: `pip install streamlit`
- [ ] Implement `app/main.py`: chat interface with conversation history
- [ ] Implement memory visualization panel: show which memories were retrieved and written per turn
- [ ] Implement `app/pages/evaluation.py`: display evaluation results with charts
- [ ] Implement `app/pages/memory_explorer.py`: browse stored memories by type and thread
- [ ] Add Phoenix tracing link in the UI
- [ ] Write `docs/ARCHITECTURE.md` with detailed system design
- [ ] Write `docs/DECISIONS.md` with Architecture Decision Records
- [ ] Polish `README.md` to portfolio standard

#### Deliverables

- [ ] Streamlit app with chat, memory visualization, and evaluation dashboard
- [ ] Documentation complete (ARCHITECTURE.md, DECISIONS.md, EVALUATION.md, README.md)
- [ ] App runs with `streamlit run app/main.py`

#### GitHub

| Item | Value |
|------|-------|
| Branch | `feature/phase-7-productization` |
| Merge strategy | Squash merge into `main` |
| PR title | `feat: Streamlit UI with memory visualization and evaluation dashboard` |
| Milestone | Phase 7 — Productization |
| Tag | `v0.8.0-productization` |
| Release | **Yes** — interactive demo, portfolio-ready |

#### Issues

---

**Issue #19: Streamlit Chat Interface**

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

---

**Issue #20: Evaluation Dashboard and Documentation**

## Context

The evaluation dashboard presents benchmark results visually. Combined with polished documentation, this makes the project portfolio-ready for v1.0.0.

## Tasks

- [ ] Implement `app/pages/evaluation.py`:
  - Load evaluation results from `data/evaluation/`
  - Display comparison table: baseline vs memory-aware
  - Bar charts for each metric (using Streamlit's built-in charting)
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

---

## Final Release: v1.0.0

After all phases are complete:

```bash
# Create final tag
git tag -a v1.0.0 -m "v1.0.0: Portfolio release — memory-aware regulatory agent with evaluation"
git push origin v1.0.0

# Create GitHub Release
gh release create v1.0.0 \
  --title "v1.0.0 — Memory-Aware Regulatory Agent" \
  --notes-file docs/releases/v1.0.0.md \
  --latest
```

**v1.0.0 release notes must include:**

- Project summary (2–3 sentences)
- Architecture diagram
- Key evaluation results (table)
- How to run the demo
- Link to live demo (if deployed)

---

## GitHub Workflow Standards

### Branch Naming Convention

```
feature/{phase-N}-{short-description}    # New functionality
fix/{short-description}                   # Bug fixes
experiment/{short-description}            # Exploratory work (may not merge)
docs/{short-description}                  # Documentation only
refactor/{short-description}              # Code restructuring (no behavior change)
```

**Examples:**

```
feature/phase-1-memory-layer
feature/phase-3-agent-loop
fix/memory-manager-null-thread
experiment/alternative-summarization
docs/evaluation-methodology
refactor/context-builder-token-counting
```

### Conventional Commits

| Prefix | Usage | Example |
|--------|-------|---------|
| `feat:` | New feature or capability | `feat: implement conversational memory with PostgreSQL` |
| `fix:` | Bug fix | `fix: handle null thread_id in MemoryManager.read_context` |
| `refactor:` | Code restructuring, no behavior change | `refactor: extract token counting into utility function` |
| `test:` | Adding or modifying tests | `test: add integration tests for semantic memory` |
| `docs:` | Documentation only | `docs: add evaluation methodology to EVALUATION.md` |
| `chore:` | Tooling, CI, dependencies | `chore: add alembic to requirements.txt` |
| `experiment:` | Exploratory work | `experiment: test alternative summarization prompt` |
| `ci:` | CI/CD changes | `ci: add integration test job with Docker services` |

### PR Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Change 1
- Change 2

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated (if applicable)
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project conventions (ruff, mypy)
- [ ] Documentation updated (if applicable)
- [ ] No TODOs or placeholder code
- [ ] CI passes
```

### Issue Templates

**Task Template (`.github/ISSUE_TEMPLATE/task.md`):**

```markdown
---
name: Task
about: A specific unit of work
labels: feature
---

## Context
Why this task exists and what problem it solves.

## Tasks
- [ ] Actionable item 1
- [ ] Actionable item 2

## Definition of Done
- Verifiable criterion 1
- Verifiable criterion 2

## References
- Relevant links
```

**Bug Template (`.github/ISSUE_TEMPLATE/bug.md`):**

```markdown
---
name: Bug
about: Report a defect
labels: bug
---

## Description
What is broken and what was expected.

## Steps to Reproduce
1. Step 1
2. Step 2

## Expected Behavior
What should happen.

## Actual Behavior
What happens instead.

## Environment
- Python version:
- OS:
- Docker versions:

## References
- Related logs, traces, or issues
```

### Labels

| Label | Color | Description |
|-------|-------|-------------|
| `feature` | `#0E8A16` (green) | New functionality |
| `bug` | `#D73A4A` (red) | Something is broken |
| `experiment` | `#7057FF` (purple) | Exploratory work, may not merge |
| `documentation` | `#0075CA` (blue) | Documentation improvements |
| `infrastructure` | `#FBCA04` (yellow) | CI, Docker, tooling |
| `memory` | `#1D76DB` (light blue) | Memory layer related |
| `agent` | `#E4E669` (lime) | Agent orchestration related |
| `evaluation` | `#D4C5F9` (lavender) | Evaluation and metrics related |
| `good-first-issue` | `#7057FF` (purple) | Easy entry point for contributors |
| `blocked` | `#B60205` (dark red) | Waiting on dependency |

---

## MCP & External Integrations

The following MCP servers and external integrations are recommended for development and demonstration purposes.

### Recommended MCP Servers

| Server | Purpose | Use Case | Repository |
|--------|---------|----------|------------|
| **Filesystem MCP** | Local file access | Read regulatory PDFs, configuration files, evaluation data | [modelcontextprotocol/servers/filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) |
| **PostgreSQL MCP** | Database queries | Inspect conversational memory, debug thread history, run ad-hoc SQL | [modelcontextprotocol/servers/postgres](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) |
| **Brave Search MCP** | Web search | Agent tool: search for regulatory updates not in the local knowledge base | [modelcontextprotocol/servers/brave-search](https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search) |
| **Memory MCP** | Knowledge graph memory | Alternative or supplement to custom memory layer for prototyping | [modelcontextprotocol/servers/memory](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) |

### Integration Notes

- **Ollama:** Local LLM server. No MCP server needed — direct HTTP API via `httpx` or the `ollama` Python package.
- **Weaviate:** Vector database. No MCP server — use the Python client directly.
- **Phoenix (Arize):** Observability. No MCP server — use the OpenTelemetry SDK.

---

## README Structure (Portfolio)

The final README.md must include these sections in order:

1. **Project Title + One-Line Description**
2. **Problem:** What gap does this project address? (stateless RAG limitations)
3. **Approach:** How does the agent solve it? (structured memory + LangGraph orchestration)
4. **Architecture:** ASCII diagram (from this roadmap) + brief component descriptions
5. **Key Features:** Bullet list of capabilities
6. **Evaluation Results:** Table comparing baseline vs memory-aware on all five metrics
7. **Quick Start:** `docker compose up -d` → `streamlit run app/main.py` (3 commands max)
8. **Tech Stack:** Table of technologies with versions
9. **Repository Structure:** Abbreviated directory tree with inline descriptions
10. **Lessons Learned:** 3–5 concise takeaways from the project
11. **License**

---

## Dependency Summary

### Production Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >=0.2 | Agent orchestration |
| `langchain-core` | >=0.3 | Message types, tool interfaces |
| `psycopg[binary]` | >=3.1 | PostgreSQL driver |
| `sqlalchemy` | >=2.0 | ORM for conversational memory |
| `alembic` | >=1.13 | Database migrations |
| `weaviate-client` | >=4.0 | Vector database client |
| `ollama` | >=0.3 | Local LLM client |
| `beautifulsoup4` | >=4.12 | Web scraping |
| `httpx` | >=0.27 | HTTP client |
| `pydantic-settings` | >=2.0 | Configuration management |
| `structlog` | >=24.0 | Structured logging |
| `opentelemetry-api` | >=1.20 | Tracing |
| `streamlit` | >=1.38 | UI framework |

### Development Dependencies (`requirements-dev.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=8.0 | Test framework |
| `pytest-cov` | >=5.0 | Coverage reporting |
| `pytest-asyncio` | >=0.23 | Async test support |
| `ruff` | >=0.6 | Linting and formatting |
| `mypy` | >=1.11 | Type checking |
| `pre-commit` | >=3.8 | Git hooks |

---

## Verification Commands

```bash
# Phase 0 — Bootstrap works
docker compose up -d
python -c "from src.config import get_settings; print(get_settings())"
pytest -m "not slow and not integration" -v
make lint

# Phase 1 — Memory layer operational
alembic upgrade head
pytest tests/unit/test_conversational_memory.py -v
pytest tests/unit/test_semantic_memory.py -v
pytest tests/integration/test_memory_roundtrip.py -v -m integration

# Phase 2 — Memory Manager unified
pytest tests/unit/test_memory_manager.py -v
pytest tests/unit/test_context_builder.py -v

# Phase 3 — Agent responds
python -c "from src.agent.graph import run_agent; print(run_agent('What is PIX?'))"
pytest tests/integration/test_agent_graph.py -v -m integration

# Phase 4 — Ingestion works
python scripts/run_ingestion.py --source pdf
python scripts/run_ingestion.py --source web

# Phase 5 — Sessions generated
python -m src.simulation.session_generator
cat data/evaluation/synthetic_sessions.json | python -m json.tool | head -50

# Phase 6 — Evaluation runs
python scripts/run_evaluation.py
cat data/evaluation/report.md

# Phase 7 — UI launches
streamlit run app/main.py

# Final — Full validation
make lint
make test
python scripts/run_evaluation.py
```
