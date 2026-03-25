.PHONY: help up down test test-unit test-integration test-all lint format typecheck

PYTHON := python
PYTEST := pytest

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Memory Agent Regulatory — Development Commands"
	@echo ""
	@echo "  Infrastructure"
	@echo "    make up                Start Weaviate + PostgreSQL via Docker Compose"
	@echo "    make down              Stop all Docker services"
	@echo ""
	@echo "  Quality"
	@echo "    make test              Run unit tests (excludes slow/integration)"
	@echo "    make test-unit         Alias for make test"
	@echo "    make test-integration  Run integration tests only"
	@echo "    make test-all          Run all tests"
	@echo "    make lint              Check code style with ruff"
	@echo "    make format            Auto-format code with ruff"
	@echo "    make typecheck         Run mypy type checker"

# ── Infrastructure ────────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

# ── Quality ───────────────────────────────────────────────────────────────────

test:
	$(PYTEST) -m "not slow and not integration" -v

test-unit: test

test-integration:
	$(PYTEST) -m "integration" -v

test-all:
	$(PYTEST) -v

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/
