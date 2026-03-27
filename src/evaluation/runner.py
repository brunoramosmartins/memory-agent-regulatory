"""Evaluation runner — orchestrates baseline vs memory-aware comparison.

Runs synthetic sessions through both pipelines, collects logs, and
computes all metrics.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.models import (
    EvaluationResult,
    MetricResult,
    SessionLog,
    TurnLog,
)
from src.simulation.models import Session


@dataclass
class PipelineConfig:
    """Configuration for a pipeline to be evaluated.

    Args:
        name: Pipeline identifier ("baseline" or "memory").
        run_fn: Function that takes (query, thread_id) and returns response string.
        token_counter: Optional function to count tokens in a string.
    """

    name: str
    run_fn: Callable[[str, str], str]
    token_counter: Callable[[str], int] | None = None


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (4 chars per token)."""
    return max(1, len(text) // 4)


def _run_session(
    session: Session,
    pipeline: PipelineConfig,
) -> SessionLog:
    """Run a single session through a pipeline and collect logs.

    Args:
        session: Synthetic session with turns.
        pipeline: Pipeline configuration with run_fn.

    Returns:
        SessionLog with per-turn metrics.
    """
    token_fn = pipeline.token_counter or _estimate_tokens
    log = SessionLog(
        session_id=session.session_id,
        topic=session.topic,
        pipeline=pipeline.name,
    )

    thread_id = f"eval-{pipeline.name}-{session.session_id}"

    for turn in session.turns:
        if turn.role != "user":
            continue

        start = time.perf_counter()
        try:
            response = pipeline.run_fn(turn.content, thread_id)
        except Exception as e:
            response = f"[error: {e}]"
        elapsed_ms = (time.perf_counter() - start) * 1000

        turn_log = TurnLog(
            query=turn.content,
            response=response,
            latency_ms=round(elapsed_ms, 2),
            tokens_used=token_fn(turn.content) + token_fn(response),
            memory_reads=1 if pipeline.name == "memory" else 0,
        )
        log.turns.append(turn_log)

    return log


@dataclass
class EvaluationRunner:
    """Orchestrates evaluation of baseline vs memory-aware pipelines.

    Args:
        baseline: Baseline pipeline configuration.
        memory: Memory-aware pipeline configuration.
        sessions: List of synthetic sessions.
        embed_fn: Optional embedding function for consistency metric.
    """

    baseline: PipelineConfig
    memory: PipelineConfig
    sessions: list[Session]
    embed_fn: Callable[[str], list[float]] | None = None
    _baseline_logs: list[SessionLog] = field(default_factory=list, init=False)
    _memory_logs: list[SessionLog] = field(default_factory=list, init=False)

    def run(self, on_progress: Callable[[str, int, int], None] | None = None) -> EvaluationResult:
        """Run all sessions through both pipelines and compute metrics.

        Args:
            on_progress: Optional callback(pipeline_name, current, total).

        Returns:
            EvaluationResult with all metrics and per-session details.
        """
        total = len(self.sessions)
        self._baseline_logs.clear()
        self._memory_logs.clear()

        # Run baseline pipeline
        for i, session in enumerate(self.sessions):
            if on_progress:
                on_progress(self.baseline.name, i + 1, total)
            log = _run_session(session, self.baseline)
            self._baseline_logs.append(log)

        # Run memory-aware pipeline
        for i, session in enumerate(self.sessions):
            if on_progress:
                on_progress(self.memory.name, i + 1, total)
            log = _run_session(session, self.memory)
            self._memory_logs.append(log)

        # Compute metrics
        all_metrics = compute_all_metrics(
            self._baseline_logs,
            self._memory_logs,
            embed_fn=self.embed_fn,
        )

        # Build per-session comparison
        per_session = self._build_per_session()

        # Build result
        total_tok = sum(
            log.total_tokens for log in self._baseline_logs
        )
        avg_lat = sum(
            log.avg_latency for log in self._baseline_logs
        ) / max(len(self._baseline_logs), 1)
        baseline_metrics = [
            MetricResult(name="total_tokens", value=total_tok),
            MetricResult(name="avg_latency", value=round(avg_lat, 2)),
        ]
        memory_metrics = list(all_metrics.values())

        comparison = {
            metric_name: {
                "value": m.value,
                "details": m.details,
            }
            for metric_name, m in all_metrics.items()
        }

        return EvaluationResult(
            baseline_metrics=baseline_metrics,
            memory_metrics=memory_metrics,
            comparison=comparison,
            per_session=per_session,
            metadata={
                "total_sessions": total,
                "baseline_pipeline": self.baseline.name,
                "memory_pipeline": self.memory.name,
            },
        )

    def _build_per_session(self) -> list[dict[str, Any]]:
        """Build per-session comparison details."""
        results = []
        for bl, ml in zip(self._baseline_logs, self._memory_logs, strict=False):
            results.append({
                "session_id": bl.session_id,
                "topic": bl.topic,
                "baseline_tokens": bl.total_tokens,
                "memory_tokens": ml.total_tokens,
                "baseline_avg_latency": round(bl.avg_latency, 2),
                "memory_avg_latency": round(ml.avg_latency, 2),
                "baseline_turns": len(bl.turns),
                "memory_turns": len(ml.turns),
            })
        return results
