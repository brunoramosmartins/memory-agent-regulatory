"""Data models for evaluation logs, metrics, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnLog:
    """Log entry for a single turn in an evaluation session."""

    query: str
    response: str
    latency_ms: float = 0.0
    tokens_used: int = 0
    memory_reads: int = 0
    retrieved_docs: list[str] = field(default_factory=list)
    relevant_docs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionLog:
    """Log for a complete evaluation session."""

    session_id: str
    topic: str
    turns: list[TurnLog] = field(default_factory=list)
    pipeline: str = ""  # "baseline" or "memory"

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens_used for t in self.turns)

    @property
    def avg_latency(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.latency_ms for t in self.turns) / len(self.turns)


@dataclass
class MetricResult:
    """Result of a single metric computation."""

    name: str
    value: float | dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result comparing baseline vs memory-aware."""

    baseline_metrics: list[MetricResult] = field(default_factory=list)
    memory_metrics: list[MetricResult] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    per_session: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metric(self, name: str, pipeline: str = "memory") -> MetricResult | None:
        """Get a specific metric by name and pipeline."""
        metrics = self.memory_metrics if pipeline == "memory" else self.baseline_metrics
        for m in metrics:
            if m.name == name:
                return m
        return None
