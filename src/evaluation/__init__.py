"""Evaluation engine — metrics, runner, and report generation."""

from src.evaluation.metrics import (
    compute_all_metrics,
    consistency_score,
    context_reuse_rate,
    latency_impact,
    retrieval_precision,
    token_efficiency,
)
from src.evaluation.models import (
    EvaluationResult,
    MetricResult,
    SessionLog,
    TurnLog,
)
from src.evaluation.report import generate_json, generate_markdown, save_report
from src.evaluation.runner import EvaluationRunner, PipelineConfig

__all__ = [
    "EvaluationResult",
    "EvaluationRunner",
    "MetricResult",
    "PipelineConfig",
    "SessionLog",
    "TurnLog",
    "compute_all_metrics",
    "consistency_score",
    "context_reuse_rate",
    "generate_json",
    "generate_markdown",
    "latency_impact",
    "retrieval_precision",
    "save_report",
    "token_efficiency",
]
