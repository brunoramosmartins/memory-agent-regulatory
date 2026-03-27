"""Report generator — Markdown and JSON outputs for evaluation results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.models import EvaluationResult, MetricResult


def _format_metric_value(value: float | dict[str, float]) -> str:
    """Format a metric value for display."""
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items()]
        return " | ".join(parts)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_to_row(metric: MetricResult) -> str:
    """Convert a MetricResult to a Markdown table row."""
    return f"| {metric.name} | {_format_metric_value(metric.value)} |"


def generate_markdown(result: EvaluationResult) -> str:
    """Generate a Markdown report from evaluation results.

    Args:
        result: Complete evaluation result.

    Returns:
        Portfolio-quality Markdown string.
    """
    lines: list[str] = []
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"> Generated: {timestamp}")
    lines.append(f"> Sessions evaluated: {result.metadata.get('total_sessions', 0)}")
    lines.append("")

    # Summary comparison table
    lines.append("## Metric Comparison")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    for metric_name, data in result.comparison.items():
        value = data.get("value", "N/A")
        formatted = _format_metric_value(value)
        lines.append(f"| {metric_name} | {formatted} |")

    lines.append("")

    # Baseline summary
    lines.append("## Baseline Pipeline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for m in result.baseline_metrics:
        lines.append(_metric_to_row(m))
    lines.append("")

    # Memory-aware summary
    lines.append("## Memory-Aware Pipeline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for m in result.memory_metrics:
        lines.append(_metric_to_row(m))
    lines.append("")

    # Per-session table (first 20)
    if result.per_session:
        lines.append("## Per-Session Details")
        lines.append("")
        lines.append(
            "| Session | Topic | BL Tokens | Mem Tokens"
            " | BL Latency (ms) | Mem Latency (ms) |"
        )
        lines.append(
            "|---------|-------|-----------|-----------"
            "|-----------------|-------------------|"
        )
        for ps in result.per_session[:20]:
            lines.append(
                f"| {ps['session_id'][:8]}... | {ps['topic']} | "
                f"{ps['baseline_tokens']} | {ps['memory_tokens']} | "
                f"{ps['baseline_avg_latency']} | {ps['memory_avg_latency']} |"
            )
        if len(result.per_session) > 20:
            lines.append(f"| ... | ({len(result.per_session) - 20} more) | | | | |")
        lines.append("")

    # Methodology note
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- **Consistency Score**: Pairwise cosine similarity"
        " of responses grouped by topic"
    )
    lines.append(
        "- **Context Reuse Rate**: Fraction of turns where"
        " memory was read and influenced response"
    )
    lines.append(
        "- **Token Efficiency**: Ratio of baseline tokens"
        " / memory tokens (>1.0 = memory saves tokens)"
    )
    lines.append(
        "- **Retrieval Precision**: Relevant documents"
        " / retrieved documents per query, averaged"
    )
    lines.append(
        "- **Latency Impact**: Avg, P50, P95 latency"
        " comparison between pipelines"
    )
    lines.append("")

    return "\n".join(lines)


def generate_json(result: EvaluationResult) -> dict[str, Any]:
    """Generate a JSON-serializable dict from evaluation results.

    Args:
        result: Complete evaluation result.

    Returns:
        Dict suitable for json.dumps().
    """
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "metadata": result.metadata,
        "comparison": result.comparison,
        "baseline_metrics": [
            {"name": m.name, "value": m.value, "details": m.details}
            for m in result.baseline_metrics
        ],
        "memory_metrics": [
            {"name": m.name, "value": m.value, "details": m.details}
            for m in result.memory_metrics
        ],
        "per_session": result.per_session,
    }


def save_report(result: EvaluationResult, output_dir: str = "data/evaluation") -> dict[str, str]:
    """Save evaluation report in both Markdown and JSON formats.

    Args:
        result: Complete evaluation result.
        output_dir: Directory to save reports.

    Returns:
        Dict with paths to saved files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "report.md"
    json_path = out / "report.json"

    md_content = generate_markdown(result)
    md_path.write_text(md_content, encoding="utf-8")

    json_content = generate_json(result)
    json_path.write_text(
        json.dumps(json_content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"markdown": str(md_path), "json": str(json_path)}
