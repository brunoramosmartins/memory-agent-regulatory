"""Evaluation metrics — pure functions over structured logs.

Five core metrics:
1. consistency_score — topic-grouped pairwise response similarity
2. context_reuse_rate — fraction of turns that reference memory content
3. token_efficiency — baseline tokens / memory tokens ratio
4. retrieval_precision — relevant / retrieved docs averaged per query
5. latency_impact — latency comparison (avg, p50, p95)
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from src.evaluation.models import MetricResult, SessionLog

# ── Helpers ──────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _simple_embedding(text: str, dim: int = 64) -> list[float]:
    """Create a simple character-frequency embedding for text similarity.

    This is a lightweight fallback when no real embedding model is available.
    For production use, replace with actual embeddings from the model.
    """
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    for ch in text.lower():
        idx = ord(ch) % dim
        vec[idx] += 1.0
    # Normalize
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


def _percentile(data: list[float], pct: float) -> float:
    """Compute percentile value from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# ── Metric functions ─────────────────────────────────────────────────────────


def consistency_score(
    session_logs: list[SessionLog],
    embed_fn: callable | None = None,
    threshold: float = 0.85,
) -> MetricResult:
    """Compute cross-session consistency for same-topic responses.

    Groups responses by topic, computes pairwise embedding similarity,
    and returns the fraction of pairs above the threshold.

    Args:
        session_logs: List of session execution logs.
        embed_fn: Optional embedding function (text -> list[float]).
                  Falls back to simple character-frequency embedding.
        threshold: Cosine similarity threshold for "consistent" (default 0.85).

    Returns:
        MetricResult with consistency score (0.0 - 1.0).
    """
    if not session_logs:
        return MetricResult(name="consistency_score", value=0.0)

    embed = embed_fn or _simple_embedding

    # Group responses by topic
    topic_responses: dict[str, list[str]] = defaultdict(list)
    for log in session_logs:
        for turn in log.turns:
            if turn.response:
                topic_responses[log.topic].append(turn.response)

    total_pairs = 0
    consistent_pairs = 0
    per_topic: dict[str, float] = {}

    for topic, responses in topic_responses.items():
        if len(responses) < 2:
            per_topic[topic] = 1.0
            continue

        embeddings = [embed(r) for r in responses]
        topic_total = 0
        topic_consistent = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                topic_total += 1
                total_pairs += 1
                if sim >= threshold:
                    topic_consistent += 1
                    consistent_pairs += 1

        per_topic[topic] = topic_consistent / topic_total if topic_total > 0 else 0.0

    score = consistent_pairs / total_pairs if total_pairs > 0 else 0.0
    return MetricResult(
        name="consistency_score",
        value=round(score, 4),
        details={"per_topic": per_topic, "threshold": threshold},
    )


def context_reuse_rate(session_logs: list[SessionLog]) -> MetricResult:
    """Compute fraction of turns where memory was read and influenced the response.

    A turn counts as "reuse" if memory_reads > 0.

    Returns:
        MetricResult with reuse rate (0.0 - 1.0).
    """
    if not session_logs:
        return MetricResult(name="context_reuse_rate", value=0.0)

    total_turns = 0
    reuse_turns = 0

    for log in session_logs:
        for turn in log.turns:
            total_turns += 1
            if turn.memory_reads > 0:
                reuse_turns += 1

    rate = reuse_turns / total_turns if total_turns > 0 else 0.0
    return MetricResult(
        name="context_reuse_rate",
        value=round(rate, 4),
        details={"reuse_turns": reuse_turns, "total_turns": total_turns},
    )


def token_efficiency(
    baseline_logs: list[SessionLog],
    memory_logs: list[SessionLog],
) -> MetricResult:
    """Compute token efficiency ratio: baseline_tokens / memory_tokens.

    A ratio > 1.0 means the memory-aware pipeline uses fewer tokens.

    Returns:
        MetricResult with efficiency ratio.
    """
    baseline_tokens = sum(log.total_tokens for log in baseline_logs)
    memory_tokens = sum(log.total_tokens for log in memory_logs)

    ratio = 0.0 if memory_tokens == 0 else baseline_tokens / memory_tokens

    return MetricResult(
        name="token_efficiency",
        value=round(ratio, 4),
        details={
            "baseline_tokens": baseline_tokens,
            "memory_tokens": memory_tokens,
        },
    )


def retrieval_precision(session_logs: list[SessionLog]) -> MetricResult:
    """Compute average precision: relevant_docs / retrieved_docs per query.

    Returns:
        MetricResult with average precision (0.0 - 1.0).
    """
    if not session_logs:
        return MetricResult(name="retrieval_precision", value=0.0)

    precisions: list[float] = []
    for log in session_logs:
        for turn in log.turns:
            if turn.retrieved_docs:
                relevant = set(turn.relevant_docs)
                retrieved = set(turn.retrieved_docs)
                if retrieved:
                    precision = len(relevant & retrieved) / len(retrieved)
                    precisions.append(precision)

    avg = statistics.mean(precisions) if precisions else 0.0
    return MetricResult(
        name="retrieval_precision",
        value=round(avg, 4),
        details={
            "num_queries_with_retrieval": len(precisions),
            "precisions": [round(p, 4) for p in precisions[:20]],  # cap detail
        },
    )


def latency_impact(
    baseline_logs: list[SessionLog],
    memory_logs: list[SessionLog],
) -> MetricResult:
    """Compare latency between baseline and memory-aware pipelines.

    Returns:
        MetricResult with dict value: {baseline_avg, memory_avg, baseline_p50, ...}.
    """
    baseline_lats = [t.latency_ms for log in baseline_logs for t in log.turns]
    memory_lats = [t.latency_ms for log in memory_logs for t in log.turns]

    def _stats(lats: list[float]) -> dict[str, float]:
        if not lats:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
        return {
            "avg": round(statistics.mean(lats), 2),
            "p50": round(_percentile(lats, 50), 2),
            "p95": round(_percentile(lats, 95), 2),
        }

    baseline_stats = _stats(baseline_lats)
    memory_stats = _stats(memory_lats)

    overhead = memory_stats["avg"] - baseline_stats["avg"]

    return MetricResult(
        name="latency_impact",
        value={
            "baseline_avg": baseline_stats["avg"],
            "baseline_p50": baseline_stats["p50"],
            "baseline_p95": baseline_stats["p95"],
            "memory_avg": memory_stats["avg"],
            "memory_p50": memory_stats["p50"],
            "memory_p95": memory_stats["p95"],
            "overhead_ms": round(overhead, 2),
        },
        details={
            "baseline_turns": len(baseline_lats),
            "memory_turns": len(memory_lats),
        },
    )


# ── Convenience ──────────────────────────────────────────────────────────────


def compute_all_metrics(
    baseline_logs: list[SessionLog],
    memory_logs: list[SessionLog],
    embed_fn: callable | None = None,
) -> dict[str, MetricResult]:
    """Compute all five metrics and return as a dict keyed by metric name."""
    return {
        "consistency_score": consistency_score(memory_logs, embed_fn=embed_fn),
        "context_reuse_rate": context_reuse_rate(memory_logs),
        "token_efficiency": token_efficiency(baseline_logs, memory_logs),
        "retrieval_precision": retrieval_precision(memory_logs),
        "latency_impact": latency_impact(baseline_logs, memory_logs),
    }
