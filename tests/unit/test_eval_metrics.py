"""Tests for evaluation metrics."""

from src.evaluation.metrics import (
    _cosine_similarity,
    _percentile,
    _simple_embedding,
    consistency_score,
    context_reuse_rate,
    latency_impact,
    retrieval_precision,
    token_efficiency,
)
from src.evaluation.models import SessionLog, TurnLog

# ── Helper factories ─────────────────────────────────────────────────────────


def _make_session_log(
    session_id: str = "s1",
    topic: str = "pix_fees",
    pipeline: str = "memory",
    turns: list[TurnLog] | None = None,
) -> SessionLog:
    if turns is None:
        turns = [
            TurnLog(
                query="What are PIX fees?",
                response="PIX is free for individuals.",
                tokens_used=50,
                latency_ms=100.0,
            ),
            TurnLog(
                query="And for businesses?",
                response="Businesses may be charged.",
                tokens_used=40,
                latency_ms=120.0,
            ),
        ]
    return SessionLog(session_id=session_id, topic=topic, pipeline=pipeline, turns=turns)


# ── Helper function tests ────────────────────────────────────────────────────


class TestHelpers:
    def test_cosine_identical_vectors(self):
        v = [1.0, 0.0, 1.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_cosine_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_cosine_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_simple_embedding_dimension(self):
        emb = _simple_embedding("hello world", dim=64)
        assert len(emb) == 64

    def test_simple_embedding_empty(self):
        emb = _simple_embedding("", dim=32)
        assert len(emb) == 32
        assert all(v == 0.0 for v in emb)

    def test_simple_embedding_normalized(self):
        import math
        emb = _simple_embedding("test text", dim=64)
        magnitude = math.sqrt(sum(v * v for v in emb))
        assert abs(magnitude - 1.0) < 1e-6

    def test_similar_texts_have_high_similarity(self):
        e1 = _simple_embedding("PIX fees for individuals")
        e2 = _simple_embedding("PIX fees for individuals are zero")
        sim = _cosine_similarity(e1, e2)
        assert sim > 0.8

    def test_percentile_median(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(data, 50) == 30.0

    def test_percentile_p95(self):
        data = list(range(100))
        p95 = _percentile(data, 95)
        assert 94 <= p95 <= 95

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0


# ── Consistency score ────────────────────────────────────────────────────────


class TestConsistencyScore:
    def test_empty_logs(self):
        r = consistency_score([])
        assert r.name == "consistency_score"
        assert r.value == 0.0

    def test_single_session(self):
        logs = [_make_session_log()]
        r = consistency_score(logs)
        # Only 1 session per topic — pairs within same session
        assert r.name == "consistency_score"
        assert 0.0 <= r.value <= 1.0

    def test_similar_responses_high_score(self):
        resp1 = "PIX transfers are free for individuals"
        resp2 = "PIX transfers are free for individuals always"
        logs = [
            _make_session_log(session_id="s1", turns=[
                TurnLog(query="q", response=resp1, tokens_used=10),
            ]),
            _make_session_log(session_id="s2", turns=[
                TurnLog(query="q", response=resp2, tokens_used=10),
            ]),
        ]
        r = consistency_score(logs, threshold=0.8)
        assert r.value > 0.5

    def test_low_threshold_more_consistent(self):
        resp_diff = "totally different answer about compliance"
        logs = [
            _make_session_log(session_id="s1", turns=[
                TurnLog(query="q", response="yes it is free", tokens_used=10),
            ]),
            _make_session_log(session_id="s2", turns=[
                TurnLog(query="q", response=resp_diff, tokens_used=10),
            ]),
        ]
        r_low = consistency_score(logs, threshold=0.5)
        r_high = consistency_score(logs, threshold=0.99)
        assert r_low.value >= r_high.value


# ── Context reuse rate ───────────────────────────────────────────────────────


class TestContextReuseRate:
    def test_empty_logs(self):
        r = context_reuse_rate([])
        assert r.value == 0.0

    def test_all_reuse(self):
        logs = [_make_session_log(turns=[
            TurnLog(query="q1", response="r1", memory_reads=2),
            TurnLog(query="q2", response="r2", memory_reads=1),
        ])]
        r = context_reuse_rate(logs)
        assert r.value == 1.0

    def test_no_reuse(self):
        logs = [_make_session_log(turns=[
            TurnLog(query="q1", response="r1", memory_reads=0),
            TurnLog(query="q2", response="r2", memory_reads=0),
        ])]
        r = context_reuse_rate(logs)
        assert r.value == 0.0

    def test_partial_reuse(self):
        logs = [_make_session_log(turns=[
            TurnLog(query="q1", response="r1", memory_reads=1),
            TurnLog(query="q2", response="r2", memory_reads=0),
        ])]
        r = context_reuse_rate(logs)
        assert r.value == 0.5


# ── Token efficiency ─────────────────────────────────────────────────────────


class TestTokenEfficiency:
    def test_equal_tokens(self):
        bl = [_make_session_log(pipeline="baseline", turns=[
            TurnLog(query="q", response="r", tokens_used=100),
        ])]
        ml = [_make_session_log(pipeline="memory", turns=[
            TurnLog(query="q", response="r", tokens_used=100),
        ])]
        r = token_efficiency(bl, ml)
        assert r.value == 1.0

    def test_memory_saves_tokens(self):
        bl = [_make_session_log(pipeline="baseline", turns=[
            TurnLog(query="q", response="r", tokens_used=200),
        ])]
        ml = [_make_session_log(pipeline="memory", turns=[
            TurnLog(query="q", response="r", tokens_used=100),
        ])]
        r = token_efficiency(bl, ml)
        assert r.value == 2.0

    def test_zero_memory_tokens(self):
        bl = [_make_session_log(pipeline="baseline", turns=[
            TurnLog(query="q", response="r", tokens_used=100),
        ])]
        ml = [_make_session_log(pipeline="memory", turns=[])]
        r = token_efficiency(bl, ml)
        assert r.value == 0.0

    def test_empty_logs(self):
        r = token_efficiency([], [])
        assert r.value == 0.0


# ── Retrieval precision ──────────────────────────────────────────────────────


class TestRetrievalPrecision:
    def test_empty_logs(self):
        r = retrieval_precision([])
        assert r.value == 0.0

    def test_perfect_precision(self):
        logs = [_make_session_log(turns=[
            TurnLog(
                query="q",
                response="r",
                retrieved_docs=["d1", "d2"],
                relevant_docs=["d1", "d2"],
            ),
        ])]
        r = retrieval_precision(logs)
        assert r.value == 1.0

    def test_half_precision(self):
        logs = [_make_session_log(turns=[
            TurnLog(
                query="q",
                response="r",
                retrieved_docs=["d1", "d2"],
                relevant_docs=["d1"],
            ),
        ])]
        r = retrieval_precision(logs)
        assert r.value == 0.5

    def test_no_retrieved_docs_skipped(self):
        logs = [_make_session_log(turns=[
            TurnLog(query="q", response="r", retrieved_docs=[], relevant_docs=["d1"]),
        ])]
        r = retrieval_precision(logs)
        assert r.value == 0.0  # no queries with retrieval


# ── Latency impact ───────────────────────────────────────────────────────────


class TestLatencyImpact:
    def test_empty_logs(self):
        r = latency_impact([], [])
        assert r.value["baseline_avg"] == 0.0
        assert r.value["memory_avg"] == 0.0

    def test_memory_overhead(self):
        bl = [_make_session_log(pipeline="baseline", turns=[
            TurnLog(query="q", response="r", latency_ms=100.0),
        ])]
        ml = [_make_session_log(pipeline="memory", turns=[
            TurnLog(query="q", response="r", latency_ms=150.0),
        ])]
        r = latency_impact(bl, ml)
        assert r.value["overhead_ms"] == 50.0
        assert r.value["baseline_avg"] == 100.0
        assert r.value["memory_avg"] == 150.0

    def test_percentiles(self):
        turns = [
            TurnLog(query="q", response="r", latency_ms=float(i))
            for i in range(1, 101)
        ]
        bl = [_make_session_log(pipeline="baseline", turns=turns)]
        ml = [_make_session_log(pipeline="memory", turns=turns)]
        r = latency_impact(bl, ml)
        assert r.value["baseline_p50"] > 0
        assert r.value["baseline_p95"] > r.value["baseline_p50"]
