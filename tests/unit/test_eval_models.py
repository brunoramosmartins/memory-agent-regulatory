"""Tests for evaluation data models."""

from src.evaluation.models import (
    EvaluationResult,
    MetricResult,
    SessionLog,
    TurnLog,
)


class TestTurnLog:
    def test_defaults(self):
        t = TurnLog(query="q", response="r")
        assert t.latency_ms == 0.0
        assert t.tokens_used == 0
        assert t.memory_reads == 0
        assert t.retrieved_docs == []
        assert t.relevant_docs == []

    def test_with_values(self):
        t = TurnLog(
            query="q",
            response="r",
            latency_ms=50.0,
            tokens_used=100,
            memory_reads=3,
            retrieved_docs=["d1", "d2"],
            relevant_docs=["d1"],
        )
        assert t.latency_ms == 50.0
        assert t.memory_reads == 3


class TestSessionLog:
    def test_empty_session(self):
        s = SessionLog(session_id="s1", topic="t1")
        assert s.total_tokens == 0
        assert s.avg_latency == 0.0

    def test_total_tokens(self):
        s = SessionLog(
            session_id="s1",
            topic="t1",
            turns=[
                TurnLog(query="q1", response="r1", tokens_used=50),
                TurnLog(query="q2", response="r2", tokens_used=30),
            ],
        )
        assert s.total_tokens == 80

    def test_avg_latency(self):
        s = SessionLog(
            session_id="s1",
            topic="t1",
            turns=[
                TurnLog(query="q1", response="r1", latency_ms=100.0),
                TurnLog(query="q2", response="r2", latency_ms=200.0),
            ],
        )
        assert s.avg_latency == 150.0


class TestMetricResult:
    def test_float_value(self):
        m = MetricResult(name="score", value=0.85)
        assert m.value == 0.85

    def test_dict_value(self):
        m = MetricResult(name="latency", value={"avg": 50.0, "p95": 100.0})
        assert m.value["avg"] == 50.0

    def test_with_details(self):
        m = MetricResult(name="score", value=0.9, details={"per_topic": {"t1": 0.8}})
        assert m.details["per_topic"]["t1"] == 0.8


class TestEvaluationResult:
    def test_get_metric(self):
        r = EvaluationResult(
            memory_metrics=[
                MetricResult(name="consistency_score", value=0.9),
                MetricResult(name="token_efficiency", value=1.2),
            ],
            baseline_metrics=[
                MetricResult(name="total_tokens", value=5000),
            ],
        )
        assert r.get_metric("consistency_score", "memory").value == 0.9
        assert r.get_metric("total_tokens", "baseline").value == 5000
        assert r.get_metric("nonexistent") is None
