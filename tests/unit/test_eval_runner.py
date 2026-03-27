"""Tests for evaluation runner."""

from src.evaluation.models import EvaluationResult
from src.evaluation.runner import (
    EvaluationRunner,
    PipelineConfig,
    _estimate_tokens,
    _run_session,
)
from src.simulation.models import Session, Turn

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_session(num_user_turns: int = 3) -> Session:
    turns = []
    for i in range(num_user_turns):
        turns.append(Turn(role="user", content=f"Question {i}?", expected_behavior="direct"))
        turns.append(Turn(role="assistant", content=f"Answer {i}.", expected_behavior="answer"))
    return Session(session_id="test-s1", topic="pix_fees", turns=turns)


def _baseline_fn(query: str, thread_id: str) -> str:
    return f"Baseline: {query[:20]}"


def _memory_fn(query: str, thread_id: str) -> str:
    return f"Memory: {query[:20]}"


def _error_fn(query: str, thread_id: str) -> str:
    raise RuntimeError("pipeline error")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("hello world") >= 1

    def test_empty(self):
        assert _estimate_tokens("") == 1  # min 1

    def test_long_text(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


class TestRunSession:
    def test_runs_user_turns_only(self):
        session = _make_session(num_user_turns=3)
        pipeline = PipelineConfig(name="baseline", run_fn=_baseline_fn)
        log = _run_session(session, pipeline)
        assert len(log.turns) == 3  # only user turns
        assert log.pipeline == "baseline"

    def test_collects_latency(self):
        session = _make_session(num_user_turns=2)
        pipeline = PipelineConfig(name="test", run_fn=_baseline_fn)
        log = _run_session(session, pipeline)
        for turn in log.turns:
            assert turn.latency_ms >= 0

    def test_handles_pipeline_error(self):
        session = _make_session(num_user_turns=1)
        pipeline = PipelineConfig(name="error", run_fn=_error_fn)
        log = _run_session(session, pipeline)
        assert len(log.turns) == 1
        assert "[error:" in log.turns[0].response

    def test_custom_token_counter(self):
        session = _make_session(num_user_turns=1)
        pipeline = PipelineConfig(
            name="test",
            run_fn=_baseline_fn,
            token_counter=lambda t: 42,
        )
        log = _run_session(session, pipeline)
        assert log.turns[0].tokens_used == 84  # query + response

    def test_memory_pipeline_has_memory_reads(self):
        session = _make_session(num_user_turns=2)
        pipeline = PipelineConfig(name="memory", run_fn=_memory_fn)
        log = _run_session(session, pipeline)
        for turn in log.turns:
            assert turn.memory_reads == 1


class TestEvaluationRunner:
    def test_run_returns_result(self):
        sessions = [_make_session(num_user_turns=2)]
        runner = EvaluationRunner(
            baseline=PipelineConfig(name="baseline", run_fn=_baseline_fn),
            memory=PipelineConfig(name="memory", run_fn=_memory_fn),
            sessions=sessions,
        )
        result = runner.run()
        assert isinstance(result, EvaluationResult)
        assert result.metadata["total_sessions"] == 1

    def test_all_metrics_present(self):
        sessions = [_make_session()]
        runner = EvaluationRunner(
            baseline=PipelineConfig(name="baseline", run_fn=_baseline_fn),
            memory=PipelineConfig(name="memory", run_fn=_memory_fn),
            sessions=sessions,
        )
        result = runner.run()
        metric_names = {m.name for m in result.memory_metrics}
        assert "consistency_score" in metric_names
        assert "context_reuse_rate" in metric_names
        assert "token_efficiency" in metric_names
        assert "retrieval_precision" in metric_names
        assert "latency_impact" in metric_names

    def test_per_session_details(self):
        sessions = [_make_session(), _make_session()]
        sessions[1].session_id = "test-s2"
        runner = EvaluationRunner(
            baseline=PipelineConfig(name="baseline", run_fn=_baseline_fn),
            memory=PipelineConfig(name="memory", run_fn=_memory_fn),
            sessions=sessions,
        )
        result = runner.run()
        assert len(result.per_session) == 2

    def test_comparison_has_all_metrics(self):
        sessions = [_make_session()]
        runner = EvaluationRunner(
            baseline=PipelineConfig(name="baseline", run_fn=_baseline_fn),
            memory=PipelineConfig(name="memory", run_fn=_memory_fn),
            sessions=sessions,
        )
        result = runner.run()
        assert "consistency_score" in result.comparison
        assert "token_efficiency" in result.comparison
        assert "latency_impact" in result.comparison

    def test_progress_callback(self):
        calls = []

        def on_progress(name, current, total):
            calls.append((name, current, total))

        sessions = [_make_session()]
        runner = EvaluationRunner(
            baseline=PipelineConfig(name="baseline", run_fn=_baseline_fn),
            memory=PipelineConfig(name="memory", run_fn=_memory_fn),
            sessions=sessions,
        )
        runner.run(on_progress=on_progress)
        assert len(calls) == 2  # 1 session × 2 pipelines
        assert calls[0][0] == "baseline"
        assert calls[1][0] == "memory"

    def test_multiple_sessions(self):
        sessions = []
        for i in range(5):
            s = _make_session(num_user_turns=2)
            s.session_id = f"test-{i}"
            sessions.append(s)
        runner = EvaluationRunner(
            baseline=PipelineConfig(name="baseline", run_fn=_baseline_fn),
            memory=PipelineConfig(name="memory", run_fn=_memory_fn),
            sessions=sessions,
        )
        result = runner.run()
        assert result.metadata["total_sessions"] == 5
        assert len(result.per_session) == 5
