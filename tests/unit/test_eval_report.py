"""Tests for evaluation report generation."""

import json
import os
import tempfile

from src.evaluation.models import EvaluationResult, MetricResult
from src.evaluation.report import generate_json, generate_markdown, save_report

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_result() -> EvaluationResult:
    return EvaluationResult(
        baseline_metrics=[
            MetricResult(name="total_tokens", value=5000),
            MetricResult(name="avg_latency", value=120.5),
        ],
        memory_metrics=[
            MetricResult(name="consistency_score", value=0.87),
            MetricResult(name="context_reuse_rate", value=0.65),
            MetricResult(name="token_efficiency", value=1.35),
            MetricResult(name="retrieval_precision", value=0.72),
            MetricResult(
                name="latency_impact",
                value={"baseline_avg": 100.0, "memory_avg": 130.0, "overhead_ms": 30.0},
            ),
        ],
        comparison={
            "consistency_score": {"value": 0.87, "details": {}},
            "token_efficiency": {"value": 1.35, "details": {}},
        },
        per_session=[
            {
                "session_id": "abcdef12-3456-7890",
                "topic": "pix_fees",
                "baseline_tokens": 500,
                "memory_tokens": 370,
                "baseline_avg_latency": 100.0,
                "memory_avg_latency": 130.0,
                "baseline_turns": 3,
                "memory_turns": 3,
            },
        ],
        metadata={"total_sessions": 1},
    )


# ── Markdown tests ───────────────────────────────────────────────────────────


class TestGenerateMarkdown:
    def test_contains_title(self):
        md = generate_markdown(_make_result())
        assert "# Evaluation Report" in md

    def test_contains_metrics_table(self):
        md = generate_markdown(_make_result())
        assert "| Metric | Value |" in md
        assert "consistency_score" in md

    def test_contains_baseline_section(self):
        md = generate_markdown(_make_result())
        assert "## Baseline Pipeline" in md
        assert "total_tokens" in md

    def test_contains_memory_section(self):
        md = generate_markdown(_make_result())
        assert "## Memory-Aware Pipeline" in md

    def test_contains_per_session(self):
        md = generate_markdown(_make_result())
        assert "## Per-Session Details" in md
        assert "pix_fees" in md

    def test_contains_methodology(self):
        md = generate_markdown(_make_result())
        assert "## Methodology" in md
        assert "Consistency Score" in md

    def test_dict_metric_formatted(self):
        md = generate_markdown(_make_result())
        assert "overhead_ms" in md

    def test_empty_result(self):
        md = generate_markdown(EvaluationResult())
        assert "# Evaluation Report" in md


# ── JSON tests ───────────────────────────────────────────────────────────────


class TestGenerateJson:
    def test_valid_json_structure(self):
        data = generate_json(_make_result())
        assert "generated_at" in data
        assert "metadata" in data
        assert "comparison" in data
        assert "baseline_metrics" in data
        assert "memory_metrics" in data

    def test_serializable(self):
        data = generate_json(_make_result())
        serialized = json.dumps(data)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["metadata"]["total_sessions"] == 1

    def test_metrics_in_json(self):
        data = generate_json(_make_result())
        memory_names = [m["name"] for m in data["memory_metrics"]]
        assert "consistency_score" in memory_names
        assert "token_efficiency" in memory_names

    def test_per_session_included(self):
        data = generate_json(_make_result())
        assert len(data["per_session"]) == 1
        assert data["per_session"][0]["topic"] == "pix_fees"


# ── Save report tests ────────────────────────────────────────────────────────


class TestSaveReport:
    def test_saves_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_report(_make_result(), output_dir=tmp)
            assert os.path.exists(paths["markdown"])
            assert os.path.exists(paths["json"])

    def test_markdown_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_report(_make_result(), output_dir=tmp)
            with open(paths["markdown"], encoding="utf-8") as f:
                content = f.read()
            assert "# Evaluation Report" in content

    def test_json_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_report(_make_result(), output_dir=tmp)
            with open(paths["json"], encoding="utf-8") as f:
                data = json.load(f)
            assert data["metadata"]["total_sessions"] == 1

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "nested", "dir")
            paths = save_report(_make_result(), output_dir=nested)
            assert os.path.exists(paths["markdown"])
