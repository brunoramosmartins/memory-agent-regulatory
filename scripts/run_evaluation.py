"""CLI script to run the full evaluation pipeline.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --sessions data/evaluation/synthetic_sessions.json
    python scripts/run_evaluation.py --output data/evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.report import save_report
from src.evaluation.runner import EvaluationRunner, PipelineConfig
from src.simulation.session_generator import generate_sessions, load_sessions


def _make_baseline_fn():
    """Create a simple baseline pipeline (stateless, no memory)."""

    def run(query: str, thread_id: str) -> str:
        # Placeholder: in production, this calls the LLM without memory
        return f"[baseline response to: {query[:50]}]"

    return run


def _make_memory_fn():
    """Create a memory-aware pipeline function."""

    def run(query: str, thread_id: str) -> str:
        # Placeholder: in production, this runs through the full agent graph
        return f"[memory-aware response to: {query[:50]}]"

    return run


def _progress(pipeline: str, current: int, total: int) -> None:
    print(f"  [{pipeline}] {current}/{total}", end="\r")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation: baseline vs memory-aware comparison"
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="data/evaluation/synthetic_sessions.json",
        help="Path to synthetic sessions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate sessions if file doesn't exist",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Sessions per topic if generating (default: 10)",
    )

    args = parser.parse_args()

    # Load or generate sessions
    sessions_path = Path(args.sessions)
    if sessions_path.exists():
        print(f"Loading sessions from {sessions_path}...")
        sessions = load_sessions(str(sessions_path))
    elif args.generate:
        print(f"Generating sessions (count_per_topic={args.count})...")
        sessions = generate_sessions(count_per_topic=args.count, seed=42)
    else:
        print(f"Sessions file not found: {sessions_path}")
        print("Run with --generate to create sessions, or use scripts/generate_sessions.py first")
        sys.exit(1)

    print(f"Loaded {len(sessions)} sessions")

    # Configure pipelines
    baseline = PipelineConfig(name="baseline", run_fn=_make_baseline_fn())
    memory = PipelineConfig(name="memory", run_fn=_make_memory_fn())

    # Run evaluation
    runner = EvaluationRunner(
        baseline=baseline,
        memory=memory,
        sessions=sessions,
    )

    print("\nRunning evaluation...")
    result = runner.run(on_progress=_progress)

    # Save reports
    print("\n\nSaving reports...")
    paths = save_report(result, output_dir=args.output)
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON:     {paths['json']}")

    # Print summary
    print("\n--- Summary ---")
    for metric_name, data in result.comparison.items():
        value = data.get("value", "N/A")
        if isinstance(value, dict):
            print(f"  {metric_name}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {metric_name}: {value}")

    print(f"\nTotal sessions: {result.metadata.get('total_sessions', 0)}")
    print("Done.")


if __name__ == "__main__":
    main()
