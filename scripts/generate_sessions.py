"""CLI script to generate synthetic sessions for evaluation.

Usage:
    python scripts/generate_sessions.py
    python scripts/generate_sessions.py --count 20 --seed 123
    python scripts/generate_sessions.py --topics pix_fees pix_compliance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.session_generator import generate_sessions, save_sessions
from src.simulation.validator import session_stats, validate_sessions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-turn sessions for evaluation"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of sessions per topic (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Topics to generate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/synthetic_sessions.json",
        help="Output file path",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=3,
        help="Minimum user turns per session (default: 3)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum user turns per session (default: 8)",
    )

    args = parser.parse_args()

    print(f"Generating sessions (seed={args.seed}, count_per_topic={args.count})...")

    sessions = generate_sessions(
        topics=args.topics,
        count_per_topic=args.count,
        turns_range=(args.min_turns, args.max_turns),
        seed=args.seed,
    )

    # Validate
    errors = validate_sessions(sessions)
    if errors:
        print(f"\n⚠ {len(errors)} sessions have validation issues:")
        for sid, errs in errors.items():
            print(f"  {sid}: {', '.join(errs)}")
    else:
        print("All sessions passed validation.")

    # Stats
    stats = session_stats(sessions)
    print("\nStatistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    # Save
    save_sessions(sessions, args.output)
    print(f"\nSaved {len(sessions)} sessions to {args.output}")


if __name__ == "__main__":
    main()
