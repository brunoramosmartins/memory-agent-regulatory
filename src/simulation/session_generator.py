"""Generate synthetic multi-turn sessions for evaluation.

Supports two modes:
1. Template-based (default): deterministic, no LLM required
2. LLM-assisted: uses Ollama to generate more natural sessions (future)
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Any

from src.simulation.models import Session, Turn

# ── Regulatory topics and templates ──────────────────────────────────────────

TOPICS: dict[str, dict[str, Any]] = {
    "pix_fees": {
        "description": "PIX transaction fees, costs, and pricing rules",
        "seed_questions": [
            "What are the fees for PIX transactions?",
            "Are PIX transfers free for individuals?",
            "Can banks charge fees for PIX?",
            "What is the fee policy for businesses using PIX?",
            "Are there any hidden costs in PIX transactions?",
        ],
        "follow_ups": [
            "What about for businesses?",
            "Does that apply to all bank types?",
            "Is there a limit on those fees?",
            "And what about the fees for receiving payments?",
            "Has that changed recently?",
        ],
        "implicit_refs": [
            "And what about the deadline for that?",
            "Is that the same for credit unions?",
            "Can you explain that in more detail?",
            "What happens if they exceed that amount?",
            "Are there exceptions to that rule?",
        ],
    },
    "pix_deadlines": {
        "description": "PIX regulatory deadlines, compliance timelines",
        "seed_questions": [
            "What are the deadlines for PIX compliance?",
            "When must institutions implement PIX instant payments?",
            "What is the timeline for PIX regulatory updates?",
            "Are there penalties for missing PIX deadlines?",
            "When was the latest PIX regulation published?",
        ],
        "follow_ups": [
            "What happens after that deadline?",
            "Is there an extension available?",
            "Does that apply to smaller institutions?",
            "What was the previous deadline?",
            "How does that compare to international standards?",
        ],
        "implicit_refs": [
            "And what are the consequences of that?",
            "Does the same rule apply here?",
            "Can you clarify that timeline?",
            "What about the second phase?",
            "Is there any flexibility on that?",
        ],
    },
    "pix_compliance": {
        "description": "PIX compliance requirements, security rules, reporting",
        "seed_questions": [
            "What are the security requirements for PIX?",
            "How should institutions report PIX fraud?",
            "What compliance certifications are needed for PIX?",
            "What are the anti-fraud measures required for PIX?",
            "How does PIX handle data protection?",
        ],
        "follow_ups": [
            "What documentation is required?",
            "How often must reports be submitted?",
            "What are the penalties for non-compliance?",
            "Is there a grace period for new participants?",
            "Who audits these requirements?",
        ],
        "implicit_refs": [
            "And how does that affect the reporting?",
            "Is that mandatory or recommended?",
            "Can smaller institutions be exempt from that?",
            "What if they already have that certification?",
            "Does that cover international transfers too?",
        ],
    },
    "pix_comparisons": {
        "description": "PIX vs other payment methods, regulatory comparisons",
        "seed_questions": [
            "How does PIX compare to TED and DOC?",
            "What advantages does PIX have over credit cards?",
            "How does PIX regulation compare to international instant payments?",
            "What are the differences between PIX and boleto?",
            "Is PIX more regulated than other payment methods?",
        ],
        "follow_ups": [
            "Which one is cheaper for businesses?",
            "And in terms of settlement time?",
            "Does that advantage hold for large transactions?",
            "What about from a security perspective?",
            "Is there a transaction limit difference?",
        ],
        "implicit_refs": [
            "And what about the other one?",
            "Does that difference really matter in practice?",
            "Can you quantify that advantage?",
            "Is that expected to change?",
            "How does that affect consumers?",
        ],
    },
    "cross_topic": {
        "description": "Questions spanning multiple regulatory areas",
        "seed_questions": [
            "Give me an overview of PIX regulations.",
            "What should a new fintech know about PIX?",
            "What are the most important PIX rules for a payment institution?",
            "How has PIX regulation evolved since launch?",
            "What are the upcoming changes to PIX rules?",
        ],
        "follow_ups": [
            "What about the fee structure specifically?",
            "And the compliance deadlines?",
            "How does security fit into that?",
            "Can you compare that to the previous version?",
            "What are the enforcement mechanisms?",
        ],
        "implicit_refs": [
            "Going back to the first point, can you elaborate?",
            "And how does that relate to what you mentioned earlier?",
            "Is that connected to the compliance requirements?",
            "Does that fee structure apply in that case too?",
            "Can you tie those points together?",
        ],
    },
}


def _make_session_id(topic: str, index: int, seed: int) -> str:
    """Create a deterministic, short session ID."""
    raw = f"{topic}-{index}-{seed}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _build_turns(
    topic_data: dict[str, Any],
    num_turns: int,
    rng: random.Random,
) -> list[Turn]:
    """Build a list of turns with mixed behaviors."""
    turns: list[Turn] = []

    # First turn is always a direct question
    seed_q = rng.choice(topic_data["seed_questions"])
    turns.append(Turn(role="user", content=seed_q, expected_behavior="direct"))
    turns.append(
        Turn(
            role="assistant",
            content="[agent response placeholder]",
            expected_behavior="answer",
        )
    )

    user_turns_needed = num_turns - 1  # first user turn already added
    has_implicit = False

    for i in range(user_turns_needed):
        # Decide behavior: at least one implicit reference per session
        if i == user_turns_needed - 1 and not has_implicit:
            behavior = "implicit_reference"
        else:
            behavior = rng.choice(
                ["follow_up", "follow_up", "implicit_reference", "direct"]
            )

        if behavior == "implicit_reference":
            content = rng.choice(topic_data["implicit_refs"])
            has_implicit = True
        elif behavior == "follow_up":
            content = rng.choice(topic_data["follow_ups"])
        else:
            content = rng.choice(topic_data["seed_questions"])

        turns.append(Turn(role="user", content=content, expected_behavior=behavior))
        turns.append(
            Turn(
                role="assistant",
                content="[agent response placeholder]",
                expected_behavior="answer",
            )
        )

    return turns


def generate_sessions(
    topics: list[str] | None = None,
    count_per_topic: int = 10,
    turns_range: tuple[int, int] = (3, 8),
    seed: int = 42,
    llm_fn: Callable[[list[dict[str, str]]], str] | None = None,
) -> list[Session]:
    """Generate synthetic multi-turn sessions.

    Args:
        topics: Topic keys to generate for. None = all topics.
        count_per_topic: Number of sessions per topic.
        turns_range: (min_user_turns, max_user_turns) per session.
        seed: Random seed for reproducibility.
        llm_fn: Optional LLM function for enhanced generation (future).

    Returns:
        List of Session objects.
    """
    rng = random.Random(seed)
    selected_topics = topics or list(TOPICS.keys())
    sessions: list[Session] = []

    for topic in selected_topics:
        if topic not in TOPICS:
            continue
        topic_data = TOPICS[topic]

        for i in range(count_per_topic):
            num_user_turns = rng.randint(turns_range[0], turns_range[1])
            session_id = _make_session_id(topic, i, seed)

            turns = _build_turns(topic_data, num_user_turns, rng)

            session = Session(
                session_id=session_id,
                topic=topic,
                turns=turns,
                metadata={
                    "seed": seed,
                    "index": i,
                    "topic_description": topic_data["description"],
                },
            )
            sessions.append(session)

    return sessions


def save_sessions(sessions: list[Session], path: str) -> None:
    """Save sessions to a JSON file."""
    import json
    from pathlib import Path

    output = [s.to_dict() for s in sessions]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def load_sessions(path: str) -> list[Session]:
    """Load sessions from a JSON file."""
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Session.from_dict(d) for d in data]
