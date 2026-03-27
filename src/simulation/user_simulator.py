"""User simulator for dynamic multi-turn interaction with the agent.

Generates contextually appropriate follow-up turns based on agent responses,
simulating different user personas (confused, expert, comparative).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class PersonaProfile:
    """Behavioral profile for a simulated user persona."""

    name: str
    description: str
    follow_up_style: str  # "clarification", "deep_dive", "comparison"
    templates: list[str] = field(default_factory=list)
    implicit_ref_chance: float = 0.3  # probability of implicit reference


# ── Pre-defined personas ─────────────────────────────────────────────────────

PERSONAS: dict[str, PersonaProfile] = {
    "confused_user": PersonaProfile(
        name="confused_user",
        description="A user who frequently asks for clarifications and simpler explanations",
        follow_up_style="clarification",
        templates=[
            "I don't understand that. Can you explain it more simply?",
            "What do you mean by '{keyword}'?",
            "Can you give me an example of that?",
            "I'm confused about the part where you mentioned {keyword}. Can you clarify?",
            "So in simple terms, what does that mean for me?",
            "Wait, does that mean {keyword} is required?",
            "Can you break that down step by step?",
            "I still don't get it. What should I actually do?",
        ],
        implicit_ref_chance=0.2,
    ),
    "expert_user": PersonaProfile(
        name="expert_user",
        description="A knowledgeable user who asks deep technical/regulatory questions",
        follow_up_style="deep_dive",
        templates=[
            "What is the legal basis for that requirement?",
            "How does that interact with {keyword} regulation?",
            "What are the edge cases for {keyword}?",
            "Can you cite the specific normative for that?",
            "What is the enforcement mechanism for {keyword}?",
            "How has that interpretation evolved since the original circular?",
            "What are the implications for {keyword} in cross-border scenarios?",
            "Does that align with the BCB's interpretation in Resolution {keyword}?",
        ],
        implicit_ref_chance=0.4,
    ),
    "comparative_user": PersonaProfile(
        name="comparative_user",
        description="A user who constantly compares different payment methods and regulations",
        follow_up_style="comparison",
        templates=[
            "How does that compare to {keyword}?",
            "Is that the same for TED and DOC?",
            "Which option is better for {keyword}?",
            "What about compared to international standards?",
            "Is {keyword} cheaper than the alternative?",
            "How does the settlement time compare?",
            "Does the same regulation apply to {keyword}?",
            "What are the pros and cons compared to {keyword}?",
        ],
        implicit_ref_chance=0.3,
    ),
}

# Keywords used for template filling
_REGULATORY_KEYWORDS = [
    "PIX", "TED", "DOC", "boleto", "instant payment",
    "settlement", "compliance", "BCB", "SPI", "DICT",
    "anti-fraud", "transaction limit", "fee structure",
    "data protection", "LGPD", "open banking",
]


def _extract_keywords(text: str) -> list[str]:
    """Extract potential keywords from agent response for template filling."""
    words = text.split()
    # Return nouns/proper-nouns (crude heuristic: capitalized words, 4+ chars)
    candidates = [w.strip(".,;:!?()\"'") for w in words if len(w) >= 4]
    return candidates[:10] if candidates else _REGULATORY_KEYWORDS[:5]


@dataclass
class UserSimulator:
    """Simulates a user persona in a multi-turn conversation.

    Args:
        persona: Name of the persona (confused_user, expert_user, comparative_user).
        topic: Regulatory topic for the conversation.
        seed: Random seed for reproducibility.
        llm_fn: Optional LLM function for more natural generation.
    """

    persona: str
    topic: str
    seed: int = 42
    llm_fn: Callable[[list[dict[str, str]]], str] | None = None
    _rng: random.Random = field(init=False)
    _profile: PersonaProfile = field(init=False)
    _history: list[dict[str, str]] = field(default_factory=list)
    _turn_count: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.persona not in PERSONAS:
            raise ValueError(
                f"Unknown persona '{self.persona}'. "
                f"Available: {list(PERSONAS.keys())}"
            )
        self._profile = PERSONAS[self.persona]

    @property
    def profile(self) -> PersonaProfile:
        return self._profile

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def generate_initial_question(self) -> str:
        """Generate the first question to start the conversation."""
        from src.simulation.session_generator import TOPICS

        topic_data = TOPICS.get(self.topic, {})
        questions = topic_data.get("seed_questions", [
            f"Tell me about {self.topic} regulations.",
        ])
        question = self._rng.choice(questions)
        self._history.append({"role": "user", "content": question})
        self._turn_count += 1
        return question

    def generate_next_turn(self, agent_response: str) -> str:
        """Generate the next user turn based on the agent's response.

        Args:
            agent_response: The agent's last response text.

        Returns:
            The next user message.
        """
        self._history.append({"role": "assistant", "content": agent_response})

        # If LLM function provided, use it for more natural generation
        if self.llm_fn is not None:
            return self._generate_with_llm(agent_response)

        return self._generate_from_templates(agent_response)

    def _generate_from_templates(self, agent_response: str) -> str:
        """Generate next turn using template-based approach."""
        keywords = _extract_keywords(agent_response)
        keyword = self._rng.choice(keywords) if keywords else "that"

        # Decide if this is an implicit reference
        is_implicit = self._rng.random() < self._profile.implicit_ref_chance

        if is_implicit and self._turn_count > 1:
            from src.simulation.session_generator import TOPICS
            topic_data = TOPICS.get(self.topic, {})
            implicit_refs = topic_data.get("implicit_refs", [])
            if implicit_refs:
                message = self._rng.choice(implicit_refs)
            else:
                message = self._rng.choice(self._profile.templates)
                message = message.format(keyword=keyword)
        else:
            template = self._rng.choice(self._profile.templates)
            message = template.format(keyword=keyword)

        self._history.append({"role": "user", "content": message})
        self._turn_count += 1
        return message

    def _generate_with_llm(self, agent_response: str) -> str:
        """Generate next turn using LLM for more natural language."""
        assert self.llm_fn is not None

        system_prompt = (
            f"You are simulating a {self._profile.description}. "
            f"Topic: {self.topic}. "
            f"Generate a natural follow-up question in the style of a "
            f"{self._profile.follow_up_style}. "
            f"Keep it concise (1-2 sentences). Only output the question."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *self._history,
            {"role": "system", "content": "Generate the next user message:"},
        ]

        message = self.llm_fn(messages)
        self._history.append({"role": "user", "content": message})
        self._turn_count += 1
        return message

    def get_history(self) -> list[dict[str, str]]:
        """Return the full conversation history."""
        return list(self._history)

    def reset(self) -> None:
        """Reset the simulator state."""
        self._history.clear()
        self._turn_count = 0
        self._rng = random.Random(self.seed)
