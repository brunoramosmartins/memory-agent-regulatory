"""Tests for user simulator."""

import pytest

from src.simulation.user_simulator import PERSONAS, UserSimulator


class TestUserSimulator:
    def test_create_simulator(self):
        sim = UserSimulator(persona="confused_user", topic="pix_fees")
        assert sim.persona == "confused_user"
        assert sim.topic == "pix_fees"
        assert sim.turn_count == 0

    def test_invalid_persona_raises(self):
        with pytest.raises(ValueError, match="Unknown persona"):
            UserSimulator(persona="nonexistent", topic="pix_fees")

    def test_generate_initial_question(self):
        sim = UserSimulator(persona="confused_user", topic="pix_fees", seed=42)
        question = sim.generate_initial_question()
        assert isinstance(question, str)
        assert len(question) > 0
        assert sim.turn_count == 1

    def test_generate_next_turn(self):
        sim = UserSimulator(persona="expert_user", topic="pix_fees", seed=42)
        sim.generate_initial_question()
        follow_up = sim.generate_next_turn("PIX transfers are free for individuals.")
        assert isinstance(follow_up, str)
        assert len(follow_up) > 0
        assert sim.turn_count == 2

    def test_history_tracks_conversation(self):
        sim = UserSimulator(persona="confused_user", topic="pix_fees", seed=42)
        sim.generate_initial_question()
        sim.generate_next_turn("Response from agent.")
        history = sim.get_history()
        assert len(history) == 3  # initial + agent response + follow-up
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"

    def test_reproducible_with_same_seed(self):
        sim1 = UserSimulator(persona="expert_user", topic="pix_fees", seed=99)
        sim2 = UserSimulator(persona="expert_user", topic="pix_fees", seed=99)
        q1 = sim1.generate_initial_question()
        q2 = sim2.generate_initial_question()
        assert q1 == q2

    def test_different_seeds_may_differ(self):
        sim1 = UserSimulator(persona="expert_user", topic="pix_fees", seed=1)
        sim2 = UserSimulator(persona="expert_user", topic="pix_fees", seed=2)
        q1 = sim1.generate_initial_question()
        q2 = sim2.generate_initial_question()
        # With different seeds, they may or may not differ (small pool)
        # Just verify both produce valid output
        assert isinstance(q1, str) and isinstance(q2, str)

    def test_reset_clears_state(self):
        sim = UserSimulator(persona="confused_user", topic="pix_fees", seed=42)
        sim.generate_initial_question()
        sim.generate_next_turn("Response.")
        assert sim.turn_count == 2
        sim.reset()
        assert sim.turn_count == 0
        assert sim.get_history() == []

    def test_multi_turn_conversation(self):
        sim = UserSimulator(persona="comparative_user", topic="pix_comparisons", seed=42)
        sim.generate_initial_question()
        for i in range(4):
            sim.generate_next_turn(f"Agent response {i}")
        assert sim.turn_count == 5
        history = sim.get_history()
        # 5 user turns + 4 agent responses = 9
        assert len(history) == 9

    def test_all_personas_exist(self):
        assert "confused_user" in PERSONAS
        assert "expert_user" in PERSONAS
        assert "comparative_user" in PERSONAS

    def test_persona_profile_accessible(self):
        sim = UserSimulator(persona="expert_user", topic="pix_fees")
        assert sim.profile.name == "expert_user"
        assert sim.profile.follow_up_style == "deep_dive"

    def test_with_mock_llm(self):
        def mock_llm(messages):
            return "What about the specific regulation number?"

        sim = UserSimulator(
            persona="expert_user",
            topic="pix_fees",
            seed=42,
            llm_fn=mock_llm,
        )
        sim.generate_initial_question()
        response = sim.generate_next_turn("PIX is regulated by BCB.")
        assert response == "What about the specific regulation number?"
        assert sim.turn_count == 2


class TestValidatorIntegration:
    """Test validator with generated sessions."""

    def test_validate_generated_sessions(self):
        from src.simulation.session_generator import generate_sessions
        from src.simulation.validator import validate_sessions

        sessions = generate_sessions(count_per_topic=5, seed=42)
        errors = validate_sessions(sessions)
        assert errors == {}, f"Validation errors: {errors}"

    def test_session_stats(self):
        from src.simulation.session_generator import generate_sessions
        from src.simulation.validator import session_stats

        sessions = generate_sessions(count_per_topic=5, seed=42)
        stats = session_stats(sessions)
        assert stats["total_sessions"] == 25
        assert stats["unique_topics"] == 5
        assert stats["implicit_ref_pct"] == 100.0

    def test_empty_session_stats(self):
        from src.simulation.validator import session_stats
        stats = session_stats([])
        assert stats == {"total": 0}
