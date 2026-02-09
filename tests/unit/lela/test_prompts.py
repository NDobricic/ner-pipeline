"""Unit tests for LELA prompts module."""

import pytest

from el_pipeline.lela.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    create_disambiguation_messages,
    mark_mention_in_text,
)
from el_pipeline.types import Candidate


class TestDefaultSystemPrompt:
    """Tests for the default system prompt."""

    def test_prompt_is_nonempty_string(self):
        assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
        assert len(DEFAULT_SYSTEM_PROMPT) > 0

    def test_prompt_mentions_disambiguation(self):
        assert "disambiguate" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_answer_format(self):
        assert "answer" in DEFAULT_SYSTEM_PROMPT.lower()


class TestMarkMentionInText:
    """Tests for mark_mention_in_text function."""

    def test_marks_mention_with_default_brackets(self):
        text = "Barack Obama was president."
        result = mark_mention_in_text(text, 0, 12)
        assert result == "[Barack Obama] was president."

    def test_marks_mention_in_middle(self):
        text = "The president Barack Obama gave a speech."
        result = mark_mention_in_text(text, 14, 26)
        assert result == "The president [Barack Obama] gave a speech."

    def test_marks_mention_at_end(self):
        text = "The president was Barack Obama"
        result = mark_mention_in_text(text, 18, 30)
        assert result == "The president was [Barack Obama]"

    def test_custom_markers(self):
        text = "Barack Obama was president."
        result = mark_mention_in_text(text, 0, 12, open_marker="<<", close_marker=">>")
        assert result == "<<Barack Obama>> was president."

    def test_empty_text(self):
        result = mark_mention_in_text("", 0, 0)
        assert result == "[]"

    def test_single_character_mention(self):
        text = "A is a letter."
        result = mark_mention_in_text(text, 0, 1)
        assert result == "[A] is a letter."


class TestCreateDisambiguationMessages:
    """Tests for create_disambiguation_messages function."""

    @pytest.fixture
    def sample_candidates(self):
        return [
            Candidate(entity_id="Barack Obama", description="44th US President"),
            Candidate(entity_id="Michelle Obama", description="Former First Lady"),
            Candidate(entity_id="Obama Foundation", description="Non-profit organization"),
        ]

    def test_returns_list_of_dicts(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
        )
        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)

    def test_first_message_is_system(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
        )
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
        )
        assert messages[-1]["role"] == "user"

    def test_user_message_contains_input_text(self, sample_candidates):
        marked_text = "[Obama] gave a speech."
        messages = create_disambiguation_messages(
            marked_text=marked_text,
            candidates=sample_candidates,
        )
        user_msg = messages[-1]["content"]
        assert marked_text in user_msg

    def test_user_message_contains_candidates(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
        )
        user_msg = messages[-1]["content"]
        assert "Barack Obama" in user_msg
        assert "Michelle Obama" in user_msg

    def test_candidates_are_numbered(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
        )
        user_msg = messages[-1]["content"]
        assert "1." in user_msg
        assert "2." in user_msg
        assert "3." in user_msg

    def test_add_none_candidate_option(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
            add_none_candidate=True,
        )
        user_msg = messages[-1]["content"]
        assert "0." in user_msg
        assert "None" in user_msg

    def test_no_none_candidate_by_default(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
            add_none_candidate=False,
        )
        user_msg = messages[-1]["content"]
        # Should not have "0. None" option
        assert "0. None" not in user_msg

    def test_descriptions_included_by_default(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
            add_descriptions=True,
        )
        user_msg = messages[-1]["content"]
        assert "44th US President" in user_msg
        assert "Former First Lady" in user_msg

    def test_descriptions_can_be_excluded(self, sample_candidates):
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
            add_descriptions=False,
        )
        user_msg = messages[-1]["content"]
        assert "44th US President" not in user_msg
        assert "Former First Lady" not in user_msg

    def test_custom_system_prompt(self, sample_candidates):
        custom_prompt = "You are a custom disambiguation system."
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
            system_prompt=custom_prompt,
        )
        assert messages[0]["content"] == custom_prompt

    def test_query_prompt_adds_message(self, sample_candidates):
        query_prompt = "Focus on political figures."
        messages = create_disambiguation_messages(
            marked_text="[Obama] gave a speech.",
            candidates=sample_candidates,
            query_prompt=query_prompt,
        )
        # Should have system, query, and user messages
        assert len(messages) == 3
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == query_prompt

    def test_empty_candidates_list(self):
        messages = create_disambiguation_messages(
            marked_text="[Test] mention.",
            candidates=[],
        )
        # Should still return valid messages
        assert len(messages) >= 2

    def test_candidate_with_empty_description(self):
        candidates = [
            Candidate(entity_id="Entity A", description=""),
            Candidate(entity_id="Entity B", description="Has description"),
        ]
        messages = create_disambiguation_messages(
            marked_text="[Test] mention.",
            candidates=candidates,
            add_descriptions=True,
        )
        user_msg = messages[-1]["content"]
        assert "Entity A" in user_msg
        assert "Entity B" in user_msg
