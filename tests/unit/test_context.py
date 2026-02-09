"""Unit tests for context extraction utilities."""

import pytest

from el_pipeline.context import (
    extract_context,
    extract_sentence_context,
    extract_window_context,
)


class TestExtractSentenceContext:
    """Tests for extract_sentence_context function."""

    def test_single_sentence(self):
        text = "Barack Obama was the 44th President."
        # Mention: "Barack Obama" at 0-12
        context = extract_sentence_context(text, 0, 12)
        assert "Barack Obama" in context
        assert "President" in context

    def test_multiple_sentences(self):
        text = "First sentence. Barack Obama was the President. Third sentence."
        # Mention: "Barack Obama" at 16-28
        context = extract_sentence_context(text, 16, 28)
        assert "Barack Obama" in context
        assert "President" in context

    def test_mention_at_start(self):
        text = "Obama graduated from Harvard. He became President."
        context = extract_sentence_context(text, 0, 5)
        assert "Obama" in context

    def test_mention_at_end(self):
        text = "The president was Barack Obama."
        context = extract_sentence_context(text, 18, 30)
        assert "Barack Obama" in context

    def test_with_max_sentences(self):
        text = "First. Second has mention. Third. Fourth."
        # Get surrounding sentences
        context = extract_sentence_context(text, 7, 13, max_sentences=1)
        assert "Second" in context

    def test_empty_text(self):
        text = ""
        context = extract_sentence_context(text, 0, 0)
        assert context == ""

    def test_text_without_sentence_boundaries(self):
        text = "no sentence boundary here"
        context = extract_sentence_context(text, 3, 11)
        assert context == text


class TestExtractWindowContext:
    """Tests for extract_window_context function."""

    def test_basic_window(self):
        text = "A" * 100 + "TARGET" + "B" * 100
        context = extract_window_context(text, 100, 106, window_chars=50)
        assert "TARGET" in context
        assert len(context) <= 106 + 50  # Can't exceed original bounds

    def test_mention_at_start(self):
        text = "Obama is the president of the country."
        context = extract_window_context(text, 0, 5, window_chars=20)
        assert "Obama" in context
        assert context.startswith("Obama")

    def test_mention_at_end(self):
        text = "The president is Barack Obama"
        context = extract_window_context(text, 17, 29, window_chars=20)
        assert "Barack Obama" in context

    def test_short_text(self):
        text = "Short text"
        context = extract_window_context(text, 0, 5, window_chars=100)
        assert "Short" in context

    def test_word_boundary_alignment(self):
        text = "word1 word2 TARGET word3 word4"
        context = extract_window_context(text, 12, 18, window_chars=10)
        # Should try to align to word boundaries
        assert "TARGET" in context

    def test_empty_text(self):
        text = ""
        context = extract_window_context(text, 0, 0, window_chars=50)
        assert context == ""


class TestExtractContext:
    """Tests for extract_context dispatcher function."""

    def test_sentence_mode(self):
        text = "First sentence. Second sentence. Third sentence."
        context = extract_context(text, 16, 22, mode="sentence")
        assert "Second" in context

    def test_window_mode(self):
        text = "A" * 50 + "TARGET" + "B" * 50
        context = extract_context(text, 50, 56, mode="window", window_chars=20)
        assert "TARGET" in context

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown context mode"):
            extract_context("text", 0, 4, mode="invalid")

    def test_passes_kwargs_to_sentence(self):
        text = "S1. S2 mention. S3. S4."
        context = extract_context(text, 4, 6, mode="sentence", max_sentences=0)
        assert "S2" in context

    def test_passes_kwargs_to_window(self):
        text = "A" * 100 + "X" + "B" * 100
        context = extract_context(text, 100, 101, mode="window", window_chars=10)
        assert "X" in context
        assert len(context) < 50
