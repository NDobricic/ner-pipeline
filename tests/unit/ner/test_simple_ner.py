"""Unit tests for SimpleRegexNER."""

import pytest

from ner_pipeline.ner.simple import SimpleRegexNER
from ner_pipeline.types import Mention


class TestSimpleRegexNER:
    """Tests for SimpleRegexNER class."""

    @pytest.fixture
    def ner(self) -> SimpleRegexNER:
        return SimpleRegexNER(min_len=3)

    def test_extract_capitalized_words(self, ner: SimpleRegexNER):
        text = "Barack Obama is the president."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        assert "Barack Obama" in texts

    def test_extract_single_capitalized_word(self, ner: SimpleRegexNER):
        text = "London is a great city."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        assert "London" in texts

    def test_min_length_filter(self):
        ner = SimpleRegexNER(min_len=5)
        text = "Al is here. Barack is too."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        assert "Al" not in texts  # Too short
        assert "Barack" in texts  # Long enough

    def test_returns_mention_objects(self, ner: SimpleRegexNER):
        text = "Obama spoke."
        mentions = ner.extract(text)
        assert len(mentions) > 0
        assert all(isinstance(m, Mention) for m in mentions)

    def test_mention_offsets_are_correct(self, ner: SimpleRegexNER):
        # Note: regex matches consecutive capitalized words together
        # "Hello Barack Obama" would be one match, so use lowercase prefix
        text = "the Barack Obama there."
        mentions = ner.extract(text)
        obama_mentions = [m for m in mentions if "Barack Obama" in m.text]
        assert len(obama_mentions) == 1
        m = obama_mentions[0]
        assert text[m.start:m.end] == m.text

    def test_all_mentions_have_label(self, ner: SimpleRegexNER):
        text = "Barack Obama visited London."
        mentions = ner.extract(text)
        for m in mentions:
            assert m.label == "ENT"

    def test_all_mentions_have_context(self, ner: SimpleRegexNER):
        text = "Barack Obama was the president. He lived in Washington."
        mentions = ner.extract(text)
        for m in mentions:
            assert m.context is not None
            assert m.text in m.context or m.context != ""

    def test_no_mentions_in_lowercase_text(self, ner: SimpleRegexNER):
        text = "this is all lowercase text without any entities."
        mentions = ner.extract(text)
        assert len(mentions) == 0

    def test_hyphenated_names(self, ner: SimpleRegexNER):
        text = "Mary-Jane went home."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        assert "Mary-Jane" in texts

    def test_consecutive_capitals(self, ner: SimpleRegexNER):
        # Note: regex matches consecutive capitalized words together
        # "The CIA" is one match since both start with capitals
        text = "The CIA is an agency. FBI too."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        # "The CIA" is matched as one entity, "FBI" as another
        assert any("CIA" in t for t in texts)
        assert "FBI" in texts

    def test_sentence_start_detection(self, ner: SimpleRegexNER):
        # Words at sentence start that are capitalized but not entities
        # The regex is simple and will match them anyway
        text = "The president spoke."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        assert "The" in texts  # Simple regex matches this

    def test_context_mode_parameter(self):
        ner = SimpleRegexNER(min_len=3, context_mode="window")
        # Use lowercase padding to ensure "Barack Obama" is a separate match
        text = "aaa aaa aaa " + "Barack Obama" + " bbb bbb bbb"
        mentions = ner.extract(text)
        obama_mentions = [m for m in mentions if "Barack Obama" in m.text]
        assert len(obama_mentions) == 1
        # Window mode should produce context

    def test_empty_text(self, ner: SimpleRegexNER):
        mentions = ner.extract("")
        assert len(mentions) == 0

    def test_whitespace_only_text(self, ner: SimpleRegexNER):
        mentions = ner.extract("   \n\t  ")
        assert len(mentions) == 0

    def test_multiple_mentions_same_entity(self, ner: SimpleRegexNER):
        # Use lowercase words after periods to ensure "Obama" is extracted separately
        text = "Obama said hello. then Obama left."
        mentions = ner.extract(text)
        obama_mentions = [m for m in mentions if m.text == "Obama"]
        assert len(obama_mentions) == 2
        # Different offsets
        offsets = [(m.start, m.end) for m in obama_mentions]
        assert len(set(offsets)) == 2
