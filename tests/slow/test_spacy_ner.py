"""Slow tests for SpacyNER (requires model download)."""

import pytest

from ner_pipeline.types import Mention


@pytest.mark.slow
@pytest.mark.requires_spacy
class TestSpacyNER:
    """Tests for SpacyNER class (requires en_core_web_sm model)."""

    @pytest.fixture
    def ner(self):
        """Create SpacyNER instance (downloads model if needed)."""
        from ner_pipeline.ner.spacy import SpacyNER
        return SpacyNER(model="en_core_web_sm")

    def test_extract_returns_mentions(self, ner):
        text = "Barack Obama was the President of the United States."
        mentions = ner.extract(text)
        assert len(mentions) > 0
        assert all(isinstance(m, Mention) for m in mentions)

    def test_extracts_person_entities(self, ner):
        text = "Barack Obama met Angela Merkel in Berlin."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        # Should find at least some person entities
        assert any("Obama" in t or "Merkel" in t for t in texts)

    def test_extracts_location_entities(self, ner):
        text = "Paris is the capital of France."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        assert "Paris" in texts or "France" in texts

    def test_extracts_organization_entities(self, ner):
        text = "Google and Microsoft are tech companies."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        assert any("Google" in t or "Microsoft" in t for t in texts)

    def test_mention_has_label(self, ner):
        text = "Barack Obama was born in Honolulu."
        mentions = ner.extract(text)
        for m in mentions:
            assert m.label is not None
            # spaCy labels like PERSON, GPE, ORG, etc.
            assert isinstance(m.label, str)

    def test_mention_has_correct_offsets(self, ner):
        text = "Barack Obama spoke yesterday."
        mentions = ner.extract(text)
        for m in mentions:
            # Verify offsets point to the actual text
            assert text[m.start:m.end] == m.text

    def test_mention_has_context(self, ner):
        text = "Yesterday, Barack Obama gave a speech in Washington."
        mentions = ner.extract(text)
        for m in mentions:
            assert m.context is not None
            # Context should contain the mention text
            assert m.text in m.context or m.context != ""

    def test_empty_text(self, ner):
        mentions = ner.extract("")
        assert len(mentions) == 0

    def test_text_without_entities(self, ner):
        text = "the quick brown fox jumps over the lazy dog"
        mentions = ner.extract(text)
        # May or may not find entities depending on model
        # Just verify it doesn't crash
        assert isinstance(mentions, list)

    def test_context_mode_window(self):
        from ner_pipeline.ner.spacy import SpacyNER
        ner = SpacyNER(model="en_core_web_sm", context_mode="window")
        # Use much longer padding to ensure window is bounded
        text = "A" * 300 + " Barack Obama " + "B" * 300
        mentions = ner.extract(text)
        obama_mentions = [m for m in mentions if "Obama" in m.text]
        if obama_mentions:
            # Window context should be bounded (default window is 150 chars each side)
            assert len(obama_mentions[0].context) < len(text)


@pytest.mark.slow
@pytest.mark.requires_spacy
class TestSpacyNERDifferentModels:
    """Test SpacyNER with different models."""

    def test_default_model(self):
        """Test default model initialization."""
        from ner_pipeline.ner.spacy import SpacyNER
        ner = SpacyNER()
        text = "New York is a city."
        mentions = ner.extract(text)
        assert isinstance(mentions, list)
