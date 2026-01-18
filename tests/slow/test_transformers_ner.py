"""Slow tests for TransformersNER (requires model download)."""

import pytest

from ner_pipeline.types import Mention


@pytest.mark.slow
@pytest.mark.requires_transformers
class TestTransformersNER:
    """Tests for TransformersNER class (requires HuggingFace model download)."""

    @pytest.fixture
    def ner(self):
        """Create TransformersNER instance (downloads model if needed)."""
        from ner_pipeline.ner.transformers import TransformersNER
        return TransformersNER(model_name="dslim/bert-base-NER")

    def test_extract_returns_mentions(self, ner):
        text = "Barack Obama was the President of the United States."
        mentions = ner.extract(text)
        assert len(mentions) > 0
        assert all(isinstance(m, Mention) for m in mentions)

    def test_extracts_person_entities(self, ner):
        text = "Barack Obama met Angela Merkel in Berlin."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        # BERT NER should find person entities
        assert any("Obama" in t or "Barack" in t for t in texts)

    def test_extracts_location_entities(self, ner):
        text = "Paris is the capital of France."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        # Should find location entities
        assert any("Paris" in t or "France" in t for t in texts)

    def test_extracts_organization_entities(self, ner):
        text = "Google and Microsoft are tech companies."
        mentions = ner.extract(text)
        texts = [m.text for m in mentions]
        # Should find organization entities
        assert any("Google" in t or "Microsoft" in t for t in texts)

    def test_mention_has_label(self, ner):
        text = "Barack Obama was born in Honolulu."
        mentions = ner.extract(text)
        for m in mentions:
            assert m.label is not None
            # Transformers NER uses labels like PER, LOC, ORG, MISC
            assert isinstance(m.label, str)

    def test_mention_has_correct_offsets(self, ner):
        text = "Barack Obama spoke yesterday."
        mentions = ner.extract(text)
        for m in mentions:
            # Note: transformers NER may have slight offset differences
            # due to tokenization, but text should match approximately
            extracted = text[m.start:m.end]
            # Allow for whitespace trimming differences
            assert m.text.strip() in extracted or extracted in m.text

    def test_mention_has_context(self, ner):
        text = "Yesterday, Barack Obama gave a speech in Washington."
        mentions = ner.extract(text)
        for m in mentions:
            assert m.context is not None

    def test_empty_text(self, ner):
        mentions = ner.extract("")
        assert len(mentions) == 0

    def test_long_text(self, ner):
        """Test handling of longer text."""
        text = "Barack Obama " * 50 + "was president."
        mentions = ner.extract(text)
        # Should handle long text without crashing
        assert isinstance(mentions, list)


@pytest.mark.slow
@pytest.mark.requires_transformers
class TestTransformersNERContextModes:
    """Test TransformersNER with different context modes."""

    def test_sentence_context_mode(self):
        from ner_pipeline.ner.transformers import TransformersNER
        ner = TransformersNER(
            model_name="dslim/bert-base-NER",
            context_mode="sentence"
        )
        text = "First sentence. Barack Obama spoke. Third sentence."
        mentions = ner.extract(text)
        obama_mentions = [m for m in mentions if "Obama" in m.text or "Barack" in m.text]
        if obama_mentions:
            # Sentence context should include the sentence
            assert "spoke" in obama_mentions[0].context

    def test_window_context_mode(self):
        from ner_pipeline.ner.transformers import TransformersNER
        ner = TransformersNER(
            model_name="dslim/bert-base-NER",
            context_mode="window"
        )
        text = "A" * 200 + " Barack Obama " + "B" * 200
        mentions = ner.extract(text)
        obama_mentions = [m for m in mentions if "Obama" in m.text or "Barack" in m.text]
        if obama_mentions:
            # Window context should be bounded
            assert len(obama_mentions[0].context) < len(text)
