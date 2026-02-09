"""Unit tests for SimpleNERComponent (spaCy-based)."""

import pytest
import spacy
from spacy.tokens import Span

from el_pipeline import spacy_components  # Register factories
from el_pipeline.spacy_components.ner import SimpleNERComponent


class TestSimpleNERComponent:
    """Tests for SimpleNERComponent class."""

    @pytest.fixture
    def nlp(self) -> spacy.language.Language:
        nlp = spacy.blank("en")
        nlp.add_pipe("el_pipeline_simple", config={"min_len": 3})
        return nlp

    def test_extract_capitalized_words(self, nlp):
        text = "Barack Obama is the president."
        doc = nlp(text)
        texts = [ent.text for ent in doc.ents]
        assert "Barack Obama" in texts

    def test_extract_single_capitalized_word(self, nlp):
        text = "London is a great city."
        doc = nlp(text)
        texts = [ent.text for ent in doc.ents]
        assert "London" in texts

    def test_min_length_filter(self):
        nlp = spacy.blank("en")
        nlp.add_pipe("el_pipeline_simple", config={"min_len": 5})
        text = "Al is here. Barack is too."
        doc = nlp(text)
        texts = [ent.text for ent in doc.ents]
        assert "Al" not in texts  # Too short
        assert "Barack" in texts  # Long enough

    def test_returns_span_objects(self, nlp):
        text = "Obama spoke."
        doc = nlp(text)
        assert len(doc.ents) > 0
        assert all(isinstance(ent, Span) for ent in doc.ents)

    def test_mention_offsets_are_correct(self, nlp):
        text = "the Barack Obama there."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Barack Obama" in ent.text]
        assert len(obama_ents) == 1
        ent = obama_ents[0]
        assert text[ent.start_char:ent.end_char] == ent.text

    def test_all_entities_have_label(self, nlp):
        text = "Barack Obama visited London."
        doc = nlp(text)
        for ent in doc.ents:
            assert ent.label_ == "ENT"

    def test_all_entities_have_context(self, nlp):
        text = "Barack Obama was the president. He lived in Washington."
        doc = nlp(text)
        for ent in doc.ents:
            assert hasattr(ent._, "context")
            assert ent._.context is not None

    def test_no_entities_in_lowercase_text(self, nlp):
        text = "this is all lowercase text without any entities."
        doc = nlp(text)
        assert len(doc.ents) == 0

    def test_hyphenated_names(self, nlp):
        text = "Mary-Jane went home."
        doc = nlp(text)
        texts = [ent.text for ent in doc.ents]
        assert "Mary-Jane" in texts

    def test_consecutive_capitals(self, nlp):
        text = "The CIA is an agency. FBI too."
        doc = nlp(text)
        texts = [ent.text for ent in doc.ents]
        assert any("CIA" in t for t in texts)
        assert "FBI" in texts

    def test_sentence_start_detection(self, nlp):
        text = "The president spoke."
        doc = nlp(text)
        texts = [ent.text for ent in doc.ents]
        assert "The" in texts  # Simple regex matches this

    def test_context_mode_parameter(self):
        nlp = spacy.blank("en")
        nlp.add_pipe("el_pipeline_simple", config={"min_len": 3, "context_mode": "window"})
        text = "aaa aaa aaa " + "Barack Obama" + " bbb bbb bbb"
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Barack Obama" in ent.text]
        assert len(obama_ents) == 1

    def test_empty_text(self, nlp):
        doc = nlp("")
        assert len(doc.ents) == 0

    def test_whitespace_only_text(self, nlp):
        doc = nlp("   \n\t  ")
        assert len(doc.ents) == 0

    def test_multiple_mentions_same_entity(self, nlp):
        text = "Obama said hello. then Obama left."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if ent.text == "Obama"]
        assert len(obama_ents) == 2
        offsets = [(ent.start_char, ent.end_char) for ent in obama_ents]
        assert len(set(offsets)) == 2
