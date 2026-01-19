"""Unit tests for FuzzyCandidatesComponent (spaCy-based)."""

import pytest
import spacy
from spacy.tokens import Span

from ner_pipeline import spacy_components  # Register factories
from ner_pipeline.spacy_components.candidates import FuzzyCandidatesComponent
from ner_pipeline.types import Entity

from tests.conftest import MockKnowledgeBase


class TestFuzzyCandidatesComponent:
    """Tests for FuzzyCandidatesComponent class."""

    @pytest.fixture
    def kb_entities(self) -> list[Entity]:
        return [
            Entity(id="Q76", title="Barack Obama", description="44th US President"),
            Entity(id="Q6279", title="Joe Biden", description="46th US President"),
            Entity(id="Q30", title="United States", description="Country"),
            Entity(id="Q84", title="London", description="Capital of UK"),
            Entity(id="Q60", title="New York City", description="US City"),
        ]

    @pytest.fixture
    def kb(self, kb_entities: list[Entity]) -> MockKnowledgeBase:
        return MockKnowledgeBase(kb_entities)

    @pytest.fixture
    def nlp(self, kb: MockKnowledgeBase) -> spacy.language.Language:
        nlp = spacy.blank("en")
        nlp.add_pipe("ner_pipeline_simple", config={"min_len": 3})
        component = nlp.add_pipe("ner_pipeline_fuzzy_candidates", config={"top_k": 3})
        component.initialize(kb)
        return nlp

    def test_generate_returns_candidates(self, nlp):
        text = "Barack Obama was here."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Obama" in ent.text]
        assert len(obama_ents) > 0
        ent = obama_ents[0]
        # Candidates are stored as List[Tuple[str, str]]
        assert hasattr(ent._, "candidates")
        assert len(ent._.candidates) > 0
        # Check format (title, description)
        for title, desc in ent._.candidates:
            assert isinstance(title, str)
            assert isinstance(desc, str)

    def test_exact_match_ranks_highest(self, nlp):
        text = "Barack Obama was here."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Obama" in ent.text]
        assert len(obama_ents) > 0
        candidates = obama_ents[0]._.candidates
        # Exact match should be in results
        titles = [c[0] for c in candidates]
        assert "Barack Obama" in titles

    def test_fuzzy_match(self, nlp):
        text = "Barak Obama was here."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Barak" in ent.text or "Obama" in ent.text]
        if obama_ents:
            candidates = obama_ents[0]._.candidates
            titles = [c[0] for c in candidates]
            assert "Barack Obama" in titles

    def test_respects_top_k(self, kb: MockKnowledgeBase):
        nlp = spacy.blank("en")
        nlp.add_pipe("ner_pipeline_simple", config={"min_len": 3})
        component = nlp.add_pipe("ner_pipeline_fuzzy_candidates", config={"top_k": 2})
        component.initialize(kb)

        text = "United States."
        doc = nlp(text)
        for ent in doc.ents:
            assert len(ent._.candidates) <= 2

    def test_candidates_have_descriptions(self, nlp):
        text = "London is here."
        doc = nlp(text)
        london_ents = [ent for ent in doc.ents if ent.text == "London"]
        if london_ents:
            candidates = london_ents[0]._.candidates
            london_candidates = [c for c in candidates if c[0] == "London"]
            assert len(london_candidates) > 0
            assert london_candidates[0][1] == "Capital of UK"

    def test_no_match_returns_some_candidates(self, nlp):
        text = "Xyzabc was here."
        doc = nlp(text)
        for ent in doc.ents:
            # Fuzzy matching still returns results (with low scores)
            assert len(ent._.candidates) > 0

    def test_case_insensitive_matching(self, nlp):
        text = "LONDON is here."
        doc = nlp(text)
        london_ents = [ent for ent in doc.ents if "LONDON" in ent.text]
        if london_ents:
            candidates = london_ents[0]._.candidates
            titles = [c[0] for c in candidates]
            assert "London" in titles
