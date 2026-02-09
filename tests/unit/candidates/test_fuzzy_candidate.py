"""Unit tests for FuzzyCandidatesComponent (spaCy-based)."""

import pytest
import spacy
from spacy.tokens import Span

from el_pipeline import spacy_components  # Register factories
from el_pipeline.spacy_components.candidates import FuzzyCandidatesComponent
from el_pipeline.types import Candidate, Entity

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
        nlp.add_pipe("el_pipeline_simple", config={"min_len": 3})
        component = nlp.add_pipe("el_pipeline_fuzzy_candidates", config={"top_k": 3})
        component.initialize(kb)
        return nlp

    def test_generate_returns_candidates(self, nlp):
        text = "Barack Obama was here."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Obama" in ent.text]
        assert len(obama_ents) > 0
        ent = obama_ents[0]
        # Candidates are stored as List[Candidate]
        assert hasattr(ent._, "candidates")
        assert len(ent._.candidates) > 0
        # Check format is Candidate objects
        for candidate in ent._.candidates:
            assert isinstance(candidate, Candidate)
            assert isinstance(candidate.entity_id, str)

    def test_exact_match_ranks_highest(self, nlp):
        text = "Barack Obama was here."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Obama" in ent.text]
        assert len(obama_ents) > 0
        candidates = obama_ents[0]._.candidates
        # Exact match should be in results (entity_id is the entity ID like Q76)
        entity_ids = [c.entity_id for c in candidates]
        assert "Q76" in entity_ids

    def test_fuzzy_match(self, nlp):
        text = "Barak Obama was here."
        doc = nlp(text)
        obama_ents = [ent for ent in doc.ents if "Barak" in ent.text or "Obama" in ent.text]
        if obama_ents:
            candidates = obama_ents[0]._.candidates
            entity_ids = [c.entity_id for c in candidates]
            assert "Q76" in entity_ids

    def test_respects_top_k(self, kb: MockKnowledgeBase):
        nlp = spacy.blank("en")
        nlp.add_pipe("el_pipeline_simple", config={"min_len": 3})
        component = nlp.add_pipe("el_pipeline_fuzzy_candidates", config={"top_k": 2})
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
            london_candidates = [c for c in candidates if c.entity_id == "Q84"]
            assert len(london_candidates) > 0
            assert london_candidates[0].description == "Capital of UK"

    def test_no_match_returns_low_score_candidates(self, nlp):
        text = "Xyzabc was here."
        doc = nlp(text)
        for ent in doc.ents:
            # Gibberish text may still return candidates but with low scores
            for candidate in ent._.candidates:
                assert candidate.score < 0.5

    def test_case_insensitive_matching(self, nlp):
        text = "LONDON is here."
        doc = nlp(text)
        london_ents = [ent for ent in doc.ents if "LONDON" in ent.text]
        if london_ents:
            candidates = london_ents[0]._.candidates
            entity_ids = [c.entity_id for c in candidates]
            assert "Q84" in entity_ids
