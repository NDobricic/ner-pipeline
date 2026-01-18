"""Unit tests for FuzzyCandidateGenerator."""

import pytest

from ner_pipeline.candidates.fuzzy import FuzzyCandidateGenerator
from ner_pipeline.types import Candidate, Document, Entity, Mention

from tests.conftest import MockKnowledgeBase


class TestFuzzyCandidateGenerator:
    """Tests for FuzzyCandidateGenerator class."""

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
    def generator(self, kb: MockKnowledgeBase) -> FuzzyCandidateGenerator:
        return FuzzyCandidateGenerator(kb=kb, top_k=3)

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(id="test-doc", text="Test document.")

    def test_requires_knowledge_base(self):
        with pytest.raises(ValueError, match="requires a knowledge base"):
            FuzzyCandidateGenerator(kb=None)

    def test_generate_returns_candidates(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        mention = Mention(start=0, end=12, text="Barack Obama")
        candidates = generator.generate(mention, sample_doc)
        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_exact_match_ranks_highest(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        mention = Mention(start=0, end=12, text="Barack Obama")
        candidates = generator.generate(mention, sample_doc)
        # Exact match should have highest score
        top_candidate = candidates[0]
        assert top_candidate.entity_id == "Q76"
        assert top_candidate.score > 90  # High fuzzy score

    def test_fuzzy_match(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        # Slight misspelling
        mention = Mention(start=0, end=11, text="Barak Obama")
        candidates = generator.generate(mention, sample_doc)
        # Should still find Barack Obama
        entity_ids = [c.entity_id for c in candidates]
        assert "Q76" in entity_ids

    def test_respects_top_k(self, kb: MockKnowledgeBase, sample_doc: Document):
        generator = FuzzyCandidateGenerator(kb=kb, top_k=2)
        mention = Mention(start=0, end=6, text="United")
        candidates = generator.generate(mention, sample_doc)
        assert len(candidates) <= 2

    def test_candidates_have_scores(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        mention = Mention(start=0, end=6, text="London")
        candidates = generator.generate(mention, sample_doc)
        for c in candidates:
            assert c.score is not None
            assert isinstance(c.score, float)

    def test_candidates_have_descriptions(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        mention = Mention(start=0, end=6, text="London")
        candidates = generator.generate(mention, sample_doc)
        london_candidates = [c for c in candidates if c.entity_id == "Q84"]
        assert len(london_candidates) > 0
        assert london_candidates[0].description == "Capital of UK"

    def test_no_match_returns_some_candidates(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        # Even gibberish should return some candidates (fuzzy will find something)
        mention = Mention(start=0, end=6, text="Xyzabc")
        candidates = generator.generate(mention, sample_doc)
        # Fuzzy matching still returns results (with low scores)
        assert len(candidates) > 0

    def test_mention_context_not_used(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        # Fuzzy matching uses only mention text, not context
        mention1 = Mention(
            start=0, end=6, text="London", context="In the UK"
        )
        mention2 = Mention(
            start=0, end=6, text="London", context="Completely different context"
        )
        candidates1 = generator.generate(mention1, sample_doc)
        candidates2 = generator.generate(mention2, sample_doc)
        # Results should be the same regardless of context
        assert [c.entity_id for c in candidates1] == [c.entity_id for c in candidates2]

    def test_case_insensitive_matching(
        self, generator: FuzzyCandidateGenerator, sample_doc: Document
    ):
        mention = Mention(start=0, end=6, text="LONDON")
        candidates = generator.generate(mention, sample_doc)
        entity_ids = [c.entity_id for c in candidates]
        assert "Q84" in entity_ids
