"""Unit tests for FirstCandidateDisambiguator."""

import pytest

from ner_pipeline.disambiguators.first import FirstCandidateDisambiguator
from ner_pipeline.types import Candidate, Document, Entity, Mention

from tests.conftest import MockKnowledgeBase


class TestFirstCandidateDisambiguator:
    """Tests for FirstCandidateDisambiguator class."""

    @pytest.fixture
    def entities(self) -> list[Entity]:
        return [
            Entity(id="Q1", title="Entity One", description="First entity"),
            Entity(id="Q2", title="Entity Two", description="Second entity"),
            Entity(id="Q3", title="Entity Three", description="Third entity"),
        ]

    @pytest.fixture
    def kb(self, entities: list[Entity]) -> MockKnowledgeBase:
        return MockKnowledgeBase(entities)

    @pytest.fixture
    def disambiguator(self, kb: MockKnowledgeBase) -> FirstCandidateDisambiguator:
        return FirstCandidateDisambiguator(kb=kb)

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(id="test", text="Test document")

    @pytest.fixture
    def sample_mention(self) -> Mention:
        return Mention(start=0, end=4, text="Test")

    def test_returns_first_candidate(
        self,
        disambiguator: FirstCandidateDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [
            Candidate(entity_id="Q1", score=0.5),
            Candidate(entity_id="Q2", score=0.9),
            Candidate(entity_id="Q3", score=0.7),
        ]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.id == "Q1"  # First candidate, not highest score

    def test_empty_candidates_returns_none(
        self,
        disambiguator: FirstCandidateDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = []
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is None

    def test_returns_entity_from_kb(
        self,
        disambiguator: FirstCandidateDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [Candidate(entity_id="Q2", score=0.8)]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.title == "Entity Two"
        assert result.description == "Second entity"

    def test_unknown_entity_returns_none(
        self,
        disambiguator: FirstCandidateDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [Candidate(entity_id="Q999", score=0.8)]  # Not in KB
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is None

    def test_ignores_scores(
        self,
        disambiguator: FirstCandidateDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        # Even with very different scores, returns first
        candidates = [
            Candidate(entity_id="Q3", score=0.1),
            Candidate(entity_id="Q1", score=0.99),
        ]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result.id == "Q3"

    def test_single_candidate(
        self,
        disambiguator: FirstCandidateDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [Candidate(entity_id="Q1", score=0.5)]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.id == "Q1"
