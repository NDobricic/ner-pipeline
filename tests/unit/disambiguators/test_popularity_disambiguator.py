"""Unit tests for PopularityDisambiguator."""

import pytest

from ner_pipeline.disambiguators.popularity import PopularityDisambiguator
from ner_pipeline.types import Candidate, Document, Entity, Mention

from tests.conftest import MockKnowledgeBase


class TestPopularityDisambiguator:
    """Tests for PopularityDisambiguator class."""

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
    def disambiguator(self, kb: MockKnowledgeBase) -> PopularityDisambiguator:
        return PopularityDisambiguator(kb=kb)

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(id="test", text="Test document")

    @pytest.fixture
    def sample_mention(self) -> Mention:
        return Mention(start=0, end=4, text="Test")

    def test_returns_highest_scored_candidate(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [
            Candidate(entity_id="Q1", score=0.5),
            Candidate(entity_id="Q2", score=0.9),  # Highest
            Candidate(entity_id="Q3", score=0.7),
        ]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.id == "Q2"

    def test_empty_candidates_returns_none(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = []
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is None

    def test_returns_entity_from_kb(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [Candidate(entity_id="Q3", score=0.8)]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.title == "Entity Three"
        assert result.description == "Third entity"

    def test_unknown_entity_returns_none(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [Candidate(entity_id="Q999", score=0.99)]  # Not in KB
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is None

    def test_handles_none_scores(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [
            Candidate(entity_id="Q1", score=None),
            Candidate(entity_id="Q2", score=0.5),
            Candidate(entity_id="Q3", score=None),
        ]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.id == "Q2"  # Only one with non-None score

    def test_all_none_scores_picks_first(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [
            Candidate(entity_id="Q1", score=None),
            Candidate(entity_id="Q2", score=None),
        ]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        # With all None scores (treated as 0.0), max() returns first
        assert result is not None

    def test_single_candidate(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [Candidate(entity_id="Q1", score=0.5)]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.id == "Q1"

    def test_tie_breaking_returns_any_max(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [
            Candidate(entity_id="Q1", score=0.8),
            Candidate(entity_id="Q2", score=0.8),  # Tie
        ]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.id in ["Q1", "Q2"]

    def test_negative_scores(
        self,
        disambiguator: PopularityDisambiguator,
        sample_mention: Mention,
        sample_doc: Document,
    ):
        candidates = [
            Candidate(entity_id="Q1", score=-0.5),
            Candidate(entity_id="Q2", score=-0.1),  # Highest (least negative)
            Candidate(entity_id="Q3", score=-0.9),
        ]
        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)
        assert result is not None
        assert result.id == "Q2"
