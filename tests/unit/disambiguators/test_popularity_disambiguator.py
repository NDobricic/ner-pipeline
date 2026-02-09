"""Unit tests for PopularityDisambiguatorComponent (spaCy-based)."""

import pytest
import spacy
from spacy.tokens import Span

from el_pipeline import spacy_components  # Register factories
from el_pipeline.spacy_components.disambiguators import PopularityDisambiguatorComponent
from el_pipeline.types import Candidate, Entity

from tests.conftest import MockKnowledgeBase


class TestPopularityDisambiguatorComponent:
    """Tests for PopularityDisambiguatorComponent class.

    Note: The PopularityDisambiguatorComponent in spaCy-based architecture
    uses LELA format candidates which don't have explicit scores.
    Candidates are already sorted by score from retrieval, so first = best.
    """

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
    def nlp(self, kb: MockKnowledgeBase) -> spacy.language.Language:
        nlp = spacy.blank("en")
        nlp.add_pipe("el_pipeline_simple", config={"min_len": 3})
        nlp.add_pipe("el_pipeline_fuzzy_candidates", config={"top_k": 3})
        component = nlp.add_pipe("el_pipeline_popularity_disambiguator")

        # Initialize components with KB
        for name, proc in nlp.pipeline:
            if hasattr(proc, "initialize"):
                proc.initialize(kb)

        return nlp

    def test_returns_highest_scored_candidate(self, nlp, kb):
        # In LELA format, candidates are pre-sorted by score
        # So first candidate has highest score
        doc = nlp.make_doc("Entity here.")

        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        span = doc.char_span(0, 6, label="ENT")
        if span:
            doc.ents = [span]
            # Candidates are sorted by score (highest first)
            span._.candidates = [
                Candidate(entity_id="Q2", description="Second entity"),  # Highest score
                Candidate(entity_id="Q1", description="First entity"),
                Candidate(entity_id="Q3", description="Third entity"),
            ]

        disambiguator = nlp.get_pipe("el_pipeline_popularity_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            # Should select first candidate (highest scored)
            assert doc.ents[0]._.resolved_entity is not None
            assert doc.ents[0]._.resolved_entity.id == "Q2"

    def test_empty_candidates_returns_none(self, nlp, kb):
        doc = nlp.make_doc("Nothing.")

        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        span = doc.char_span(0, 7, label="ENT")
        if span:
            doc.ents = [span]
            span._.candidates = []

        disambiguator = nlp.get_pipe("el_pipeline_popularity_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            assert doc.ents[0]._.resolved_entity is None

    def test_returns_entity_from_kb(self, nlp, kb):
        doc = nlp.make_doc("Entity Three here.")

        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        span = doc.char_span(0, 12, label="ENT")
        if span:
            doc.ents = [span]
            span._.candidates = [Candidate(entity_id="Q3", description="Third entity")]

        disambiguator = nlp.get_pipe("el_pipeline_popularity_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            ent = doc.ents[0]
            assert ent._.resolved_entity is not None
            assert ent._.resolved_entity.id == "Q3"
            assert ent._.resolved_entity.title == "Entity Three"

    def test_unknown_entity_returns_none(self, nlp, kb):
        doc = nlp.make_doc("Unknown.")

        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        span = doc.char_span(0, 7, label="ENT")
        if span:
            doc.ents = [span]
            span._.candidates = [Candidate(entity_id="unknown", description="Not in KB")]

        disambiguator = nlp.get_pipe("el_pipeline_popularity_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            assert doc.ents[0]._.resolved_entity is None

    def test_single_candidate(self, nlp, kb):
        doc = nlp.make_doc("Entity One.")

        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        span = doc.char_span(0, 10, label="ENT")
        if span:
            doc.ents = [span]
            span._.candidates = [Candidate(entity_id="Q1", description="First entity")]

        disambiguator = nlp.get_pipe("el_pipeline_popularity_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            assert doc.ents[0]._.resolved_entity is not None
            assert doc.ents[0]._.resolved_entity.id == "Q1"
