"""Unit tests for FirstDisambiguatorComponent (spaCy-based)."""

import pytest
import spacy
from spacy.tokens import Span

from el_pipeline import spacy_components  # Register factories
from el_pipeline.spacy_components.disambiguators import FirstDisambiguatorComponent
from el_pipeline.types import Candidate, Entity

from tests.conftest import MockKnowledgeBase


class TestFirstDisambiguatorComponent:
    """Tests for FirstDisambiguatorComponent class."""

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
        component = nlp.add_pipe("el_pipeline_first_disambiguator")

        # Initialize components with KB
        for name, proc in nlp.pipeline:
            if hasattr(proc, "initialize"):
                proc.initialize(kb)

        return nlp

    def test_returns_first_candidate(self, nlp, kb):
        # Create a doc and manually set candidates
        doc = nlp.make_doc("Entity One is here.")

        # Register extension if not already
        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        # Add an entity span
        span = doc.char_span(0, 10, label="ENT")
        if span:
            doc.ents = [span]
            # Set candidates manually (List[Candidate])
            span._.candidates = [
                Candidate(entity_id="Q1", description="First entity"),
                Candidate(entity_id="Q2", description="Second entity"),
                Candidate(entity_id="Q3", description="Third entity"),
            ]

        # Get the disambiguator component and call it
        disambiguator = nlp.get_pipe("el_pipeline_first_disambiguator")
        doc = disambiguator(doc)

        # Should select first candidate
        if doc.ents:
            ent = doc.ents[0]
            assert ent._.resolved_entity is not None
            assert ent._.resolved_entity.id == "Q1"

    def test_empty_candidates_no_resolution(self, nlp, kb):
        doc = nlp.make_doc("Nothing here.")

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

        disambiguator = nlp.get_pipe("el_pipeline_first_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            assert doc.ents[0]._.resolved_entity is None

    def test_returns_entity_from_kb(self, nlp, kb):
        doc = nlp.make_doc("Entity Two here.")

        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        span = doc.char_span(0, 10, label="ENT")
        if span:
            doc.ents = [span]
            span._.candidates = [Candidate(entity_id="Q2", description="Second entity")]

        disambiguator = nlp.get_pipe("el_pipeline_first_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            ent = doc.ents[0]
            assert ent._.resolved_entity is not None
            assert ent._.resolved_entity.id == "Q2"
            assert ent._.resolved_entity.title == "Entity Two"

    def test_unknown_entity_returns_none(self, nlp, kb):
        doc = nlp.make_doc("Unknown here.")

        if not Span.has_extension("candidates"):
            Span.set_extension("candidates", default=[])
        if not Span.has_extension("resolved_entity"):
            Span.set_extension("resolved_entity", default=None)
        if not Span.has_extension("context"):
            Span.set_extension("context", default=None)

        span = doc.char_span(0, 7, label="ENT")
        if span:
            doc.ents = [span]
            span._.candidates = [Candidate(entity_id="unknown_entity", description="Not in KB")]

        disambiguator = nlp.get_pipe("el_pipeline_first_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            # Entity not in KB, so resolved_entity should be None
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

        disambiguator = nlp.get_pipe("el_pipeline_first_disambiguator")
        doc = disambiguator(doc)

        if doc.ents:
            assert doc.ents[0]._.resolved_entity is not None
            assert doc.ents[0]._.resolved_entity.id == "Q1"
