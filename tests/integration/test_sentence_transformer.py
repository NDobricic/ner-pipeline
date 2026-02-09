"""Integration tests for SentenceTransformer components.

These tests actually load models and test real functionality.
They require model downloads and are marked as slow.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import spacy
from spacy.tokens import Span

from el_pipeline.types import Candidate, Entity
from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase


# Use a small, fast model for testing
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.slow
class TestSentenceTransformerPool:
    """Integration tests for SentenceTransformer pool functions."""

    def test_get_sentence_transformer_instance_loads_model(self):
        """Test that get_sentence_transformer_instance actually loads a model."""
        from el_pipeline.lela.llm_pool import (
            get_sentence_transformer_instance,
            clear_sentence_transformer_instances,
        )

        # Clear any cached instances
        clear_sentence_transformer_instances(force=True)

        model = get_sentence_transformer_instance(TEST_MODEL)

        # Model should be a SentenceTransformer instance
        assert model is not None
        assert hasattr(model, "encode")

    def test_model_can_encode_texts(self):
        """Test that the loaded model can encode texts."""
        from el_pipeline.lela.llm_pool import get_sentence_transformer_instance

        model = get_sentence_transformer_instance(TEST_MODEL)

        texts = ["Hello world", "This is a test"]
        embeddings = model.encode(texts, convert_to_numpy=True)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Has embedding dimensions

    def test_model_embeddings_are_meaningful(self):
        """Test that embeddings capture semantic similarity."""
        from el_pipeline.lela.llm_pool import get_sentence_transformer_instance

        model = get_sentence_transformer_instance(TEST_MODEL)

        # Similar texts
        texts = [
            "Barack Obama was the president",
            "Obama served as US president",
            "The weather is nice today",
        ]
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        # Compute cosine similarities
        sim_0_1 = np.dot(embeddings[0], embeddings[1])  # Similar texts
        sim_0_2 = np.dot(embeddings[0], embeddings[2])  # Different texts

        # Similar texts should have higher similarity
        assert sim_0_1 > sim_0_2

    def test_singleton_returns_same_instance(self):
        """Test that the same model instance is returned."""
        from el_pipeline.lela.llm_pool import (
            get_sentence_transformer_instance,
            clear_sentence_transformer_instances,
        )

        clear_sentence_transformer_instances(force=True)

        model1 = get_sentence_transformer_instance(TEST_MODEL)
        model2 = get_sentence_transformer_instance(TEST_MODEL)

        assert model1 is model2


@pytest.mark.slow
class TestLELAEmbedderRerankerIntegration:
    """Integration tests for LELAEmbedderRerankerComponent with real model."""

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @pytest.fixture
    def sample_candidates(self) -> list[Candidate]:
        return [
            Candidate(entity_id="E1", description="44th President of the United States"),
            Candidate(entity_id="E2", description="A type of tropical fruit"),
            Candidate(entity_id="E3", description="Capital city of France"),
            Candidate(entity_id="E4", description="American politician and former senator"),
            Candidate(entity_id="E5", description="A brand of computer software"),
        ]

    def test_reranker_sorts_by_semantic_similarity(self, nlp, sample_candidates):
        """Test that reranker sorts candidates by semantic similarity."""
        from el_pipeline.lela.llm_pool import clear_sentence_transformer_instances
        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent

        clear_sentence_transformer_instances(force=True)

        reranker = LELAEmbedderRerankerComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=3,
        )

        # Create a document with "Obama" as an entity
        doc = nlp("Barack Obama was the 44th president of the United States.")
        doc.ents = [Span(doc, 0, 2, label="PERSON")]  # "Barack Obama"
        doc.ents[0]._.candidates = sample_candidates

        # Rerank
        doc = reranker(doc)

        result = doc.ents[0]._.candidates

        # Should return top 3
        assert len(result) == 3

        # The "44th President" candidate should be ranked highly
        entity_ids = [c.entity_id for c in result]
        # E1 (44th President) should be in top 3 given context about president
        assert "E1" in entity_ids

    def test_reranker_assigns_similarity_scores(self, nlp):
        """Test that reranker assigns meaningful similarity scores."""
        from el_pipeline.lela.llm_pool import clear_sentence_transformer_instances
        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent

        clear_sentence_transformer_instances(force=True)

        # Need more candidates than top_k to trigger reranking
        many_candidates = [
            Candidate(entity_id="E1", description="44th President of the United States"),
            Candidate(entity_id="E2", description="A type of tropical fruit"),
            Candidate(entity_id="E3", description="Capital city of France"),
            Candidate(entity_id="E4", description="American politician and former senator"),
            Candidate(entity_id="E5", description="A brand of computer software"),
            Candidate(entity_id="E6", description="A popular music artist"),
            Candidate(entity_id="E7", description="A sports team from Chicago"),
        ]

        reranker = LELAEmbedderRerankerComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=3,  # Less than number of candidates to trigger reranking
        )

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="PERSON")]
        doc.ents[0]._.candidates = many_candidates

        doc = reranker(doc)
        result = doc.ents[0]._.candidates

        # Should return top_k candidates
        assert len(result) == 3

        # All candidates should have scores
        for candidate in result:
            assert candidate.score is not None
            assert isinstance(candidate.score, float)

        # Scores should be sorted in descending order
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_reranker_handles_empty_candidates(self, nlp):
        """Test that reranker handles entities with no candidates."""
        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent

        reranker = LELAEmbedderRerankerComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=3,
        )

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="PERSON")]
        doc.ents[0]._.candidates = []

        doc = reranker(doc)
        assert doc.ents[0]._.candidates == []


@pytest.mark.slow
class TestLELADenseCandidatesIntegration:
    """Integration tests for LELADenseCandidatesComponent with real model."""

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @pytest.fixture
    def kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th President of the United States"},
            {"title": "Michelle Obama", "description": "Former First Lady of the United States"},
            {"title": "Joe Biden", "description": "46th President of the United States"},
            {"title": "Paris", "description": "Capital city of France"},
            {"title": "Apple Inc.", "description": "American technology company"},
            {"title": "Harvard University", "description": "Private university in Cambridge, Massachusetts"},
        ]

    @pytest.fixture
    def temp_kb_file(self, kb_data: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def kb(self, temp_kb_file: str) -> CustomJSONLKnowledgeBase:
        return CustomJSONLKnowledgeBase(path=temp_kb_file)

    def test_dense_candidates_returns_semantically_similar(self, nlp, kb):
        """Test that dense retrieval returns semantically similar candidates."""
        from el_pipeline.lela.llm_pool import clear_sentence_transformer_instances
        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent

        clear_sentence_transformer_instances(force=True)

        component = LELADenseCandidatesComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=3,
            use_context=False,
        )
        component.initialize(kb)

        # Search for "Obama" - should return Obama-related entities
        doc = nlp("Obama gave a speech about education.")
        doc.ents = [Span(doc, 0, 1, label="PERSON")]

        doc = component(doc)
        candidates = doc.ents[0]._.candidates

        assert len(candidates) > 0
        assert len(candidates) <= 3

        # Check that Obama-related entities are returned
        entity_ids = [c.entity_id for c in candidates]
        # At least one Obama should be in results
        assert any("Obama" in eid for eid in entity_ids)

    def test_dense_candidates_uses_context(self, nlp, kb):
        """Test that context improves retrieval results."""
        from el_pipeline.lela.llm_pool import clear_sentence_transformer_instances
        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        from spacy.tokens import Span as SpacySpan

        if not SpacySpan.has_extension("context"):
            SpacySpan.set_extension("context", default=None)

        clear_sentence_transformer_instances(force=True)

        component = LELADenseCandidatesComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=3,
            use_context=True,
        )
        component.initialize(kb)

        # Search for "Obama" with context about being president
        doc = nlp("Obama was the 44th president.")
        doc.ents = [Span(doc, 0, 1, label="PERSON")]
        doc.ents[0]._.context = "was the 44th president"

        doc = component(doc)
        candidates = doc.ents[0]._.candidates

        assert len(candidates) > 0

        # Barack Obama (the president) should be ranked higher than Michelle Obama
        entity_ids = [c.entity_id for c in candidates]
        if "Barack Obama" in entity_ids and "Michelle Obama" in entity_ids:
            barack_idx = entity_ids.index("Barack Obama")
            michelle_idx = entity_ids.index("Michelle Obama")
            assert barack_idx < michelle_idx, "Barack should rank higher with president context"

    def test_dense_candidates_assigns_scores(self, nlp, kb):
        """Test that candidates have similarity scores."""
        from el_pipeline.lela.llm_pool import clear_sentence_transformer_instances
        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent

        clear_sentence_transformer_instances(force=True)

        component = LELADenseCandidatesComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=5,
            use_context=False,
        )
        component.initialize(kb)

        doc = nlp("The president spoke today.")
        doc.ents = [Span(doc, 1, 2, label="PERSON")]  # "president"

        doc = component(doc)
        candidates = doc.ents[0]._.candidates

        # All candidates should have scores in descending order
        scores = [c.score for c in candidates]
        assert all(isinstance(s, float) for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_dense_candidates_handles_no_matches(self, nlp, kb):
        """Test that component handles queries with poor matches gracefully."""
        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent

        component = LELADenseCandidatesComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=3,
            use_context=False,
        )
        component.initialize(kb)

        # Query something not in KB
        doc = nlp("XYZ123 is a random string.")
        doc.ents = [Span(doc, 0, 1, label="MISC")]

        doc = component(doc)
        candidates = doc.ents[0]._.candidates

        # Should still return some candidates (closest matches)
        assert len(candidates) > 0

    def test_index_built_correctly(self, nlp, kb):
        """Test that the FAISS index is built with correct dimensions."""
        from el_pipeline.lela.llm_pool import clear_sentence_transformer_instances
        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent

        clear_sentence_transformer_instances(force=True)

        component = LELADenseCandidatesComponent(
            nlp=nlp,
            model_name=TEST_MODEL,
            top_k=3,
            use_context=False,
        )
        component.initialize(kb)

        # Index should have 6 entities
        assert component.index.ntotal == 6
        assert len(component.entities) == 6
