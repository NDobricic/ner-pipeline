"""Unit tests for LELAEmbedderRerankerComponent."""

from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import spacy
from spacy.tokens import Span

from ner_pipeline.types import Candidate, Document, Mention


class TestLELAEmbedderRerankerComponent:
    """Tests for LELAEmbedderRerankerComponent class."""

    @pytest.fixture
    def sample_candidates(self) -> list[tuple[str, str]]:
        return [
            ("E1", "Description 1"),
            ("E2", "Description 2"),
            ("E3", "Description 3"),
            ("E4", "Description 4"),
            ("E5", "Description 5"),
        ]

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_rerank_returns_candidates(
        self, mock_pool, sample_candidates, nlp
    ):
        # Mock embeddings
        mock_pool.embed.return_value = [
            [0.1, 0.2, 0.3],  # query
            [0.2, 0.3, 0.4],  # E1
            [0.3, 0.4, 0.5],  # E2
            [0.4, 0.5, 0.6],  # E3
            [0.5, 0.6, 0.7],  # E4
            [0.6, 0.7, 0.8],  # E5
        ]

        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert len(result) == 3
        assert all(isinstance(c, tuple) and len(c) == 2 for c in result)

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_rerank_respects_top_k(
        self, mock_pool, sample_candidates, nlp
    ):
        mock_pool.embed.return_value = [[0.1, 0.2, 0.3]] * 6

        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=2)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert len(result) == 2

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_rerank_returns_all_if_fewer_than_top_k(
        self, mock_pool, nlp
    ):
        candidates = [("E1", "Desc 1")]

        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=5)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        # Should return all candidates without calling embeddings
        assert len(result) == 1

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_rerank_empty_candidates(
        self, mock_pool, nlp
    ):
        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = []
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert result == []

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_rerank_sorts_by_similarity(
        self, mock_pool, sample_candidates, nlp
    ):
        # Create embeddings where E3 is most similar to query
        query_emb = [1.0, 0.0, 0.0]
        mock_pool.embed.return_value = [
            query_emb,
            [0.1, 0.9, 0.0],  # E1 - low similarity
            [0.2, 0.8, 0.0],  # E2
            [0.9, 0.1, 0.0],  # E3 - high similarity
            [0.3, 0.7, 0.0],  # E4
            [0.4, 0.6, 0.0],  # E5
        ]

        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        # E3 should be first (highest similarity)
        assert result[0][0] == "E3"

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_query_includes_marked_mention(
        self, mock_pool, sample_candidates, nlp
    ):
        embed_calls = []
        def capture_embed(texts, **kwargs):
            embed_calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)
        mock_pool.embed.side_effect = capture_embed

        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        # First text in embed call is the query
        query_text = embed_calls[0][0]
        assert "[Obama]" in query_text  # Mention should be marked
        assert "Instruct:" in query_text

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_candidates_formatted_for_embedding(
        self, mock_pool, nlp
    ):
        candidates = [
            ("Entity A", "Description A"),
            ("Entity B", "Description B"),
            ("Entity C", None),  # No description
        ]

        embed_calls = []
        def capture_embed(texts, **kwargs):
            embed_calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)
        mock_pool.embed.side_effect = capture_embed

        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=2)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = reranker(doc)

        # Check candidate formatting
        texts = embed_calls[0]
        # Query + 3 candidates = 4 texts
        assert len(texts) == 4
        assert "Entity A: Description A" in texts[1]
        assert "Entity B: Description B" in texts[2]
        assert "Entity C" in texts[3]  # No description

    @patch("ner_pipeline.spacy_components.rerankers.embedder_pool")
    def test_preserves_descriptions(
        self, mock_pool, sample_candidates, nlp
    ):
        mock_pool.embed.return_value = [[0.1, 0.2, 0.3]] * 6

        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        # Descriptions should be preserved
        for title, desc in result:
            original = next(o for o in sample_candidates if o[0] == title)
            assert desc == original[1]

    def test_initialization_with_custom_params(self):
        nlp = spacy.blank("en")
        from ner_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(
            nlp=nlp,
            model_name="custom-model",
            top_k=5,
            base_url="http://custom-host",
            port=9000,
        )

        assert reranker.model_name == "custom-model"
        assert reranker.top_k == 5
        assert reranker.base_url == "http://custom-host"
        assert reranker.port == 9000
