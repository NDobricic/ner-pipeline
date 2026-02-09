"""Unit tests for LELAEmbedderRerankerComponent."""

from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import spacy
from spacy.tokens import Span

from el_pipeline.types import Candidate, Document, Mention


class TestLELAEmbedderRerankerComponent:
    """Tests for LELAEmbedderRerankerComponent class."""

    @pytest.fixture
    def sample_candidates(self) -> list[Candidate]:
        return [
            Candidate(entity_id="E1", description="Description 1"),
            Candidate(entity_id="E2", description="Description 2"),
            Candidate(entity_id="E3", description="Description 3"),
            Candidate(entity_id="E4", description="Description 4"),
            Candidate(entity_id="E5", description="Description 5"),
        ]

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_rerank_returns_candidates(
        self, mock_get_st, sample_candidates, nlp
    ):
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],  # query
            [0.2, 0.3, 0.4],  # E1
            [0.3, 0.4, 0.5],  # E2
            [0.4, 0.5, 0.6],  # E3
            [0.5, 0.6, 0.7],  # E4
            [0.6, 0.7, 0.8],  # E5
        ])
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert len(result) == 3
        assert all(isinstance(c, Candidate) for c in result)

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_rerank_respects_top_k(
        self, mock_get_st, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]] * 6)
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=2)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert len(result) == 2

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_rerank_returns_all_if_fewer_than_top_k(
        self, mock_get_st, nlp
    ):
        mock_model = MagicMock()
        mock_get_st.return_value = mock_model
        candidates = [Candidate(entity_id="E1", description="Desc 1")]

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=5)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        # Should return all candidates without calling embeddings
        assert len(result) == 1

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_rerank_empty_candidates(
        self, mock_get_st, nlp
    ):
        mock_model = MagicMock()
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = []
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert result == []

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_rerank_sorts_by_similarity(
        self, mock_get_st, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        # Create embeddings where E3 is most similar to query
        query_emb = [1.0, 0.0, 0.0]
        mock_model.encode.return_value = np.array([
            query_emb,
            [0.1, 0.9, 0.0],  # E1 - low similarity
            [0.2, 0.8, 0.0],  # E2
            [0.9, 0.1, 0.0],  # E3 - high similarity
            [0.3, 0.7, 0.0],  # E4
            [0.4, 0.6, 0.0],  # E5
        ])
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        # E3 should be first (highest similarity)
        assert result[0].entity_id == "E3"

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_query_includes_marked_mention(
        self, mock_get_st, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        encode_calls = []
        def capture_encode(texts, **kwargs):
            encode_calls.append(texts)
            return np.array([[0.1, 0.2, 0.3]] * len(texts))
        mock_model.encode.side_effect = capture_encode
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        # First text in encode call is the query
        query_text = encode_calls[0][0]
        assert "[Obama]" in query_text  # Mention should be marked
        assert "Instruct:" in query_text

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_candidates_formatted_for_embedding(
        self, mock_get_st, nlp
    ):
        candidates = [
            Candidate(entity_id="Entity A", description="Description A"),
            Candidate(entity_id="Entity B", description="Description B"),
            Candidate(entity_id="Entity C", description=None),  # No description
        ]

        mock_model = MagicMock()
        encode_calls = []
        def capture_encode(texts, **kwargs):
            encode_calls.append(texts)
            return np.array([[0.1, 0.2, 0.3]] * len(texts))
        mock_model.encode.side_effect = capture_encode
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=2)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = reranker(doc)

        # Check candidate formatting
        texts = encode_calls[0]
        # Query + 3 candidates = 4 texts
        assert len(texts) == 4
        assert "Entity A: Description A" in texts[1]
        assert "Entity B: Description B" in texts[2]
        assert "Entity C" in texts[3]  # No description

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_preserves_descriptions(
        self, mock_get_st, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]] * 6)
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        # Descriptions should be preserved
        for candidate in result:
            original = next(o for o in sample_candidates if o.entity_id == candidate.entity_id)
            assert candidate.description == original.description

    @patch("el_pipeline.spacy_components.rerankers.get_sentence_transformer_instance")
    def test_initialization_with_custom_params(self, mock_get_st):
        import numpy as np
        from spacy.tokens import Span
        from el_pipeline.types import Candidate
        from el_pipeline.utils import ensure_candidates_extension

        ensure_candidates_extension()

        mock_model = MagicMock()
        # Mock model.encode to return embeddings (1 query + 10 candidates = 11)
        mock_model.encode.return_value = np.array([[0.5, 0.5]] * 11)
        mock_get_st.return_value = mock_model

        nlp = spacy.blank("en")
        from el_pipeline.spacy_components.rerankers import LELAEmbedderRerankerComponent
        reranker = LELAEmbedderRerankerComponent(
            nlp=nlp,
            model_name="custom-model",
            top_k=5,
            device="cuda",
        )

        assert reranker.model_name == "custom-model"
        assert reranker.top_k == 5
        assert reranker.device == "cuda"

        # Create a doc with entity and candidates to trigger lazy loading
        # Need more candidates than top_k (5) to trigger reranking
        doc = nlp.make_doc("Test Entity is here.")
        doc.ents = [Span(doc, 0, 2, label="TEST")]
        # Set candidates on the Span - need > top_k candidates to trigger model loading
        doc.ents[0]._.candidates = [
            Candidate(entity_id=str(i), score=1.0, description=f"Test {i}")
            for i in range(10)  # 10 > top_k (5)
        ]

        # Call the component to trigger lazy model loading
        reranker(doc)

        mock_get_st.assert_called_once_with("custom-model", "cuda")
