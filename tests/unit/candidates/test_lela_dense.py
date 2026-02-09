"""Unit tests for LELADenseCandidatesComponent."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import spacy
from spacy.tokens import Span

from el_pipeline.types import Candidate, Document, Mention
from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase


class TestLELADenseCandidatesComponent:
    """Tests for LELADenseCandidatesComponent class."""

    @pytest.fixture
    def lela_kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Michelle Obama", "description": "Former First Lady"},
            {"title": "Joe Biden", "description": "46th US President"},
        ]

    @pytest.fixture
    def temp_kb_file(self, lela_kb_data: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in lela_kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def kb(self, temp_kb_file: str) -> CustomJSONLKnowledgeBase:
        return CustomJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(id="test-doc", text="Test document about Obama.")

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
    def test_requires_knowledge_base(self, mock_get_st, mock_faiss, nlp):
        mock_model = MagicMock()
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        # Component returns doc unchanged when not initialized (logs warning)
        doc = nlp("Test")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        result = component(doc)
        # Candidates should remain empty since KB not initialized
        assert result.ents[0]._.candidates == []

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
    def test_initialization_embeds_entities(self, mock_get_st, mock_faiss, kb, nlp):
        # Setup mocks
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        # Setup model mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        # Should have called encode with entity texts
        mock_model.encode.assert_called_once()
        encode_args = mock_model.encode.call_args[0][0]
        assert len(encode_args) == 3  # 3 entities

        # Index should have been created
        mock_faiss_module.IndexFlatIP.assert_called_once_with(3)  # dim=3

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
    def test_generate_returns_candidates(self, mock_get_st, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        # Setup model mock with different return values for init and query
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            # Initial embedding for entities
            np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]),
            # Query embedding
            np.array([[0.2, 0.3, 0.4]]),
        ]
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        # Search results
        mock_index.search.return_value = (
            np.array([[0.95, 0.85]]),  # scores
            np.array([[0, 1]]),  # indices
        )

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc = component(doc)

        candidates = doc.ents[0]._.candidates
        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
    def test_candidates_have_descriptions(self, mock_get_st, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3]] * 3),
            np.array([[0.2, 0.3, 0.4]]),
        ]
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        mock_index.search.return_value = (
            np.array([[0.95]]),
            np.array([[0]]),  # First entity
        )

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc = component(doc)

        candidates = doc.ents[0]._.candidates
        # First entity is "Barack Obama"
        assert candidates[0].entity_id == "Barack Obama"
        assert candidates[0].description == "44th US President"

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
    def test_query_includes_task_instruction(self, mock_get_st, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_model = MagicMock()
        encode_calls = []
        def capture_encode(texts, **kwargs):
            encode_calls.append(texts)
            return np.array([[0.1, 0.2, 0.3]] * len(texts))
        mock_model.encode.side_effect = capture_encode
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc = component(doc)

        # Second call is the query embedding
        query_text = encode_calls[1][0]
        assert "Instruct:" in query_text
        assert "Query:" in query_text
        assert "Obama" in query_text

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
    def test_use_context_includes_context(self, mock_get_st, mock_faiss, kb, sample_doc, nlp):
        from spacy.tokens import Span as SpacySpan
        if not SpacySpan.has_extension("context"):
            SpacySpan.set_extension("context", default=None)

        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_model = MagicMock()
        encode_calls = []
        def capture_encode(texts, **kwargs):
            encode_calls.append(texts)
            return np.array([[0.1, 0.2, 0.3]] * len(texts))
        mock_model.encode.side_effect = capture_encode
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=True)
        component.initialize(kb)

        mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))

        doc = nlp("Obama was the 44th President")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.context = "was the 44th President"
        doc = component(doc)

        query_text = encode_calls[1][0]
        assert "Obama" in query_text
        assert "44th President" in query_text

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
    def test_respects_top_k(self, mock_get_st, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3]] * 3),
            np.array([[0.2, 0.3, 0.4]]),
        ]
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=2, use_context=False)
        component.initialize(kb)

        mock_index.search.return_value = (
            np.array([[0.95, 0.85]]),
            np.array([[0, 1]]),
        )

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc = component(doc)

        # Check search was called with correct k
        mock_index.search.assert_called_once()
        call_args = mock_index.search.call_args[0]
        assert call_args[1] == 2  # k parameter
