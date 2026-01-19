"""Unit tests for LELADenseCandidatesComponent."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import spacy
from spacy.tokens import Span

from ner_pipeline.types import Candidate, Document, Mention
from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase


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
    def kb(self, temp_kb_file: str) -> LELAJSONLKnowledgeBase:
        return LELAJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(id="test-doc", text="Test document about Obama.")

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("ner_pipeline.spacy_components.candidates._get_faiss")
    @patch("ner_pipeline.spacy_components.candidates.embedder_pool")
    def test_requires_knowledge_base(self, mock_pool, mock_faiss, nlp):
        from ner_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        # Component returns doc unchanged when not initialized (logs warning)
        doc = nlp("Test")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        result = component(doc)
        # Candidates should remain empty since KB not initialized
        assert result.ents[0]._.candidates == []

    @patch("ner_pipeline.spacy_components.candidates._get_faiss")
    @patch("ner_pipeline.spacy_components.candidates.embedder_pool")
    def test_initialization_embeds_entities(self, mock_pool, mock_faiss, kb, nlp):
        # Setup mocks
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        # Return fake embeddings
        mock_pool.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        from ner_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        # Should have called embed with entity texts
        mock_pool.embed.assert_called_once()
        embed_args = mock_pool.embed.call_args[0][0]
        assert len(embed_args) == 3  # 3 entities

        # Index should have been created
        mock_faiss_module.IndexFlatIP.assert_called_once_with(3)  # dim=3

    @patch("ner_pipeline.spacy_components.candidates._get_faiss")
    @patch("ner_pipeline.spacy_components.candidates.embedder_pool")
    def test_generate_returns_candidates(self, mock_pool, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        # Initial embedding for entities
        mock_pool.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        from ner_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        # Query embedding
        mock_pool.embed.return_value = [[0.2, 0.3, 0.4]]

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
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)

    @patch("ner_pipeline.spacy_components.candidates._get_faiss")
    @patch("ner_pipeline.spacy_components.candidates.embedder_pool")
    def test_candidates_have_descriptions(self, mock_pool, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_pool.embed.return_value = [[0.1, 0.2, 0.3]] * 3

        from ner_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        mock_pool.embed.return_value = [[0.2, 0.3, 0.4]]
        mock_index.search.return_value = (
            np.array([[0.95]]),
            np.array([[0]]),  # First entity
        )

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc = component(doc)

        candidates = doc.ents[0]._.candidates
        # First entity is "Barack Obama"
        assert candidates[0][0] == "Barack Obama"
        assert candidates[0][1] == "44th US President"

    @patch("ner_pipeline.spacy_components.candidates._get_faiss")
    @patch("ner_pipeline.spacy_components.candidates.embedder_pool")
    def test_query_includes_task_instruction(self, mock_pool, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        embed_calls = []
        def capture_embed(texts, **kwargs):
            embed_calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)
        mock_pool.embed.side_effect = capture_embed

        from ner_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc = component(doc)

        # Second call is the query embedding
        query_text = embed_calls[1][0]
        assert "Instruct:" in query_text
        assert "Query:" in query_text
        assert "Obama" in query_text

    @patch("ner_pipeline.spacy_components.candidates._get_faiss")
    @patch("ner_pipeline.spacy_components.candidates.embedder_pool")
    def test_use_context_includes_context(self, mock_pool, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        embed_calls = []
        def capture_embed(texts, **kwargs):
            embed_calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)
        mock_pool.embed.side_effect = capture_embed

        from ner_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=True)
        component.initialize(kb)

        mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))

        doc = nlp("Obama was the 44th President")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.context = "was the 44th President"
        doc = component(doc)

        query_text = embed_calls[1][0]
        assert "Obama" in query_text
        assert "44th President" in query_text

    @patch("ner_pipeline.spacy_components.candidates._get_faiss")
    @patch("ner_pipeline.spacy_components.candidates.embedder_pool")
    def test_respects_top_k(self, mock_pool, mock_faiss, kb, sample_doc, nlp):
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_pool.embed.return_value = [[0.1, 0.2, 0.3]] * 3

        from ner_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=2, use_context=False)
        component.initialize(kb)

        mock_pool.embed.return_value = [[0.2, 0.3, 0.4]]
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
