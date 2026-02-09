"""Unit tests for LELABM25CandidatesComponent."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import spacy
from spacy.tokens import Span

from el_pipeline.types import Candidate, Document, Entity, Mention
from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase


class TestLELABM25CandidatesComponent:
    """Tests for LELABM25CandidatesComponent class."""

    @pytest.fixture
    def lela_kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President from Illinois"},
            {"title": "Michelle Obama", "description": "Former First Lady and author"},
            {"title": "Joe Biden", "description": "46th US President from Delaware"},
            {"title": "United States", "description": "Country in North America"},
            {"title": "White House", "description": "Official residence of the US President"},
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
        return Document(
            id="test-doc",
            text="Barack Obama was the 44th President of the United States."
        )

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    def test_requires_knowledge_base(self, nlp):
        with patch("el_pipeline.spacy_components.candidates._get_bm25s"):
            with patch("el_pipeline.spacy_components.candidates._get_stemmer"):
                from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
                component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
                # Component returns doc unchanged when not initialized (logs warning)
                doc = nlp("Test")
                doc.ents = [Span(doc, 0, 1, label="ENTITY")]
                result = component(doc)
                # Candidates should remain empty since KB not initialized
                assert result.ents[0]._.candidates == []

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_generate_returns_candidates(
        self, mock_bm25s, mock_stemmer, kb, sample_doc, nlp
    ):
        # Setup mocks
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        # Setup retrieve response - include id field for proper entity tracking
        mock_results = MagicMock()
        mock_results.documents = [[
            {"id": "Barack Obama", "title": "Barack Obama", "description": "44th US President"},
            {"id": "Michelle Obama", "title": "Michelle Obama", "description": "Former First Lady"},
        ]]
        mock_results.scores = [[0.9, 0.7]]
        mock_retriever.retrieve.return_value = mock_results

        mock_bm25s.return_value.tokenize.return_value = [["obama"]]

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        doc = nlp("Barack Obama was president.")
        doc.ents = [Span(doc, 0, 2, label="ENTITY")]
        doc = component(doc)

        candidates = doc.ents[0]._.candidates
        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_candidates_have_entity_ids(
        self, mock_bm25s, mock_stemmer, kb, sample_doc, nlp
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[
            {"id": "Barack Obama", "title": "Barack Obama", "description": "44th US President"},
        ]]
        mock_results.scores = [[0.9]]
        mock_retriever.retrieve.return_value = mock_results
        mock_bm25s.return_value.tokenize.return_value = [["obama"]]

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        doc = nlp("Barack Obama was president.")
        doc.ents = [Span(doc, 0, 2, label="ENTITY")]
        doc = component(doc)

        candidates = doc.ents[0]._.candidates
        assert candidates[0].entity_id == "Barack Obama"

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_candidates_have_descriptions(
        self, mock_bm25s, mock_stemmer, kb, sample_doc, nlp
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[
            {"id": "Barack Obama", "title": "Barack Obama", "description": "44th US President"},
        ]]
        mock_results.scores = [[0.9]]
        mock_retriever.retrieve.return_value = mock_results
        mock_bm25s.return_value.tokenize.return_value = [["obama"]]

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        doc = nlp("Barack Obama was president.")
        doc.ents = [Span(doc, 0, 2, label="ENTITY")]
        doc = component(doc)

        candidates = doc.ents[0]._.candidates
        assert candidates[0].description == "44th US President"

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_empty_tokenization_returns_empty(
        self, mock_bm25s, mock_stemmer, kb, sample_doc, nlp
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        # Empty tokenization
        mock_bm25s.return_value.tokenize.return_value = [[]]

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        doc = nlp("...")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc = component(doc)

        candidates = doc.ents[0]._.candidates
        assert candidates == []

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_use_context_includes_context_in_query(
        self, mock_bm25s, mock_stemmer, kb, sample_doc, nlp
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[]]
        mock_results.scores = [[]]
        mock_retriever.retrieve.return_value = mock_results

        tokenize_calls = []
        def capture_tokenize(texts, **kwargs):
            tokenize_calls.append(texts)
            return [[]]
        mock_bm25s.return_value.tokenize.side_effect = capture_tokenize

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=True)
        component.initialize(kb)

        doc = nlp("Barack Obama was the 44th President")
        doc.ents = [Span(doc, 0, 2, label="ENTITY")]
        # Set context on the entity
        doc.ents[0]._.context = "was the 44th President"
        doc = component(doc)

        # Check that context was included in the query
        assert len(tokenize_calls) > 0
        query = tokenize_calls[0][0]
        assert "Barack Obama" in query
        assert "44th President" in query

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_use_context_false_excludes_context(
        self, mock_bm25s, mock_stemmer, kb, sample_doc, nlp
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[]]
        mock_results.scores = [[]]
        mock_retriever.retrieve.return_value = mock_results

        tokenize_calls = []
        def capture_tokenize(texts, **kwargs):
            tokenize_calls.append(texts)
            return [[]]
        mock_bm25s.return_value.tokenize.side_effect = capture_tokenize

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        doc = nlp("Barack Obama was the 44th President")
        doc.ents = [Span(doc, 0, 2, label="ENTITY")]
        doc.ents[0]._.context = "was the 44th President"
        doc = component(doc)

        # Context should not be in query when use_context=False
        assert len(tokenize_calls) > 0
        query = tokenize_calls[0][0]
        assert query == "Barack Obama"

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_respects_top_k(self, mock_bm25s, mock_stemmer, kb, sample_doc, nlp):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_bm25s.return_value.tokenize.return_value = [["test"]]

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=3, use_context=False)
        component.initialize(kb)

        # Check that top_k is stored
        assert component.top_k == 3
