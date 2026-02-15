"""Unit tests for candidate generator caching."""

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import spacy

from lela.knowledge_bases.jsonl import JSONLKnowledgeBase
from lela.types import Entity


class TestLELADenseCache:
    """Tests for LELADenseCandidatesComponent caching."""

    @pytest.fixture
    def kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Joe Biden", "description": "46th US President"},
            {"title": "United States", "description": "Country in North America"},
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
    def kb(self, temp_kb_file: str) -> JSONLKnowledgeBase:
        return JSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("lela.spacy_components.candidates._get_faiss")
    @patch("lela.spacy_components.candidates.release_sentence_transformer")
    @patch("lela.spacy_components.candidates.get_sentence_transformer_instance")
    def test_initialize_saves_cache(
        self, mock_get_st, mock_release_st, mock_faiss, kb, cache_dir, nlp
    ):
        """First initialize with cache_dir writes FAISS index file."""
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        mock_get_st.return_value = (mock_model, False)

        from lela.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb, cache_dir=Path(cache_dir))

        # faiss.write_index should have been called
        mock_faiss_module.write_index.assert_called_once()
        write_args = mock_faiss_module.write_index.call_args[0]
        assert write_args[0] is mock_index
        assert "dense_" in write_args[1]
        assert write_args[1].endswith("index.faiss")

    @patch("lela.spacy_components.candidates._get_faiss")
    @patch("lela.spacy_components.candidates.release_sentence_transformer")
    @patch("lela.spacy_components.candidates.get_sentence_transformer_instance")
    def test_initialize_loads_from_cache(
        self, mock_get_st, mock_release_st, mock_faiss, kb, cache_dir, nlp
    ):
        """Second initialize loads FAISS index from cache, skips model load."""
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
        mock_get_st.return_value = (mock_model, False)

        from lela.spacy_components.candidates import LELADenseCandidatesComponent

        # First call: builds and saves
        component1 = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component1.initialize(kb, cache_dir=Path(cache_dir))

        # write_index was called, so we need to make the file exist
        # Extract the path that write_index was called with
        write_path = mock_faiss_module.write_index.call_args[0][1]
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        # Create the file so the cache check finds it
        with open(write_path, "wb") as f:
            f.write(b"fake faiss index")

        mock_cached_index = MagicMock()
        mock_cached_index.ntotal = 3
        mock_faiss_module.read_index.return_value = mock_cached_index

        # Reset mocks
        mock_get_st.reset_mock()
        mock_model.encode.reset_mock()

        # Second call: should load from cache
        component2 = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component2.initialize(kb, cache_dir=Path(cache_dir))

        # read_index should have been called (cache hit)
        mock_faiss_module.read_index.assert_called_once_with(write_path)
        # SentenceTransformer should NOT have been loaded for the second init
        mock_get_st.assert_not_called()
        mock_model.encode.assert_not_called()
        # The cached index should be assigned
        assert component2.index is mock_cached_index

    @patch("lela.spacy_components.candidates._get_faiss")
    @patch("lela.spacy_components.candidates.release_sentence_transformer")
    @patch("lela.spacy_components.candidates.get_sentence_transformer_instance")
    def test_no_cache_dir_skips_caching(
        self, mock_get_st, mock_release_st, mock_faiss, kb, nlp
    ):
        """Without cache_dir, no cache operations occur."""
        mock_faiss_module = MagicMock()
        mock_faiss.return_value = mock_faiss_module

        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_faiss_module.IndexFlatIP.return_value = mock_index

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]] * 3)
        mock_get_st.return_value = (mock_model, False)

        from lela.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        # Should build normally
        mock_model.encode.assert_called_once()
        mock_faiss_module.IndexFlatIP.assert_called_once()
        # No read/write cache calls
        mock_faiss_module.read_index.assert_not_called()
        mock_faiss_module.write_index.assert_not_called()


class TestBM25Cache:
    """Tests for BM25CandidatesComponent (rank-bm25) caching."""

    @pytest.fixture
    def kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Joe Biden", "description": "46th US President"},
            {"title": "United States", "description": "Country in North America"},
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
    def kb(self, temp_kb_file: str) -> JSONLKnowledgeBase:
        return JSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    def test_initialize_saves_cache(self, kb, cache_dir, nlp):
        """First initialize with cache_dir creates a pickle file."""
        from lela.spacy_components.candidates import BM25CandidatesComponent
        component = BM25CandidatesComponent(nlp=nlp, top_k=5)
        component.initialize(kb, cache_dir=Path(cache_dir))

        # Check cache file was created
        index_dir = os.path.join(cache_dir, "index")
        assert os.path.isdir(index_dir)
        cache_files = [f for f in os.listdir(index_dir) if f.startswith("bm25_") and f.endswith(".pkl")]
        assert len(cache_files) == 1

        # Verify pickled contents
        cache_path = os.path.join(index_dir, cache_files[0])
        with open(cache_path, "rb") as f:
            bm25, corpus = pickle.load(f)
        assert len(corpus) == 3

    def test_initialize_loads_from_cache(self, kb, cache_dir, nlp):
        """Second initialize loads from pickle cache."""
        from lela.spacy_components.candidates import BM25CandidatesComponent

        # First call: builds and saves
        component1 = BM25CandidatesComponent(nlp=nlp, top_k=5)
        component1.initialize(kb, cache_dir=Path(cache_dir))
        original_bm25 = component1.bm25
        original_corpus = component1.corpus

        # Second call: loads from cache
        component2 = BM25CandidatesComponent(nlp=nlp, top_k=5)
        component2.initialize(kb, cache_dir=Path(cache_dir))

        # Data should match
        assert len(component2.corpus) == len(original_corpus)
        assert component2.corpus == original_corpus
        assert component2.bm25 is not None

    def test_no_cache_dir_works(self, kb, nlp):
        """Without cache_dir, initialization works normally."""
        from lela.spacy_components.candidates import BM25CandidatesComponent
        component = BM25CandidatesComponent(nlp=nlp, top_k=5)
        component.initialize(kb)
        assert component.bm25 is not None
        assert len(component.corpus) == 3

    def test_corrupt_cache_falls_back_to_build(self, kb, cache_dir, nlp):
        """Corrupt cache file triggers rebuild."""
        from lela.spacy_components.candidates import BM25CandidatesComponent

        # First call to create cache
        component1 = BM25CandidatesComponent(nlp=nlp, top_k=5)
        component1.initialize(kb, cache_dir=Path(cache_dir))

        # Corrupt the cache file
        index_dir = os.path.join(cache_dir, "index")
        cache_files = [f for f in os.listdir(index_dir) if f.startswith("bm25_")]
        cache_path = os.path.join(index_dir, cache_files[0])
        with open(cache_path, "wb") as f:
            f.write(b"corrupted")

        # Should still work, rebuilding from scratch
        component2 = BM25CandidatesComponent(nlp=nlp, top_k=5)
        component2.initialize(kb, cache_dir=Path(cache_dir))
        assert component2.bm25 is not None
        assert len(component2.corpus) == 3


class TestFuzzyCache:
    """Tests for FuzzyCandidatesComponent cache_dir parameter."""

    @pytest.fixture
    def kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Joe Biden", "description": "46th US President"},
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
    def kb(self, temp_kb_file: str) -> JSONLKnowledgeBase:
        return JSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    def test_initialize_accepts_cache_dir(self, kb, nlp):
        """FuzzyCandidatesComponent.initialize accepts cache_dir without error."""
        with tempfile.TemporaryDirectory() as cache_dir:
            from lela.spacy_components.candidates import FuzzyCandidatesComponent
            component = FuzzyCandidatesComponent(nlp=nlp, top_k=5)
            component.initialize(kb, cache_dir=Path(cache_dir))
            assert component.entities is not None
            assert len(component.entities) == 2

    def test_initialize_without_cache_dir(self, kb, nlp):
        """FuzzyCandidatesComponent.initialize works without cache_dir."""
        from lela.spacy_components.candidates import FuzzyCandidatesComponent
        component = FuzzyCandidatesComponent(nlp=nlp, top_k=5)
        component.initialize(kb)
        assert component.entities is not None
        assert len(component.entities) == 2


class TestCacheInvalidation:
    """Tests for cache invalidation across components."""

    @pytest.fixture
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    def _make_kb_file(self, data: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            return f.name

    def test_bm25_cache_invalidated_on_kb_change(self, cache_dir, nlp):
        """BM25 cache is invalidated when the KB file changes."""
        from lela.spacy_components.candidates import BM25CandidatesComponent

        data_v1 = [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Joe Biden", "description": "46th US President"},
        ]
        path = self._make_kb_file(data_v1)

        try:
            kb1 = JSONLKnowledgeBase(path=path)
            comp1 = BM25CandidatesComponent(nlp=nlp, top_k=5)
            comp1.initialize(kb1, cache_dir=Path(cache_dir))
            assert len(comp1.corpus) == 2

            # Modify file to add an entity
            import time
            time.sleep(0.05)
            with open(path, "a") as f:
                f.write(json.dumps({"title": "New Entity", "description": "Desc"}) + "\n")

            kb2 = JSONLKnowledgeBase(path=path)
            comp2 = BM25CandidatesComponent(nlp=nlp, top_k=5)
            comp2.initialize(kb2, cache_dir=Path(cache_dir))

            # Should have rebuilt with 3 entities
            assert len(comp2.corpus) == 3

            # Should have two cache files now
            index_dir = os.path.join(cache_dir, "index")
            cache_files = [f for f in os.listdir(index_dir) if f.startswith("bm25_")]
            assert len(cache_files) == 2
        finally:
            os.unlink(path)

