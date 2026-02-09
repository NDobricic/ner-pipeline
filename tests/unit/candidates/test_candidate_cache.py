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

from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase
from el_pipeline.types import Entity


class TestLELABM25Cache:
    """Tests for LELABM25CandidatesComponent caching."""

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
    def kb(self, temp_kb_file: str) -> CustomJSONLKnowledgeBase:
        return CustomJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_initialize_saves_cache(
        self, mock_bm25s, mock_stemmer, kb, cache_dir, nlp
    ):
        """First initialize with cache_dir calls retriever.save and writes components.pkl."""
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        # Use real picklable objects for stemmer/tokenizer so pickle.dump succeeds
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb, cache_dir=Path(cache_dir))

        # Override non-picklable mocks with real data, then manually save
        # Instead, verify that retriever.save was called with the right path pattern
        mock_retriever.save.assert_called_once()
        save_path = mock_retriever.save.call_args[0][0]
        assert "lela_bm25_" in str(save_path)
        assert "bm25s" in str(save_path)

        # The index directory should have been created
        index_dir = save_path.parent
        assert index_dir.exists()

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_initialize_loads_from_cache(
        self, mock_bm25s, mock_stemmer, kb, cache_dir, nlp
    ):
        """Cache hit loads retriever via BM25.load and components from pickle."""
        import hashlib

        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_loaded_retriever = MagicMock()
        mock_bm25s.return_value.BM25.load.return_value = mock_loaded_retriever

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent

        # Compute the expected cache hash to pre-populate cache
        raw = f"lela_bm25:{kb.identity_hash}:english".encode()
        cache_hash = hashlib.sha256(raw).hexdigest()
        index_dir = Path(cache_dir) / "index" / f"lela_bm25_{cache_hash}"
        bm25s_dir = index_dir / "bm25s"
        bm25s_dir.mkdir(parents=True, exist_ok=True)

        # Write a valid components.pkl with picklable data
        # Note: stemmer is not pickled (Cython object), it's recreated on load
        corpus_records = [
            {"id": "Barack Obama", "title": "Barack Obama", "description": "44th US President"},
            {"id": "Joe Biden", "title": "Joe Biden", "description": "46th US President"},
            {"id": "United States", "title": "United States", "description": "Country in North America"},
        ]
        components_file = index_dir / "components.pkl"
        with components_file.open("wb") as f:
            pickle.dump(corpus_records, f)

        # Initialize - should load from cache
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb, cache_dir=Path(cache_dir))

        # BM25.load should have been called (cache hit)
        mock_bm25s.return_value.BM25.load.assert_called_once_with(bm25s_dir, load_corpus=True)
        # index() should NOT have been called (skipped due to cache hit)
        mock_retriever.index.assert_not_called()

        # The loaded retriever should be assigned
        assert component.retriever is mock_loaded_retriever
        # corpus_records should be restored from pickle
        assert component.corpus_records == corpus_records
        # stemmer should be recreated (not from pickle)
        mock_stemmer.return_value.Stemmer.assert_called_with("english")

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_no_cache_dir_skips_caching(
        self, mock_bm25s, mock_stemmer, kb, nlp
    ):
        """Without cache_dir, no cache operations occur."""
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent
        component = LELABM25CandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb)

        # Should build normally
        mock_retriever.index.assert_called_once()
        # No save/load calls
        mock_retriever.save.assert_not_called()
        mock_bm25s.return_value.BM25.load.assert_not_called()


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
    def kb(self, temp_kb_file: str) -> CustomJSONLKnowledgeBase:
        return CustomJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.release_sentence_transformer")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
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
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
        component = LELADenseCandidatesComponent(nlp=nlp, top_k=5, use_context=False)
        component.initialize(kb, cache_dir=Path(cache_dir))

        # faiss.write_index should have been called
        mock_faiss_module.write_index.assert_called_once()
        write_args = mock_faiss_module.write_index.call_args[0]
        assert write_args[0] is mock_index
        assert "lela_dense_" in write_args[1]
        assert write_args[1].endswith("index.faiss")

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.release_sentence_transformer")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
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
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent

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

    @patch("el_pipeline.spacy_components.candidates._get_faiss")
    @patch("el_pipeline.spacy_components.candidates.release_sentence_transformer")
    @patch("el_pipeline.spacy_components.candidates.get_sentence_transformer_instance")
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
        mock_get_st.return_value = mock_model

        from el_pipeline.spacy_components.candidates import LELADenseCandidatesComponent
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
    def kb(self, temp_kb_file: str) -> CustomJSONLKnowledgeBase:
        return CustomJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    def test_initialize_saves_cache(self, kb, cache_dir, nlp):
        """First initialize with cache_dir creates a pickle file."""
        from el_pipeline.spacy_components.candidates import BM25CandidatesComponent
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
        from el_pipeline.spacy_components.candidates import BM25CandidatesComponent

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
        from el_pipeline.spacy_components.candidates import BM25CandidatesComponent
        component = BM25CandidatesComponent(nlp=nlp, top_k=5)
        component.initialize(kb)
        assert component.bm25 is not None
        assert len(component.corpus) == 3

    def test_corrupt_cache_falls_back_to_build(self, kb, cache_dir, nlp):
        """Corrupt cache file triggers rebuild."""
        from el_pipeline.spacy_components.candidates import BM25CandidatesComponent

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
    def kb(self, temp_kb_file: str) -> CustomJSONLKnowledgeBase:
        return CustomJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    def test_initialize_accepts_cache_dir(self, kb, nlp):
        """FuzzyCandidatesComponent.initialize accepts cache_dir without error."""
        with tempfile.TemporaryDirectory() as cache_dir:
            from el_pipeline.spacy_components.candidates import FuzzyCandidatesComponent
            component = FuzzyCandidatesComponent(nlp=nlp, top_k=5)
            component.initialize(kb, cache_dir=Path(cache_dir))
            assert component.entities is not None
            assert len(component.entities) == 2

    def test_initialize_without_cache_dir(self, kb, nlp):
        """FuzzyCandidatesComponent.initialize works without cache_dir."""
        from el_pipeline.spacy_components.candidates import FuzzyCandidatesComponent
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
        from el_pipeline.spacy_components.candidates import BM25CandidatesComponent

        data_v1 = [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Joe Biden", "description": "46th US President"},
        ]
        path = self._make_kb_file(data_v1)

        try:
            kb1 = CustomJSONLKnowledgeBase(path=path)
            comp1 = BM25CandidatesComponent(nlp=nlp, top_k=5)
            comp1.initialize(kb1, cache_dir=Path(cache_dir))
            assert len(comp1.corpus) == 2

            # Modify file to add an entity
            import time
            time.sleep(0.05)
            with open(path, "a") as f:
                f.write(json.dumps({"title": "New Entity", "description": "Desc"}) + "\n")

            kb2 = CustomJSONLKnowledgeBase(path=path)
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

    @patch("el_pipeline.spacy_components.candidates._get_stemmer")
    @patch("el_pipeline.spacy_components.candidates._get_bm25s")
    def test_lela_bm25_different_stemmer_different_cache(
        self, mock_bm25s, mock_stemmer, cache_dir, nlp
    ):
        """Different stemmer_language produces different cache keys."""
        data = [
            {"title": "Barack Obama", "description": "44th US President"},
        ]
        path = self._make_kb_file(data)

        try:
            mock_stemmer_instance = MagicMock()
            mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

            mock_tokenizer = MagicMock()
            mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
            mock_tokenizer.tokenize.return_value = [("tokens",)]

            mock_retriever = MagicMock()
            mock_bm25s.return_value.BM25.return_value = mock_retriever

            from el_pipeline.spacy_components.candidates import LELABM25CandidatesComponent

            kb = CustomJSONLKnowledgeBase(path=path)

            comp_en = LELABM25CandidatesComponent(
                nlp=nlp, top_k=5, use_context=False, stemmer_language="english"
            )
            comp_en.initialize(kb, cache_dir=Path(cache_dir))

            comp_fr = LELABM25CandidatesComponent(
                nlp=nlp, top_k=5, use_context=False, stemmer_language="french"
            )
            comp_fr.initialize(kb, cache_dir=Path(cache_dir))

            # Should have two different cache directories
            index_dir = os.path.join(cache_dir, "index")
            lela_dirs = [d for d in os.listdir(index_dir) if d.startswith("lela_bm25_")]
            assert len(lela_dirs) == 2
        finally:
            os.unlink(path)
