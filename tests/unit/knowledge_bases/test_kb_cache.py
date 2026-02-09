"""Unit tests for CustomJSONLKnowledgeBase caching."""

import json
import os
import pickle
import tempfile
import time

import pytest

from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase


class TestKBCaching:
    """Tests for knowledge base disk caching."""

    @pytest.fixture
    def kb_data(self) -> list[dict]:
        return [
            {"id": "Q1", "title": "Barack Obama", "description": "44th US President"},
            {"id": "Q2", "title": "Joe Biden", "description": "46th US President"},
            {"id": "Q3", "title": "United States", "description": "Country in North America"},
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
    def cache_dir(self) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_no_cache_dir_works_normally(self, temp_kb_file: str):
        """KB loads normally without cache_dir (backward compatible)."""
        kb = CustomJSONLKnowledgeBase(path=temp_kb_file)
        assert len(list(kb.all_entities())) == 3
        assert kb.get_entity("Q1").title == "Barack Obama"

    def test_first_load_creates_cache_file(self, temp_kb_file: str, cache_dir: str):
        """First load with cache_dir creates a .pkl file under kb/ subdir."""
        kb = CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)
        assert len(list(kb.all_entities())) == 3

        kb_cache_dir = os.path.join(cache_dir, "kb")
        assert os.path.isdir(kb_cache_dir)
        cache_files = os.listdir(kb_cache_dir)
        assert len(cache_files) == 1
        assert cache_files[0].endswith(".pkl")

    def test_second_load_uses_cache(self, temp_kb_file: str, cache_dir: str):
        """Second load reads from cache, producing identical data."""
        kb1 = CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)
        kb2 = CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)

        entities1 = {e.id: e for e in kb1.all_entities()}
        entities2 = {e.id: e for e in kb2.all_entities()}

        assert entities1.keys() == entities2.keys()
        for eid in entities1:
            assert entities1[eid].title == entities2[eid].title
            assert entities1[eid].description == entities2[eid].description

        assert kb1.titles == kb2.titles

    def test_cached_data_matches_original(self, temp_kb_file: str, cache_dir: str):
        """Entities and titles from cache are identical to JSONL parse."""
        kb_no_cache = CustomJSONLKnowledgeBase(path=temp_kb_file)
        # Populate cache
        CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)
        # Load from cache
        kb_cached = CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)

        assert kb_no_cache.titles == kb_cached.titles
        for eid in kb_no_cache.entities:
            orig = kb_no_cache.entities[eid]
            cached = kb_cached.entities[eid]
            assert orig.id == cached.id
            assert orig.title == cached.title
            assert orig.description == cached.description

    def test_identity_hash_is_stable(self, temp_kb_file: str):
        """Same file produces the same identity_hash on repeated calls."""
        kb = CustomJSONLKnowledgeBase(path=temp_kb_file)
        hash1 = kb.identity_hash
        hash2 = kb.identity_hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_identity_hash_changes_on_modification(self, kb_data: list[dict]):
        """Modifying the file (changing mtime/size) changes the hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name

        try:
            kb1 = CustomJSONLKnowledgeBase(path=path)
            hash1 = kb1.identity_hash

            # Wait a bit and append data to change mtime and size
            time.sleep(0.05)
            with open(path, "a") as f:
                f.write(json.dumps({"id": "Q99", "title": "New Entity", "description": "Added"}) + "\n")

            kb2 = CustomJSONLKnowledgeBase(path=path)
            hash2 = kb2.identity_hash

            assert hash1 != hash2
        finally:
            os.unlink(path)

    def test_cache_invalidated_on_file_change(self, kb_data: list[dict], cache_dir: str):
        """Changing the JSONL file causes a cache miss and rebuild."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name

        try:
            # Populate cache
            kb1 = CustomJSONLKnowledgeBase(path=path, cache_dir=cache_dir)
            assert len(list(kb1.all_entities())) == 3

            # Modify file
            time.sleep(0.05)
            with open(path, "a") as f:
                f.write(json.dumps({"id": "Q99", "title": "New Entity", "description": "Added"}) + "\n")

            # Load again - should rebuild and include new entity
            kb2 = CustomJSONLKnowledgeBase(path=path, cache_dir=cache_dir)
            assert len(list(kb2.all_entities())) == 4
            assert kb2.get_entity("Q99") is not None

            # Should now have two cache files (old and new)
            kb_cache_dir = os.path.join(cache_dir, "kb")
            cache_files = os.listdir(kb_cache_dir)
            assert len(cache_files) == 2
        finally:
            os.unlink(path)

    def test_corrupt_cache_falls_back_to_jsonl(self, temp_kb_file: str, cache_dir: str):
        """Corrupt cache file triggers a fallback to full JSONL parse."""
        # First load to create cache
        kb1 = CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)
        hash_val = kb1.identity_hash

        # Corrupt the cache file
        cache_file = os.path.join(cache_dir, "kb", f"{hash_val}.pkl")
        assert os.path.exists(cache_file)
        with open(cache_file, "wb") as f:
            f.write(b"corrupted data")

        # Should still load successfully from JSONL
        kb2 = CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)
        assert len(list(kb2.all_entities())) == 3
        assert kb2.get_entity("Q1").title == "Barack Obama"

    def test_source_path_stored(self, temp_kb_file: str):
        """source_path attribute is set to the original file path."""
        kb = CustomJSONLKnowledgeBase(path=temp_kb_file)
        assert kb.source_path == temp_kb_file

    def test_cache_dir_does_not_affect_search(self, temp_kb_file: str, cache_dir: str):
        """Search still works after loading from cache."""
        # Populate cache
        CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)
        # Load from cache
        kb = CustomJSONLKnowledgeBase(path=temp_kb_file, cache_dir=cache_dir)

        results = kb.search("Obama", top_k=5)
        assert len(results) > 0
        assert any(e.title == "Barack Obama" for e in results)
