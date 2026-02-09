"""Unit tests for CustomJSONLKnowledgeBase."""

import json
import os
import tempfile

import pytest

from el_pipeline.knowledge_bases.custom import CustomJSONLKnowledgeBase
from el_pipeline.types import Entity


class TestCustomJSONLKnowledgeBase:
    """Tests for CustomJSONLKnowledgeBase class."""

    @pytest.fixture
    def kb_data(self) -> list[dict]:
        return [
            {"id": "Q1", "title": "Barack Obama", "description": "44th US President"},
            {"id": "Q2", "title": "Joe Biden", "description": "46th US President"},
            {"id": "Q3", "title": "United States", "description": "Country in North America"},
            {"id": "Q4", "title": "New York", "description": "City in the United States"},
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

    def test_load_entities_from_file(self, kb: CustomJSONLKnowledgeBase):
        entities = list(kb.all_entities())
        assert len(entities) == 4

    def test_get_entity_by_id(self, kb: CustomJSONLKnowledgeBase):
        entity = kb.get_entity("Q1")
        assert entity is not None
        assert entity.id == "Q1"
        assert entity.title == "Barack Obama"
        assert entity.description == "44th US President"

    def test_get_nonexistent_entity(self, kb: CustomJSONLKnowledgeBase):
        entity = kb.get_entity("Q999")
        assert entity is None

    def test_search_finds_matches(self, kb: CustomJSONLKnowledgeBase):
        results = kb.search("Obama", top_k=5)
        assert len(results) > 0
        titles = [e.title for e in results]
        assert "Barack Obama" in titles

    def test_search_top_k_limit(self, kb: CustomJSONLKnowledgeBase):
        results = kb.search("United", top_k=2)
        assert len(results) <= 2

    def test_all_entities_returns_iterable(self, kb: CustomJSONLKnowledgeBase):
        entities = kb.all_entities()
        assert hasattr(entities, "__iter__")
        entity_list = list(entities)
        assert all(isinstance(e, Entity) for e in entity_list)

    def test_entity_metadata_preserved(self):
        data = [
            {"id": "Q1", "title": "Test", "description": "Desc", "extra_field": "value", "count": 42}
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = CustomJSONLKnowledgeBase(path=path)
            entity = kb.get_entity("Q1")
            assert entity.metadata["extra_field"] == "value"
            assert entity.metadata["count"] == 42
        finally:
            os.unlink(path)

    def test_missing_title_uses_id(self):
        data = [{"id": "Q1", "description": "No title entity"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = CustomJSONLKnowledgeBase(path=path)
            entity = kb.get_entity("Q1")
            assert entity.title == "Q1"  # Falls back to ID
        finally:
            os.unlink(path)

    def test_missing_description(self):
        data = [{"id": "Q1", "title": "Test Entity"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = CustomJSONLKnowledgeBase(path=path)
            entity = kb.get_entity("Q1")
            assert entity.description is None
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            kb = CustomJSONLKnowledgeBase(path=path)
            entities = list(kb.all_entities())
            assert len(entities) == 0
        finally:
            os.unlink(path)

    def test_search_case_insensitive(self, kb: CustomJSONLKnowledgeBase):
        # Search behavior depends on rapidfuzz implementation
        results = kb.search("OBAMA", top_k=5)
        # Should find Barack Obama
        assert len(results) > 0

    def test_fixture_based_kb(self, temp_jsonl_kb: str):
        """Test using the conftest fixture."""
        kb = CustomJSONLKnowledgeBase(path=temp_jsonl_kb)
        entities = list(kb.all_entities())
        assert len(entities) > 0
        # Check one of the sample entities
        obama = kb.get_entity("Q76")
        assert obama is not None
        assert obama.title == "Barack Obama"
