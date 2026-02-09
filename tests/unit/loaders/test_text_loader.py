"""Unit tests for text loader."""

import json
import os
import tempfile

import pytest

from el_pipeline.loaders.text import JSONLoader, JSONLLoader, TextLoader


class TestTextLoader:
    """Tests for TextLoader class."""

    def test_load_text_file(self, temp_text_file: str):
        loader = TextLoader()
        docs = list(loader.load(temp_text_file))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.text is not None
        assert len(doc.text) > 0
        assert "Obama" in doc.text  # Sample text contains Obama
        assert doc.meta["source"] == temp_text_file

    def test_load_returns_iterator(self, temp_text_file: str):
        loader = TextLoader()
        result = loader.load(temp_text_file)
        # Check it's an iterator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_document_id_from_filename(self, temp_text_file: str):
        loader = TextLoader()
        docs = list(loader.load(temp_text_file))
        # ID should be the stem of the filename (without extension)
        expected_stem = os.path.splitext(os.path.basename(temp_text_file))[0]
        assert docs[0].id == expected_stem


class TestJSONLLoader:
    """Tests for JSONLLoader class."""

    @pytest.fixture
    def temp_jsonl_file(self) -> str:
        data = [
            {"id": "doc1", "text": "First document text."},
            {"id": "doc2", "text": "Second document text."},
            {"text": "No ID document."},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        os.unlink(path)

    def test_load_jsonl_file(self, temp_jsonl_file: str):
        loader = JSONLLoader()
        docs = list(loader.load(temp_jsonl_file))
        assert len(docs) == 3
        assert docs[0].id == "doc1"
        assert docs[0].text == "First document text."
        assert docs[1].id == "doc2"

    def test_auto_generated_id(self, temp_jsonl_file: str):
        loader = JSONLLoader()
        docs = list(loader.load(temp_jsonl_file))
        # Third document has no ID, should be auto-generated
        assert docs[2].id is not None
        assert "2" in docs[2].id  # Index-based ID

    def test_custom_text_field(self):
        data = [{"content": "Custom field text."}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            loader = JSONLLoader(text_field="content")
            docs = list(loader.load(path))
            assert docs[0].text == "Custom field text."
        finally:
            os.unlink(path)

    def test_missing_text_field_returns_empty(self):
        data = [{"other_field": "value"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            loader = JSONLLoader()
            docs = list(loader.load(path))
            assert docs[0].text == ""
        finally:
            os.unlink(path)


class TestJSONLoader:
    """Tests for JSONLoader class."""

    @pytest.fixture
    def temp_json_array_file(self) -> str:
        data = [
            {"id": "doc1", "text": "First document."},
            {"id": "doc2", "text": "Second document."},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_json_object_file(self) -> str:
        data = {"id": "single", "text": "Single document."}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        yield path
        os.unlink(path)

    def test_load_json_array(self, temp_json_array_file: str):
        loader = JSONLoader()
        docs = list(loader.load(temp_json_array_file))
        assert len(docs) == 2
        assert docs[0].id == "doc1"
        assert docs[1].text == "Second document."

    def test_load_json_object(self, temp_json_object_file: str):
        loader = JSONLoader()
        docs = list(loader.load(temp_json_object_file))
        assert len(docs) == 1
        assert docs[0].id == "single"
        assert docs[0].text == "Single document."

    def test_custom_text_field(self):
        data = [{"body": "Custom body text."}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            loader = JSONLoader(text_field="body")
            docs = list(loader.load(path))
            assert docs[0].text == "Custom body text."
        finally:
            os.unlink(path)
