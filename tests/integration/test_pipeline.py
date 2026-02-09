"""Integration tests for the EL pipeline."""

import json
import os
import tempfile

import pytest

from el_pipeline.config import PipelineConfig
from el_pipeline.pipeline import ELPipeline
from el_pipeline.types import Document


@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end pipeline integration tests using lightweight components."""

    @pytest.fixture
    def pipeline(self, minimal_config_dict: dict) -> ELPipeline:
        """Create a pipeline with lightweight components."""
        config = PipelineConfig.from_dict(minimal_config_dict)
        return ELPipeline(config)

    @pytest.fixture
    def text_file_with_entities(self) -> str:
        """Create a text file with named entities."""
        content = (
            "Barack Obama was the 44th President of the United States. "
            "He was born in Honolulu, Hawaii. "
            "Obama graduated from Columbia University and Harvard Law School."
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        yield path
        os.unlink(path)

    def test_pipeline_initialization(self, minimal_config_dict: dict):
        """Test that pipeline initializes with minimal config."""
        config = PipelineConfig.from_dict(minimal_config_dict)
        pipeline = ELPipeline(config)
        assert pipeline.loader is not None
        assert pipeline.nlp is not None
        assert pipeline.kb is not None

    def test_process_single_document(self, pipeline: ELPipeline):
        """Test processing a single document."""
        doc = Document(
            id="test-doc",
            text="Barack Obama visited New York City yesterday.",
        )
        result = pipeline.process_document(doc)
        assert result is not None
        assert result["id"] == "test-doc"
        assert "text" in result
        assert "entities" in result
        assert isinstance(result["entities"], list)

    def test_entities_extracted(self, pipeline: ELPipeline):
        """Test that entities are extracted from document."""
        doc = Document(
            id="test-doc",
            text="Barack Obama met with Joe Biden in Washington.",
        )
        result = pipeline.process_document(doc)
        entities = result["entities"]
        # Should find some capitalized names with SimpleRegexNER
        assert len(entities) > 0
        entity_texts = [e["text"] for e in entities]
        assert any("Obama" in t for t in entity_texts)

    def test_entity_structure(self, pipeline: ELPipeline):
        """Test the structure of extracted entities."""
        doc = Document(
            id="test-doc",
            text="Barack Obama was president.",
        )
        result = pipeline.process_document(doc)
        if result["entities"]:
            entity = result["entities"][0]
            # Check required fields
            assert "text" in entity
            assert "start" in entity
            assert "end" in entity
            assert "label" in entity
            assert "candidates" in entity
            # Check offsets are valid
            assert entity["start"] >= 0
            assert entity["end"] > entity["start"]

    def test_candidates_generated(self, pipeline: ELPipeline):
        """Test that candidates are generated for mentions."""
        doc = Document(
            id="test-doc",
            text="Barack Obama gave a speech.",
        )
        result = pipeline.process_document(doc)
        obama_entities = [
            e for e in result["entities"] if "Obama" in e["text"]
        ]
        if obama_entities:
            entity = obama_entities[0]
            # Should have candidates from fuzzy matching
            assert "candidates" in entity
            assert isinstance(entity["candidates"], list)

    def test_disambiguation_resolves_entity(self, pipeline: ELPipeline):
        """Test that disambiguation assigns entity ID."""
        doc = Document(
            id="test-doc",
            text="Barack Obama is a former president.",
        )
        result = pipeline.process_document(doc)
        obama_entities = [
            e for e in result["entities"] if "Obama" in e["text"]
        ]
        if obama_entities and obama_entities[0]["candidates"]:
            entity = obama_entities[0]
            # Disambiguation should assign an entity_id
            # (may be None if no good match, but field should exist)
            assert "entity_id" in entity

    def test_run_with_file_path(
        self, pipeline: ELPipeline, text_file_with_entities: str
    ):
        """Test running pipeline on a file path."""
        results = pipeline.run([text_file_with_entities])
        assert len(results) == 1
        result = results[0]
        assert "entities" in result
        assert len(result["entities"]) > 0

    def test_run_with_output_file(
        self, pipeline: ELPipeline, text_file_with_entities: str
    ):
        """Test running pipeline with output to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            output_path = f.name

        try:
            results = pipeline.run([text_file_with_entities], output_path=output_path)
            # Results should be returned
            assert len(results) == 1

            # Output file should exist and contain valid JSONL
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            output_data = json.loads(lines[0])
            assert "entities" in output_data
        finally:
            os.unlink(output_path)

    def test_run_multiple_files(self, pipeline: ELPipeline):
        """Test running pipeline on multiple files."""
        texts = [
            "Barack Obama was president.",
            "Joe Biden is the current president.",
        ]
        paths = []
        for i, text in enumerate(texts):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(text)
                paths.append(f.name)

        try:
            results = pipeline.run(paths)
            assert len(results) == 2
            for result in results:
                assert "entities" in result
        finally:
            for path in paths:
                os.unlink(path)

    def test_caching_works(
        self, pipeline: ELPipeline, text_file_with_entities: str
    ):
        """Test that document caching works."""
        # Run twice on same file
        results1 = pipeline.run([text_file_with_entities])
        results2 = pipeline.run([text_file_with_entities])
        # Should get same results (from cache)
        assert len(results1) == len(results2)
        assert results1[0]["text"] == results2[0]["text"]

    def test_empty_document(self, pipeline: ELPipeline):
        """Test processing an empty document."""
        doc = Document(id="empty", text="")
        result = pipeline.process_document(doc)
        assert result["id"] == "empty"
        assert result["entities"] == []

    def test_document_without_entities(self, pipeline: ELPipeline):
        """Test processing text without named entities."""
        doc = Document(
            id="no-entities",
            text="the quick brown fox jumps over the lazy dog",
        )
        result = pipeline.process_document(doc)
        # Lowercase text should have no entities with SimpleRegexNER
        assert result["entities"] == []


@pytest.mark.integration
class TestPipelineWithoutDisambiguator:
    """Test pipeline without disambiguation component."""

    @pytest.fixture
    def config_no_disambiguator(
        self, temp_jsonl_kb: str, temp_cache_dir: str
    ) -> dict:
        return {
            "loader": {"name": "text"},
            "ner": {"name": "simple", "params": {"min_len": 3}},
            "candidate_generator": {"name": "fuzzy", "params": {"top_k": 5}},
            "knowledge_base": {"name": "custom", "params": {"path": temp_jsonl_kb}},
            "cache_dir": temp_cache_dir,
        }

    def test_pipeline_without_disambiguator(self, config_no_disambiguator: dict):
        """Test pipeline works without disambiguator."""
        config = PipelineConfig.from_dict(config_no_disambiguator)
        pipeline = ELPipeline(config)
        # In spaCy architecture, there's no separate disambiguator attribute
        # The pipeline just doesn't have a disambiguator component

        doc = Document(id="test", text="Barack Obama spoke today.")
        result = pipeline.process_document(doc)
        # Should still have entities with candidates, but no entity_id
        for entity in result["entities"]:
            assert entity["entity_id"] is None


@pytest.mark.integration
class TestPipelineConfiguration:
    """Test various pipeline configurations."""

    def test_custom_ner_params(
        self, temp_jsonl_kb: str, temp_cache_dir: str
    ):
        """Test custom NER parameters."""
        config_dict = {
            "loader": {"name": "text"},
            "ner": {"name": "simple", "params": {"min_len": 10}},  # High min_len
            "candidate_generator": {"name": "fuzzy", "params": {"top_k": 3}},
            "knowledge_base": {"name": "custom", "params": {"path": temp_jsonl_kb}},
            "cache_dir": temp_cache_dir,
        }
        config = PipelineConfig.from_dict(config_dict)
        pipeline = ELPipeline(config)

        doc = Document(id="test", text="Obama spoke.")
        result = pipeline.process_document(doc)
        # "Obama" is only 5 chars, should not be extracted with min_len=10
        entity_texts = [e["text"] for e in result["entities"]]
        assert "Obama" not in entity_texts

    def test_custom_candidate_top_k(
        self, temp_jsonl_kb: str, temp_cache_dir: str
    ):
        """Test custom candidate generation parameters."""
        config_dict = {
            "loader": {"name": "text"},
            "ner": {"name": "simple", "params": {"min_len": 3}},
            "candidate_generator": {"name": "fuzzy", "params": {"top_k": 2}},
            "knowledge_base": {"name": "custom", "params": {"path": temp_jsonl_kb}},
            "cache_dir": temp_cache_dir,
        }
        config = PipelineConfig.from_dict(config_dict)
        pipeline = ELPipeline(config)

        doc = Document(id="test", text="Barack Obama spoke.")
        result = pipeline.process_document(doc)
        obama_entities = [e for e in result["entities"] if "Obama" in e["text"]]
        if obama_entities:
            # Should have at most 2 candidates
            assert len(obama_entities[0]["candidates"]) <= 2
