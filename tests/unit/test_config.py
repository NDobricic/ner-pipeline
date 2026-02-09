"""Unit tests for configuration parsing."""

import pytest

from el_pipeline.config import ComponentConfig, PipelineConfig


class TestComponentConfig:
    """Tests for ComponentConfig dataclass."""

    def test_create_with_name_only(self):
        config = ComponentConfig(name="simple")
        assert config.name == "simple"
        assert config.params == {}

    def test_create_with_params(self):
        config = ComponentConfig(name="spacy", params={"model": "en_core_web_sm"})
        assert config.name == "spacy"
        assert config.params["model"] == "en_core_web_sm"


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass and from_dict parsing."""

    def test_from_dict_minimal(self):
        data = {
            "loader": {"name": "text", "params": {}},
            "ner": {"name": "simple", "params": {}},
            "candidate_generator": {"name": "fuzzy", "params": {}},
        }
        config = PipelineConfig.from_dict(data)
        assert config.loader.name == "text"
        assert config.ner.name == "simple"
        assert config.candidate_generator.name == "fuzzy"
        assert config.reranker is None
        assert config.disambiguator is None
        assert config.knowledge_base is None
        assert config.cache_dir == ".ner_cache"
        assert config.batch_size == 1

    def test_from_dict_full(self):
        data = {
            "loader": {"name": "jsonl", "params": {"text_field": "content"}},
            "ner": {"name": "spacy", "params": {"model": "en_core_web_sm"}},
            "candidate_generator": {"name": "dense", "params": {"top_k": 10}},
            "reranker": {"name": "cross_encoder", "params": {}},
            "disambiguator": {"name": "popularity", "params": {}},
            "knowledge_base": {"name": "custom", "params": {"path": "/path/to/kb.jsonl"}},
            "cache_dir": "/tmp/cache",
            "batch_size": 4,
        }
        config = PipelineConfig.from_dict(data)
        assert config.loader.name == "jsonl"
        assert config.loader.params["text_field"] == "content"
        assert config.ner.name == "spacy"
        assert config.candidate_generator.name == "dense"
        assert config.candidate_generator.params["top_k"] == 10
        assert config.reranker.name == "cross_encoder"
        assert config.disambiguator.name == "popularity"
        assert config.knowledge_base.name == "custom"
        assert config.cache_dir == "/tmp/cache"
        assert config.batch_size == 4

    def test_from_dict_with_none_optional_components(self):
        data = {
            "loader": {"name": "text", "params": {}},
            "ner": {"name": "simple", "params": {}},
            "candidate_generator": {"name": "fuzzy", "params": {}},
            "reranker": None,
            "disambiguator": None,
            "knowledge_base": None,
        }
        config = PipelineConfig.from_dict(data)
        assert config.reranker is None
        assert config.disambiguator is None
        assert config.knowledge_base is None

    def test_from_dict_params_default_to_empty_dict(self):
        data = {
            "loader": {"name": "text"},
            "ner": {"name": "simple"},
            "candidate_generator": {"name": "fuzzy"},
        }
        config = PipelineConfig.from_dict(data)
        assert config.loader.params == {}
        assert config.ner.params == {}
        assert config.candidate_generator.params == {}

    def test_from_dict_custom_cache_dir(self):
        data = {
            "loader": {"name": "text"},
            "ner": {"name": "simple"},
            "candidate_generator": {"name": "fuzzy"},
            "cache_dir": "/custom/cache",
        }
        config = PipelineConfig.from_dict(data)
        assert config.cache_dir == "/custom/cache"

    def test_from_dict_custom_batch_size(self):
        data = {
            "loader": {"name": "text"},
            "ner": {"name": "simple"},
            "candidate_generator": {"name": "fuzzy"},
            "batch_size": 8,
        }
        config = PipelineConfig.from_dict(data)
        assert config.batch_size == 8
