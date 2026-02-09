"""Unit tests for LELA config module."""

import pytest

from el_pipeline.lela import config


class TestLELAConfig:
    """Tests for LELA configuration constants."""

    def test_ner_labels_is_list(self):
        assert isinstance(config.NER_LABELS, list)
        assert len(config.NER_LABELS) > 0

    def test_ner_labels_contains_expected_types(self):
        expected = {"person", "organization", "location"}
        actual = set(config.NER_LABELS)
        assert expected.issubset(actual)

    def test_default_models_are_strings(self):
        assert isinstance(config.DEFAULT_GLINER_MODEL, str)
        assert isinstance(config.DEFAULT_LLM_MODEL, str)
        assert isinstance(config.DEFAULT_EMBEDDER_MODEL, str)
        assert isinstance(config.DEFAULT_RERANKER_MODEL, str)

    def test_default_models_have_org_prefix(self):
        # Model IDs should have org/model format
        assert "/" in config.DEFAULT_GLINER_MODEL
        assert "/" in config.DEFAULT_LLM_MODEL
        assert "/" in config.DEFAULT_EMBEDDER_MODEL

    def test_top_k_values_are_positive_integers(self):
        assert isinstance(config.CANDIDATES_TOP_K, int)
        assert isinstance(config.RERANKER_TOP_K, int)
        assert config.CANDIDATES_TOP_K > 0
        assert config.RERANKER_TOP_K > 0

    def test_candidates_top_k_greater_than_reranker(self):
        # We retrieve more candidates than we keep after reranking
        assert config.CANDIDATES_TOP_K > config.RERANKER_TOP_K

    def test_span_markers_are_strings(self):
        assert isinstance(config.SPAN_OPEN, str)
        assert isinstance(config.SPAN_CLOSE, str)
        assert len(config.SPAN_OPEN) == 1
        assert len(config.SPAN_CLOSE) == 1

    def test_not_an_entity_value(self):
        assert config.NOT_AN_ENTITY == "None"

    def test_task_descriptions_are_nonempty(self):
        assert len(config.RETRIEVER_TASK) > 0
        assert len(config.RERANKER_TASK) > 0

    def test_generation_config_has_max_tokens(self):
        assert "max_tokens" in config.DEFAULT_GENERATION_CONFIG
        assert config.DEFAULT_GENERATION_CONFIG["max_tokens"] > 0

    def test_tensor_parallel_size_default(self):
        assert config.DEFAULT_TENSOR_PARALLEL_SIZE >= 1
