"""Unit tests for LELAEmbedderVLLMRerankerComponent."""

from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import spacy
from spacy.tokens import Span

from el_pipeline.types import Candidate
from el_pipeline.utils import ensure_candidates_extension


@pytest.fixture
def nlp():
    return spacy.blank("en")


@pytest.fixture
def sample_candidates() -> list:
    return [
        Candidate(entity_id="E1", description="Description 1"),
        Candidate(entity_id="E2", description="Description 2"),
        Candidate(entity_id="E3", description="Description 3"),
        Candidate(entity_id="E4", description="Description 4"),
        Candidate(entity_id="E5", description="Description 5"),
    ]


def _make_encode_output(embedding: list):
    """Create a mock vLLM encode output."""
    output = MagicMock()
    output.outputs.embedding = embedding
    return output


class TestLELAEmbedderVLLMRerankerComponent:
    """Tests for LELAEmbedderVLLMRerankerComponent."""

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_rerank_returns_candidates(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            _make_encode_output([1.0, 0.0, 0.0]),
            _make_encode_output([0.1, 0.9, 0.0]),
            _make_encode_output([0.2, 0.8, 0.0]),
            _make_encode_output([0.9, 0.1, 0.0]),
            _make_encode_output([0.3, 0.7, 0.0]),
            _make_encode_output([0.4, 0.6, 0.0]),
        ]
        mock_get_instance.return_value = (mock_model, False)

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELAEmbedderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert len(result) == 3
        assert all(isinstance(c, Candidate) for c in result)

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_rerank_sorts_by_cosine_similarity(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            _make_encode_output([1.0, 0.0, 0.0]),
            _make_encode_output([0.1, 0.9, 0.0]),
            _make_encode_output([0.2, 0.8, 0.0]),
            _make_encode_output([0.95, 0.05, 0.0]),
            _make_encode_output([0.3, 0.7, 0.0]),
            _make_encode_output([0.5, 0.5, 0.0]),
        ]
        mock_get_instance.return_value = (mock_model, False)

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELAEmbedderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert result[0].entity_id == "E3"

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_skips_reranking_when_candidates_below_top_k(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, nlp
    ):
        mock_model = MagicMock()
        mock_get_instance.return_value = (mock_model, False)

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELAEmbedderVLLMRerankerComponent(nlp=nlp, top_k=5)

        candidates = [Candidate(entity_id=f"E{i}", description=f"Desc {i}") for i in range(3)]
        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = reranker(doc)

        mock_model.encode.assert_not_called()
        assert len(doc.ents[0]._.candidates) == 3

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_query_format_matches_embedder_pattern(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            _make_encode_output([0.5, 0.5, 0.0]),
        ] * 6
        mock_get_instance.return_value = (mock_model, False)

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELAEmbedderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        query = texts[0]
        assert "[Obama]" in query
        assert "Instruct:" in query
        assert "Query:" in query

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_releases_vllm_after_use(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            _make_encode_output([0.5, 0.5, 0.0]),
        ] * 6
        mock_get_instance.return_value = (mock_model, False)

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELAEmbedderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        mock_release.assert_called_once_with(reranker.model_name, convert="embed", gpu_memory_utilization=reranker.gpu_memory_utilization)

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_loads_vllm_with_embed_task(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            _make_encode_output([0.5, 0.5, 0.0]),
        ] * 6
        mock_get_instance.return_value = (mock_model, False)

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELAEmbedderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        mock_get_instance.assert_called_once_with(
            model_name=reranker.model_name,
            convert="embed",
            max_model_len=None,
            gpu_memory_utilization=None,
            estimated_vram_gb=10.0,
        )

    def test_initialization_with_custom_params(self, nlp):
        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        reranker = LELAEmbedderVLLMRerankerComponent(
            nlp=nlp,
            model_name="custom-embed-model",
            top_k=7,
        )
        assert reranker.model_name == "custom-embed-model"
        assert reranker.top_k == 7
        assert reranker.max_model_len is None
        assert reranker.model is None

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_passes_custom_max_model_len(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            _make_encode_output([0.5, 0.5, 0.0]),
        ] * 6
        mock_get_instance.return_value = (mock_model, False)
        mock_get_vllm_mod.return_value = MagicMock()

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent

        ensure_candidates_extension()
        reranker = LELAEmbedderVLLMRerankerComponent(
            nlp=nlp,
            top_k=3,
            max_model_len=4096,
        )

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        assert mock_get_instance.call_args.kwargs["max_model_len"] == 4096

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_normalizes_embeddings(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            _make_encode_output([3.0, 0.0, 0.0]),
            _make_encode_output([0.0, 4.0, 0.0]),
            _make_encode_output([0.0, 0.0, 5.0]),
            _make_encode_output([6.0, 0.0, 0.0]),
            _make_encode_output([0.0, 7.0, 0.0]),
            _make_encode_output([2.0, 0.0, 0.0]),
        ]
        mock_get_instance.return_value = (mock_model, False)

        from el_pipeline.spacy_components.rerankers import LELAEmbedderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELAEmbedderVLLMRerankerComponent(nlp=nlp, top_k=2)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert result[0].entity_id in ("E3", "E5")
        assert result[1].entity_id in ("E3", "E5")
        assert abs(result[0].score - 1.0) < 0.01
