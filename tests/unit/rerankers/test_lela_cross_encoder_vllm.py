"""Unit tests for LELACrossEncoderVLLMRerankerComponent."""

from unittest.mock import MagicMock, patch

import pytest
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


def _make_score_output(score: float):
    """Create a mock vLLM score output."""
    output = MagicMock()
    output.outputs.score = score
    return output


class TestLELACrossEncoderVLLMRerankerComponent:
    """Tests for LELACrossEncoderVLLMRerankerComponent."""

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_rerank_returns_candidates(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.score.return_value = [
            _make_score_output(0.1),
            _make_score_output(0.5),
            _make_score_output(0.9),
            _make_score_output(0.3),
            _make_score_output(0.7),
        ]
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

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
    def test_rerank_sorts_by_score(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.score.return_value = [
            _make_score_output(0.1),  # E1
            _make_score_output(0.5),  # E2
            _make_score_output(0.9),  # E3 - highest
            _make_score_output(0.3),  # E4
            _make_score_output(0.7),  # E5
        ]
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        result = doc.ents[0]._.candidates
        assert result[0].entity_id == "E3"
        assert result[1].entity_id == "E5"
        assert result[2].entity_id == "E2"

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_skips_reranking_when_candidates_below_top_k(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, nlp
    ):
        """Should not call .score() if all entities have <= top_k candidates."""
        mock_model = MagicMock()
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=5)

        candidates = [Candidate(entity_id=f"E{i}", description=f"Desc {i}") for i in range(3)]
        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = reranker(doc)

        mock_model.score.assert_not_called()
        assert len(doc.ents[0]._.candidates) == 3

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_score_called_with_batched_queries_and_documents(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        """score() should be called with batched N->N queries and documents."""
        mock_model = MagicMock()
        mock_model.score.return_value = [_make_score_output(0.5)] * 5
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        mock_model.score.assert_called_once()
        call_args = mock_model.score.call_args
        queries = call_args[0][0]
        documents = call_args[0][1]
        # Batched: queries is a list of repeated query strings (one per candidate)
        assert isinstance(queries, list)
        assert len(queries) == 5
        assert all("[Obama]" in q for q in queries)
        assert isinstance(documents, list)
        assert len(documents) == 5
        assert "<Document>" in documents[0]
        assert "E1 (Description 1)" in documents[0]

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_preserves_descriptions(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.score.return_value = [
            _make_score_output(float(i) / 5) for i in range(5)
        ]
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = reranker(doc)

        for candidate in doc.ents[0]._.candidates:
            original = next(o for o in sample_candidates if o.entity_id == candidate.entity_id)
            assert candidate.description == original.description

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_releases_vllm_after_use(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        mock_model = MagicMock()
        mock_model.score.return_value = [_make_score_output(0.5)] * 5
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        mock_release.assert_called_once_with(reranker.model_name, task="score")

    def test_initialization_with_custom_params(self, nlp):
        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        reranker = LELACrossEncoderVLLMRerankerComponent(
            nlp=nlp,
            model_name="custom-reranker-model",
            top_k=5,
        )
        assert reranker.model_name == "custom-reranker-model"
        assert reranker.top_k == 5
        assert reranker.model is None

    @patch("el_pipeline.spacy_components.rerankers.release_vllm")
    @patch("el_pipeline.spacy_components.rerankers.get_vllm_instance")
    @patch("el_pipeline.spacy_components.rerankers._get_vllm")
    def test_loads_vllm_with_score_task(
        self, mock_get_vllm_mod, mock_get_instance, mock_release, sample_candidates, nlp
    ):
        """Model should be loaded with task='score' and hf_overrides for seq-cls."""
        mock_model = MagicMock()
        mock_model.score.return_value = [_make_score_output(0.5)] * 5
        mock_get_instance.return_value = (mock_model, False)

        mock_vllm_mod = MagicMock()
        mock_get_vllm_mod.return_value = mock_vllm_mod

        from el_pipeline.spacy_components.rerankers import LELACrossEncoderVLLMRerankerComponent
        ensure_candidates_extension()

        reranker = LELACrossEncoderVLLMRerankerComponent(nlp=nlp, top_k=3)

        doc = nlp("Obama was the president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        reranker(doc)

        mock_get_instance.assert_called_once_with(
            model_name=reranker.model_name,
            task="score",
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
        )
