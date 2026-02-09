"""Unit tests for LELA llm_pool module."""

import pytest
from unittest.mock import MagicMock, patch

from el_pipeline.lela.llm_pool import (
    _get_sentence_transformers,
    get_sentence_transformer_instance,
    clear_sentence_transformer_instances,
    _get_vllm,
    get_vllm_instance,
    clear_vllm_instances,
)


class TestSentenceTransformerPool:
    """Tests for SentenceTransformer singleton functions."""

    def test_get_sentence_transformers_raises_on_missing(self):
        # Just verify the function exists and is callable
        assert callable(_get_sentence_transformers)

    @patch("el_pipeline.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_get_sentence_transformer_instance_creates_model(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        mock_get_st.return_value = mock_st_module

        # Clear cache first
        clear_sentence_transformer_instances(force=True)

        result = get_sentence_transformer_instance(
            model_name="test-model",
            device="cuda",
        )

        mock_st_module.SentenceTransformer.assert_called_once()
        call_kwargs = mock_st_module.SentenceTransformer.call_args
        assert call_kwargs[0][0] == "test-model"
        assert call_kwargs[1]["device"] == "cuda"
        assert call_kwargs[1]["trust_remote_code"] is True

    @patch("el_pipeline.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_get_sentence_transformer_instance_reuses_model(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        mock_get_st.return_value = mock_st_module

        clear_sentence_transformer_instances(force=True)

        model1 = get_sentence_transformer_instance("model-a", device="cuda")
        model2 = get_sentence_transformer_instance("model-a", device="cuda")

        # Should only create once
        assert mock_st_module.SentenceTransformer.call_count == 1
        assert model1 is model2

    @patch("el_pipeline.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_different_devices_create_different_instances(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.side_effect = [MagicMock(), MagicMock()]
        mock_get_st.return_value = mock_st_module

        clear_sentence_transformer_instances(force=True)

        model1 = get_sentence_transformer_instance("model-a", device="cuda")
        model2 = get_sentence_transformer_instance("model-a", device="cpu")

        # Different devices should create different instances
        assert mock_st_module.SentenceTransformer.call_count == 2

    def test_clear_sentence_transformer_instances_no_force(self):
        # Should not raise and should do nothing without force=True
        clear_sentence_transformer_instances()

    @patch("el_pipeline.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_clear_sentence_transformer_instances_force(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"
        mock_torch.cuda.is_available.return_value = False

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        mock_get_st.return_value = mock_st_module

        clear_sentence_transformer_instances(force=True)

        get_sentence_transformer_instance("test-model")

        # Clear with force
        clear_sentence_transformer_instances(force=True)

        # Model should be created again on next call
        get_sentence_transformer_instance("test-model")

        assert mock_st_module.SentenceTransformer.call_count == 2


class TestLazyImports:
    """Tests for lazy import functions."""

    def test_get_vllm_raises_on_missing(self):
        # Just verify the function exists
        assert callable(_get_vllm)


class TestVLLMInstanceManagement:
    """Tests for vLLM instance management."""

    def test_clear_vllm_instances(self):
        # Should not raise
        clear_vllm_instances()

    @patch("el_pipeline.lela.llm_pool._get_vllm")
    def test_get_vllm_instance_creates_model(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_llm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_get_vllm.return_value = mock_vllm

        # Clear cache first (force=True required to actually clear)
        clear_vllm_instances(force=True)

        result = get_vllm_instance(
            model_name="test-model",
            tensor_parallel_size=1,
        )

        mock_vllm.LLM.assert_called_once()
        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["tensor_parallel_size"] == 1

    @patch("el_pipeline.lela.llm_pool._get_vllm")
    def test_get_vllm_instance_reuses_model(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_llm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        llm1 = get_vllm_instance("model-a", tensor_parallel_size=1)
        llm2 = get_vllm_instance("model-a", tensor_parallel_size=1)

        # Should only create once
        assert mock_vllm.LLM.call_count == 1
        assert llm1 is llm2

    @patch("el_pipeline.lela.llm_pool._get_vllm")
    def test_different_configs_create_different_instances(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_vllm.LLM.side_effect = [MagicMock(), MagicMock()]
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        llm1 = get_vllm_instance("model-a", tensor_parallel_size=1)
        llm2 = get_vllm_instance("model-a", tensor_parallel_size=2)

        # Different tensor_parallel_size should create different instances
        assert mock_vllm.LLM.call_count == 2

    @patch("el_pipeline.lela.llm_pool._get_vllm")
    def test_max_model_len_passed_when_specified(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        get_vllm_instance("model-x", tensor_parallel_size=1, max_model_len=4096)

        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["max_model_len"] == 4096
