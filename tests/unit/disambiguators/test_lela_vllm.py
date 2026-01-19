"""Unit tests for LELAvLLMDisambiguatorComponent."""

from unittest.mock import MagicMock, patch
import json
import os
import tempfile

import pytest
import spacy
from spacy.tokens import Span

from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase


class TestLELAvLLMDisambiguatorComponent:
    """Tests for LELAvLLMDisambiguatorComponent class."""

    @pytest.fixture
    def lela_kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Michelle Obama", "description": "Former First Lady"},
            {"title": "Joe Biden", "description": "46th US President"},
        ]

    @pytest.fixture
    def temp_kb_file(self, lela_kb_data: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in lela_kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def kb(self, temp_kb_file: str) -> LELAJSONLKnowledgeBase:
        return LELAJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def sample_candidates(self) -> list[tuple[str, str]]:
        return [
            ("Barack Obama", "44th US President"),
            ("Michelle Obama", "Former First Lady"),
            ("Joe Biden", "46th US President"),
        ]

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_requires_knowledge_base(self, mock_get_instance, mock_get_vllm, nlp):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp)
        # Component returns doc unchanged when KB not initialized (logs warning)
        doc = nlp("Test")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = [("Test", "Desc")]
        result = component(doc)
        # resolved_entity should remain None since KB is not set
        assert result.ents[0]._.resolved_entity is None

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_disambiguate_returns_entity(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, nlp
    ):
        mock_vllm = MagicMock()
        mock_sampling_params = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, mock_sampling_params)

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns "answer": 1 (first candidate)
        mock_output = MagicMock()
        mock_output.text = '"answer": 1'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp)
        component.initialize(kb)

        doc = nlp("Obama was the 44th president of the United States.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        assert result is not None
        assert result.id == "Barack Obama"

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_disambiguate_empty_candidates(
        self, mock_get_instance, mock_get_vllm, kb, nlp
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp)
        component.initialize(kb)

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = []
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        assert result is None

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_single_candidate_no_none_option(
        self, mock_get_instance, mock_get_vllm, kb, nlp
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp, add_none_candidate=False)
        component.initialize(kb)

        candidates = [("Barack Obama", "44th US President")]

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = candidates
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        # Should return the single candidate directly without LLM call
        assert result is not None
        assert result.id == "Barack Obama"

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_answer_zero_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, nlp
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns "answer": 0 (none option)
        mock_output = MagicMock()
        mock_output.text = '"answer": 0'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp, add_none_candidate=True)
        component.initialize(kb)

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        assert result is None

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_selects_second_candidate(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, nlp
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns "answer": 2 (second candidate)
        mock_output = MagicMock()
        mock_output.text = '"answer": 2'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp)
        component.initialize(kb)

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        assert result is not None
        assert result.id == "Michelle Obama"

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_out_of_range_answer_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, nlp
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns answer out of range
        mock_output = MagicMock()
        mock_output.text = '"answer": 99'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp)
        component.initialize(kb)

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        assert result is None

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_invalid_output_format_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, nlp
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns unparseable output
        mock_output = MagicMock()
        mock_output.text = 'I think the answer is Barack Obama'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp)
        component.initialize(kb)

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        # Should return None (answer 0) when parsing fails
        assert result is None

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_llm_error_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, nlp
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm
        mock_llm.chat.side_effect = Exception("LLM error")

        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        component = LELAvLLMDisambiguatorComponent(nlp=nlp)
        component.initialize(kb)

        doc = nlp("Obama was president.")
        doc.ents = [Span(doc, 0, 1, label="ENTITY")]
        doc.ents[0]._.candidates = sample_candidates
        doc = component(doc)

        result = doc.ents[0]._.resolved_entity
        assert result is None


class TestLELAvLLMDisambiguatorParsing:
    """Tests for output parsing logic."""

    def test_parse_output_standard_format(self):
        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        assert LELAvLLMDisambiguatorComponent._parse_output('"answer": 1') == 1
        assert LELAvLLMDisambiguatorComponent._parse_output('"answer": 2') == 2
        assert LELAvLLMDisambiguatorComponent._parse_output('"answer": 0') == 0

    def test_parse_output_without_quotes(self):
        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        assert LELAvLLMDisambiguatorComponent._parse_output('answer: 1') == 1
        assert LELAvLLMDisambiguatorComponent._parse_output('answer: 3') == 3

    def test_parse_output_with_surrounding_text(self):
        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        assert LELAvLLMDisambiguatorComponent._parse_output('Based on context, "answer": 2') == 2
        assert LELAvLLMDisambiguatorComponent._parse_output('{"answer": 1}') == 1

    def test_parse_output_invalid_returns_zero(self):
        from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
        assert LELAvLLMDisambiguatorComponent._parse_output('no answer here') == 0
        assert LELAvLLMDisambiguatorComponent._parse_output('') == 0
        assert LELAvLLMDisambiguatorComponent._parse_output('Barack Obama') == 0


class TestLELAvLLMDisambiguatorConfig:
    """Tests for disambiguator configuration."""

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_custom_model_name(self, mock_get_instance, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        # Create minimal KB
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Test", "description": "Test"}\n')
            path = f.name

        try:
            kb = LELAJSONLKnowledgeBase(path=path)

            from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
            nlp = spacy.blank("en")
            component = LELAvLLMDisambiguatorComponent(nlp=nlp, model_name="custom/model")
            component.initialize(kb)

            # Check that get_vllm_instance was called with correct model
            mock_get_instance.assert_called_once()
            call_kwargs = mock_get_instance.call_args[1]
            assert call_kwargs["model_name"] == "custom/model"
        finally:
            os.unlink(path)

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_tensor_parallel_size(self, mock_get_instance, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Test", "description": "Test"}\n')
            path = f.name

        try:
            kb = LELAJSONLKnowledgeBase(path=path)

            from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
            nlp = spacy.blank("en")
            component = LELAvLLMDisambiguatorComponent(nlp=nlp, tensor_parallel_size=4)
            component.initialize(kb)

            call_kwargs = mock_get_instance.call_args[1]
            assert call_kwargs["tensor_parallel_size"] == 4
        finally:
            os.unlink(path)

    @patch("ner_pipeline.spacy_components.disambiguators._get_vllm")
    @patch("ner_pipeline.spacy_components.disambiguators.get_vllm_instance")
    def test_custom_system_prompt(self, mock_get_instance, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Test", "description": "Test"}\n')
            path = f.name

        try:
            kb = LELAJSONLKnowledgeBase(path=path)

            from ner_pipeline.spacy_components.disambiguators import LELAvLLMDisambiguatorComponent
            nlp = spacy.blank("en")
            custom_prompt = "Custom disambiguation prompt"
            component = LELAvLLMDisambiguatorComponent(nlp=nlp, system_prompt=custom_prompt)

            assert component.system_prompt == custom_prompt
        finally:
            os.unlink(path)
