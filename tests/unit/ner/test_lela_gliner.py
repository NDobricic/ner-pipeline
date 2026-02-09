"""Unit tests for LELAGLiNERComponent."""

from unittest.mock import MagicMock, patch

import pytest
import spacy

from el_pipeline.types import Mention


class TestLELAGLiNERComponent:
    """Tests for LELAGLiNERComponent class with mocked GLiNER."""

    @pytest.fixture
    def nlp(self):
        return spacy.blank("en")

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_initialization_loads_model(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.5,
            context_mode="sentence",
        )

        mock_gliner_class.from_pretrained.assert_called_once()

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_initialization_with_custom_model(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="custom/model",
            labels=["person"],
            threshold=0.5,
            context_mode="sentence",
        )

        mock_gliner_class.from_pretrained.assert_called_once_with("custom/model")

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_initialization_with_custom_labels(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        custom_labels = ["person", "company"]
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=custom_labels,
            threshold=0.5,
            context_mode="sentence",
        )

        assert component.labels == custom_labels

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_initialization_with_custom_threshold(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.7,
            context_mode="sentence",
        )

        assert component.threshold == 0.7

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_call_returns_doc_with_entities(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        # Mock predictions
        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
        ]

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.5,
            context_mode="sentence",
        )

        doc = nlp("Barack Obama was president.")
        doc = component(doc)

        assert len(doc.ents) == 1
        assert doc.ents[0].text == "Barack Obama"

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_entity_has_correct_label(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
        ]

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.5,
            context_mode="sentence",
        )

        doc = nlp("Barack Obama was president.")
        doc = component(doc)

        assert doc.ents[0].label_ == "person"

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_entity_has_context_extension(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
        ]

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.5,
            context_mode="sentence",
        )

        doc = nlp("Barack Obama was president.")
        doc = component(doc)

        # Context should be extracted
        assert doc.ents[0]._.context is not None

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_extract_multiple_entities(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
            {"start": 30, "end": 43, "text": "United States", "label": "location"},
        ]

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person", "location"],
            threshold=0.5,
            context_mode="sentence",
        )

        doc = nlp("Barack Obama was president of United States.")
        doc = component(doc)

        assert len(doc.ents) == 2
        assert doc.ents[0].text == "Barack Obama"
        assert doc.ents[1].text == "United States"

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_extract_empty_text(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.5,
            context_mode="sentence",
        )

        doc = nlp("")
        doc = component(doc)

        assert len(doc.ents) == 0
        # Should not call predict on empty text
        mock_model.predict_entities.assert_not_called()

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_context_mode_parameter(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.5,
            context_mode="window",
        )

        assert component.context_mode == "window"

    @patch("el_pipeline.spacy_components.ner._get_gliner")
    def test_threshold_passed_to_predict(self, mock_get_gliner, nlp):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = []

        from el_pipeline.spacy_components.ner import LELAGLiNERComponent
        component = LELAGLiNERComponent(
            nlp=nlp,
            model_name="test/model",
            labels=["person"],
            threshold=0.7,
            context_mode="sentence",
        )

        doc = nlp("Test text")
        doc = component(doc)

        mock_model.predict_entities.assert_called_once_with(
            "Test text",
            labels=["person"],
            threshold=0.7,
        )
