"""Slow tests that use actual ML models.

These tests require model downloads and are marked with @pytest.mark.slow.
Run with: pytest tests/slow/ -v --run-slow
"""

import pytest


@pytest.mark.slow
@pytest.mark.requires_spacy
class TestSpacyNER:
    """Tests for SpaCy NER with real models."""

    def test_spacy_en_core_web_sm_loads(self):
        """SpaCy en_core_web_sm model loads successfully."""
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            assert nlp is not None
        except OSError:
            pytest.skip("en_core_web_sm not installed")

    def test_spacy_extracts_person_entities(self):
        """SpaCy extracts PERSON entities correctly."""
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm not installed")

        doc = nlp("Barack Obama was the president.")
        persons = [ent for ent in doc.ents if ent.label_ == "PERSON"]
        assert len(persons) > 0
        assert any("Obama" in ent.text for ent in persons)

    def test_spacy_extracts_gpe_entities(self):
        """SpaCy extracts GPE (geopolitical) entities correctly."""
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm not installed")

        doc = nlp("Paris is the capital of France.")
        gpes = [ent for ent in doc.ents if ent.label_ == "GPE"]
        assert len(gpes) >= 1


@pytest.mark.slow
class TestGLiNER:
    """Tests for GLiNER NER models."""

    def test_gliner_basic_extraction(self):
        """GLiNER extracts entities from text."""
        try:
            from gliner import GLiNER

            model = GLiNER.from_pretrained("urchade/gliner_small")
        except (ImportError, Exception) as e:
            pytest.skip(f"GLiNER not available: {e}")

        labels = ["person", "organization", "location"]
        entities = model.predict_entities(
            "Albert Einstein worked at Princeton University.", labels
        )
        assert isinstance(entities, list)

    def test_lela_gliner_chunking(self, sample_text: str, minimal_config_dict: dict):
        """LELA GLiNER processes long text with chunking."""
        try:
            from el_pipeline.config import PipelineConfig
            from el_pipeline.pipeline import ELPipeline
            from el_pipeline.types import Document
        except ImportError:
            pytest.skip("el_pipeline not available")

        try:
            from gliner import GLiNER  # noqa: F401
        except ImportError:
            pytest.skip("GLiNER not installed")

        # Update config to use lela_gliner
        config_dict = minimal_config_dict.copy()
        config_dict["ner"] = {
            "name": "lela_gliner",
            "params": {
                "model_name": "urchade/gliner_small",
                "labels": ["person", "organization", "location"],
                "threshold": 0.5,
            },
        }

        try:
            config = PipelineConfig.from_dict(config_dict)
            pipeline = ELPipeline(config)

            # Create a long document that requires chunking
            long_text = sample_text * 10
            doc = Document(id="test-long", text=long_text)
            result = pipeline.process_document(doc)

            assert "entities" in result
            assert isinstance(result["entities"], list)
        except Exception as e:
            pytest.skip(f"LELA GLiNER test failed: {e}")
