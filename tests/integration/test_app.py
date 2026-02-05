"""Integration tests for the Gradio web application (app.py)."""

import pytest

from app import (
    compute_linking_stats,
    format_highlighted_text,
    get_available_components,
    highlighted_to_html,
    run_pipeline,
)
from tests.conftest import MockGradioFile, MockGradioProgress


@pytest.mark.integration
class TestFormatHighlightedText:
    """Tests for format_highlighted_text function."""

    def test_format_empty_entities(self):
        """No entities returns single tuple with full text."""
        result = {"text": "Hello world", "entities": []}
        highlighted, color_map = format_highlighted_text(result)
        assert len(highlighted) == 1
        assert highlighted[0][0] == "Hello world"
        assert highlighted[0][1] is None
        assert highlighted[0][2] is None

    def test_format_single_entity(self):
        """One entity is highlighted correctly."""
        result = {
            "text": "Barack Obama was president.",
            "entities": [
                {
                    "text": "Barack Obama",
                    "start": 0,
                    "end": 12,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                }
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        assert len(highlighted) == 2
        # First element: (text, label, entity_info)
        assert highlighted[0][0] == "Barack Obama"
        assert highlighted[0][1] == "PERSON: Barack Obama"
        assert highlighted[0][2] is not None  # entity_info dict
        assert highlighted[0][2]["kb_title"] == "Barack Obama"
        # Second element: plain text
        assert highlighted[1][0] == " was president."
        assert highlighted[1][1] is None

    def test_format_multiple_entities(self):
        """Multiple entities with gaps are formatted correctly."""
        result = {
            "text": "Albert Einstein was born in Germany.",
            "entities": [
                {
                    "text": "Albert Einstein",
                    "start": 0,
                    "end": 15,
                    "label": "PERSON",
                    "entity_title": "Albert Einstein",
                },
                {
                    "text": "Germany",
                    "start": 28,
                    "end": 35,
                    "label": "GPE",
                    "entity_title": "Germany",
                },
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        assert len(highlighted) == 4
        assert highlighted[0][0] == "Albert Einstein"
        assert highlighted[0][1] == "PERSON: Albert Einstein"
        assert highlighted[1][0] == " was born in "
        assert highlighted[1][1] is None
        assert highlighted[2][0] == "Germany"
        assert highlighted[2][1] == "GPE: Germany"
        assert highlighted[3][0] == "."
        assert highlighted[3][1] is None

    def test_format_linked_entity(self):
        """Linked entity shows 'LABEL: Title' format."""
        result = {
            "text": "Obama",
            "entities": [
                {
                    "text": "Obama",
                    "start": 0,
                    "end": 5,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                }
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        assert highlighted[0][0] == "Obama"
        assert highlighted[0][1] == "PERSON: Barack Obama"

    def test_format_unlinked_entity(self):
        """Unlinked entity shows 'LABEL [NOT IN KB]' format."""
        result = {
            "text": "John Doe is here.",
            "entities": [
                {
                    "text": "John Doe",
                    "start": 0,
                    "end": 8,
                    "label": "PERSON",
                    "entity_title": None,
                }
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        assert highlighted[0][0] == "John Doe"
        assert highlighted[0][1] == "PERSON [NOT IN KB]"

    def test_format_entity_at_boundaries(self):
        """Entity at start and end of text are handled correctly."""
        result = {
            "text": "Obama",
            "entities": [
                {
                    "text": "Obama",
                    "start": 0,
                    "end": 5,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                }
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        assert len(highlighted) == 1
        assert highlighted[0][0] == "Obama"
        assert highlighted[0][1] == "PERSON: Barack Obama"

    def test_format_missing_label_uses_ent(self):
        """Missing label defaults to 'ENT'."""
        result = {
            "text": "Entity here.",
            "entities": [
                {
                    "text": "Entity",
                    "start": 0,
                    "end": 6,
                    "entity_title": None,
                }
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        assert highlighted[0][0] == "Entity"
        assert highlighted[0][1] == "ENT [NOT IN KB]"

    def test_returns_color_map(self):
        """Returns a color_map with colors for each label."""
        result = {
            "text": "Barack Obama and Germany",
            "entities": [
                {
                    "text": "Barack Obama",
                    "start": 0,
                    "end": 12,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                },
                {
                    "text": "Germany",
                    "start": 17,
                    "end": 24,
                    "label": "GPE",
                    "entity_title": "Germany",
                },
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        assert "PERSON: Barack Obama" in color_map
        assert "GPE: Germany" in color_map
        # Colors should be hex strings
        assert color_map["PERSON: Barack Obama"].startswith("#")
        assert color_map["GPE: Germany"].startswith("#")

    def test_entity_info_contains_expected_fields(self):
        """Entity info dict contains expected fields for popup."""
        result = {
            "text": "Obama",
            "entities": [
                {
                    "text": "Obama",
                    "start": 0,
                    "end": 5,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                    "entity_id": "Q76",
                    "entity_description": "44th President of the United States",
                }
            ],
        }
        highlighted, color_map = format_highlighted_text(result)
        entity_info = highlighted[0][2]
        assert entity_info["mention"] == "Obama"
        assert entity_info["type"] == "PERSON"
        assert entity_info["kb_id"] == "Q76"
        assert entity_info["kb_title"] == "Barack Obama"
        assert entity_info["kb_description"] == "44th President of the United States"


@pytest.mark.integration
class TestHighlightedToHtml:
    """Tests for highlighted_to_html function."""

    def test_converts_to_html_string(self):
        """Returns an HTML string."""
        highlighted = [("Hello ", None, None), ("World", "ENTITY", {"kb_title": "World"})]
        color_map = {"ENTITY": "#FF0000"}
        html = highlighted_to_html(highlighted, color_map)
        assert isinstance(html, str)
        assert "<div" in html

    def test_includes_entity_marks(self):
        """HTML includes mark elements for entities."""
        highlighted = [("Obama", "PERSON: Barack Obama", {"kb_title": "Barack Obama"})]
        color_map = {"PERSON: Barack Obama": "#1F77B4"}
        html = highlighted_to_html(highlighted, color_map)
        assert "<mark" in html
        assert "Obama" in html

    def test_includes_popup_div(self):
        """HTML includes a popup div for hover info."""
        highlighted = [("Obama", "PERSON: Barack Obama", {"kb_title": "Barack Obama"})]
        color_map = {"PERSON: Barack Obama": "#1F77B4"}
        html = highlighted_to_html(highlighted, color_map)
        assert "popup" in html.lower() or "display: none" in html


@pytest.mark.integration
class TestComputeLinkingStats:
    """Tests for compute_linking_stats function."""

    def test_stats_no_entities(self):
        """No entities returns appropriate message."""
        result = {"entities": []}
        stats = compute_linking_stats(result)
        assert stats == "No entities found."

    def test_stats_all_linked(self):
        """All entities linked shows 100%."""
        result = {
            "entities": [
                {"entity_title": "Entity 1"},
                {"entity_title": "Entity 2"},
            ]
        }
        stats = compute_linking_stats(result)
        assert "Linked to KB: 2 (100.0%)" in stats
        assert "Not in KB: 0 (0.0%)" in stats

    def test_stats_none_linked(self):
        """No entities linked shows 0%."""
        result = {
            "entities": [
                {"entity_title": None},
                {"entity_title": None},
            ]
        }
        stats = compute_linking_stats(result)
        assert "Linked to KB: 0 (0.0%)" in stats
        assert "Not in KB: 2 (100.0%)" in stats

    def test_stats_partial_linked(self):
        """Mixed linking shows correct percentages."""
        result = {
            "entities": [
                {"entity_title": "Entity 1"},
                {"entity_title": None},
                {"entity_title": "Entity 3"},
                {"entity_title": None},
            ]
        }
        stats = compute_linking_stats(result)
        assert "Total entities: 4" in stats
        assert "Linked to KB: 2 (50.0%)" in stats
        assert "Not in KB: 2 (50.0%)" in stats

    def test_stats_empty_result_dict(self):
        """Empty result dict returns no entities message."""
        result = {}
        stats = compute_linking_stats(result)
        assert stats == "No entities found."


@pytest.mark.integration
class TestGetAvailableComponents:
    """Tests for get_available_components function."""

    def test_returns_all_categories(self):
        """Returns all expected component categories."""
        components = get_available_components()
        expected_keys = {"loaders", "ner", "candidates", "rerankers", "disambiguators", "knowledge_bases"}
        assert set(components.keys()) == expected_keys

    def test_disambiguators_includes_vllm(self):
        """Disambiguators always includes lela_vllm."""
        components = get_available_components()
        assert "lela_vllm" in components["disambiguators"]

    def test_ner_includes_expected_types(self):
        """NER includes expected types."""
        components = get_available_components()
        assert "simple" in components["ner"]
        assert "spacy" in components["ner"]
        assert "gliner" in components["ner"]

    def test_candidates_includes_expected_types(self):
        """Candidate generators include expected types."""
        components = get_available_components()
        assert "fuzzy" in components["candidates"]
        assert "bm25" in components["candidates"]


@pytest.mark.integration
class TestRunPipeline:
    """Tests for run_pipeline generator function."""

    def _exhaust_generator(self, gen):
        """Exhaust a generator and return the final result."""
        result = None
        for result in gen:
            pass
        return result

    def test_run_pipeline_text_input(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Pipeline processes text input correctly."""
        text_input = "Barack Obama was president."
        gen = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_embedding_model="Qwen/Qwen3-Embedding-4B",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_embedding_model="Qwen/Qwen3-Embedding-4B",
            reranker_top_k=10,
            disambig_type="first",
            llm_model="Qwen/Qwen3-4B",
            lela_thinking=False,
            lela_none_candidate=True,
            kb_type="custom",
            progress=mock_progress,
        )
        html_output, stats, result, _ = self._exhaust_generator(gen)
        assert isinstance(html_output, str)
        assert isinstance(stats, str)
        assert isinstance(result, dict)

    def test_run_pipeline_returns_tuple(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Pipeline generator yields 4-tuples of (html, stats, result, tab_switch)."""
        text_input = "Test text."
        gen = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_embedding_model="Qwen/Qwen3-Embedding-4B",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_embedding_model="Qwen/Qwen3-Embedding-4B",
            reranker_top_k=10,
            disambig_type="first",
            llm_model="Qwen/Qwen3-4B",
            lela_thinking=False,
            lela_none_candidate=True,
            kb_type="custom",
            progress=mock_progress,
        )
        output = self._exhaust_generator(gen)
        assert len(output) == 4

    def test_run_pipeline_result_structure(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Result has text and entities keys."""
        text_input = "Barack Obama was president."
        gen = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_embedding_model="Qwen/Qwen3-Embedding-4B",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_embedding_model="Qwen/Qwen3-Embedding-4B",
            reranker_top_k=10,
            disambig_type="first",
            llm_model="Qwen/Qwen3-4B",
            lela_thinking=False,
            lela_none_candidate=True,
            kb_type="custom",
            progress=mock_progress,
        )
        _, _, result, _ = self._exhaust_generator(gen)
        assert "text" in result
        assert "entities" in result

    def test_run_pipeline_no_kb_uses_default(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Without KB file, pipeline uses default YAGO KB."""
        from unittest.mock import patch
        # Mock the downloader to return the sample KB instead of downloading YAGO
        with patch("ner_pipeline.knowledge_bases.yago_downloader.ensure_yago_kb", return_value=mock_kb_file.name):
            gen = run_pipeline(
                text_input="Some text",
                file_input=None,
                kb_file=None,
                loader_type="text",
                ner_type="simple",
                spacy_model="en_core_web_sm",
                gliner_model="urchade/gliner_large",
                gliner_labels="",
                gliner_threshold=0.5,
                simple_min_len=3,
                cand_type="fuzzy",
                cand_embedding_model="Qwen/Qwen3-Embedding-4B",
                cand_top_k=10,
                cand_use_context=True,
                reranker_type="none",
                reranker_embedding_model="Qwen/Qwen3-Embedding-4B",
                reranker_top_k=10,
                disambig_type="first",
                llm_model="Qwen/Qwen3-4B",
                lela_thinking=False,
                lela_none_candidate=True,
                kb_type="custom",
                progress=mock_progress,
            )
            html_output, stats, result, _ = self._exhaust_generator(gen)
            assert isinstance(result, dict)
            assert "text" in result

    def test_run_pipeline_no_input_error(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Returns error without text or file input."""
        gen = run_pipeline(
            text_input="",
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_embedding_model="Qwen/Qwen3-Embedding-4B",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_embedding_model="Qwen/Qwen3-Embedding-4B",
            reranker_top_k=10,
            disambig_type="first",
            llm_model="Qwen/Qwen3-4B",
            lela_thinking=False,
            lela_none_candidate=True,
            kb_type="custom",
            progress=mock_progress,
        )
        html_output, stats, result, _ = self._exhaust_generator(gen)
        assert "error" in result
        assert "Input" in result["error"]

    def test_run_pipeline_html_output_is_string(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """HTML output is a string, not a list."""
        text_input = "Barack Obama was president."
        gen = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_embedding_model="Qwen/Qwen3-Embedding-4B",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_embedding_model="Qwen/Qwen3-Embedding-4B",
            reranker_top_k=10,
            disambig_type="first",
            llm_model="Qwen/Qwen3-4B",
            lela_thinking=False,
            lela_none_candidate=True,
            kb_type="custom",
            progress=mock_progress,
        )
        html_output, _, _, _ = self._exhaust_generator(gen)
        assert isinstance(html_output, str)
        # Should contain HTML markup
        assert "<" in html_output or html_output == ""
